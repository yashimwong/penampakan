from __future__ import annotations

import asyncio
import gc
import threading
import weakref
from typing import cast

import pytest
from PIL import Image

from penampakan import client as client_module
from penampakan.client import AsyncPenampakan
from penampakan.config import AgentSettings, CacheSettings, Settings
from penampakan.errors import (
    BackendError,
    ConfigurationError,
    InvalidImageError,
    OperationTimeoutError,
    SessionClosedError,
)
from penampakan.image.assets import AssetStore
from penampakan.models import (
    AnswerAction,
    BackendDescriptor,
    BackendImage,
    Capability,
    CapabilityDescriptor,
    CaptionPayload,
    CaptionRequest,
    InspectionOperation,
    InspectionPlan,
    InspectionResult,
    LLMRequest,
    LLMResponse,
    MetadataPayload,
    MetadataRequest,
    ObservationDraft,
    PolicyAction,
    PolicyInput,
    TraceEvent,
    VisionAnswer,
    VisionRequest,
    VisionResult,
)
from penampakan.perception.cache import MemoryLRUCache, NullCache, SingleFlightCoordinator
from penampakan.perception.router import BackendRouter
from penampakan.protocols import ActionPolicy, Cache, TraceSink
from penampakan.reasoning import supported_prompt_versions
from penampakan.session import AsyncVisionSession
from tests.fixtures.images import encode_image
from tests.unit.reasoning.helpers import ScriptedPolicy


def _image_bytes(
    size: tuple[int, int] = (6, 4),
    mode: str = "RGBA",
) -> bytes:
    color = (12, 34, 56, 128) if mode == "RGBA" else (12, 34, 56)
    image = Image.new(mode, size, color)
    try:
        return encode_image(image)
    finally:
        image.close()


class RecordingBackend:
    def __init__(
        self,
        name: str,
        capability: Capability = Capability.CAPTION,
        *,
        result: VisionResult | None = None,
        analyze_error: BaseException | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self._descriptor = BackendDescriptor(
            name=name,
            version="1.0",
            capabilities=(CapabilityDescriptor(capability=capability),),
            max_concurrency=2,
        )
        self._capability = capability
        self._result = result
        self.analyze_error = analyze_error
        self.close_error = close_error
        self.requests: list[VisionRequest] = []
        self.close_calls = 0
        self.started = asyncio.Event()
        self.gate: asyncio.Event | None = None

    @property
    def descriptor(self) -> BackendDescriptor:
        return self._descriptor

    def supports(self, request: VisionRequest) -> bool:
        return request.capability is self._capability

    async def analyze(
        self,
        image: BackendImage,
        request: VisionRequest,
    ) -> VisionResult:
        self.requests.append(request)
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        if self.analyze_error is not None:
            raise self.analyze_error
        if self._result is not None:
            return self._result
        if self._capability is Capability.METADATA:
            return VisionResult(
                observations=(
                    ObservationDraft(
                        payload=MetadataPayload(
                            width=999,
                            height=999,
                            aspect_ratio=1.0,
                            has_alpha=False,
                        )
                    ),
                )
            )
        return VisionResult(
            observations=(ObservationDraft(payload=CaptionPayload(text="A test image.")),)
        )

    async def aclose(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class RecordingCache:
    def __init__(self, close_error: BaseException | None = None) -> None:
        self.close_error = close_error
        self.get_calls = 0
        self.set_calls = 0
        self.close_calls = 0

    async def get(self, key: str) -> bytes | None:
        self.get_calls += 1
        return None

    async def set(self, key: str, value: bytes, *, size: int) -> None:
        self.set_calls += 1

    async def aclose(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class RecordingSink:
    def __init__(self, close_error: BaseException | None = None) -> None:
        self.close_error = close_error
        self.events: list[TraceEvent] = []
        self.close_calls = 0

    async def emit(self, event: TraceEvent) -> None:
        self.events.append(event)

    async def aclose(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class MinimalTextLLM:
    async def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(text="{}")


class SlowPolicy:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = 0

    async def next_action(self, input: PolicyInput) -> PolicyAction:
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        raise AssertionError


class FakeLateStore:
    def __init__(self) -> None:
        self.close_calls = 0
        self.closed = threading.Event()

    def close(self) -> None:
        self.close_calls += 1
        self.closed.set()


class FailingOneShotSession:
    def __init__(
        self,
        *,
        inspect_error: BaseException | None = None,
        ask_error: BaseException | None = None,
        block_ask: bool = False,
    ) -> None:
        self.inspect_error = inspect_error
        self.ask_error = ask_error
        self.block_ask = block_ask
        self.ask_started = asyncio.Event()
        self.close_calls = 0

    async def inspect(
        self,
        plan: InspectionPlan | None = None,
        *,
        timeout_s: float | None = None,
    ) -> InspectionResult:
        if self.inspect_error is not None:
            raise self.inspect_error
        raise AssertionError

    async def ask(
        self,
        question: str,
        *,
        timeout_s: float | None = None,
    ) -> VisionAnswer:
        self.ask_started.set()
        if self.block_ask:
            await asyncio.Event().wait()
        if self.ask_error is not None:
            raise self.ask_error
        raise AssertionError

    async def aclose(self) -> None:
        self.close_calls += 1


async def test_constructor_configuration_validation() -> None:
    llm = MinimalTextLLM()
    policy = ScriptedPolicy(())
    unsupported = Settings(
        agent=AgentSettings(prompt_version="agent-v2"),
    )
    cases: tuple[tuple[dict[str, object], str], ...] = (
        ({"settings": object()}, "invalid_settings"),
        ({"llm": llm, "policy": policy}, "llm_and_policy_conflict"),
        ({"llm": object()}, "invalid_llm"),
        ({"policy": object()}, "invalid_policy"),
        ({"cache": object()}, "invalid_cache"),
        ({"trace_sinks": (object(),)}, "invalid_trace_sink"),
        ({"settings": unsupported}, "unsupported_prompt_version"),
    )

    for arguments, code in cases:
        with pytest.raises(ConfigurationError) as captured:
            AsyncPenampakan(**arguments)
        assert captured.value.code == code


async def test_constructor_accepts_every_supported_prompt_version() -> None:
    for version in supported_prompt_versions():
        settings = Settings(agent=AgentSettings(prompt_version=version))
        client = AsyncPenampakan(settings=settings, llm=MinimalTextLLM())
        try:
            assert client.settings.agent.prompt_version == version
        finally:
            await client.aclose()


async def test_settings_are_deep_snapshots_and_cache_selection_is_exact() -> None:
    supplied = Settings(backend_preferences={})
    disabled = AsyncPenampakan(settings=supplied)
    supplied.backend_preferences[Capability.CAPTION] = ("example.future",)
    first_snapshot = disabled.settings
    first_snapshot.backend_preferences[Capability.OCR] = ("example.ocr",)

    assert disabled.settings.backend_preferences == {}
    assert isinstance(disabled._cache, NullCache)

    enabled_settings = Settings(cache=CacheSettings(mode="memory", max_entries=3, max_bytes=4096))
    enabled = AsyncPenampakan(settings=enabled_settings)
    assert isinstance(enabled._cache, MemoryLRUCache)
    assert enabled._cache.max_entries == 3
    assert enabled._cache.max_bytes == 4096

    custom_cache = RecordingCache()
    custom = AsyncPenampakan(cache=custom_cache)
    assert custom._cache is custom_cache

    await asyncio.gather(disabled.aclose(), enabled.aclose(), custom.aclose())

    assert custom_cache.close_calls == 1


async def test_caller_backend_ownership_is_exclusive_and_released() -> None:
    backend = RecordingBackend("example.owned")
    first = AsyncPenampakan(backends=(backend,))

    with pytest.raises(ConfigurationError) as live_conflict:
        AsyncPenampakan(backends=(backend,))
    with pytest.raises(ConfigurationError) as duplicate:
        AsyncPenampakan(backends=(backend, backend))

    assert live_conflict.value.code == "backend_already_owned"
    assert duplicate.value.code == "duplicate_backend_instance"

    await asyncio.gather(first.aclose(), first.aclose(), first.aclose())
    second = AsyncPenampakan(backends=(backend,))
    await second.aclose()

    assert backend.close_calls == 2


async def test_constructor_router_failure_releases_backend_claims() -> None:
    first = RecordingBackend("example.same")
    duplicate_name = RecordingBackend("example.same")

    with pytest.raises(ConfigurationError) as captured:
        AsyncPenampakan(backends=(first, duplicate_name))

    assert captured.value.code == "duplicate_backend_name"

    client = AsyncPenampakan(backends=(first,))
    await client.aclose()

    assert first.close_calls == 1


async def test_pillow_metadata_is_authoritative_over_caller_backend() -> None:
    caller = RecordingBackend("example.metadata", Capability.METADATA)
    client = AsyncPenampakan(backends=(caller,))
    plan = InspectionPlan(
        operations=(InspectionOperation(request=MetadataRequest(), required=True),),
        include_available_overview=False,
    )

    result = await client.inspect(_image_bytes(), plan)

    payload = result.observations[0].payload
    assert isinstance(payload, MetadataPayload)
    assert payload.width == 6
    assert payload.height == 4
    assert payload.aspect_ratio == 1.5
    assert payload.has_alpha is True
    assert result.observations[0].provenance.backend_name == "penampakan.pillow"
    assert caller.requests == []

    await client.aclose()


async def test_open_image_and_context_managers_close_session_and_client() -> None:
    client = AsyncPenampakan()

    async with client as entered:
        assert entered is client
        session = await client.open_image(_image_bytes((3, 2), "RGB"))
        async with session as entered_session:
            assert entered_session is session
            assert session.root_asset.width == 3
            assert session.root_asset.height == 2
            assert session in client._sessions
        assert session.closed is True
        assert session not in client._sessions

        with pytest.raises(InvalidImageError):
            await client.open_image(b"not an image")
        replacement = await client.open_image(_image_bytes())
        await replacement.aclose()

    assert client.closed is True
    with pytest.raises(SessionClosedError):
        await client.open_image(_image_bytes())


async def test_one_shot_results_are_deep_detached_before_session_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action = AnswerAction(
        status="insufficient_evidence",
        answer="The requested fact is not established by the available observations.",
        uncertainties=("No semantic caption or text observation is available.",),
    )
    policy = ScriptedPolicy((action,))
    client = AsyncPenampakan(
        policy=policy,
        settings=Settings(agent=AgentSettings(initial_capabilities=(Capability.METADATA,))),
    )
    captured_inspections: list[InspectionResult] = []
    captured_answers: list[VisionAnswer] = []
    original_inspect = AsyncVisionSession.inspect
    original_ask = AsyncVisionSession.ask

    async def inspect_and_capture(
        self: AsyncVisionSession,
        plan: InspectionPlan | None = None,
        *,
        timeout_s: float | None = None,
    ) -> InspectionResult:
        result = await original_inspect(self, plan, timeout_s=timeout_s)
        captured_inspections.append(result)
        return result

    async def ask_and_capture(
        self: AsyncVisionSession,
        question: str,
        *,
        timeout_s: float | None = None,
    ) -> VisionAnswer:
        result = await original_ask(self, question, timeout_s=timeout_s)
        captured_answers.append(result)
        return result

    monkeypatch.setattr(AsyncVisionSession, "inspect", inspect_and_capture)
    monkeypatch.setattr(AsyncVisionSession, "ask", ask_and_capture)
    plan = InspectionPlan(
        operations=(InspectionOperation(request=MetadataRequest()),),
        include_available_overview=False,
    )

    inspection = await client.inspect(_image_bytes(), plan)
    answer = await client.ask(_image_bytes(), "What does the image say?")

    assert inspection == captured_inspections[0]
    assert inspection is not captured_inspections[0]
    assert inspection.root_asset is not captured_inspections[0].root_asset
    assert inspection.trace is not captured_inspections[0].trace
    assert answer == captured_answers[0]
    assert answer is not captured_answers[0]
    assert answer.trace is not captured_answers[0].trace
    assert inspection.root_asset.width == 6
    assert answer.answer.startswith("The requested fact")
    assert client._sessions == set()

    await client.aclose()


async def test_one_shot_inspect_and_ask_errors_always_close_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspect_error = RuntimeError("inspection sentinel")
    ask_error = RuntimeError("ask sentinel")
    inspect_session = FailingOneShotSession(inspect_error=inspect_error)
    ask_session = FailingOneShotSession(ask_error=ask_error)
    sessions = [inspect_session, ask_session]
    client = AsyncPenampakan(policy=ScriptedPolicy(()))

    async def create_session(source: object, deadline: float) -> FailingOneShotSession:
        return sessions.pop(0)

    monkeypatch.setattr(client, "_create_session_before", create_session)

    with pytest.raises(RuntimeError) as inspected:
        await client.inspect(b"unused")
    with pytest.raises(RuntimeError) as asked:
        await client.ask(b"unused", "question")

    assert inspected.value is inspect_error
    assert asked.value is ask_error
    assert inspect_session.close_calls == 1
    assert ask_session.close_calls == 1

    await client.aclose()


async def test_one_shot_cancellation_and_timeout_close_private_sessions() -> None:
    settings = Settings(agent=AgentSettings(initial_capabilities=(Capability.METADATA,)))
    cancellation_policy = SlowPolicy()
    cancelled_client = AsyncPenampakan(policy=cancellation_policy, settings=settings)
    cancelled = asyncio.create_task(cancelled_client.ask(_image_bytes(), "What is visible?"))
    await asyncio.wait_for(cancellation_policy.started.wait(), timeout=1.0)
    cancelled.cancel()

    with pytest.raises(asyncio.CancelledError):
        await cancelled

    assert cancellation_policy.cancelled == 1
    assert cancelled_client._sessions == set()

    timeout_policy = SlowPolicy()
    timeout_client = AsyncPenampakan(policy=timeout_policy, settings=settings)

    with pytest.raises(OperationTimeoutError):
        await timeout_client.ask(
            _image_bytes(),
            "What is visible?",
            timeout_s=0.02,
        )

    assert timeout_policy.cancelled == 1
    assert timeout_client._sessions == set()

    await asyncio.gather(cancelled_client.aclose(), timeout_client.aclose())


async def test_timed_out_threaded_load_closes_late_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = AsyncPenampakan()
    started = threading.Event()
    release = threading.Event()
    store = FakeLateStore()

    def load_store(source: object) -> tuple[AssetStore, tuple[()]]:
        started.set()
        release.wait(timeout=1.0)
        return cast(AssetStore, store), ()

    monkeypatch.setattr(client, "_load_store", load_store)
    operation = asyncio.create_task(client.inspect(b"unused", timeout_s=0.02))
    assert await asyncio.to_thread(started.wait, 1.0)

    try:
        with pytest.raises(OperationTimeoutError):
            await operation
    finally:
        release.set()

    assert await asyncio.to_thread(store.closed.wait, 1.0)
    assert store.close_calls == 1
    assert client._active_operations == 0
    assert client._sessions == set()

    await client.aclose()


async def test_client_close_drains_multiple_sessions_and_is_concurrently_idempotent() -> None:
    backend = RecordingBackend("example.slow")
    backend.gate = asyncio.Event()
    settings = Settings(agent=AgentSettings(initial_capabilities=()))
    client = AsyncPenampakan(backends=(backend,), settings=settings)
    first = await client.open_image(_image_bytes())
    second = await client.open_image(_image_bytes())
    plan = InspectionPlan(
        operations=(InspectionOperation(request=CaptionRequest(), required=True),),
        include_available_overview=False,
    )
    operation = asyncio.create_task(first.inspect(plan))
    await asyncio.wait_for(backend.started.wait(), timeout=1.0)
    closers = [asyncio.create_task(client.aclose()) for _ in range(5)]
    await asyncio.sleep(0)

    assert not all(task.done() for task in closers)

    backend.gate.set()
    result = await operation
    await asyncio.gather(*closers)

    assert isinstance(result.observations[0].payload, CaptionPayload)
    assert first.closed is True
    assert second.closed is True
    assert client.closed is True
    assert backend.close_calls == 1
    with pytest.raises(SessionClosedError):
        first.get_asset(first.root_asset.id)

    await client.aclose()
    assert backend.close_calls == 1


async def test_backend_cache_and_sink_close_failures_are_best_effort() -> None:
    backend = RecordingBackend(
        "example.close_failure",
        close_error=RuntimeError("backend close sentinel"),
    )
    cache = RecordingCache(RuntimeError("cache close sentinel"))
    failing_sink = RecordingSink(RuntimeError("sink close sentinel"))
    succeeding_sink = RecordingSink()
    client = AsyncPenampakan(
        backends=(backend,),
        cache=cache,
        trace_sinks=(failing_sink, succeeding_sink),
    )

    await asyncio.gather(client.aclose(), client.aclose(), client.aclose())

    assert client.closed is True
    assert backend.close_calls == 1
    assert cache.close_calls == 1
    assert failing_sink.close_calls == 1
    assert succeeding_sink.close_calls == 1

    successor = AsyncPenampakan(backends=(backend,))
    await successor.aclose()

    assert backend.close_calls == 2


class RecordingCloser:
    """An owned resource that records its close attempt and may fail."""

    def __init__(
        self,
        name: str,
        log: list[str],
        error: BaseException | None = None,
        *,
        started: asyncio.Event | None = None,
        gate: asyncio.Event | None = None,
    ) -> None:
        self.name = name
        self.log = log
        self.error = error
        self.started = started
        self.gate = gate
        self.close_calls = 0
        self.finished = asyncio.Event()

    async def aclose(self) -> None:
        self.close_calls += 1
        self.log.append(self.name)
        if self.started is not None:
            self.started.set()
        try:
            if self.gate is not None:
                await self.gate.wait()
            if self.error is not None:
                raise self.error
        finally:
            self.finished.set()


class CancellingCloser(RecordingCloser):
    """An owned resource whose close aborts the running close task itself.

    Cancelling from inside the task leaves the request pending until the
    coroutine returns, so the internal close task ends in the cancelled state
    while every remaining close step still runs. That is the state an outer
    ``TaskGroup`` abort or an ``asyncio.run`` shutdown leaves behind.
    """

    async def aclose(self) -> None:
        await super().aclose()
        running = asyncio.current_task()
        assert running is not None
        running.cancel()


class RetainedBaseException(BaseException):
    """A base exception a task hands to its awaiting caller unchanged.

    ``KeyboardInterrupt`` and ``SystemExit`` are handed to the event loop by
    ``asyncio.Task`` instead, so neither can stand in for a retained primary
    exception that a caller is expected to observe.
    """


class RecordingIdle:
    """An idle gate that records the drain attempt and may fail."""

    def __init__(self, log: list[str], error: BaseException | None = None) -> None:
        self.log = log
        self.error = error

    async def wait(self) -> bool:
        self.log.append("active_operations")
        if self.error is not None:
            raise self.error
        return True


_CLOSE_POSITIONS = (
    "active_operations",
    "session",
    "singleflight",
    "router",
    "cache",
    "trace_sink_0",
)

_CLOSE_POSITIONS_WITH_POLICY = (
    "active_operations",
    "session",
    "singleflight",
    "router",
    "policy",
    "cache",
    "trace_sink_0",
)


def _instrument_close(
    client: AsyncPenampakan,
    failures: dict[str, BaseException],
) -> list[str]:
    log: list[str] = []
    client._idle = cast(asyncio.Event, RecordingIdle(log, failures.get("active_operations")))
    session = RecordingCloser("session", log, failures.get("session"))
    client._sessions.add(cast(AsyncVisionSession, session))
    client._singleflight = cast(
        SingleFlightCoordinator[bytes],
        RecordingCloser("singleflight", log, failures.get("singleflight")),
    )
    client._router = cast(BackendRouter, RecordingCloser("router", log, failures.get("router")))
    if client._owns_policy and callable(getattr(client._policy, "aclose", None)):
        client._policy = cast(
            ActionPolicy,
            RecordingCloser("policy", log, failures.get("policy")),
        )
    client._cache = cast(Cache, RecordingCloser("cache", log, failures.get("cache")))
    client._trace_sinks = (
        cast(TraceSink, RecordingCloser("trace_sink_0", log, failures.get("trace_sink_0"))),
    )
    return log


@pytest.mark.parametrize("position", _CLOSE_POSITIONS)
async def test_ordinary_close_failure_warns_and_attempts_every_remaining_resource(
    position: str,
) -> None:
    backend = RecordingBackend(f"example.ordinary_{position}")
    client = AsyncPenampakan(backends=(backend,))
    log = _instrument_close(client, {position: RuntimeError("close sentinel")})

    await client.aclose()

    assert log == list(_CLOSE_POSITIONS)
    assert client.closed is True
    assert [warning.details["resource"] for warning in client.close_warnings] == [position]
    assert client.close_warnings[0].code == "owned_resource_close_failed"
    assert client.close_warnings[0].details["error_type"] == "RuntimeError"
    assert id(backend) not in client_module._BACKEND_OWNERS


@pytest.mark.parametrize("position", _CLOSE_POSITIONS)
@pytest.mark.parametrize(
    "factory",
    (asyncio.CancelledError, KeyboardInterrupt, SystemExit, GeneratorExit),
    ids=("cancelled", "keyboard_interrupt", "system_exit", "generator_exit"),
)
async def test_base_close_failure_propagates_after_complete_cleanup(
    position: str,
    factory: type[BaseException],
) -> None:
    backend = RecordingBackend(f"example.base_{factory.__name__.lower()}_{position}")
    client = AsyncPenampakan(backends=(backend,))
    injected = factory()
    log = _instrument_close(client, {position: injected})

    with pytest.raises(factory) as captured:
        await client.aclose()

    assert captured.value is injected
    assert log == list(_CLOSE_POSITIONS)
    assert client.closed is True
    assert client.close_warnings == ()
    assert id(backend) not in client_module._BACKEND_OWNERS

    successor = AsyncPenampakan(backends=(backend,))
    await successor.aclose()


async def test_first_base_exception_survives_later_close_failures() -> None:
    client = AsyncPenampakan()
    primary = asyncio.CancelledError()
    log = _instrument_close(
        client,
        {
            "session": primary,
            "router": SystemExit(),
            "cache": RuntimeError("late ordinary sentinel"),
        },
    )

    with pytest.raises(asyncio.CancelledError) as captured:
        await client.aclose()

    assert captured.value is primary
    assert log == list(_CLOSE_POSITIONS)
    assert [warning.details["resource"] for warning in client.close_warnings] == ["cache"]


async def _close_outcome(client: AsyncPenampakan) -> BaseException | None:
    try:
        await client.aclose()
    except BaseException as error:
        return error
    return None


async def test_concurrent_and_repeated_closes_share_one_attempt_and_outcome() -> None:
    client = AsyncPenampakan()
    primary = KeyboardInterrupt()
    log = _instrument_close(client, {"singleflight": primary})

    # A task that lets KeyboardInterrupt escape hands it to the event loop, so
    # each concurrent close reports its own outcome from inside its coroutine.
    results = await asyncio.gather(*(_close_outcome(client) for _ in range(3)))
    repeated = await asyncio.gather(*(_close_outcome(client) for _ in range(1)))

    assert log == list(_CLOSE_POSITIONS)
    assert all(result is primary for result in (*results, *repeated))


async def test_cancelled_caller_waits_for_protected_cleanup_then_re_raises() -> None:
    client = AsyncPenampakan()
    started = asyncio.Event()
    gate = asyncio.Event()
    log: list[str] = []
    client._singleflight = cast(
        SingleFlightCoordinator[bytes],
        RecordingCloser("singleflight", log, started=started, gate=gate),
    )
    client._router = cast(BackendRouter, RecordingCloser("router", log))
    client._cache = cast(Cache, RecordingCloser("cache", log))
    closer = asyncio.create_task(client.aclose())
    await asyncio.wait_for(started.wait(), timeout=1.0)
    closer.cancel()
    for _ in range(5):
        await asyncio.sleep(0)

    assert not closer.done()
    assert client.closed is False

    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await closer

    assert log == ["singleflight", "router", "cache"]
    assert client.closed is True


async def test_a_cancelled_close_task_still_reports_the_retained_primary() -> None:
    backend = RecordingBackend("example.cancelled_close_task")
    client = AsyncPenampakan(backends=(backend,))
    primary = RetainedBaseException()
    log = _instrument_close(client, {"session": primary})
    client._router = cast(BackendRouter, CancellingCloser("router", log))

    with pytest.raises(RetainedBaseException) as first:
        await client.aclose()
    with pytest.raises(RetainedBaseException) as repeated:
        await client.aclose()

    assert client._close_task is not None
    assert client._close_task.cancelled() is True
    assert first.value is primary
    assert repeated.value is primary
    assert log == list(_CLOSE_POSITIONS)
    assert client.closed is True
    assert client.close_warnings == ()
    assert id(backend) not in client_module._BACKEND_OWNERS


async def test_a_base_policy_close_failure_still_attempts_every_later_resource() -> None:
    policy = ClosablePolicy()
    client = AsyncPenampakan(policy=policy, owns_policy=True)
    primary = RetainedBaseException()
    log = _instrument_close(client, {"policy": primary})

    with pytest.raises(RetainedBaseException) as captured:
        await client.aclose()

    assert captured.value is primary
    assert log == list(_CLOSE_POSITIONS_WITH_POLICY)
    assert log[-2:] == ["cache", "trace_sink_0"]
    assert client.closed is True
    assert client.close_warnings == ()


def _install_sessions(client: AsyncPenampakan, sessions: tuple[RecordingCloser, ...]) -> None:
    client._sessions.update(cast(AsyncVisionSession, session) for session in sessions)


async def test_session_close_failures_aggregate_across_every_owned_session() -> None:
    client = AsyncPenampakan()
    log: list[str] = []
    primary = RetainedBaseException()
    ordinary = RecordingCloser("session_ordinary", log, RuntimeError("session close sentinel"))
    base = RecordingCloser("session_base", log, primary)
    _install_sessions(client, (ordinary, base))

    with pytest.raises(RetainedBaseException) as captured:
        await client.aclose()

    assert captured.value is primary
    assert sorted(log) == ["session_base", "session_ordinary"]
    assert ordinary.close_calls == 1
    assert base.close_calls == 1
    assert [warning.details["resource"] for warning in client.close_warnings] == ["session"]
    assert client.close_warnings[0].details["error_type"] == "RuntimeError"
    assert client.closed is True


async def test_a_cancelled_session_gather_keeps_the_failures_already_reported() -> None:
    client = AsyncPenampakan()
    log: list[str] = []
    ordinary = RecordingCloser("session_ordinary", log, RuntimeError("session close sentinel"))
    started = asyncio.Event()
    gate = asyncio.Event()
    parked = RecordingCloser("session_parked", log, started=started, gate=gate)
    _install_sessions(client, (ordinary, parked))
    closer = asyncio.create_task(client.aclose())
    await asyncio.wait_for(ordinary.finished.wait(), timeout=1.0)
    await asyncio.wait_for(started.wait(), timeout=1.0)
    assert client._close_task is not None
    client._close_task.cancel()
    gate.set()

    with pytest.raises(asyncio.CancelledError):
        await closer

    assert sorted(log) == ["session_ordinary", "session_parked"]
    assert parked.finished.is_set() is True
    assert [warning.details["resource"] for warning in client.close_warnings] == ["session"]
    assert client.close_warnings[0].details["error_type"] == "RuntimeError"
    assert client.closed is True


async def test_a_base_failure_at_one_trace_sink_still_attempts_the_next() -> None:
    client = AsyncPenampakan()
    primary = RetainedBaseException()
    log = _instrument_close(client, {})
    failing = RecordingCloser("trace_sink_0", log, primary)
    successor = RecordingCloser("trace_sink_1", log)
    client._trace_sinks = (cast(TraceSink, failing), cast(TraceSink, successor))

    with pytest.raises(RetainedBaseException) as captured:
        await client.aclose()

    assert captured.value is primary
    assert log == [*_CLOSE_POSITIONS, "trace_sink_1"]
    assert successor.close_calls == 1
    assert client.closed is True
    assert client.close_warnings == ()


async def test_a_failing_ownership_release_warns_without_hiding_the_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = RecordingBackend("example.release_failure")
    client = AsyncPenampakan(backends=(backend,))
    primary = RetainedBaseException()
    log = _instrument_close(client, {"router": primary})

    def release_then_fail() -> None:
        AsyncPenampakan._release_backend_ownership(client)
        raise RuntimeError("ownership release sentinel")

    monkeypatch.setattr(client, "_release_backend_ownership", release_then_fail)

    with pytest.raises(RetainedBaseException) as captured:
        await client.aclose()

    assert captured.value is primary
    assert log == list(_CLOSE_POSITIONS)
    assert client.closed is True
    assert [warning.details["resource"] for warning in client.close_warnings] == [
        "backend_ownership"
    ]
    assert client.close_warnings[0].details["error_type"] == "RuntimeError"
    assert id(backend) not in client_module._BACKEND_OWNERS


class UnhashableBackend:
    """A backend that is unhashable and equal to everything."""

    __hash__ = None  # type: ignore[assignment]

    def __init__(self, name: str) -> None:
        self._descriptor = BackendDescriptor(
            name=name,
            version="1.0",
            capabilities=(CapabilityDescriptor(capability=Capability.CAPTION),),
        )
        self.close_calls = 0

    def __eq__(self, other: object) -> bool:
        return True

    @property
    def descriptor(self) -> BackendDescriptor:
        return self._descriptor

    def supports(self, request: VisionRequest) -> bool:
        return request.capability is Capability.CAPTION

    async def analyze(self, image: BackendImage, request: VisionRequest) -> VisionResult:
        raise AssertionError

    async def aclose(self) -> None:
        self.close_calls += 1


class SlottedBackend:
    """A backend whose layout offers no weak-reference support."""

    __slots__ = ("_descriptor", "close_calls")

    def __init__(self, name: str) -> None:
        self._descriptor = BackendDescriptor(
            name=name,
            version="1.0",
            capabilities=(CapabilityDescriptor(capability=Capability.CAPTION),),
        )
        self.close_calls = 0

    @property
    def descriptor(self) -> BackendDescriptor:
        return self._descriptor

    def supports(self, request: VisionRequest) -> bool:
        return request.capability is Capability.CAPTION

    async def analyze(self, image: BackendImage, request: VisionRequest) -> VisionResult:
        raise AssertionError

    async def aclose(self) -> None:
        self.close_calls += 1


async def test_ownership_uses_identity_not_equality_or_hashing() -> None:
    first_backend = UnhashableBackend("example.unhashable_first")
    second_backend = UnhashableBackend("example.unhashable_second")
    client = AsyncPenampakan(backends=(first_backend, second_backend))

    with pytest.raises(ConfigurationError) as duplicate:
        AsyncPenampakan(backends=(first_backend, first_backend))
    with pytest.raises(ConfigurationError) as shared:
        AsyncPenampakan(backends=(second_backend,))

    assert duplicate.value.code == "duplicate_backend_instance"
    assert shared.value.code == "backend_already_owned"

    await client.aclose()

    assert first_backend.close_calls == 1
    assert second_backend.close_calls == 1
    assert id(first_backend) not in client_module._BACKEND_OWNERS
    assert id(second_backend) not in client_module._BACKEND_OWNERS


async def test_backend_without_weak_reference_support_is_accepted() -> None:
    backend = SlottedBackend("example.slotted")
    first = AsyncPenampakan(backends=(backend,))
    # The cross-client convenience guard needs a weak reference, so it is
    # skipped rather than rejecting an otherwise valid backend.
    second = AsyncPenampakan(backends=(backend,))

    with pytest.raises(ConfigurationError) as duplicate:
        AsyncPenampakan(backends=(backend, backend))

    assert duplicate.value.code == "duplicate_backend_instance"
    assert id(backend) not in client_module._BACKEND_OWNERS

    await asyncio.gather(first.aclose(), second.aclose())

    assert backend.close_calls == 2


async def test_a_slotted_backend_mixes_with_a_weak_referenceable_one() -> None:
    slotted = SlottedBackend("example.mixed_slotted")
    normal = RecordingBackend("example.mixed_normal")
    client = AsyncPenampakan(backends=(slotted, normal))

    with pytest.raises(ConfigurationError) as duplicate:
        AsyncPenampakan(backends=(slotted, slotted))
    with pytest.raises(ConfigurationError) as shared:
        AsyncPenampakan(backends=(normal,))

    assert duplicate.value.code == "duplicate_backend_instance"
    assert shared.value.code == "backend_already_owned"
    # Only the weak-referenceable backend takes part in the cross-client guard.
    assert id(slotted) not in client_module._BACKEND_OWNERS
    assert client_module._BACKEND_OWNERS[id(normal)].backend_ref() is normal

    await client.aclose()

    assert slotted.close_calls == 1
    assert normal.close_calls == 1
    assert id(normal) not in client_module._BACKEND_OWNERS


async def test_a_rejected_claim_leaves_every_other_backend_unclaimed() -> None:
    owned = RecordingBackend("example.claim_owned")
    fresh = RecordingBackend("example.claim_fresh")
    owner = AsyncPenampakan(backends=(owned,))

    with pytest.raises(ConfigurationError) as captured:
        AsyncPenampakan(backends=(fresh, owned))

    assert captured.value.code == "backend_already_owned"
    assert id(fresh) not in client_module._BACKEND_OWNERS

    successor = AsyncPenampakan(backends=(fresh,))
    await asyncio.gather(owner.aclose(), successor.aclose())

    assert owned.close_calls == 1
    assert fresh.close_calls == 1
    assert id(owned) not in client_module._BACKEND_OWNERS
    assert id(fresh) not in client_module._BACKEND_OWNERS


async def test_a_constructor_failure_after_the_claim_releases_the_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = RecordingBackend("example.constructor_failure")

    def unavailable_backend() -> object:
        raise RuntimeError("pillow backend sentinel")

    monkeypatch.setattr(client_module, "PillowBackend", unavailable_backend)

    with pytest.raises(RuntimeError):
        AsyncPenampakan(backends=(backend,))

    assert id(backend) not in client_module._BACKEND_OWNERS

    monkeypatch.undo()
    successor = AsyncPenampakan(backends=(backend,))
    await successor.aclose()

    assert backend.close_calls == 1
    assert id(backend) not in client_module._BACKEND_OWNERS


async def test_an_authoritative_preference_conflict_releases_the_backend() -> None:
    backend = RecordingBackend("example.preference_conflict")
    settings = Settings(backend_preferences={Capability.METADATA: ("example.other",)})

    with pytest.raises(ConfigurationError) as captured:
        AsyncPenampakan(backends=(backend,), settings=settings)

    assert captured.value.code == "authoritative_backend_preference"
    assert id(backend) not in client_module._BACKEND_OWNERS

    successor = AsyncPenampakan(backends=(backend,))
    await successor.aclose()

    assert backend.close_calls == 1
    assert id(backend) not in client_module._BACKEND_OWNERS


async def test_owner_collected_without_close_releases_the_claim() -> None:
    backend = RecordingBackend("example.collected_owner")
    abandoned = AsyncPenampakan(backends=(backend,))
    key = id(backend)

    assert client_module._BACKEND_OWNERS[key].owner_ref() is abandoned

    del abandoned
    gc.collect()
    successor = AsyncPenampakan(backends=(backend,))

    assert client_module._BACKEND_OWNERS[key].owner_ref() is successor

    await successor.aclose()

    assert key not in client_module._BACKEND_OWNERS
    assert backend.close_calls == 1


async def test_stale_address_entry_never_rejects_an_unrelated_backend() -> None:
    owner = AsyncPenampakan()
    backend = RecordingBackend("example.address_reuse")
    unrelated = RecordingBackend("example.address_previous")
    key = id(backend)
    stale_token = object()
    client_module._BACKEND_OWNERS[key] = client_module._OwnershipEntry(
        backend_ref=weakref.ref(unrelated),
        owner_ref=weakref.ref(owner),
        token=stale_token,
    )
    try:
        client = AsyncPenampakan(backends=(backend,))
        entry = client_module._BACKEND_OWNERS[key]

        assert entry.backend_ref() is backend
        assert entry.token is not stale_token

        await client.aclose()

        assert key not in client_module._BACKEND_OWNERS
    finally:
        client_module._BACKEND_OWNERS.pop(key, None)
        await owner.aclose()


async def test_stale_weakref_callback_never_evicts_a_newer_owner() -> None:
    backend = RecordingBackend("example.token_race")
    client = AsyncPenampakan(backends=(backend,))
    key = id(backend)
    entry = client_module._BACKEND_OWNERS[key]

    client_module._forget_backend(key, object(), None)

    assert client_module._BACKEND_OWNERS[key] is entry

    client_module._forget_backend(key, entry.token, None)

    assert key not in client_module._BACKEND_OWNERS

    await client.aclose()

    assert backend.close_calls == 1


async def test_close_warnings_carry_only_redacted_library_details() -> None:
    class SecretError(Exception):
        code = "s3cr3t-connection-string"

    client = AsyncPenampakan()
    _instrument_close(
        client,
        {
            "router": SecretError("secret close sentinel"),
            "cache": BackendError(code="backend_close_failed"),
        },
    )

    await client.aclose()

    warnings = client.close_warnings
    assert [warning.details["resource"] for warning in warnings] == ["router", "cache"]
    assert "error_code" not in warnings[0].details
    assert warnings[0].details["error_type"] == "SecretError"
    assert warnings[1].details["error_code"] == "backend_close_failed"
    assert all("secret" not in warning.model_dump_json() for warning in warnings)


class ClosableTextLLM:
    """A minimal language model that records close requests."""

    def __init__(self) -> None:
        self.close_calls = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Return one syntactically valid but unused response."""

        return LLMResponse(text="{}")

    async def aclose(self) -> None:
        """Record one close request."""

        self.close_calls += 1


class ClosablePolicy(ScriptedPolicy):
    """A scripted policy that records close requests."""

    def __init__(self) -> None:
        super().__init__(())
        self.close_calls = 0

    async def aclose(self) -> None:
        """Record one close request."""

        self.close_calls += 1


async def test_convenience_language_model_stays_caller_owned_by_default() -> None:
    llm = ClosableTextLLM()
    client = AsyncPenampakan(llm=llm)

    await client.aclose()

    assert client.closed is True
    assert llm.close_calls == 0
    assert client.close_warnings == ()


async def test_convenience_path_cascades_to_an_owned_language_model() -> None:
    llm = ClosableTextLLM()
    client = AsyncPenampakan(llm=llm, owns_llm=True)

    await client.aclose()

    assert llm.close_calls == 1


async def test_caller_supplied_policy_is_never_closed_implicitly() -> None:
    policy = ClosablePolicy()
    client = AsyncPenampakan(policy=policy)

    await client.aclose()

    assert policy.close_calls == 0


async def test_owned_policy_is_closed_exactly_once_in_dependency_order() -> None:
    policy = ClosablePolicy()
    client = AsyncPenampakan(policy=policy, owns_policy=True)
    positions = [name for name, _ in client._close_sequence()]

    await client.aclose()
    await client.aclose()

    assert positions.index("policy") > positions.index("session")
    assert positions.index("policy") < positions.index("cache")
    assert policy.close_calls == 1


async def test_owned_policy_close_failure_is_recorded_and_cleanup_continues() -> None:
    class FailingPolicy(ClosablePolicy):
        async def aclose(self) -> None:
            await super().aclose()
            raise RuntimeError("policy close sentinel")

    policy = FailingPolicy()
    client = AsyncPenampakan(policy=policy, owns_policy=True)

    await client.aclose()

    assert client.closed is True
    assert policy.close_calls == 1
    assert [warning.details["resource"] for warning in client.close_warnings] == ["policy"]


async def test_a_policy_without_aclose_is_skipped_when_owned() -> None:
    policy = ScriptedPolicy(())
    client = AsyncPenampakan(policy=policy, owns_policy=True)

    await client.aclose()

    assert client.closed is True
    assert client.close_warnings == ()


async def test_ownership_flags_require_the_matching_resource() -> None:
    with pytest.raises(ConfigurationError) as missing_llm:
        AsyncPenampakan(owns_llm=True)
    assert missing_llm.value.code == "invalid_ownership"

    with pytest.raises(ConfigurationError) as missing_policy:
        AsyncPenampakan(owns_policy=True)
    assert missing_policy.value.code == "invalid_ownership"

    with pytest.raises(ConfigurationError) as wrong_type:
        AsyncPenampakan(llm=ClosableTextLLM(), owns_llm=cast(bool, "yes"))
    assert wrong_type.value.code == "invalid_ownership"
