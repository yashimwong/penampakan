from __future__ import annotations

import asyncio
import threading
from typing import cast

import pytest
from PIL import Image

from penampakan.client import AsyncPenampakan
from penampakan.config import AgentSettings, CacheSettings, Settings
from penampakan.errors import (
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
from penampakan.perception.cache import MemoryLRUCache, NullCache
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


async def test_settings_are_deep_snapshots_and_cache_selection_is_exact() -> None:
    supplied = Settings(backend_preferences={})
    disabled = AsyncPenampakan(settings=supplied)
    supplied.backend_preferences[Capability.CAPTION] = ("example.future",)
    first_snapshot = disabled.settings
    first_snapshot.backend_preferences[Capability.OCR] = ("example.ocr",)

    assert disabled.settings.backend_preferences == {}
    assert isinstance(disabled._cache, NullCache)

    enabled_settings = Settings(cache=CacheSettings(enabled=True, max_entries=3, max_bytes=4096))
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
