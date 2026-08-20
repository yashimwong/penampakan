from __future__ import annotations

import asyncio
import gc
import sys
import threading
import types
import warnings
from collections.abc import Callable, Coroutine, Iterator
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TypeVar, cast

import pytest

from penampakan.config import Settings
from penampakan.errors import SessionClosedError, SyncInAsyncContextError
from penampakan.models import (
    ImageAsset,
    InspectionResult,
    Observation,
    VisionAnswer,
)
from penampakan.protocols import ActionPolicy, Cache, TextLLM, TraceSink, VisionBackend
from penampakan.sync import Penampakan, VisionSession

_T = TypeVar("_T")
_ROOT_ASSET = cast(ImageAsset, object())
_DERIVED_ASSET = cast(ImageAsset, object())
_OBSERVATION = cast(Observation, object())
_INSPECTION = cast(InspectionResult, object())
_ANSWER = cast(VisionAnswer, object())


def _loop_thread_ids() -> set[int]:
    return {
        thread.ident
        for thread in threading.enumerate()
        if thread.name == "penampakan-sync-loop" and thread.ident is not None
    }


def _concurrently(
    call: Callable[[], _T],
    *,
    count: int = 6,
) -> tuple[list[_T], list[BaseException]]:
    barrier = threading.Barrier(count)
    results: list[_T] = []
    errors: list[BaseException] = []
    result_lock = threading.Lock()

    def invoke() -> None:
        barrier.wait()
        try:
            result = call()
        except BaseException as error:
            with result_lock:
                errors.append(error)
        else:
            with result_lock:
                results.append(result)

    threads = [threading.Thread(target=invoke) for _ in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3.0)
    assert all(not thread.is_alive() for thread in threads)
    return results, errors


class FakeAsyncSession:
    def __init__(self, owner: FakeAsyncClient) -> None:
        self._owner = owner
        self.close_calls = 0
        self.closed = False
        self.coroutine_creations = 0
        self.close_delay = 0.0

    @property
    def root_asset(self) -> ImageAsset:
        self._owner.record("session.root_asset")
        return _ROOT_ASSET

    @property
    def assets(self) -> tuple[ImageAsset, ...]:
        self._owner.record("session.assets")
        return _ROOT_ASSET, _DERIVED_ASSET

    @property
    def observations(self) -> tuple[Observation, ...]:
        self._owner.record("session.observations")
        return (_OBSERVATION,)

    def inspect(
        self,
        plan: object = None,
        *,
        timeout_s: float | None = None,
    ) -> Coroutine[object, object, InspectionResult]:
        return self._operation("session.inspect", _INSPECTION)

    def ask(
        self,
        question: str,
        *,
        timeout_s: float | None = None,
    ) -> Coroutine[object, object, VisionAnswer]:
        return self._operation("session.ask", _ANSWER)

    def get_asset(self, asset_id: str) -> ImageAsset:
        self._owner.record("session.get_asset")
        return _DERIVED_ASSET

    def get_observation(self, observation_id: str) -> Observation:
        self._owner.record("session.get_observation")
        return _OBSERVATION

    def aclose(self) -> Coroutine[object, object, None]:
        self.coroutine_creations += 1

        async def close() -> None:
            self._owner.record("session.aclose")
            if self.closed:
                return
            if self.close_delay:
                await asyncio.sleep(self.close_delay)
            self.close_calls += 1
            self.closed = True

        return close()

    def __aenter__(self) -> Coroutine[object, object, FakeAsyncSession]:
        return self._operation("session.__aenter__", self)

    def _operation(
        self,
        name: str,
        result: _T,
    ) -> Coroutine[object, object, _T]:
        self.coroutine_creations += 1

        async def execute() -> _T:
            self._owner.record(name)
            return result

        return execute()


class FakeAsyncClient:
    def __init__(self, arguments: dict[str, object]) -> None:
        selected_settings = arguments["settings"]
        self.settings = selected_settings if isinstance(selected_settings, Settings) else Settings()
        self.arguments = arguments
        self.closed = False
        self.close_calls = 0
        self.coroutine_creations = 0
        self.records: list[tuple[str, int, int]] = []
        self.session = FakeAsyncSession(self)
        self.open_error: BaseException | None = None
        self.inspect_error: BaseException | None = None
        self.close_error: BaseException | None = None
        self.close_delay = 0.0
        self.operation_delay = 0.0
        self.operation_started = threading.Event()
        self.active_operations = 0
        self.maximum_active_operations = 0

    def record(self, operation: str) -> None:
        self.records.append((operation, id(asyncio.get_running_loop()), threading.get_ident()))

    def open_image(
        self,
        source: object,
    ) -> Coroutine[object, object, FakeAsyncSession]:
        return self._operation("client.open_image", self.session, self.open_error)

    def inspect(
        self,
        source: object,
        plan: object = None,
        *,
        timeout_s: float | None = None,
    ) -> Coroutine[object, object, InspectionResult]:
        return self._operation("client.inspect", _INSPECTION, self.inspect_error)

    def ask(
        self,
        source: object,
        question: str,
        *,
        timeout_s: float | None = None,
    ) -> Coroutine[object, object, VisionAnswer]:
        return self._operation("client.ask", _ANSWER)

    def aclose(self) -> Coroutine[object, object, None]:
        self.coroutine_creations += 1

        async def close() -> None:
            self.record("client.aclose")
            if self.close_delay:
                await asyncio.sleep(self.close_delay)
            self.close_calls += 1
            if self.close_error is not None:
                raise self.close_error
            await self.session.aclose()
            self.closed = True

        return close()

    def __aenter__(self) -> Coroutine[object, object, FakeAsyncClient]:
        return self._operation("client.__aenter__", self)

    def _operation(
        self,
        name: str,
        result: _T,
        error: BaseException | None = None,
    ) -> Coroutine[object, object, _T]:
        self.coroutine_creations += 1

        async def execute() -> _T:
            self.record(name)
            self.active_operations += 1
            self.maximum_active_operations = max(
                self.maximum_active_operations,
                self.active_operations,
            )
            self.operation_started.set()
            try:
                if self.operation_delay:
                    await asyncio.sleep(self.operation_delay)
                if error is not None:
                    raise error
                return result
            finally:
                self.active_operations -= 1

        return execute()


@dataclass
class FakeClientHarness:
    monkeypatch: pytest.MonkeyPatch
    async_clients: list[FakeAsyncClient] = field(default_factory=list)
    sync_clients: list[Penampakan] = field(default_factory=list)
    constructor_error: BaseException | None = None

    def install(self) -> None:
        module = types.ModuleType("penampakan.client")
        module.AsyncPenampakan = self._construct
        self.monkeypatch.setitem(sys.modules, "penampakan.client", module)

    def _construct(self, **arguments: object) -> FakeAsyncClient:
        if self.constructor_error is not None:
            raise self.constructor_error
        client = FakeAsyncClient(arguments)
        self.async_clients.append(client)
        return client

    def create(self, **arguments: object) -> Penampakan:
        client = Penampakan(**arguments)
        self.sync_clients.append(client)
        return client


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeClientHarness]:
    selected = FakeClientHarness(monkeypatch)
    selected.install()
    yield selected
    for client in selected.sync_clients:
        with suppress(BaseException):
            client.close()


def test_constructor_is_thread_lazy_and_forwards_public_configuration(
    harness: FakeClientHarness,
) -> None:
    baseline = _loop_thread_ids()
    settings = Settings()
    llm = cast(TextLLM, object())
    backend = cast(VisionBackend, object())
    policy = cast(ActionPolicy, object())
    cache = cast(Cache, object())
    sink = cast(TraceSink, object())

    client = harness.create(
        llm=llm,
        backends=(backend,),
        policy=policy,
        cache=cache,
        settings=settings,
        trace_sinks=(sink,),
    )
    asynchronous = harness.async_clients[0]

    assert _loop_thread_ids() == baseline
    assert client.settings is settings
    assert client.closed is False
    assert asynchronous.arguments == {
        "llm": llm,
        "backends": (backend,),
        "policy": policy,
        "cache": cache,
        "settings": settings,
        "trace_sinks": (sink,),
        "owns_policy": False,
        "owns_llm": False,
        "owns_trace_sinks": False,
    }

    client.close()

    assert client.closed is True
    assert asynchronous.close_calls == 1
    assert _loop_thread_ids() == baseline


def test_all_operations_properties_and_lookups_reuse_one_loop_thread(
    harness: FakeClientHarness,
) -> None:
    caller_thread = threading.get_ident()
    client = harness.create()

    assert client.inspect(b"image", timeout_s=1.0) is _INSPECTION
    assert client.ask(b"image", "question", timeout_s=2.0) is _ANSWER
    session = client.open_image(b"image")
    assert isinstance(session, VisionSession)
    assert session.root_asset is _ROOT_ASSET
    assert session.assets == (_ROOT_ASSET, _DERIVED_ASSET)
    assert session.observations == (_OBSERVATION,)
    assert session.get_asset("img_0123456789abcdef") is _DERIVED_ASSET
    assert session.get_observation("obs_000001") is _OBSERVATION
    assert session.inspect(timeout_s=3.0) is _INSPECTION
    assert session.ask("another question", timeout_s=4.0) is _ANSWER

    asynchronous = harness.async_clients[0]
    identities = {(loop_id, thread_id) for _, loop_id, thread_id in asynchronous.records}

    assert len(identities) == 1
    assert next(iter(identities))[1] != caller_thread
    assert len(_loop_thread_ids()) == 1

    session.close()
    client.close()

    assert asynchronous.session.close_calls == 1
    assert asynchronous.close_calls == 1
    assert not _loop_thread_ids()


def test_synchronous_contexts_enter_async_objects_and_close_once(
    harness: FakeClientHarness,
) -> None:
    client = harness.create()
    asynchronous = harness.async_clients[0]

    with client as entered_client:
        assert entered_client is client
        with client.open_image(b"image") as session:
            assert session.root_asset is _ROOT_ASSET

    client.close()

    assert asynchronous.closed is True
    assert asynchronous.close_calls == 1
    assert asynchronous.session.close_calls == 1
    assert "client.__aenter__" in {item[0] for item in asynchronous.records}
    assert "session.__aenter__" in {item[0] for item in asynchronous.records}
    assert not _loop_thread_ids()


def test_construction_and_operation_failures_preserve_cleanup_and_reuse(
    harness: FakeClientHarness,
) -> None:
    baseline = _loop_thread_ids()
    construction_error = ValueError("construction failed")
    harness.constructor_error = construction_error

    with pytest.raises(ValueError) as construction:
        harness.create()

    assert construction.value is construction_error
    assert _loop_thread_ids() == baseline

    harness.constructor_error = None
    client = harness.create()
    asynchronous = harness.async_clients[0]
    open_error = ValueError("open failed")
    inspect_error = RuntimeError("inspect failed")
    asynchronous.open_error = open_error
    asynchronous.inspect_error = inspect_error

    with pytest.raises(ValueError) as opened:
        client.open_image(b"bad")
    with pytest.raises(RuntimeError) as inspected:
        client.inspect(b"bad")

    assert opened.value is open_error
    assert inspected.value is inspect_error
    assert len(_loop_thread_ids() - baseline) == 1

    asynchronous.open_error = None
    asynchronous.inspect_error = None
    session = client.open_image(b"good")
    assert client.inspect(b"good") is _INSPECTION
    session.close()
    client.close()

    assert asynchronous.close_calls == 1
    assert _loop_thread_ids() == baseline


def test_running_loop_rejection_precedes_coroutine_creation_without_warnings(
    harness: FakeClientHarness,
) -> None:
    client = harness.create()
    session = client.open_image(b"image")
    asynchronous = harness.async_clients[0]
    client_creations = asynchronous.coroutine_creations
    session_creations = asynchronous.session.coroutine_creations
    records = tuple(asynchronous.records)

    async def invoke_blocking_api() -> None:
        calls: tuple[Callable[[], object], ...] = (
            lambda: client.open_image(b"image"),
            lambda: client.inspect(b"image"),
            lambda: client.ask(b"image", "question"),
            client.__enter__,
            client.close,
            lambda: session.root_asset,
            lambda: session.assets,
            lambda: session.observations,
            lambda: session.inspect(),
            lambda: session.ask("question"),
            lambda: session.get_asset("img_0123456789abcdef"),
            lambda: session.get_observation("obs_000001"),
            session.__enter__,
            session.close,
        )
        for call in calls:
            with pytest.raises(SyncInAsyncContextError):
                call()
        await asyncio.sleep(0)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        asyncio.run(invoke_blocking_api())
        gc.collect()

    assert asynchronous.coroutine_creations == client_creations
    assert asynchronous.session.coroutine_creations == session_creations
    assert tuple(asynchronous.records) == records
    assert not [item for item in captured if issubclass(item.category, RuntimeWarning)]

    session.close()
    client.close()


def test_concurrent_calls_share_the_loop_and_client_close_drains_them(
    harness: FakeClientHarness,
) -> None:
    client = harness.create()
    asynchronous = harness.async_clients[0]
    asynchronous.operation_delay = 0.05

    results, errors = _concurrently(lambda: client.ask(b"image", "question"), count=8)

    assert not errors
    assert results == [_ANSWER] * 8
    assert asynchronous.maximum_active_operations > 1
    assert len({item[1:] for item in asynchronous.records}) == 1

    asynchronous.operation_started.clear()
    operation_results: list[VisionAnswer] = []
    operation_thread = threading.Thread(
        target=lambda: operation_results.append(client.ask(b"image", "question"))
    )
    operation_thread.start()
    assert asynchronous.operation_started.wait(timeout=1.0)

    close_results, close_errors = _concurrently(client.close, count=5)
    operation_thread.join(timeout=1.0)

    assert not operation_thread.is_alive()
    assert operation_results == [_ANSWER]
    assert close_results == [None] * 5
    assert not close_errors
    assert asynchronous.close_calls == 1
    assert asynchronous.closed is True
    assert not _loop_thread_ids()

    client.close()


def test_concurrent_session_close_and_failure_outcomes_are_idempotent(
    harness: FakeClientHarness,
) -> None:
    client = harness.create()
    session = client.open_image(b"image")
    asynchronous = harness.async_clients[0]
    asynchronous.session.close_delay = 0.05

    results, errors = _concurrently(session.close, count=5)

    assert results == [None] * 5
    assert not errors
    assert asynchronous.session.close_calls == 1
    with pytest.raises(SessionClosedError):
        session.ask("question")

    close_error = RuntimeError("close failed")
    asynchronous.close_error = close_error
    asynchronous.close_delay = 0.05
    results, errors = _concurrently(client.close, count=5)

    assert not results
    assert len(errors) == 5
    assert all(error is close_error for error in errors)
    assert asynchronous.close_calls == 1
    assert not _loop_thread_ids()

    with pytest.raises(RuntimeError) as repeated:
        client.close()
    assert repeated.value is close_error


class RetainedBaseException(BaseException):
    """A base exception the asynchronous client retains and re-raises."""


def test_client_close_propagates_and_repeats_a_retained_base_exception(
    harness: FakeClientHarness,
) -> None:
    client = harness.create()
    asynchronous = harness.async_clients[0]
    close_error = RetainedBaseException()
    asynchronous.close_error = close_error

    with pytest.raises(RetainedBaseException) as first:
        client.close()
    with pytest.raises(RetainedBaseException) as repeated:
        client.close()

    assert first.value is close_error
    assert repeated.value is close_error
    assert asynchronous.close_calls == 1
    assert asynchronous.closed is False
    assert not _loop_thread_ids()


def test_concurrent_client_closes_share_one_retained_base_exception(
    harness: FakeClientHarness,
) -> None:
    client = harness.create()
    asynchronous = harness.async_clients[0]
    close_error = RetainedBaseException()
    asynchronous.close_error = close_error
    asynchronous.close_delay = 0.05

    results, errors = _concurrently(client.close, count=5)

    assert not results
    assert len(errors) == 5
    assert all(error is close_error for error in errors)
    assert asynchronous.close_calls == 1
    assert not _loop_thread_ids()
