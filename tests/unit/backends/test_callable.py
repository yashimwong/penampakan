import asyncio
import functools
import threading
from collections.abc import Awaitable, Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager

import pytest
from PIL import Image

from penampakan.backends.callable import AnalyzeCallable, CallableVisionBackend, CloseCallable
from penampakan.image.assets import AssetStore
from penampakan.models import (
    BackendDescriptor,
    Capability,
    CapabilityDescriptor,
    CaptionPayload,
    CaptionRequest,
    ColorsRequest,
    DetectionRequest,
    MetadataRequest,
    ObservationDraft,
    VisionResult,
)


def descriptor(*features: str) -> BackendDescriptor:
    return BackendDescriptor(
        name="application.caption",
        version="1.0",
        capabilities=(
            CapabilityDescriptor(
                capability=Capability.CAPTION,
                features=frozenset(features),
            ),
        ),
        max_concurrency=2,
    )


def result() -> VisionResult:
    return VisionResult(
        observations=(ObservationDraft(payload=CaptionPayload(text="A red square.")),)
    )


class RecordingExecutor(ThreadPoolExecutor):
    def __init__(self) -> None:
        super().__init__(max_workers=2)
        self.submissions = 0

    def submit(  # type: ignore[override]
        self,
        fn: Callable[..., object],
        /,
        *args: object,
        **kwargs: object,
    ) -> Future[object]:
        self.submissions += 1
        return super().submit(fn, *args, **kwargs)


@contextmanager
def recording_executor() -> Iterator[RecordingExecutor]:
    """Install a default executor that counts every offloaded callable."""

    loop = asyncio.get_event_loop()
    executor = RecordingExecutor()
    loop.set_default_executor(executor)
    try:
        yield executor
    finally:
        executor.shutdown(wait=False)


@contextmanager
def opened_store() -> Iterator[AssetStore]:
    source = Image.new("RGB", (2, 2), "red")
    store = AssetStore.create(source)
    try:
        yield store
    finally:
        store.close()
        source.close()


async def test_sync_analyzer_runs_in_worker_thread() -> None:
    source = Image.new("RGB", (2, 2), "red")
    store = AssetStore.create(source)
    caller_thread = threading.get_ident()
    analyzer_threads: list[int] = []

    def analyze(image: object, request: object) -> VisionResult:
        analyzer_threads.append(threading.get_ident())
        return result()

    backend = CallableVisionBackend(descriptor(), analyze)

    response = await backend.analyze(store.backend_image(store.root_id), CaptionRequest())

    assert response == result()
    assert analyzer_threads and analyzer_threads[0] != caller_thread
    await backend.aclose()
    store.close()
    source.close()


async def test_async_analyzer_is_awaited() -> None:
    source = Image.new("RGB", (2, 2), "red")
    store = AssetStore.create(source)
    calls = 0

    async def analyze(image: object, request: object) -> VisionResult:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        return result()

    backend = CallableVisionBackend(descriptor(), analyze)

    assert await backend.analyze(store.backend_image(store.root_id), CaptionRequest()) == result()
    assert calls == 1
    await backend.aclose()
    store.close()
    source.close()


def test_descriptor_derived_support_rejects_unsupported_options() -> None:
    backend = CallableVisionBackend(descriptor(), lambda image, request: result())
    focused = CallableVisionBackend(
        descriptor("caption.focus"),
        lambda image, request: result(),
    )

    assert backend.supports(CaptionRequest())
    assert not backend.supports(CaptionRequest(focus="display"))
    assert focused.supports(CaptionRequest(focus="display"))
    assert not backend.supports(CaptionRequest(mark_indices=(1,)))
    mark_aware = CallableVisionBackend(
        descriptor("caption.mark_references"),
        lambda image, request: result(),
    )
    assert mark_aware.supports(CaptionRequest(mark_indices=(1,)))
    assert not backend.supports(MetadataRequest())
    assert not backend.supports(ColorsRequest())
    assert not backend.supports(DetectionRequest(labels=("car",)))


async def test_close_callable_runs_once_for_concurrent_close() -> None:
    close_calls = 0

    async def close() -> None:
        nonlocal close_calls
        close_calls += 1
        await asyncio.sleep(0)

    backend = CallableVisionBackend(
        descriptor(),
        lambda image, request: result(),
        close=close,
    )

    await asyncio.gather(backend.aclose(), backend.aclose(), backend.aclose())

    assert close_calls == 1


@pytest.mark.parametrize(
    "shape",
    [
        "coroutine_function",
        "partial",
        "async_call_object",
        "partial_of_async_call_object",
        "nested_partial_of_async_call_object",
    ],
)
async def test_async_analyzers_run_on_the_loop_thread(shape: str) -> None:
    loop_thread = threading.get_ident()
    analyzer_threads: list[int] = []

    async def analyze(image: object, request: object) -> VisionResult:
        analyzer_threads.append(threading.get_ident())
        await asyncio.sleep(0)
        return result()

    class AsyncAnalyzer:
        async def __call__(self, image: object, request: object) -> VisionResult:
            analyzer_threads.append(threading.get_ident())
            await asyncio.sleep(0)
            return result()

    async_analyzer = AsyncAnalyzer()
    analyzers: dict[str, AnalyzeCallable] = {
        "coroutine_function": analyze,
        "partial": functools.partial(analyze),
        "async_call_object": async_analyzer,
        "partial_of_async_call_object": functools.partial(async_analyzer),
        "nested_partial_of_async_call_object": functools.partial(functools.partial(async_analyzer)),
    }

    with opened_store() as store, recording_executor() as executor:
        backend = CallableVisionBackend(descriptor(), analyzers[shape])

        response = await backend.analyze(store.backend_image(store.root_id), CaptionRequest())

        assert response == result()
        assert analyzer_threads == [loop_thread]
        assert executor.submissions == 0
        await backend.aclose()


async def test_sync_analyzer_returning_awaitable_awaits_on_the_loop_thread() -> None:
    loop_thread = threading.get_ident()
    observed: dict[str, int] = {}

    def analyze(image: object, request: object) -> Awaitable[VisionResult]:
        observed["call"] = threading.get_ident()

        async def finish() -> VisionResult:
            observed["await"] = threading.get_ident()
            return result()

        return finish()

    with opened_store() as store, recording_executor() as executor:
        backend = CallableVisionBackend(descriptor(), analyze)

        response = await backend.analyze(store.backend_image(store.root_id), CaptionRequest())

        assert response == result()
        assert observed["call"] != loop_thread
        assert observed["await"] == loop_thread
        assert executor.submissions == 1
        await backend.aclose()


@pytest.mark.parametrize(
    "shape",
    [
        "coroutine_function",
        "partial",
        "async_call_object",
        "partial_of_async_call_object",
        "nested_partial_of_async_call_object",
    ],
)
async def test_async_close_callables_run_on_the_loop_thread(shape: str) -> None:
    loop_thread = threading.get_ident()
    close_threads: list[int] = []

    async def close() -> None:
        close_threads.append(threading.get_ident())
        await asyncio.sleep(0)

    class AsyncCloser:
        async def __call__(self) -> None:
            close_threads.append(threading.get_ident())
            await asyncio.sleep(0)

    async_closer = AsyncCloser()
    closers: dict[str, CloseCallable] = {
        "coroutine_function": close,
        "partial": functools.partial(close),
        "async_call_object": async_closer,
        "partial_of_async_call_object": functools.partial(async_closer),
        "nested_partial_of_async_call_object": functools.partial(functools.partial(async_closer)),
    }
    backend = CallableVisionBackend(
        descriptor(),
        lambda image, request: result(),
        close=closers[shape],
    )

    with recording_executor() as executor:
        await backend.aclose()

    assert close_threads == [loop_thread]
    assert executor.submissions == 0


async def test_sync_close_callable_runs_in_worker_thread() -> None:
    loop_thread = threading.get_ident()
    close_threads: list[int] = []

    def close() -> None:
        close_threads.append(threading.get_ident())

    backend = CallableVisionBackend(descriptor(), lambda image, request: result(), close=close)

    await backend.aclose()

    assert close_threads and close_threads[0] != loop_thread


async def test_sync_close_callable_returning_awaitable_awaits_on_the_loop_thread() -> None:
    loop_thread = threading.get_ident()
    observed: dict[str, int] = {}

    def close() -> Awaitable[None]:
        observed["call"] = threading.get_ident()

        async def finish() -> None:
            observed["await"] = threading.get_ident()

        return finish()

    backend = CallableVisionBackend(descriptor(), lambda image, request: result(), close=close)

    with recording_executor() as executor:
        await backend.aclose()

    assert observed["call"] != loop_thread
    assert observed["await"] == loop_thread
    assert executor.submissions == 1


async def test_cancelling_async_analyze_propagates_cancellation() -> None:
    started = asyncio.Event()

    async def analyze(image: object, request: object) -> VisionResult:
        started.set()
        await asyncio.sleep(30)
        return result()

    with opened_store() as store:
        backend = CallableVisionBackend(descriptor(), analyze)
        task = asyncio.create_task(
            backend.analyze(store.backend_image(store.root_id), CaptionRequest())
        )

        await started.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task


async def test_cancelling_threaded_analyze_propagates_cancellation() -> None:
    started = threading.Event()
    release = threading.Event()

    def analyze(image: object, request: object) -> VisionResult:
        started.set()
        release.wait(30)
        return result()

    with opened_store() as store:
        backend = CallableVisionBackend(descriptor(), analyze)
        task = asyncio.create_task(
            backend.analyze(store.backend_image(store.root_id), CaptionRequest())
        )
        try:
            while not started.is_set():
                await asyncio.sleep(0.01)
            task.cancel()

            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()


async def test_cancelling_async_close_propagates_cancellation() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def close() -> None:
        started.set()
        await release.wait()

    backend = CallableVisionBackend(descriptor(), lambda image, request: result(), close=close)
    task = asyncio.create_task(backend.aclose())

    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    release.set()
    await backend.aclose()


async def test_cancelling_threaded_close_propagates_cancellation() -> None:
    started = threading.Event()
    release = threading.Event()

    def close() -> None:
        started.set()
        release.wait(30)

    backend = CallableVisionBackend(descriptor(), lambda image, request: result(), close=close)
    task = asyncio.create_task(backend.aclose())
    try:
        while not started.is_set():
            await asyncio.sleep(0.01)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()

    await backend.aclose()
