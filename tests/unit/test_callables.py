from __future__ import annotations

import asyncio
import functools
import gc
import threading
import warnings
import weakref
from collections.abc import Awaitable, Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager

import pytest

from penampakan._callables import call_async_or_thread, is_async_callable


class _RecordingExecutor(ThreadPoolExecutor):
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
def _recording_executor() -> Iterator[_RecordingExecutor]:
    """Install a default executor that counts every offloaded callable."""

    loop = asyncio.get_event_loop()
    executor = _RecordingExecutor()
    loop.set_default_executor(executor)
    try:
        yield executor
    finally:
        executor.shutdown(wait=False)


async def _coroutine_function() -> str:
    return "async"


def _plain_function() -> str:
    return "sync"


class _AsyncCallable:
    async def __call__(self) -> str:
        return "async-object"


class _SyncCallable:
    def __call__(self) -> str:
        return "sync-object"


_ASYNC_INSTANCE = _AsyncCallable()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (_plain_function, False),
        (_coroutine_function, True),
        (functools.partial(_coroutine_function), True),
        (functools.partial(functools.partial(_coroutine_function)), True),
        (_ASYNC_INSTANCE, True),
        (functools.partial(_ASYNC_INSTANCE), True),
        (functools.partial(functools.partial(_ASYNC_INSTANCE)), True),
        (_SyncCallable(), False),
        (object(), False),
        (functools.partial(len, ""), False),
    ],
    ids=[
        "plain_function",
        "coroutine_function",
        "partial_of_coroutine_function",
        "nested_partial_of_coroutine_function",
        "async_call_object",
        "partial_of_async_call_object",
        "nested_partial_of_async_call_object",
        "sync_call_object",
        "not_callable",
        "partial_of_builtin",
    ],
)
def test_is_async_callable_reports_whether_invocation_yields_a_coroutine(
    value: object,
    expected: bool,
) -> None:
    assert is_async_callable(value) is expected


async def test_async_callable_runs_on_the_loop_thread_without_offloading() -> None:
    loop_thread = threading.get_ident()
    call_threads: list[int] = []

    async def call() -> str:
        call_threads.append(threading.get_ident())
        await asyncio.sleep(0)
        return "async"

    with _recording_executor() as executor:
        assert await call_async_or_thread(call) == "async"

    assert call_threads == [loop_thread]
    assert executor.submissions == 0


async def test_partial_of_an_async_call_object_runs_on_the_loop_thread() -> None:
    loop_thread = threading.get_ident()
    call_threads: list[int] = []

    class AsyncCaller:
        async def __call__(self, marker: str) -> str:
            call_threads.append(threading.get_ident())
            await asyncio.sleep(0)
            return marker

    caller = AsyncCaller()

    with _recording_executor() as executor:
        assert await call_async_or_thread(functools.partial(caller, "one")) == "one"
        assert await call_async_or_thread(functools.partial(functools.partial(caller), "two")) == (
            "two"
        )

    assert call_threads == [loop_thread, loop_thread]
    assert executor.submissions == 0


async def test_sync_callable_runs_in_a_worker_thread() -> None:
    loop_thread = threading.get_ident()
    call_threads: list[int] = []

    def call() -> str:
        call_threads.append(threading.get_ident())
        return "sync"

    with _recording_executor() as executor:
        assert await call_async_or_thread(call) == "sync"

    assert call_threads and call_threads[0] != loop_thread
    assert executor.submissions == 1


async def test_sync_callable_returning_an_awaitable_awaits_on_the_loop_thread() -> None:
    loop_thread = threading.get_ident()
    observed: dict[str, int] = {}

    def call() -> Awaitable[str]:
        observed["call"] = threading.get_ident()

        async def finish() -> str:
            observed["await"] = threading.get_ident()
            return "late"

        return finish()

    with _recording_executor() as executor:
        assert await call_async_or_thread(call) == "late"

    assert observed["call"] != loop_thread
    assert observed["await"] == loop_thread
    assert executor.submissions == 1


async def test_cancelling_an_async_callable_propagates_cancellation() -> None:
    started = asyncio.Event()

    async def call() -> str:
        started.set()
        await asyncio.sleep(30)
        return "async"

    task = asyncio.create_task(call_async_or_thread(call))

    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


async def test_cancelling_a_threaded_callable_closes_the_late_awaitable() -> None:
    started = threading.Event()
    release = threading.Event()
    returned = threading.Event()
    references: list[weakref.ref[object]] = []

    def call() -> Awaitable[str]:
        started.set()
        release.wait(30)

        async def finish() -> str:
            return "late"

        pending = finish()
        references.append(weakref.ref(pending))
        returned.set()
        return pending

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        task = asyncio.create_task(call_async_or_thread(call))
        try:
            while not started.is_set():
                await asyncio.sleep(0.01)
            task.cancel()

            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()

        while not returned.is_set():
            await asyncio.sleep(0.01)
        for _ in range(200):
            await asyncio.sleep(0.01)
            gc.collect()
            if references and references[0]() is None:
                break

    assert references and references[0]() is None
    assert not [item for item in captured if issubclass(item.category, RuntimeWarning)]
