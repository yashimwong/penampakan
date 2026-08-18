from __future__ import annotations

import asyncio
import functools
import threading
from collections.abc import Awaitable, Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from typing import cast

import pytest
from pydantic import ValidationError

from penampakan.llms.callable import CallableTextLLM, CloseCallable, CompleteCallable
from penampakan.models import LLMRequest, LLMResponse, Message, MessageRole


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


def _request() -> LLMRequest:
    return LLMRequest(
        messages=(Message(role=MessageRole.USER, content="Return JSON."),),
        response_json_schema={"type": "object"},
    )


@pytest.mark.asyncio
async def test_sync_callable_runs_in_worker_thread_and_wraps_string() -> None:
    calling_thread = threading.get_ident()
    worker_threads: list[int] = []

    def complete(request: LLMRequest) -> str:
        worker_threads.append(threading.get_ident())
        assert request == _request()
        return '{"type":"answer"}'

    llm = CallableTextLLM(complete)

    response = await llm.complete(_request())

    assert response == LLMResponse(text='{"type":"answer"}')
    assert worker_threads and worker_threads[0] != calling_thread


@pytest.mark.asyncio
async def test_async_callable_is_awaited_and_preserves_response() -> None:
    expected = LLMResponse(text="{}", model_id="async-model", finish_reason="stop")

    async def complete(request: LLMRequest) -> LLMResponse:
        await asyncio.sleep(0)
        assert request == _request()
        return expected

    llm = CallableTextLLM(complete)

    assert await llm.complete(_request()) == expected


@pytest.mark.asyncio
async def test_invalid_callable_result_is_rejected_strictly() -> None:
    def complete(request: LLMRequest) -> object:
        return {"text": 42}

    llm = CallableTextLLM(cast(CompleteCallable, complete))

    with pytest.raises(ValidationError):
        await llm.complete(_request())


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_close_callable_runs_once_and_post_close_completion_fails(
    asynchronous: bool,
) -> None:
    close_calls = 0

    async def async_close() -> None:
        nonlocal close_calls
        await asyncio.sleep(0)
        close_calls += 1

    def sync_close() -> None:
        nonlocal close_calls
        close_calls += 1

    llm = CallableTextLLM(
        lambda request: "{}",
        close=async_close if asynchronous else sync_close,
    )

    await asyncio.gather(llm.aclose(), llm.aclose(), llm.aclose())

    assert close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        await llm.complete(_request())


@pytest.mark.asyncio
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
async def test_async_completions_run_on_the_loop_thread(shape: str) -> None:
    loop_thread = threading.get_ident()
    completion_threads: list[int] = []

    async def complete(request: LLMRequest) -> LLMResponse:
        completion_threads.append(threading.get_ident())
        await asyncio.sleep(0)
        return LLMResponse(text="{}")

    class AsyncCompleter:
        async def __call__(self, request: LLMRequest) -> LLMResponse:
            completion_threads.append(threading.get_ident())
            await asyncio.sleep(0)
            return LLMResponse(text="{}")

    async_completer = AsyncCompleter()
    completers: dict[str, CompleteCallable] = {
        "coroutine_function": complete,
        "partial": functools.partial(complete),
        "async_call_object": async_completer,
        "partial_of_async_call_object": functools.partial(async_completer),
        "nested_partial_of_async_call_object": functools.partial(
            functools.partial(async_completer)
        ),
    }
    llm = CallableTextLLM(completers[shape])

    with _recording_executor() as executor:
        assert await llm.complete(_request()) == LLMResponse(text="{}")

    assert completion_threads == [loop_thread]
    assert executor.submissions == 0


@pytest.mark.asyncio
async def test_sync_completion_returning_awaitable_awaits_on_the_loop_thread() -> None:
    loop_thread = threading.get_ident()
    observed: dict[str, int] = {}

    def complete(request: LLMRequest) -> Awaitable[LLMResponse]:
        observed["call"] = threading.get_ident()

        async def finish() -> LLMResponse:
            observed["await"] = threading.get_ident()
            return LLMResponse(text="{}")

        return finish()

    llm = CallableTextLLM(complete)

    with _recording_executor() as executor:
        assert await llm.complete(_request()) == LLMResponse(text="{}")

    assert observed["call"] != loop_thread
    assert observed["await"] == loop_thread
    assert executor.submissions == 1


@pytest.mark.asyncio
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
    llm = CallableTextLLM(lambda request: "{}", close=closers[shape])

    with _recording_executor() as executor:
        await llm.aclose()

    assert close_threads == [loop_thread]
    assert executor.submissions == 0


@pytest.mark.asyncio
async def test_sync_close_callable_runs_in_worker_thread() -> None:
    loop_thread = threading.get_ident()
    close_threads: list[int] = []

    def close() -> None:
        close_threads.append(threading.get_ident())

    llm = CallableTextLLM(lambda request: "{}", close=close)

    await llm.aclose()

    assert close_threads and close_threads[0] != loop_thread


@pytest.mark.asyncio
async def test_sync_close_callable_returning_awaitable_awaits_on_the_loop_thread() -> None:
    loop_thread = threading.get_ident()
    observed: dict[str, int] = {}

    def close() -> Awaitable[None]:
        observed["call"] = threading.get_ident()

        async def finish() -> None:
            observed["await"] = threading.get_ident()

        return finish()

    llm = CallableTextLLM(lambda request: "{}", close=close)

    with _recording_executor() as executor:
        await llm.aclose()

    assert observed["call"] != loop_thread
    assert observed["await"] == loop_thread
    assert executor.submissions == 1


@pytest.mark.asyncio
async def test_cancelling_async_completion_propagates_cancellation() -> None:
    started = asyncio.Event()

    async def complete(request: LLMRequest) -> LLMResponse:
        started.set()
        await asyncio.sleep(30)
        return LLMResponse(text="{}")

    llm = CallableTextLLM(complete)
    task = asyncio.create_task(llm.complete(_request()))

    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_cancelling_threaded_completion_propagates_cancellation() -> None:
    started = threading.Event()
    release = threading.Event()

    def complete(request: LLMRequest) -> str:
        started.set()
        release.wait(30)
        return "{}"

    llm = CallableTextLLM(complete)
    task = asyncio.create_task(llm.complete(_request()))
    try:
        while not started.is_set():
            await asyncio.sleep(0.01)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()


@pytest.mark.asyncio
async def test_cancelling_async_close_propagates_cancellation() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def close() -> None:
        started.set()
        await release.wait()

    llm = CallableTextLLM(lambda request: "{}", close=close)
    task = asyncio.create_task(llm.aclose())

    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    release.set()
    await llm.aclose()


@pytest.mark.asyncio
async def test_cancelling_threaded_close_propagates_cancellation() -> None:
    started = threading.Event()
    release = threading.Event()

    def close() -> None:
        started.set()
        release.wait(30)

    llm = CallableTextLLM(lambda request: "{}", close=close)
    task = asyncio.create_task(llm.aclose())
    try:
        while not started.is_set():
            await asyncio.sleep(0.01)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()

    await llm.aclose()


def test_constructor_requires_callable_boundaries() -> None:
    with pytest.raises(TypeError, match="complete must be callable"):
        CallableTextLLM(cast(CompleteCallable, object()))
    with pytest.raises(TypeError, match="close must be callable"):
        CallableTextLLM(lambda request: "{}", close=cast(CloseCallable, "not callable"))
