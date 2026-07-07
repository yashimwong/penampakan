from __future__ import annotations

import asyncio
import threading
from typing import cast

import pytest
from pydantic import ValidationError

from penampakan.llms.callable import CallableTextLLM, CloseCallable, CompleteCallable
from penampakan.models import LLMRequest, LLMResponse, Message, MessageRole


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


def test_constructor_requires_callable_boundaries() -> None:
    with pytest.raises(TypeError, match="complete must be callable"):
        CallableTextLLM(cast(CompleteCallable, object()))
    with pytest.raises(TypeError, match="close must be callable"):
        CallableTextLLM(lambda request: "{}", close=cast(CloseCallable, "not callable"))
