"""Text language model adapter for application callables."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable

from penampakan.models import LLMRequest, LLMResponse

CompleteCallable = Callable[
    [LLMRequest],
    str | LLMResponse | Awaitable[str | LLMResponse],
]
CloseCallable = Callable[[], Awaitable[None] | None]


class CallableTextLLM:
    """Adapt a synchronous or asynchronous application function as a text LLM."""

    def __init__(
        self,
        complete: CompleteCallable,
        *,
        close: CloseCallable | None = None,
    ) -> None:
        if not callable(complete):
            raise TypeError("complete must be callable")
        if close is not None and not callable(close):
            raise TypeError("close must be callable")
        self._complete = complete
        self._close = close
        self._close_task: asyncio.Task[None] | None = None

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Invoke the application callable without blocking the event loop."""
        if self._close_task is not None:
            raise RuntimeError("language model is closed")
        result = await asyncio.to_thread(self._complete, request)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, str):
            return LLMResponse(text=result)
        return LLMResponse.model_validate(result, strict=True)

    async def aclose(self) -> None:
        """Run the optional close callable exactly once."""
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._run_close())
        await asyncio.shield(self._close_task)

    async def _run_close(self) -> None:
        if self._close is None:
            return
        result = await asyncio.to_thread(self._close)
        if inspect.isawaitable(result):
            await result


__all__ = ["CallableTextLLM", "CloseCallable", "CompleteCallable"]
