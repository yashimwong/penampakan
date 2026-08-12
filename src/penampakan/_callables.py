"""Shared dispatch helpers for application-supplied callables."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import TypeVar, cast

_T = TypeVar("_T")


def is_async_callable(value: object) -> bool:
    """Report whether the callable produces a coroutine when invoked."""

    if inspect.iscoroutinefunction(value):
        return True
    # Read ``__call__`` itself: callable() cannot tell whether the object returns a coroutine.
    call = getattr(value, "__call__", None)  # noqa: B004
    return inspect.iscoroutinefunction(call)


async def call_async_or_thread(
    func: Callable[..., _T | Awaitable[_T]],
    /,
    *args: object,
) -> _T:
    """Await an async callable on the loop and offload a sync callable to a thread."""

    if is_async_callable(func):
        outcome = func(*args)
    else:
        outcome = await asyncio.to_thread(func, *args)
    if inspect.isawaitable(outcome):
        return cast("_T", await outcome)
    return outcome


__all__ = ["call_async_or_thread", "is_async_callable"]
