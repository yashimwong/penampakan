"""Shared dispatch helpers for application-supplied callables."""

from __future__ import annotations

import asyncio
import functools
import inspect
from collections.abc import Awaitable, Callable
from typing import TypeVar, cast

_T = TypeVar("_T")

# A partial cannot wrap itself, but a subclass could misreport ``func``.
_MAX_PARTIAL_DEPTH = 32


def _unwrap_partial(value: object) -> object:
    """Return the callable a chain of partial applications ultimately invokes."""

    for _ in range(_MAX_PARTIAL_DEPTH):
        if not isinstance(value, functools.partial):
            break
        value = value.func
    return value


def is_async_callable(value: object) -> bool:
    """Report whether the callable produces a coroutine when invoked."""

    if inspect.iscoroutinefunction(value):
        return True
    # ``iscoroutinefunction`` unwraps a partial only far enough to read a
    # function's code flags, so a partial bound to an object with an
    # asynchronous ``__call__`` is missed unless the chain is unwrapped first.
    target = _unwrap_partial(value)
    if target is not value and inspect.iscoroutinefunction(target):
        return True
    # Read ``__call__`` itself: callable() cannot tell whether the object returns a coroutine.
    call = getattr(target, "__call__", None)  # noqa: B004
    return inspect.iscoroutinefunction(call)


async def call_async_or_thread(
    func: Callable[..., _T | Awaitable[_T]],
    /,
    *args: object,
) -> _T:
    """Await an async callable on the loop and offload a sync callable to a thread."""

    if is_async_callable(func):
        return await _awaited(func(*args))
    worker = asyncio.create_task(asyncio.to_thread(func, *args))
    try:
        outcome = await asyncio.shield(worker)
    except asyncio.CancelledError:
        # The thread cannot be cancelled, so an awaitable it returns would be
        # abandoned unawaited once the caller leaves. Dispose of it instead.
        worker.add_done_callback(_discard_late_awaitable)
        raise
    return await _awaited(outcome)


async def _awaited(outcome: object) -> _T:
    # Callable inspection cannot prove a return type, so a synchronous callable
    # that returns an awaitable is still awaited on the loop.
    if inspect.isawaitable(outcome):
        return cast("_T", await outcome)
    return cast("_T", outcome)


def _discard_late_awaitable(worker: asyncio.Task[object]) -> None:
    """Close a coroutine a worker produced after its awaiter was cancelled."""

    try:
        outcome = worker.result()
    except (asyncio.CancelledError, Exception):
        return
    if inspect.iscoroutine(outcome):
        outcome.close()


__all__ = ["call_async_or_thread", "is_async_callable"]
