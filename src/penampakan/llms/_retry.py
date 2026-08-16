"""One shared provider retry implementation with a total monotonic deadline.

Every adapter routes its single provider call through :func:`call_with_retries`
so retry accounting, deadline enforcement, and redaction are identical across
providers. Only connection failures, timeouts, 429, and 5xx are retried;
schema, refusal, truncation, authentication, and ordinary 4xx outcomes are
terminal. Owned SDK clients disable their own retries so provider attempts
cannot multiply.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from penampakan.errors import LLMError
from penampakan.models import RetryPolicy

_T = TypeVar("_T")

# A retry is only started when the backoff plus this much attempt work still
# fits inside the remaining deadline.
_MINIMUM_ATTEMPT_S = 0.05

RandomSource = Callable[[], float]


@dataclass(frozen=True, slots=True)
class ProviderFailure:
    """A redacted classification of one provider exception."""

    retryable: bool
    status: int | None = None
    code: str | None = None


@dataclass(frozen=True, slots=True)
class _AttemptFailure:
    """Keep provider exceptions distinct from ``wait_for`` expiration."""

    error: Exception


FailureClassifier = Callable[[BaseException], ProviderFailure]


class Deadline:
    """A monotonic budget covering every attempt, read, and backoff."""

    __slots__ = ("_expires_at",)

    def __init__(self, timeout_s: float | None) -> None:
        self._expires_at = None if timeout_s is None else time.monotonic() + timeout_s

    @property
    def bounded(self) -> bool:
        """Return whether this deadline constrains the operation."""
        return self._expires_at is not None

    def remaining(self) -> float | None:
        """Return the seconds left, or ``None`` when unbounded."""
        if self._expires_at is None:
            return None
        return self._expires_at - time.monotonic()

    def expired(self) -> bool:
        """Return whether the deadline has already passed."""
        remaining = self.remaining()
        return remaining is not None and remaining <= 0.0


def _jittered_delay(
    policy: RetryPolicy,
    attempt: int,
    random_source: RandomSource,
) -> float:
    capped = min(policy.max_delay_s, policy.base_delay_s * (2.0 ** (attempt - 1)))
    # Full jitter over the capped exponential delay.
    fraction = random_source()
    if not 0.0 <= fraction <= 1.0:
        fraction = 0.0
    return capped * fraction


async def _capture_attempt(operation: Callable[[], Awaitable[_T]]) -> _T | _AttemptFailure:
    """Return provider exceptions as values so deadline timeouts stay distinguishable."""
    try:
        return await operation()
    except Exception as error:
        return _AttemptFailure(error)


async def call_with_retries(
    operation: Callable[[], Awaitable[_T]],
    *,
    policy: RetryPolicy,
    classify: FailureClassifier,
    deadline: Deadline,
    provider: str,
    random_source: RandomSource | None = None,
) -> tuple[_T, int]:
    """Run one provider call under the retry policy and total deadline.

    Returns the provider result and the number of attempts actually made. The
    caller reports that count through ``LLMResponse.attempts`` so budgets and
    cost reports include retries.
    """
    source = random.random if random_source is None else random_source
    attempts = 0
    failure = ProviderFailure(retryable=False)
    last_error: BaseException | None = None
    while True:
        if deadline.expired():
            raise LLMError(
                code="llm_timeout",
                retryable=True,
                attempts=attempts or None,
                provider=provider,
                provider_status=failure.status,
                provider_code=failure.code,
                cause=last_error,
            )
        attempts += 1
        remaining = deadline.remaining()
        try:
            if remaining is None:
                outcome = await _capture_attempt(operation)
            else:
                outcome = await asyncio.wait_for(_capture_attempt(operation), timeout=remaining)
        except asyncio.TimeoutError as deadline_error:
            raise LLMError(
                code="llm_timeout",
                retryable=True,
                attempts=attempts,
                provider=provider,
                cause=deadline_error,
            ) from deadline_error
        if not isinstance(outcome, _AttemptFailure):
            return outcome, attempts
        attempt_error = outcome.error
        if isinstance(attempt_error, LLMError):
            # Adapters raise typed refusal, truncation, and schema failures that
            # are never retried.
            raise attempt_error
        failure = classify(attempt_error)
        last_error = attempt_error
        if not failure.retryable or attempts >= policy.max_attempts:
            raise LLMError(
                code="llm_request_failed" if not failure.retryable else "llm_retries_exhausted",
                retryable=failure.retryable,
                attempts=attempts,
                provider=provider,
                provider_status=failure.status,
                provider_code=failure.code,
                cause=attempt_error,
            ) from attempt_error
        delay = _jittered_delay(policy, attempts, source)
        remaining = deadline.remaining()
        if remaining is not None and delay + _MINIMUM_ATTEMPT_S >= remaining:
            # The next attempt cannot do useful work before the deadline.
            raise LLMError(
                code="llm_timeout",
                retryable=True,
                attempts=attempts,
                provider=provider,
                provider_status=failure.status,
                provider_code=failure.code,
                cause=attempt_error,
            ) from attempt_error
        if delay > 0.0:
            await asyncio.sleep(delay)


__all__ = [
    "Deadline",
    "FailureClassifier",
    "ProviderFailure",
    "RandomSource",
    "call_with_retries",
]
