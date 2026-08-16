"""Unit tests for the shared adapter retry implementation."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import pytest

from penampakan.errors import LLMError
from penampakan.llms._retry import (
    Deadline,
    ProviderFailure,
    call_with_retries,
)
from penampakan.models import RetryPolicy


class _Boom(Exception):
    """A provider failure stand-in carrying no safe metadata."""


def _classify(
    retryable: bool, status: int | None = None, code: str | None = None
) -> Callable[[BaseException], ProviderFailure]:
    def classify(_error: BaseException) -> ProviderFailure:
        return ProviderFailure(retryable=retryable, status=status, code=code)

    return classify


def _operation(outcomes: list[object]) -> Callable[[], Awaitable[object]]:
    async def operation() -> object:
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    return operation


async def test_returns_the_first_successful_result_without_retrying() -> None:
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    value, attempts = await call_with_retries(
        operation,
        policy=RetryPolicy(),
        classify=_classify(retryable=True),
        deadline=Deadline(None),
        provider="fake",
    )
    assert (value, attempts, calls) == ("ok", 1, 1)


async def test_retries_retryable_failures_up_to_the_policy_bound() -> None:
    outcomes: list[object] = [_Boom(), _Boom(), "ok"]
    value, attempts = await call_with_retries(
        _operation(outcomes),
        policy=RetryPolicy(max_attempts=3, base_delay_s=0.001, max_delay_s=0.002),
        classify=_classify(retryable=True, status=503, code="overloaded"),
        deadline=Deadline(None),
        provider="fake",
        random_source=lambda: 0.0,
    )
    assert (value, attempts) == ("ok", 3)
    assert outcomes == []


async def test_exhausted_retries_raise_a_redacted_error_with_safe_metadata() -> None:
    outcomes: list[object] = [_Boom(), _Boom()]
    with pytest.raises(LLMError) as failure:
        await call_with_retries(
            _operation(outcomes),
            policy=RetryPolicy(max_attempts=2, base_delay_s=0.001, max_delay_s=0.002),
            classify=_classify(retryable=True, status=429, code="rate_limited"),
            deadline=Deadline(None),
            provider="fake",
            random_source=lambda: 0.0,
        )
    error = failure.value
    assert error.code == "llm_retries_exhausted"
    assert error.attempts == 2
    assert error.provider == "fake"
    assert error.provider_status == 429
    assert error.provider_code == "rate_limited"
    assert error.retryable is True
    assert isinstance(error.__cause__, _Boom)


async def test_non_retryable_failures_are_reported_after_one_attempt() -> None:
    outcomes: list[object] = [_Boom(), "unreachable"]
    with pytest.raises(LLMError) as failure:
        await call_with_retries(
            _operation(outcomes),
            policy=RetryPolicy(max_attempts=4, base_delay_s=0.001, max_delay_s=0.002),
            classify=_classify(retryable=False, status=401),
            deadline=Deadline(None),
            provider="fake",
        )
    assert failure.value.code == "llm_request_failed"
    assert failure.value.attempts == 1
    assert failure.value.provider_status == 401
    assert outcomes == ["unreachable"]


async def test_typed_adapter_errors_are_never_retried() -> None:
    refusal = LLMError(code="llm_refused")
    outcomes: list[object] = [refusal, "unreachable"]
    with pytest.raises(LLMError) as failure:
        await call_with_retries(
            _operation(outcomes),
            policy=RetryPolicy(max_attempts=4, base_delay_s=0.001, max_delay_s=0.002),
            classify=_classify(retryable=True),
            deadline=Deadline(None),
            provider="fake",
        )
    assert failure.value is refusal
    assert outcomes == ["unreachable"]


async def test_unsafe_provider_metadata_is_dropped() -> None:
    with pytest.raises(LLMError) as failure:
        await call_with_retries(
            _operation([_Boom()]),
            policy=RetryPolicy(max_attempts=1),
            classify=_classify(retryable=True, status=99, code="Bearer sk-secret-token"),
            deadline=Deadline(None),
            provider="fake",
        )
    error = failure.value
    assert error.provider_status is None
    assert error.provider_code is None
    rendered = f"{error} {error!r}"
    assert "sk-secret-token" not in rendered


async def test_backoff_uses_full_jitter_over_the_capped_exponential_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    outcomes: list[object] = [_Boom(), _Boom(), _Boom(), "ok"]
    value, attempts = await call_with_retries(
        _operation(outcomes),
        policy=RetryPolicy(max_attempts=4, base_delay_s=0.5, max_delay_s=1.0),
        classify=_classify(retryable=True),
        deadline=Deadline(None),
        provider="fake",
        random_source=lambda: 1.0,
    )
    assert (value, attempts) == ("ok", 4)
    # Capped exponential: 0.5, 1.0, then held at the maximum.
    assert delays == [0.5, 1.0, 1.0]


async def test_jitter_scales_the_capped_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await call_with_retries(
        _operation([_Boom(), "ok"]),
        policy=RetryPolicy(max_attempts=2, base_delay_s=0.4, max_delay_s=1.0),
        classify=_classify(retryable=True),
        deadline=Deadline(None),
        provider="fake",
        random_source=lambda: 0.25,
    )
    assert delays == [0.1]


async def test_an_expired_deadline_prevents_any_attempt() -> None:
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    with pytest.raises(LLMError) as failure:
        await call_with_retries(
            operation,
            policy=RetryPolicy(),
            classify=_classify(retryable=True),
            deadline=Deadline(-1.0),
            provider="fake",
        )
    assert failure.value.code == "llm_timeout"
    assert calls == 0


async def test_a_retry_that_cannot_fit_the_deadline_does_not_begin() -> None:
    outcomes: list[object] = [_Boom(), "unreachable"]
    with pytest.raises(LLMError) as failure:
        await call_with_retries(
            _operation(outcomes),
            policy=RetryPolicy(max_attempts=3, base_delay_s=4.0, max_delay_s=4.0),
            classify=_classify(retryable=True, status=503),
            deadline=Deadline(0.2),
            provider="fake",
            random_source=lambda: 1.0,
        )
    assert failure.value.code == "llm_timeout"
    assert failure.value.attempts == 1
    assert outcomes == ["unreachable"]


async def test_a_hanging_provider_call_is_bounded_by_the_deadline() -> None:
    async def operation() -> str:
        await asyncio.sleep(5.0)
        return "never"

    with pytest.raises(LLMError) as failure:
        await call_with_retries(
            operation,
            policy=RetryPolicy(),
            classify=_classify(retryable=True),
            deadline=Deadline(0.05),
            provider="fake",
        )
    assert failure.value.code == "llm_timeout"
    assert failure.value.attempts == 1


async def test_provider_raised_timeout_is_classified_and_retried_with_a_deadline() -> None:
    outcomes: list[object] = [TimeoutError("transport timed out"), "ok"]

    value, attempts = await call_with_retries(
        _operation(outcomes),
        policy=RetryPolicy(max_attempts=2, base_delay_s=0.001, max_delay_s=0.002),
        classify=_classify(retryable=True, code="provider_timeout"),
        deadline=Deadline(1.0),
        provider="fake",
        random_source=lambda: 0.0,
    )

    assert (value, attempts) == ("ok", 2)
    assert outcomes == []


async def test_deadline_reports_remaining_time_and_expiry() -> None:
    unbounded = Deadline(None)
    assert unbounded.bounded is False
    assert unbounded.remaining() is None
    assert unbounded.expired() is False
    bounded = Deadline(0.5)
    assert bounded.bounded is True
    remaining = bounded.remaining()
    assert remaining is not None and 0.0 < remaining <= 0.5
    assert Deadline(0.0).expired() is True


def test_retry_policy_validates_its_delay_window() -> None:
    assert RetryPolicy(base_delay_s=1.0, max_delay_s=1.0).max_delay_s == 1.0
    with pytest.raises(ValueError, match="max_delay_s"):
        RetryPolicy(base_delay_s=2.0, max_delay_s=1.0)
    with pytest.raises(ValueError, match=r"\S"):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError, match=r"\S"):
        RetryPolicy(max_attempts=7)
    with pytest.raises(ValueError, match=r"\S"):
        RetryPolicy(base_delay_s=0.0)


def test_retry_policy_is_frozen_and_carries_no_callable_state() -> None:
    policy = RetryPolicy()
    with pytest.raises(ValueError, match=r"\S"):
        policy.max_attempts = 2  # type: ignore[misc]
    assert set(policy.model_dump()) == {"max_attempts", "base_delay_s", "max_delay_s"}
