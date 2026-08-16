"""Per-call provider metrics shared between a policy and the active run trace.

A policy owns the provider response; the session owns the trace. Instead of
widening the policy protocol, the session installs one mutable per-call record
for the duration of its policy call. The record travels through the context, so
concurrent runs never observe each other's counters, and only numeric counts and
a fixed enumeration value are ever recorded.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from penampakan.models import JsonValue, LLMResponse


@dataclass(slots=True)
class PolicyCallMetrics:
    """Provider counters observed during one policy call."""

    provider_attempts: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    schema_enforcement: str | None = None

    def as_trace_data(self) -> dict[str, JsonValue]:
        """Return the redacted counters for one trace event."""
        data: dict[str, JsonValue] = {}
        if self.provider_attempts:
            data["provider_attempts"] = self.provider_attempts
        if self.input_tokens is not None:
            data["input_tokens"] = self.input_tokens
        if self.output_tokens is not None:
            data["output_tokens"] = self.output_tokens
        if self.schema_enforcement is not None:
            data["schema_enforcement"] = self.schema_enforcement
        return data


_ACTIVE_CALL: ContextVar[PolicyCallMetrics | None] = ContextVar(
    "penampakan_policy_call_metrics",
    default=None,
)


@contextmanager
def collect_policy_call() -> Iterator[PolicyCallMetrics]:
    """Collect provider counters reported during one policy call."""
    metrics = PolicyCallMetrics()
    token = _ACTIVE_CALL.set(metrics)
    try:
        yield metrics
    finally:
        _ACTIVE_CALL.reset(token)


def record_policy_response(response: LLMResponse) -> None:
    """Record one provider response against the active policy call, if any."""
    metrics = _ACTIVE_CALL.get()
    if metrics is None:
        return
    metrics.provider_attempts += response.attempts
    metrics.schema_enforcement = response.schema_enforcement.value
    usage = response.usage
    if usage is None:
        return
    if usage.input_tokens is not None:
        metrics.input_tokens = (metrics.input_tokens or 0) + usage.input_tokens
    if usage.output_tokens is not None:
        metrics.output_tokens = (metrics.output_tokens or 0) + usage.output_tokens


__all__ = ["PolicyCallMetrics", "collect_policy_call", "record_policy_response"]
