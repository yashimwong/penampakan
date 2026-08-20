"""An optional OpenTelemetry destination for correlated trace events.

The OpenTelemetry dependency is deliberately imported only when the sink is
constructed.  Importing :mod:`penampakan`, or even this module, therefore stays
safe in a base installation.
"""

from __future__ import annotations

import asyncio
import importlib
import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from penampakan.models import JsonValue, TraceEvent

_INSTRUMENTATION_NAME = "penampakan.trace_sinks.opentelemetry"

_START_EVENTS = {
    "policy_call_started": ("policy_call_finished", "penampakan.policy"),
    "tool_call_started": ("tool_call_finished", "penampakan.tool"),
    "backend_call_started": ("backend_call_finished", "penampakan.backend"),
    "verification_started": ("verification_finished", "penampakan.verification"),
}
_FINISH_EVENTS = {finish: start for start, (finish, _name) in _START_EVENTS.items()}
_TERMINAL_EVENTS = frozenset({"run_finished", "run_failed"})
_OK_STOP_REASONS = frozenset(
    {
        "completed",
        "insufficient_evidence",
        "step_limit",
        "llm_limit",
        "tool_limit",
        "backend_limit",
        "asset_limit",
        "depth_limit",
        "context_limit",
    }
)
_SAFE_ATTRIBUTE_KEYS = frozenset(
    {
        "action_type",
        "asset_id",
        "backend_calls",
        "backend_name",
        "cache_failures",
        "cache_hit",
        "cache_hits",
        "capability",
        "derived_assets",
        "error_code",
        "llm_calls",
        "operation",
        "outcome",
        "reason",
        "repair",
        "stop_reason",
        "tool_calls",
        "tool_name",
    }
)


class OpenTelemetryUnavailableError(ImportError):
    """Raised when the optional OpenTelemetry API is not installed."""


@dataclass(frozen=True, slots=True)
class OpenTelemetryTraceSinkStats:
    """A bounded summary of sink activity and malformed input."""

    emitted_events: int
    completed_traces: int
    active_traces: int
    active_spans: int
    malformed_events: int
    out_of_order_events: int
    duplicate_events: int
    unknown_parent_events: int
    missing_start_events: int
    unsupported_schema_events: int
    expired_traces: int
    capacity_evictions: int
    incomplete_spans: int
    post_close_emits: int
    closed: bool

    @property
    def post_close_events(self) -> int:
        """Alias for events offered after the sink was closed."""
        return self.post_close_emits


@dataclass(slots=True)
class _SpanRecord:
    span: Any
    expected_finish: str
    started_ns: int


@dataclass(slots=True)
class _TraceState:
    run_span: Any
    run_invocation_id: str | None
    run_started_ns: int
    last_sequence: int
    last_seen: float
    spans: dict[str, _SpanRecord] = field(default_factory=dict)
    completed_ids: deque[str] = field(default_factory=deque)
    completed_id_set: set[str] = field(default_factory=set)
    completed_parent_spans: dict[str, Any] = field(default_factory=dict)


def _load_opentelemetry() -> tuple[Any, Any, Any, Any]:
    try:
        trace = importlib.import_module("opentelemetry.trace")
    except ImportError as error:
        raise OpenTelemetryUnavailableError(
            "OpenTelemetryTraceSink requires the optional OpenTelemetry API and SDK; "
            "install penampakan with its OpenTelemetry extra"
        ) from error
    return trace.SpanKind, trace.Status, trace.StatusCode, trace.set_span_in_context


class OpenTelemetryTraceSink:
    """Export schema-v2 trace events through an injected tracer provider.

    The sink never reads or configures OpenTelemetry's global provider.  State
    is limited by ``max_active_traces``, ``max_spans_per_trace``, and
    ``trace_ttl_s``. TTL cleanup is opportunistic: it runs on every ``emit``;
    ``aclose`` provides deterministic final cleanup.
    """

    def __init__(
        self,
        tracer_provider: object,
        *,
        trace_ttl_s: float = 300.0,
        max_active_traces: int = 256,
        max_spans_per_trace: int = 256,
        monotonic_clock: Callable[[], float] = time.monotonic,
        wall_clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        if not callable(getattr(tracer_provider, "get_tracer", None)):
            raise TypeError("tracer_provider must provide get_tracer()")
        if (
            isinstance(trace_ttl_s, bool)
            or not isinstance(trace_ttl_s, (int, float))
            or not math.isfinite(trace_ttl_s)
            or trace_ttl_s <= 0
        ):
            raise ValueError("trace_ttl_s must be a finite positive number")
        for name, value in (
            ("max_active_traces", max_active_traces),
            ("max_spans_per_trace", max_spans_per_trace),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not callable(monotonic_clock) or not callable(wall_clock_ns):
            raise TypeError("sink clocks must be callable")

        span_kind, status, status_code, set_span_in_context = _load_opentelemetry()
        provider = tracer_provider
        self._tracer = provider.get_tracer(_INSTRUMENTATION_NAME)  # type: ignore[attr-defined]
        self._span_kind = span_kind
        self._status = status
        self._status_code = status_code
        self._set_span_in_context = set_span_in_context
        self._trace_ttl_s = float(trace_ttl_s)
        self._max_active_traces = max_active_traces
        self._max_spans_per_trace = max_spans_per_trace
        self._monotonic_clock = monotonic_clock
        self._wall_clock_ns = wall_clock_ns
        self._states: dict[UUID, _TraceState] = {}
        self._lock = asyncio.Lock()
        self._closed = False

        self._emitted_events = 0
        self._completed_traces = 0
        self._malformed_events = 0
        self._out_of_order_events = 0
        self._duplicate_events = 0
        self._unknown_parent_events = 0
        self._missing_start_events = 0
        self._unsupported_schema_events = 0
        self._expired_traces = 0
        self._capacity_evictions = 0
        self._incomplete_spans = 0
        self._post_close_events = 0

    async def emit(self, event: TraceEvent) -> None:
        """Map one already-redacted schema-v2 event to OpenTelemetry."""
        async with self._lock:
            if self._closed:
                self._post_close_events += 1
                return

            now = self._validated_monotonic()
            event_ns = self._event_ns(event)
            self._evict_expired(now, event_ns)

            if getattr(event, "schema_version", 1) != 2:
                self._diagnose("unsupported_schema")
                return

            state = self._states.get(event.trace_id)
            if state is None:
                if event.event_type != "run_started":
                    self._diagnose("missing_start")
                    return
                self._make_room_for_trace(event_ns)
                self._start_run(event, now, event_ns)
                self._emitted_events += 1
                return

            if event.sequence <= state.last_sequence:
                if event.sequence == state.last_sequence:
                    self._diagnose("duplicate")
                else:
                    self._diagnose("out_of_order")
                return
            state.last_sequence = event.sequence
            state.last_seen = now

            if event.event_type == "run_started":
                self._diagnose("duplicate")
                return
            if event.event_type in _TERMINAL_EVENTS:
                self._finish_run(event, state, event_ns)
            elif event.event_type in _START_EVENTS:
                self._start_operation(event, state, event_ns)
            elif event.event_type in _FINISH_EVENTS:
                self._finish_operation(event, state, event_ns)
            else:
                self._add_point_event(event, state, event_ns)
            self._emitted_events += 1

    async def aclose(self) -> None:
        """Close every live span as incomplete; repeated calls are harmless."""
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            end_ns = self._wall_clock_ns()
            for trace_id in tuple(self._states):
                state = self._states.pop(trace_id)
                self._close_incomplete(state, end_ns)

    def stats(self) -> OpenTelemetryTraceSinkStats:
        """Return current counters without exposing trace content or identifiers."""
        return OpenTelemetryTraceSinkStats(
            emitted_events=self._emitted_events,
            completed_traces=self._completed_traces,
            active_traces=len(self._states),
            active_spans=sum(len(state.spans) for state in self._states.values()),
            malformed_events=self._malformed_events,
            out_of_order_events=self._out_of_order_events,
            duplicate_events=self._duplicate_events,
            unknown_parent_events=self._unknown_parent_events,
            missing_start_events=self._missing_start_events,
            unsupported_schema_events=self._unsupported_schema_events,
            expired_traces=self._expired_traces,
            capacity_evictions=self._capacity_evictions,
            incomplete_spans=self._incomplete_spans,
            post_close_emits=self._post_close_events,
            closed=self._closed,
        )

    @property
    def closed(self) -> bool:
        """Return whether this sink has stopped accepting events."""
        return self._closed

    def _validated_monotonic(self) -> float:
        value = self._monotonic_clock()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("monotonic clock must return a number")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("monotonic clock must return a finite number")
        return numeric

    @staticmethod
    def _event_ns(event: TraceEvent) -> int:
        return int(event.occurred_at.timestamp() * 1_000_000_000)

    def _start_run(self, event: TraceEvent, now: float, event_ns: int) -> None:
        operation = event.data.get("operation")
        name = f"penampakan.{operation}" if operation in {"ask", "inspect"} else "penampakan.run"
        attributes = self._attributes(event)
        span = self._tracer.start_span(
            name,
            kind=self._span_kind.INTERNAL,
            attributes=attributes,
            start_time=event_ns,
        )
        self._states[event.trace_id] = _TraceState(
            run_span=span,
            run_invocation_id=event.invocation_id,
            run_started_ns=event_ns,
            last_sequence=event.sequence,
            last_seen=now,
        )

    def _start_operation(self, event: TraceEvent, state: _TraceState, event_ns: int) -> None:
        invocation_id = getattr(event, "invocation_id", None)
        if not isinstance(invocation_id, str) or not invocation_id:
            self._diagnose("missing_start")
            return
        if invocation_id in state.spans or invocation_id in state.completed_id_set:
            self._diagnose("duplicate")
            return

        if len(state.spans) >= self._max_spans_per_trace:
            oldest_id = next(iter(state.spans))
            record = state.spans.pop(oldest_id)
            self._end_incomplete(record, event_ns)
            self._remember_completed(state, oldest_id, record.span)
            self._capacity_evictions += 1

        parent = self._parent_span(event, state)
        context = self._set_span_in_context(parent)
        expected_finish, name = _START_EVENTS[event.event_type]
        span = self._tracer.start_span(
            name,
            context=context,
            kind=self._span_kind.INTERNAL,
            attributes=self._attributes(event),
            start_time=event_ns,
        )
        state.spans[invocation_id] = _SpanRecord(span, expected_finish, event_ns)

    def _finish_operation(self, event: TraceEvent, state: _TraceState, event_ns: int) -> None:
        invocation_id = getattr(event, "invocation_id", None)
        if not isinstance(invocation_id, str) or not invocation_id:
            self._diagnose("missing_start")
            return
        record = state.spans.get(invocation_id)
        if record is None:
            if invocation_id in state.completed_id_set:
                self._diagnose("duplicate")
            else:
                self._diagnose("missing_start")
            return
        if record.expected_finish != event.event_type:
            self._diagnose("missing_start")
            return

        state.spans.pop(invocation_id)
        for key, value in self._attributes(event).items():
            record.span.set_attribute(key, value)
        outcome = event.data.get("outcome")
        if not isinstance(outcome, str) or not outcome:
            outcome = "incomplete"
            record.span.set_attribute("penampakan.outcome", outcome)
            self._malformed_events += 1
        self._set_outcome_status(record.span, outcome)
        record.span.end(end_time=max(event_ns, record.started_ns))
        self._remember_completed(state, invocation_id, record.span)

    def _finish_run(self, event: TraceEvent, state: _TraceState, event_ns: int) -> None:
        for invocation_id, record in reversed(tuple(state.spans.items())):
            self._end_incomplete(record, event_ns)
            self._remember_completed(state, invocation_id, record.span)
        state.spans.clear()

        for key, value in self._attributes(event).items():
            state.run_span.set_attribute(key, value)
        stop_reason = event.data.get("stop_reason")
        if not isinstance(stop_reason, str) or not stop_reason:
            stop_reason = "error" if event.event_type == "run_failed" else "completed"
        state.run_span.set_attribute("penampakan.stop_reason", stop_reason)
        self._set_run_status(state.run_span, stop_reason)
        state.run_span.end(end_time=max(event_ns, state.run_started_ns))
        self._states.pop(event.trace_id, None)
        self._completed_traces += 1

    def _add_point_event(self, event: TraceEvent, state: _TraceState, event_ns: int) -> None:
        span = self._parent_span(event, state)
        span.add_event(event.event_type, attributes=self._attributes(event), timestamp=event_ns)

    def _parent_span(self, event: TraceEvent, state: _TraceState) -> Any:
        parent_id = getattr(event, "parent_invocation_id", None)
        if parent_id is None or parent_id == state.run_invocation_id:
            return state.run_span
        if isinstance(parent_id, str) and parent_id in state.spans:
            return state.spans[parent_id].span
        if isinstance(parent_id, str) and parent_id in state.completed_parent_spans:
            return state.completed_parent_spans[parent_id]
        self._diagnose("unknown_parent")
        return state.run_span

    def _attributes(self, event: TraceEvent) -> dict[str, Any]:
        attributes: dict[str, Any] = {
            "penampakan.trace_id": str(event.trace_id),
            "penampakan.schema_version": 2,
            "penampakan.sequence": event.sequence,
        }
        if event.duration_ms is not None:
            attributes["penampakan.duration_ms"] = event.duration_ms
        for key in _SAFE_ATTRIBUTE_KEYS:
            value = event.data.get(key)
            if self._is_attribute_value(value):
                attributes[f"penampakan.{key}"] = value
        for source, target in (
            ("input_tokens", "gen_ai.usage.input_tokens"),
            ("output_tokens", "gen_ai.usage.output_tokens"),
        ):
            value = event.data.get(source)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                attributes[target] = value
        return attributes

    @staticmethod
    def _is_attribute_value(value: JsonValue | None) -> bool:
        return isinstance(value, (bool, int, float, str))

    def _set_outcome_status(self, span: Any, outcome: str) -> None:
        if outcome in {"error", "timeout", "incomplete"}:
            span.set_status(self._status(self._status_code.ERROR))
        elif outcome != "cancelled":
            span.set_status(self._status(self._status_code.OK))

    def _set_run_status(self, span: Any, stop_reason: str) -> None:
        if stop_reason in _OK_STOP_REASONS:
            span.set_status(self._status(self._status_code.OK))
        elif stop_reason in {"timeout", "error"}:
            span.set_status(self._status(self._status_code.ERROR))

    def _remember_completed(
        self,
        state: _TraceState,
        invocation_id: str,
        parent_span: Any,
    ) -> None:
        if len(state.completed_ids) >= self._max_spans_per_trace:
            removed = state.completed_ids.popleft()
            state.completed_id_set.discard(removed)
            state.completed_parent_spans.pop(removed, None)
        state.completed_ids.append(invocation_id)
        state.completed_id_set.add(invocation_id)
        state.completed_parent_spans[invocation_id] = parent_span

    def _evict_expired(self, now: float, end_ns: int) -> None:
        expired = tuple(
            trace_id
            for trace_id, state in self._states.items()
            if now - state.last_seen >= self._trace_ttl_s
        )
        for trace_id in expired:
            state = self._states.pop(trace_id)
            self._close_incomplete(state, end_ns)
            self._expired_traces += 1

    def _make_room_for_trace(self, end_ns: int) -> None:
        if len(self._states) < self._max_active_traces:
            return
        oldest_id = min(self._states, key=lambda trace_id: self._states[trace_id].last_seen)
        state = self._states.pop(oldest_id)
        self._close_incomplete(state, end_ns)
        self._capacity_evictions += 1

    def _close_incomplete(self, state: _TraceState, end_ns: int) -> None:
        for record in reversed(tuple(state.spans.values())):
            self._end_incomplete(record, end_ns)
        state.spans.clear()
        state.run_span.set_attribute("penampakan.outcome", "incomplete")
        state.run_span.set_status(self._status(self._status_code.ERROR))
        state.run_span.end(end_time=max(end_ns, state.run_started_ns))
        self._incomplete_spans += 1

    def _end_incomplete(self, record: _SpanRecord, end_ns: int) -> None:
        record.span.set_attribute("penampakan.outcome", "incomplete")
        record.span.set_status(self._status(self._status_code.ERROR))
        record.span.end(end_time=max(end_ns, record.started_ns))
        self._incomplete_spans += 1

    def _diagnose(self, kind: str) -> None:
        self._malformed_events += 1
        if kind == "out_of_order":
            self._out_of_order_events += 1
        elif kind == "duplicate":
            self._duplicate_events += 1
        elif kind == "unknown_parent":
            self._unknown_parent_events += 1
        elif kind == "missing_start":
            self._missing_start_events += 1
        elif kind == "unsupported_schema":
            self._unsupported_schema_events += 1


__all__ = [
    "OpenTelemetryTraceSink",
    "OpenTelemetryTraceSinkStats",
    "OpenTelemetryUnavailableError",
]
