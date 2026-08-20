from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import pytest

from penampakan.models import TraceEvent
from penampakan.trace_sinks import opentelemetry as otel_sink

TRACE_ID = UUID("00000000-0000-4000-8000-000000000101")
OTHER_TRACE_ID = UUID("00000000-0000-4000-8000-000000000102")
STARTED_AT = datetime(2026, 1, 1, tzinfo=timezone.utc)


class _StatusCode:
    OK = "ok"
    ERROR = "error"


@dataclass(frozen=True)
class _Status:
    status_code: str


@dataclass(frozen=True)
class _Context:
    span: _Span


class _SpanKind:
    INTERNAL = "internal"


@dataclass
class _Span:
    name: str
    parent: _Span | None
    attributes: dict[str, object]
    start_time: int
    status: _Status | None = None
    end_time: int | None = None
    events: list[tuple[str, dict[str, object], int]] = field(default_factory=list)

    def set_attribute(self, name: str, value: object) -> None:
        self.attributes[name] = value

    def set_status(self, status: _Status) -> None:
        self.status = status

    def add_event(
        self,
        name: str,
        attributes: dict[str, object],
        timestamp: int,
    ) -> None:
        self.events.append((name, attributes, timestamp))

    def end(self, *, end_time: int) -> None:
        self.end_time = end_time


class _Tracer:
    def __init__(self) -> None:
        self.spans: list[_Span] = []

    def start_span(
        self,
        name: str,
        *,
        context: _Context | None = None,
        kind: str,
        attributes: dict[str, object],
        start_time: int,
    ) -> _Span:
        assert kind == _SpanKind.INTERNAL
        span = _Span(name, None if context is None else context.span, attributes, start_time)
        self.spans.append(span)
        return span


class _Provider:
    def __init__(self) -> None:
        self.tracer = _Tracer()
        self.instrumentation_name: str | None = None

    def get_tracer(self, instrumentation_name: str) -> _Tracer:
        self.instrumentation_name = instrumentation_name
        return self.tracer


@pytest.fixture(autouse=True)
def _fake_otel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        otel_sink,
        "_load_opentelemetry",
        lambda: (_SpanKind, _Status, _StatusCode, _Context),
    )


def _event(
    sequence: int,
    event_type: str,
    *,
    trace_id: UUID = TRACE_ID,
    invocation_id: str | None = None,
    parent_invocation_id: str | None = None,
    data: dict[str, Any] | None = None,
) -> TraceEvent:
    return TraceEvent.model_validate(
        {
            "schema_version": 2,
            "trace_id": trace_id,
            "sequence": sequence,
            "event_type": event_type,
            "occurred_at": STARTED_AT + timedelta(milliseconds=sequence),
            "invocation_id": invocation_id,
            "parent_invocation_id": parent_invocation_id,
            "data": data or {},
        }
    )


async def test_correlates_span_tree_by_ids_and_maps_only_safe_attributes() -> None:
    provider = _Provider()
    sink = otel_sink.OpenTelemetryTraceSink(provider)

    await sink.emit(_event(0, "run_started", data={"operation": "ask", "question": "secret"}))
    await sink.emit(_event(1, "tool_call_started", invocation_id="tool-1"))
    await sink.emit(
        _event(
            2,
            "backend_call_started",
            invocation_id="backend-1",
            parent_invocation_id="tool-1",
            data={"backend_name": "local", "raw_output": "never-export"},
        )
    )
    await sink.emit(
        _event(
            3,
            "cache_hit",
            parent_invocation_id="backend-1",
            data={"backend_name": "local"},
        )
    )
    await sink.emit(
        _event(
            4,
            "backend_call_finished",
            invocation_id="backend-1",
            data={"outcome": "ok"},
        )
    )
    await sink.emit(_event(5, "tool_call_finished", invocation_id="tool-1", data={"outcome": "ok"}))
    await sink.emit(
        _event(
            6,
            "run_finished",
            data={
                "stop_reason": "completed",
                "input_tokens": 11,
                "output_tokens": 7,
                "answer": "never-export",
            },
        )
    )

    run, tool, backend = provider.tracer.spans
    assert [span.name for span in provider.tracer.spans] == [
        "penampakan.ask",
        "penampakan.tool",
        "penampakan.backend",
    ]
    assert tool.parent is run
    assert backend.parent is tool
    assert backend.events[0][0] == "cache_hit"
    assert run.attributes["penampakan.trace_id"] == str(TRACE_ID)
    assert run.attributes["gen_ai.usage.input_tokens"] == 11
    assert run.attributes["gen_ai.usage.output_tokens"] == 7
    serialized_attributes = repr([span.attributes for span in provider.tracer.spans])
    assert "secret" not in serialized_attributes
    assert "never-export" not in serialized_attributes
    assert run.status == _Status(_StatusCode.OK)
    assert tool.status == _Status(_StatusCode.OK)
    assert backend.status == _Status(_StatusCode.OK)
    assert sink.stats().active_traces == 0


async def test_completed_invocation_remains_a_resolvable_parent() -> None:
    provider = _Provider()
    sink = otel_sink.OpenTelemetryTraceSink(provider)

    await sink.emit(_event(0, "run_started", invocation_id="run", data={"operation": "ask"}))
    await sink.emit(
        _event(1, "policy_call_started", invocation_id="policy", parent_invocation_id="run")
    )
    await sink.emit(
        _event(
            2,
            "policy_call_finished",
            invocation_id="policy",
            parent_invocation_id="run",
            data={"outcome": "ok"},
        )
    )
    await sink.emit(
        _event(3, "tool_call_started", invocation_id="tool", parent_invocation_id="policy")
    )
    await sink.emit(_event(4, "tool_call_finished", invocation_id="tool", data={"outcome": "ok"}))
    await sink.emit(_event(5, "run_finished", data={"stop_reason": "completed"}))

    run, policy, tool = provider.tracer.spans
    assert policy.parent is run
    assert tool.parent is policy
    assert sink.stats().unknown_parent_events == 0
    assert sink.stats().malformed_events == 0


@pytest.mark.parametrize(
    ("stop_reason", "expected_status"),
    [
        ("insufficient_evidence", _StatusCode.OK),
        ("tool_limit", _StatusCode.OK),
        ("cancelled", None),
        ("timeout", _StatusCode.ERROR),
        ("error", _StatusCode.ERROR),
    ],
)
async def test_run_status_is_domain_aware(stop_reason: str, expected_status: str | None) -> None:
    provider = _Provider()
    sink = otel_sink.OpenTelemetryTraceSink(provider)

    await sink.emit(_event(0, "run_started", data={"operation": "inspect"}))
    await sink.emit(_event(1, "run_finished", data={"stop_reason": stop_reason}))

    run = provider.tracer.spans[0]
    assert run.attributes["penampakan.stop_reason"] == stop_reason
    assert run.status == (None if expected_status is None else _Status(expected_status))


async def test_malformed_ordering_is_counted_without_creating_phantom_spans() -> None:
    provider = _Provider()
    sink = otel_sink.OpenTelemetryTraceSink(provider)

    await sink.emit(
        _event(
            0,
            "backend_call_finished",
            invocation_id="never-started",
            data={"outcome": "error"},
        )
    )
    await sink.emit(_event(0, "run_started", data={"operation": "ask"}))
    await sink.emit(_event(1, "tool_call_started", invocation_id="tool"))
    await sink.emit(_event(1, "tool_call_started", invocation_id="tool"))
    await sink.emit(
        _event(
            2,
            "backend_call_started",
            invocation_id="backend",
            parent_invocation_id="unknown",
        )
    )
    await sink.emit(_event(0, "image_loaded"))
    await sink.emit(_event(3, "run_finished", data={"stop_reason": "completed"}))

    stats = sink.stats()
    assert stats.missing_start_events == 1
    assert stats.duplicate_events == 1
    assert stats.unknown_parent_events == 1
    assert stats.out_of_order_events == 1
    assert stats.malformed_events == 4
    assert len(provider.tracer.spans) == 3
    assert stats.incomplete_spans == 2


async def test_ttl_and_capacity_bound_abandoned_trace_state() -> None:
    now = 0.0

    def monotonic() -> float:
        return now

    provider = _Provider()
    sink = otel_sink.OpenTelemetryTraceSink(
        provider,
        trace_ttl_s=5,
        max_active_traces=1,
        monotonic_clock=monotonic,
    )
    await sink.emit(_event(0, "run_started", data={"operation": "ask"}))
    await sink.emit(_event(1, "tool_call_started", invocation_id="tool"))

    now = 6.0
    await sink.emit(
        _event(0, "run_started", trace_id=OTHER_TRACE_ID, data={"operation": "inspect"})
    )

    stats = sink.stats()
    assert stats.expired_traces == 1
    assert stats.active_traces == 1
    assert stats.incomplete_spans == 2
    assert provider.tracer.spans[0].end_time is not None
    assert provider.tracer.spans[1].end_time is not None


async def test_close_is_idempotent_and_post_close_emit_is_counted() -> None:
    provider = _Provider()
    sink = otel_sink.OpenTelemetryTraceSink(provider, wall_clock_ns=lambda: 10**18)
    await sink.emit(_event(0, "run_started", data={"operation": "ask"}))

    await sink.aclose()
    await sink.aclose()
    await sink.emit(_event(1, "run_finished", data={"stop_reason": "completed"}))

    stats = sink.stats()
    assert stats.closed
    assert sink.closed
    assert stats.active_traces == 0
    assert stats.incomplete_spans == 1
    assert stats.post_close_emits == 1
    assert stats.post_close_events == 1


def test_missing_optional_dependency_has_an_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing() -> tuple[object, object, object, object]:
        raise otel_sink.OpenTelemetryUnavailableError("install the extra")

    monkeypatch.setattr(otel_sink, "_load_opentelemetry", missing)

    with pytest.raises(otel_sink.OpenTelemetryUnavailableError, match="install the extra"):
        otel_sink.OpenTelemetryTraceSink(_Provider())
