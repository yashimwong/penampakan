import asyncio
from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from penampakan.models import TraceEvent
from penampakan.tracing import REQUIRED_EVENT_TYPES, TraceBuilder

TRACE_ID = UUID("00000000-0000-4000-8000-000000000001")


class ManualClock:
    def __init__(self) -> None:
        self.wall_value = datetime(2026, 2, 10, 9, 0, tzinfo=timezone.utc)
        self.monotonic_value = 100.0

    def wall(self) -> datetime:
        return self.wall_value

    def monotonic(self) -> float:
        return self.monotonic_value

    def advance(self, milliseconds: int) -> None:
        delta = timedelta(milliseconds=milliseconds)
        self.wall_value += delta
        self.monotonic_value += milliseconds / 1000.0


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []
        self.close_count = 0

    async def emit(self, event: TraceEvent) -> None:
        self.events.append(event)

    async def aclose(self) -> None:
        self.close_count += 1


class FailingSink:
    def __init__(self) -> None:
        self.emit_count = 0
        self.close_count = 0

    async def emit(self, event: TraceEvent) -> None:
        self.emit_count += 1
        raise RuntimeError("sink-secret-sentinel")

    async def aclose(self) -> None:
        self.close_count += 1


async def test_trace_events_are_sequential_utc_and_monotonic() -> None:
    clock = ManualClock()
    builder = TraceBuilder(
        trace_id=TRACE_ID,
        wall_clock=clock.wall,
        monotonic_clock=clock.monotonic,
    )

    started = await builder.start({"operation": "inspect"})
    clock.advance(125)
    backend_started = await builder.emit("backend_call_started", {"backend_name": "local"})
    clock.advance(75)
    backend_finished = await builder.emit(
        "backend_call_finished",
        {"backend_name": "local"},
        duration_ms=75,
    )
    clock.advance(150)
    trace = await builder.finish()

    assert tuple(event.sequence for event in trace.events) == (0, 1, 2, 3)
    assert tuple(event.event_type for event in trace.events) == (
        "run_started",
        "backend_call_started",
        "backend_call_finished",
        "run_finished",
    )
    assert started.occurred_at < backend_started.occurred_at < backend_finished.occurred_at
    assert all(event.occurred_at.tzinfo is timezone.utc for event in trace.events)
    assert backend_finished.duration_ms == 75
    assert trace.events[-1].duration_ms == 350
    assert trace.summary.started_at == datetime(2026, 2, 10, 9, 0, tzinfo=timezone.utc)
    assert trace.summary.duration_ms == 350
    assert trace.summary.stop_reason == "completed"


async def test_trace_counts_initial_cache_fallback_repair_and_final_calls_exactly() -> None:
    builder = TraceBuilder(trace_id=TRACE_ID)

    await builder.emit("initial_plan_started", {"phase": "initial"})
    await builder.emit("tool_call_started", {"phase": "initial"})
    await builder.emit("cache_hit", {"phase": "initial"})
    await builder.emit("backend_call_started", {"attempt": 1})
    await builder.emit("backend_call_started", {"attempt": 2, "fallback": True})
    await builder.emit("policy_call_started", {"phase": "interactive"})
    await builder.emit("policy_call_started", {"phase": "repair"})
    await builder.emit("policy_call_started", {"phase": "final", "answer_only": True})
    await builder.emit(
        "policy_call_finished",
        {"input_tokens": 20, "output_tokens": 5},
    )
    await builder.emit("asset_created", {"asset_id": "img_0123456789abcdef"})
    trace = await builder.finish()

    assert trace.summary.llm_calls == 3
    assert trace.summary.tool_calls == 1
    assert trace.summary.backend_calls == 2
    assert trace.summary.cache_hits == 1
    assert trace.summary.derived_assets == 1
    assert trace.summary.input_tokens == 20
    assert trace.summary.output_tokens == 5


async def test_external_counts_are_atomic_and_reject_updates_after_finish() -> None:
    builder = TraceBuilder(trace_id=TRACE_ID)

    await asyncio.gather(
        *(
            builder.add_counts(
                llm_calls=1,
                tool_calls=1,
                backend_calls=2,
                cache_hits=1,
                derived_assets=1,
                input_tokens=3,
                output_tokens=2,
            )
            for _ in range(20)
        )
    )
    trace = await builder.finish()

    assert trace.summary.llm_calls == 20
    assert trace.summary.tool_calls == 20
    assert trace.summary.backend_calls == 40
    assert trace.summary.cache_hits == 20
    assert trace.summary.derived_assets == 20
    assert trace.summary.input_tokens == 60
    assert trace.summary.output_tokens == 40

    with pytest.raises(RuntimeError):
        await builder.add_counts(llm_calls=1)


async def test_success_error_timeout_and_cancel_have_required_lifecycle_events() -> None:
    success = TraceBuilder(trace_id=UUID("00000000-0000-4000-8000-000000000002"))
    success_trace = await success.finish()

    failure = TraceBuilder(trace_id=UUID("00000000-0000-4000-8000-000000000003"))
    failure_trace = await failure.fail(RuntimeError("backend-secret-sentinel"))

    timeout = TraceBuilder(trace_id=UUID("00000000-0000-4000-8000-000000000004"))
    await timeout.emit("budget_stop", {"reason": "timeout"})
    timeout_trace = await timeout.finish("timeout")

    cancelled = TraceBuilder(trace_id=UUID("00000000-0000-4000-8000-000000000005"))
    cancelled_trace = await cancelled.cancel()

    assert tuple(event.event_type for event in success_trace.events) == (
        "run_started",
        "run_finished",
    )
    assert tuple(event.event_type for event in failure_trace.events) == (
        "run_started",
        "run_failed",
    )
    assert failure_trace.events[-1].data["error"] == {"error_type": "RuntimeError"}
    assert tuple(event.event_type for event in timeout_trace.events) == (
        "run_started",
        "budget_stop",
        "run_finished",
    )
    assert tuple(event.event_type for event in cancelled_trace.events) == (
        "run_started",
        "run_finished",
    )
    assert success_trace.summary.stop_reason == "completed"
    assert failure_trace.summary.stop_reason == "error"
    assert timeout_trace.summary.stop_reason == "timeout"
    assert cancelled_trace.summary.stop_reason == "cancelled"
    assert {"run_started", "run_finished", "run_failed", "budget_stop"}.issubset(
        REQUIRED_EVENT_TYPES
    )


async def test_failing_sink_never_fails_trace_and_warns_once() -> None:
    failing = FailingSink()
    recording = RecordingSink()
    builder = TraceBuilder(trace_id=TRACE_ID, sinks=(failing, recording))

    await builder.start()
    await builder.emit("image_loaded", {"asset_id": "img_0123456789abcdef"})
    trace = await builder.finish()

    assert tuple(event.event_type for event in recording.events) == tuple(
        event.event_type for event in trace.events
    )
    assert failing.emit_count == 1
    assert tuple(warning.code for warning in builder.warnings) == ("trace_sink_failed",)
    assert "sink-secret-sentinel" not in repr(builder.warnings)

    await asyncio.gather(builder.aclose(), builder.aclose())
    await builder.aclose()

    assert failing.close_count == 1
    assert recording.close_count == 1


async def test_start_finish_and_close_are_idempotent() -> None:
    sink = RecordingSink()
    builder = TraceBuilder(trace_id=TRACE_ID, sinks=(sink,))

    first_start = await builder.start()
    second_start = await builder.start()
    first_trace = await builder.finish()
    second_trace = await builder.finish()

    assert first_start is second_start
    assert first_trace is second_trace
    assert builder.finalized

    with pytest.raises(RuntimeError):
        await builder.emit("image_loaded")

    await asyncio.gather(builder.aclose(), builder.aclose(), builder.aclose())
    await builder.aclose()

    assert sink.close_count == 1
