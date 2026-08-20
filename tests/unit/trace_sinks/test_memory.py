from datetime import datetime, timezone
from uuid import UUID

import pytest

from penampakan.models import TraceEvent
from penampakan.trace_sinks.memory import InMemoryTraceSink


class ManualClock:
    def __init__(self) -> None:
        self.value = 10.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def event(trace_number: int, sequence: int, event_type: str) -> TraceEvent:
    return TraceEvent(
        schema_version=2,
        trace_id=UUID(f"00000000-0000-4000-8000-{trace_number:012d}"),
        sequence=sequence,
        event_type=event_type,
        occurred_at=datetime(2026, 8, 18, tzinfo=timezone.utc),
    )


async def test_only_complete_runs_are_exposed() -> None:
    sink = InMemoryTraceSink()
    started = event(1, 0, "run_started")
    point = event(1, 1, "image_loaded")
    finished = event(1, 2, "run_finished")

    await sink.emit(started)
    await sink.emit(point)

    assert sink.events == ()
    assert dict(sink.runs) == {}
    assert (await sink.stats()).active_runs == 1

    await sink.emit(finished)

    assert sink.events == (started, point, finished)
    assert sink.runs[started.trace_id] == (started, point, finished)
    stats = await sink.stats()
    assert stats.completed_runs == 1
    assert stats.active_runs == 0
    assert stats.retained_events == 3


async def test_completed_run_eviction_updates_runs_and_events_together() -> None:
    sink = InMemoryTraceSink(max_runs=2)
    retained_by_trace: dict[UUID, tuple[TraceEvent, TraceEvent]] = {}

    for trace_number in range(1, 4):
        events = (
            event(trace_number, 0, "run_started"),
            event(trace_number, 1, "run_finished"),
        )
        retained_by_trace[events[0].trace_id] = events
        for item in events:
            await sink.emit(item)

    first_id, second_id, third_id = retained_by_trace
    assert tuple(sink.runs) == (second_id, third_id)
    assert first_id not in sink.runs
    assert sink.events == retained_by_trace[second_id] + retained_by_trace[third_id]
    stats = await sink.stats()
    assert stats.evicted_runs == 1
    assert stats.evicted_events == 2
    assert stats.retained_runs == 2


async def test_per_run_cap_preserves_terminal_and_reports_truncation() -> None:
    sink = InMemoryTraceSink(max_events_per_run=2)
    started = event(1, 0, "run_started")
    omitted_one = event(1, 1, "image_loaded")
    omitted_two = event(1, 2, "answer_validated")
    terminal = event(1, 3, "run_finished")

    for item in (started, omitted_one, omitted_two, terminal):
        await sink.emit(item)

    assert sink.events == (started, terminal)
    stats = await sink.stats()
    assert stats.truncated_runs == 1
    assert stats.dropped_events == 2
    assert stats.retained_events == 2


async def test_active_runs_are_bounded_and_expire_after_inactivity() -> None:
    clock = ManualClock()
    sink = InMemoryTraceSink(
        max_active_runs=2,
        active_ttl_s=5,
        monotonic_clock=clock,
    )

    await sink.emit(event(1, 0, "run_started"))
    await sink.emit(event(2, 0, "run_started"))
    await sink.emit(event(3, 0, "run_started"))

    stats = await sink.stats()
    assert stats.active_runs == 2
    assert stats.overflowed_active_runs == 1
    assert stats.dropped_events == 1

    clock.advance(5)
    stats = await sink.stats()
    assert stats.active_runs == 0
    assert stats.expired_active_runs == 2
    assert stats.dropped_events == 3


async def test_close_is_idempotent_and_post_close_emit_is_counted() -> None:
    sink = InMemoryTraceSink()
    started = event(1, 0, "run_started")
    finished = event(1, 1, "run_finished")
    await sink.emit(started)
    await sink.emit(finished)
    before_close = sink.events

    await sink.aclose()
    await sink.aclose()
    await sink.emit(event(2, 0, "run_started"))
    await sink.emit(event(2, 1, "run_finished"))

    assert sink.closed
    assert sink.events == before_close
    stats = await sink.stats()
    assert stats.completed_runs == 1
    assert stats.post_close_emits == 2
    assert stats.post_close_events == 2


async def test_close_discards_incomplete_runs_without_erasing_complete_runs() -> None:
    sink = InMemoryTraceSink()
    await sink.emit(event(1, 0, "run_started"))
    await sink.emit(event(1, 1, "run_finished"))
    await sink.emit(event(2, 0, "run_started"))

    await sink.aclose()

    assert tuple(sink.runs) == (event(1, 0, "run_started").trace_id,)
    stats = await sink.stats()
    assert stats.closed_active_runs == 1
    assert stats.dropped_events == 1


async def test_future_schema_event_is_ignored_and_counted() -> None:
    sink = InMemoryTraceSink()
    unsupported = TraceEvent.model_construct(
        schema_version=3,
        trace_id=UUID("00000000-0000-4000-8000-000000000001"),
        sequence=0,
        event_type="run_finished",
        occurred_at=datetime(2026, 8, 18, tzinfo=timezone.utc),
        duration_ms=None,
        invocation_id=None,
        parent_invocation_id=None,
        data={},
    )

    await sink.emit(unsupported)

    assert sink.events == ()
    assert (await sink.stats()).unsupported_schema_events == 1


@pytest.mark.parametrize(
    ("argument", "value", "error"),
    [
        ("max_runs", 0, ValueError),
        ("max_runs", True, TypeError),
        ("max_events_per_run", -1, ValueError),
        ("max_active_runs", 0, ValueError),
        ("active_ttl_s", float("inf"), ValueError),
    ],
)
def test_limits_are_validated(argument: str, value: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        InMemoryTraceSink(**{argument: value})  # type: ignore[arg-type]
