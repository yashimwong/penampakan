"""Bounded process-local recording of complete trace runs."""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from uuid import UUID

from penampakan.models import TraceEvent

__all__ = ["InMemoryTraceSink", "InMemoryTraceSinkStats"]

_TERMINAL_EVENT_TYPES = frozenset({"run_finished", "run_failed"})
_SUPPORTED_SCHEMA_VERSIONS = frozenset({1, 2})


@dataclass(frozen=True, slots=True)
class InMemoryTraceSinkStats:
    """Immutable accounting snapshot for :class:`InMemoryTraceSink`."""

    completed_runs: int
    active_runs: int
    retained_events: int
    truncated_runs: int
    dropped_events: int
    evicted_runs: int
    evicted_events: int
    expired_active_runs: int
    overflowed_active_runs: int
    closed_active_runs: int
    post_close_emits: int
    unsupported_schema_events: int
    duplicate_terminal_events: int

    @property
    def retained_runs(self) -> int:
        """Alias describing the number of completed runs currently retained."""

        return self.completed_runs

    @property
    def post_close_events(self) -> int:
        """Alias for events offered after the sink was closed."""

        return self.post_close_emits


@dataclass(slots=True)
class _ActiveRun:
    events: list[TraceEvent]
    last_seen: float
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class _CompletedRun:
    events: tuple[TraceEvent, ...]
    truncated: bool


class InMemoryTraceSink:
    """Record bounded, complete trace runs for tests and notebooks.

    ``events`` is a flattened snapshot of ``runs``. Active (unterminated) runs
    are deliberately absent from both views, so capacity eviction can never
    leave a fragment of a formerly completed run. Active buffers are bounded
    separately and expire after ``active_ttl_s`` seconds without an event.

    Closing is idempotent and preserves completed snapshots for inspection.
    Events offered after close are ignored and counted in :meth:`stats`.
    """

    __slots__ = (
        "_active",
        "_active_ttl_s",
        "_closed",
        "_closed_active_runs",
        "_completed",
        "_dropped_events",
        "_duplicate_terminal_events",
        "_evicted_events",
        "_evicted_runs",
        "_expired_active_runs",
        "_lock",
        "_max_active_runs",
        "_max_events_per_run",
        "_max_runs",
        "_monotonic_clock",
        "_overflowed_active_runs",
        "_post_close_emits",
        "_truncated_runs",
        "_unsupported_schema_events",
    )

    def __init__(
        self,
        *,
        max_runs: int = 100,
        max_events_per_run: int | None = None,
        max_active_runs: int = 100,
        active_ttl_s: float = 300.0,
        monotonic_clock: Callable[[], float] | None = None,
    ) -> None:
        self._max_runs = _positive_int("max_runs", max_runs)
        self._max_events_per_run = (
            None
            if max_events_per_run is None
            else _positive_int("max_events_per_run", max_events_per_run)
        )
        self._max_active_runs = _positive_int("max_active_runs", max_active_runs)
        self._active_ttl_s = _positive_float("active_ttl_s", active_ttl_s)
        self._monotonic_clock = monotonic_clock or time.monotonic
        self._active: OrderedDict[UUID, _ActiveRun] = OrderedDict()
        self._completed: OrderedDict[UUID, _CompletedRun] = OrderedDict()
        self._lock = threading.Lock()
        self._closed = False
        self._truncated_runs = 0
        self._dropped_events = 0
        self._evicted_runs = 0
        self._evicted_events = 0
        self._expired_active_runs = 0
        self._overflowed_active_runs = 0
        self._closed_active_runs = 0
        self._post_close_emits = 0
        self._unsupported_schema_events = 0
        self._duplicate_terminal_events = 0

    async def emit(self, event: TraceEvent) -> None:
        """Retain an already-redacted event without blocking on external I/O."""

        if not isinstance(event, TraceEvent):
            raise TypeError("event must be a TraceEvent")
        with self._lock:
            if self._closed:
                self._post_close_emits += 1
                return
            schema_version = getattr(event, "schema_version", 1)
            if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
                self._unsupported_schema_events += 1
                return

            now = self._now()
            self._expire_active(now)
            if event.trace_id in self._completed:
                # A terminal run is immutable. In particular, a duplicate
                # terminal must not turn into a second partial run.
                self._duplicate_terminal_events += 1
                return

            active = self._active.get(event.trace_id)
            if active is None:
                active = _ActiveRun(events=[], last_seen=now)
                self._active[event.trace_id] = active
            else:
                active.last_seen = now
                self._active.move_to_end(event.trace_id)

            self._append_bounded(active, event)
            if event.event_type in _TERMINAL_EVENT_TYPES:
                self._complete(event.trace_id, active)
            else:
                self._bound_active()

    async def aclose(self) -> None:
        """Stop admission idempotently, discarding only incomplete active runs."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._closed_active_runs += len(self._active)
            self._dropped_events += sum(len(run.events) for run in self._active.values())
            self._active.clear()

    async def stats(self) -> InMemoryTraceSinkStats:
        """Return consistent retention and loss accounting."""

        with self._lock:
            if not self._closed:
                self._expire_active(self._now())
            return self._stats()

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        """Return all retained completed events, grouped in completion order."""

        with self._lock:
            if not self._closed:
                self._expire_active(self._now())
            return tuple(event for run in self._completed.values() for event in run.events)

    @property
    def runs(self) -> Mapping[UUID, tuple[TraceEvent, ...]]:
        """Return an immutable trace-ID-to-events snapshot of completed runs."""

        with self._lock:
            if not self._closed:
                self._expire_active(self._now())
            snapshot = {trace_id: run.events for trace_id, run in self._completed.items()}
            return MappingProxyType(snapshot)

    @property
    def closed(self) -> bool:
        """Return whether this sink has stopped accepting events."""

        with self._lock:
            return self._closed

    @property
    def max_runs(self) -> int:
        """Return the completed-run retention limit."""

        return self._max_runs

    @property
    def max_events_per_run(self) -> int | None:
        """Return the optional per-run event retention limit."""

        return self._max_events_per_run

    @property
    def max_active_runs(self) -> int:
        """Return the incomplete-run retention limit."""

        return self._max_active_runs

    @property
    def active_ttl_s(self) -> float:
        """Return the inactivity lifetime for incomplete runs."""

        return self._active_ttl_s

    def _append_bounded(self, run: _ActiveRun, event: TraceEvent) -> None:
        limit = self._max_events_per_run
        if limit is None or len(run.events) < limit:
            run.events.append(event)
            return

        run.truncated = True
        self._dropped_events += 1
        if event.event_type in _TERMINAL_EVENT_TYPES:
            # The terminal event carries the v2 summary and makes the retained
            # fragment intelligible. Replacing the tail stays within the cap;
            # the counter above accounts for the displaced event.
            run.events[-1] = event

    def _complete(self, trace_id: UUID, run: _ActiveRun) -> None:
        self._active.pop(trace_id, None)
        completed = _CompletedRun(tuple(run.events), run.truncated)
        self._completed[trace_id] = completed
        if run.truncated:
            self._truncated_runs += 1
        while len(self._completed) > self._max_runs:
            _, removed = self._completed.popitem(last=False)
            self._evicted_runs += 1
            self._evicted_events += len(removed.events)

    def _expire_active(self, now: float) -> None:
        expired = tuple(
            trace_id
            for trace_id, run in self._active.items()
            if now - run.last_seen >= self._active_ttl_s
        )
        for trace_id in expired:
            removed = self._active.pop(trace_id)
            self._expired_active_runs += 1
            self._dropped_events += len(removed.events)

    def _bound_active(self) -> None:
        while len(self._active) > self._max_active_runs:
            _, removed = self._active.popitem(last=False)
            self._overflowed_active_runs += 1
            self._dropped_events += len(removed.events)

    def _now(self) -> float:
        value = self._monotonic_clock()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("monotonic_clock must return a finite number")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("monotonic_clock must return a finite number")
        return result

    def _stats(self) -> InMemoryTraceSinkStats:
        return InMemoryTraceSinkStats(
            completed_runs=len(self._completed),
            active_runs=len(self._active),
            retained_events=sum(len(run.events) for run in self._completed.values()),
            truncated_runs=self._truncated_runs,
            dropped_events=self._dropped_events,
            evicted_runs=self._evicted_runs,
            evicted_events=self._evicted_events,
            expired_active_runs=self._expired_active_runs,
            overflowed_active_runs=self._overflowed_active_runs,
            closed_active_runs=self._closed_active_runs,
            post_close_emits=self._post_close_emits,
            unsupported_schema_events=self._unsupported_schema_events,
            duplicate_terminal_events=self._duplicate_terminal_events,
        )


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return result
