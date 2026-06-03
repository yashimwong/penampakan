"""Private, redacted, run-local tracing infrastructure."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel

from penampakan.config import TraceContentPolicy
from penampakan.models import JsonValue, RunTrace, TraceEvent, TraceSummary, WarningInfo
from penampakan.protocols import TraceSink

StopReason = Literal[
    "completed",
    "insufficient_evidence",
    "step_limit",
    "llm_limit",
    "tool_limit",
    "backend_limit",
    "asset_limit",
    "depth_limit",
    "context_limit",
    "timeout",
    "cancelled",
    "error",
]

_STOP_REASONS = frozenset(
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
        "timeout",
        "cancelled",
        "error",
    }
)

REQUIRED_EVENT_TYPES = frozenset(
    {
        "run_started",
        "image_loaded",
        "initial_plan_started",
        "policy_call_started",
        "policy_call_finished",
        "invalid_action",
        "tool_call_started",
        "backend_call_started",
        "backend_call_finished",
        "cache_hit",
        "asset_created",
        "observations_committed",
        "budget_stop",
        "answer_validated",
        "run_finished",
        "run_failed",
    }
)

_PATH_KEYS = frozenset(
    {
        "file",
        "file_name",
        "filename",
        "filesystem_path",
        "path",
        "paths",
        "source",
        "source_file",
        "source_path",
    }
)
_QUESTION_KEYS = frozenset({"question", "questions", "query", "user_question"})
_OBSERVATION_TEXT_KEYS = frozenset(
    {
        "caption",
        "captions",
        "focus",
        "ocr",
        "ocr_text",
        "observation_text",
        "raw_text",
        "text",
    }
)
_MODEL_OUTPUT_KEYS = frozenset(
    {"invalid_model_output", "llm_output", "model_output", "raw_model_output"}
)
_ANSWER_KEYS = frozenset({"answer", "answers", "final_answer"})
_ALWAYS_REDACTED_KEYS = frozenset(
    {
        "api_key",
        "arguments",
        "authorization",
        "authorization_header",
        "backend_request_secret",
        "bearer",
        "bytes",
        "content",
        "cookie",
        "credentials",
        "environment",
        "environment_variables",
        "headers",
        "image",
        "image_bytes",
        "message",
        "messages",
        "parameters",
        "password",
        "pixels",
        "prompt",
        "prompts",
        "request_body",
        "secret",
        "token",
    }
)


def _safe_key(key: object) -> str | None:
    if not isinstance(key, str) or "\x00" in key:
        return None
    normalized = key.strip()
    return normalized or None


def _key_category(key: str) -> str | None:
    normalized = key.casefold().replace("-", "_")
    if normalized in _ALWAYS_REDACTED_KEYS:
        return "always"
    if normalized in _PATH_KEYS or normalized.endswith("_path"):
        return "path"
    if normalized in _QUESTION_KEYS or normalized.endswith("_question"):
        return "question"
    if normalized in _OBSERVATION_TEXT_KEYS:
        return "observation_text"
    if normalized in _MODEL_OUTPUT_KEYS or normalized.endswith("_model_output"):
        return "model_output"
    if normalized in _ANSWER_KEYS or normalized.endswith("_answer"):
        return "answer"
    if normalized.endswith("api_key") or normalized.endswith("_secret"):
        return "always"
    if any(
        marker in normalized
        for marker in ("credential", "password", "private_key", "access_token", "api_secret")
    ):
        return "always"
    return None


def _category_allowed(category: str | None, policy: TraceContentPolicy) -> bool:
    if category is None:
        return True
    if category == "path":
        return policy.include_paths
    if category == "question":
        return policy.include_questions
    if category == "observation_text":
        return policy.include_observation_text
    if category == "model_output":
        return policy.include_model_output
    if category == "answer":
        return policy.include_answers
    return False


def _safe_string(value: str) -> str | None:
    if "\x00" in value:
        return None
    return value


def _safe_error(value: BaseException) -> dict[str, JsonValue]:
    data: dict[str, JsonValue] = {"error_type": type(value).__name__}
    code = getattr(value, "code", None)
    if isinstance(code, str) and _safe_string(code) is not None:
        data["code"] = code
    retryable = getattr(value, "retryable", None)
    if isinstance(retryable, bool):
        data["retryable"] = retryable
    return data


def _safe_json(value: object, policy: TraceContentPolicy) -> JsonValue | None:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return _safe_string(value)
    if isinstance(value, (bytes, bytearray, memoryview, Path)):
        return None
    if isinstance(value, BaseException):
        return _safe_error(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            return None
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, Enum):
        return _safe_json(value.value, policy)
    if isinstance(value, BaseModel):
        return _safe_json(value.model_dump(mode="python", exclude_none=True), policy)
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for raw_key, item in value.items():
            key = _safe_key(raw_key)
            category = _key_category(key) if key is not None else "always"
            if key is None or not _category_allowed(category, policy):
                continue
            if category == "path" and isinstance(item, Path):
                result[key] = str(item)
                continue
            if item is None:
                result[key] = None
                continue
            safe_item = _safe_json(item, policy)
            if safe_item is not None:
                result[key] = safe_item
        return result
    if isinstance(value, Sequence):
        result_list: list[JsonValue] = []
        for item in value:
            if item is None:
                result_list.append(None)
                continue
            safe_item = _safe_json(item, policy)
            if safe_item is not None:
                result_list.append(safe_item)
        return result_list
    return None


def redact_trace_data(
    data: Mapping[str, object] | None,
    policy: TraceContentPolicy | None = None,
) -> dict[str, JsonValue]:
    """Return strict JSON trace data with sensitive categories removed."""
    if data is None:
        return {}
    active_policy = policy or TraceContentPolicy()
    safe = _safe_json(data, active_policy)
    if not isinstance(safe, dict):
        return {}
    return safe


class TraceBuilder:
    """Build one immutable redacted run trace and safely fan out its events."""

    def __init__(
        self,
        *,
        content_policy: TraceContentPolicy | None = None,
        sinks: Sequence[TraceSink] = (),
        trace_id: UUID | None = None,
        wall_clock: Callable[[], datetime] | None = None,
        monotonic_clock: Callable[[], float] | None = None,
    ) -> None:
        selected_id = trace_id or uuid4()
        if selected_id.version != 4:
            raise ValueError("trace_id must be a UUIDv4 value")
        self._trace_id = selected_id
        self._content_policy = content_policy or TraceContentPolicy()
        self._sinks = tuple(sinks)
        self._wall_clock = wall_clock or self._utc_now
        self._monotonic_clock = monotonic_clock or time.monotonic
        self._started_at = self._validated_utc(self._wall_clock())
        self._started_monotonic = self._validated_monotonic(self._monotonic_clock())
        self._events: list[TraceEvent] = []
        self._warnings: list[WarningInfo] = []
        self._sequence = 0
        self._started = False
        self._final_trace: RunTrace | None = None
        self._llm_calls = 0
        self._tool_calls = 0
        self._backend_calls = 0
        self._cache_hits = 0
        self._derived_assets = 0
        self._input_tokens: int | None = None
        self._output_tokens: int | None = None
        self._failed_sink_ids: set[int] = set()
        self._sink_warning_emitted = False
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _validated_utc(value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("wall clock must return a timezone-aware datetime")
        return value.astimezone(timezone.utc)

    @staticmethod
    def _validated_monotonic(value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("monotonic clock must return a finite number")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("monotonic clock must return a finite number")
        return result

    @property
    def trace_id(self) -> UUID:
        """Return this run's random UUIDv4 identifier."""
        return self._trace_id

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        """Return an immutable snapshot of emitted events."""
        return tuple(self._events)

    @property
    def warnings(self) -> tuple[WarningInfo, ...]:
        """Return safe warnings produced by trace infrastructure."""
        return tuple(self._warnings)

    @property
    def finalized(self) -> bool:
        """Return whether a final run trace has been materialized."""
        return self._final_trace is not None

    def _elapsed_ms(self) -> int:
        current = self._validated_monotonic(self._monotonic_clock())
        return max(0, int((current - self._started_monotonic) * 1000.0))

    def _new_event(
        self,
        event_type: str,
        data: Mapping[str, object] | None,
        duration_ms: int | None,
        occurred_at: datetime | None = None,
    ) -> TraceEvent:
        if duration_ms is not None:
            if isinstance(duration_ms, bool) or not isinstance(duration_ms, int):
                raise TypeError("duration_ms must be an integer")
            if duration_ms < 0:
                raise ValueError("duration_ms cannot be negative")
        event = TraceEvent(
            trace_id=self._trace_id,
            sequence=self._sequence,
            event_type=event_type,
            occurred_at=self._validated_utc(occurred_at or self._wall_clock()),
            duration_ms=duration_ms,
            data=redact_trace_data(data, self._content_policy),
        )
        self._sequence += 1
        self._events.append(event)
        self._apply_event_counts(event)
        return event

    def _apply_event_counts(self, event: TraceEvent) -> None:
        if event.event_type == "policy_call_started":
            self._llm_calls += 1
        elif event.event_type == "tool_call_started":
            self._tool_calls += 1
        elif event.event_type == "backend_call_started":
            self._backend_calls += 1
        elif event.event_type == "cache_hit":
            self._cache_hits += 1
        elif event.event_type == "asset_created":
            self._derived_assets += 1
        if event.event_type == "policy_call_finished":
            self._add_optional_tokens(
                event.data.get("input_tokens"), event.data.get("output_tokens")
            )

    def _add_optional_tokens(self, input_tokens: object, output_tokens: object) -> None:
        if (
            isinstance(input_tokens, int)
            and not isinstance(input_tokens, bool)
            and input_tokens >= 0
        ):
            self._input_tokens = (self._input_tokens or 0) + input_tokens
        if (
            isinstance(output_tokens, int)
            and not isinstance(output_tokens, bool)
            and output_tokens >= 0
        ):
            self._output_tokens = (self._output_tokens or 0) + output_tokens

    async def _emit_to_sinks(self, event: TraceEvent) -> None:
        for sink in self._sinks:
            sink_id = id(sink)
            if sink_id in self._failed_sink_ids or self._closed:
                continue
            try:
                await sink.emit(event)
            except asyncio.CancelledError:
                raise
            except Exception:
                self._record_sink_failure(sink_id)

    def _record_sink_failure(self, sink_id: int) -> None:
        self._failed_sink_ids.add(sink_id)
        if not self._sink_warning_emitted:
            self._warnings.append(
                WarningInfo(
                    code="trace_sink_failed",
                    message="A trace sink failed and was disabled for this run.",
                )
            )
            self._sink_warning_emitted = True

    async def _ensure_started_locked(
        self,
        data: Mapping[str, object] | None = None,
    ) -> TraceEvent:
        if self._started:
            return self._events[0]
        event = self._new_event("run_started", data, None, occurred_at=self._started_at)
        self._started = True
        await self._emit_to_sinks(event)
        return event

    async def start(self, data: Mapping[str, object] | None = None) -> TraceEvent:
        """Start the run idempotently and emit its required first event."""
        async with self._lock:
            if self._final_trace is not None:
                return self._events[0]
            return await self._ensure_started_locked(data)

    async def emit(
        self,
        event_type: str,
        data: Mapping[str, object] | None = None,
        *,
        duration_ms: int | None = None,
    ) -> TraceEvent:
        """Append and safely deliver one already-redacted immutable event."""
        async with self._lock:
            if self._final_trace is not None:
                raise RuntimeError("cannot emit events after trace finalization")
            await self._ensure_started_locked()
            if event_type in {"run_started", "run_finished", "run_failed"}:
                raise ValueError("lifecycle events are managed by TraceBuilder")
            event = self._new_event(event_type, data, duration_ms)
            await self._emit_to_sinks(event)
            return event

    async def add_counts(
        self,
        *,
        llm_calls: int = 0,
        tool_calls: int = 0,
        backend_calls: int = 0,
        cache_hits: int = 0,
        derived_assets: int = 0,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        """Add non-event counters when integrating an external budget source."""
        values = (llm_calls, tool_calls, backend_calls, cache_hits, derived_assets)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("trace counters must be integers")
        if any(value < 0 for value in values):
            raise ValueError("trace counters cannot be negative")
        for token_count in (input_tokens, output_tokens):
            if token_count is not None and (
                isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0
            ):
                raise ValueError("token counts must be non-negative integers")
        async with self._lock:
            if self._final_trace is not None:
                raise RuntimeError("cannot update counters after trace finalization")
            self._llm_calls += llm_calls
            self._tool_calls += tool_calls
            self._backend_calls += backend_calls
            self._cache_hits += cache_hits
            self._derived_assets += derived_assets
            self._add_optional_tokens(input_tokens, output_tokens)

    def _summary(self, stop_reason: StopReason) -> TraceSummary:
        return TraceSummary(
            trace_id=self._trace_id,
            started_at=self._started_at,
            duration_ms=self._elapsed_ms(),
            llm_calls=self._llm_calls,
            tool_calls=self._tool_calls,
            backend_calls=self._backend_calls,
            cache_hits=self._cache_hits,
            derived_assets=self._derived_assets,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            stop_reason=stop_reason,
        )

    async def _finalize(
        self,
        stop_reason: StopReason,
        event_type: Literal["run_finished", "run_failed"],
        data: Mapping[str, object] | None,
    ) -> RunTrace:
        async with self._lock:
            if self._final_trace is not None:
                return self._final_trace
            await self._ensure_started_locked()
            final_data = dict(data or {})
            final_data["stop_reason"] = stop_reason
            event = self._new_event(event_type, final_data, self._elapsed_ms())
            await self._emit_to_sinks(event)
            self._final_trace = RunTrace(
                summary=self._summary(stop_reason),
                events=tuple(self._events),
            )
            return self._final_trace

    async def finish(
        self,
        stop_reason: StopReason = "completed",
        data: Mapping[str, object] | None = None,
    ) -> RunTrace:
        """Finalize a successful, insufficient, limited, timed-out, or cancelled run."""
        if stop_reason not in _STOP_REASONS:
            raise ValueError("stop_reason is invalid")
        if stop_reason == "error":
            return await self.fail(data=data)
        return await self._finalize(stop_reason, "run_finished", data)

    async def fail(
        self,
        error: BaseException | None = None,
        data: Mapping[str, object] | None = None,
    ) -> RunTrace:
        """Finalize a failed run with one safe run-failed event."""
        failure_data = dict(data or {})
        if error is not None:
            failure_data["error"] = error
        return await self._finalize("error", "run_failed", failure_data)

    async def cancel(self, data: Mapping[str, object] | None = None) -> RunTrace:
        """Finalize cancellation with a run-finished event when safe."""
        return await self._finalize("cancelled", "run_finished", data)

    async def _close_sinks(self) -> None:
        for sink in self._sinks:
            try:
                await sink.aclose()
            except asyncio.CancelledError:
                raise
            except Exception:
                self._record_sink_failure(id(sink))
        self._closed = True

    async def aclose(self) -> None:
        """Close all owned sinks once while concurrent callers await one task."""
        async with self._lock:
            if self._close_task is None:
                self._close_task = asyncio.create_task(self._close_sinks())
            close_task = self._close_task
        await asyncio.shield(close_task)


__all__ = [
    "REQUIRED_EVENT_TYPES",
    "StopReason",
    "TraceBuilder",
    "redact_trace_data",
]
