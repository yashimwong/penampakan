from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from datetime import datetime, timezone
from uuid import UUID

from penampakan.models import (
    Capability,
    LLMRequest,
    LLMResponse,
    Observation,
    ObservationPayload,
    PolicyAction,
    PolicyInput,
    Provenance,
    RemainingBudget,
    RunTrace,
    ToolSpec,
    TraceEvent,
    TraceSummary,
)


class ScriptedTextLLM:
    """A finite text-LLM script that records every validated request."""

    def __init__(self, responses: Iterable[str | LLMResponse | BaseException]) -> None:
        self._responses = deque(responses)
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Return or raise the next scripted item."""

        self.requests.append(request)
        if not self._responses:
            raise AssertionError("unexpected extra language-model call")
        response = self._responses.popleft()
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, str):
            return LLMResponse(text=response)
        return response

    @property
    def remaining(self) -> int:
        """Return the number of unused scripted responses."""

        return len(self._responses)


class ScriptedPolicy:
    """A finite typed policy script that records each policy input."""

    def __init__(self, actions: Iterable[PolicyAction | BaseException]) -> None:
        self._actions = deque(actions)
        self.inputs: list[PolicyInput] = []

    async def next_action(self, input: PolicyInput) -> PolicyAction:
        """Return or raise the next scripted action."""

        self.inputs.append(input)
        if not self._actions:
            raise AssertionError("unexpected extra policy call")
        action = self._actions.popleft()
        if isinstance(action, BaseException):
            raise action
        return action


def make_provenance(
    *,
    capability: Capability | None = Capability.CAPTION,
    tool: str = "describe_image",
    parent_observation_ids: tuple[str, ...] = (),
) -> Provenance:
    """Create deterministic safe provenance for reasoning tests."""

    return Provenance(
        tool=tool,
        capability=capability,
        backend_name="tests.backend",
        backend_version="1.0",
        request_hash="a" * 64,
        parent_observation_ids=parent_observation_ids,
        duration_ms=1,
    )


def make_observation(
    sequence: int,
    payload: ObservationPayload,
    *,
    asset_id: str = "img_aaaaaaaaaaaaaaaa",
    confidence: float | None = None,
    region: object | None = None,
    contradicts: tuple[str, ...] = (),
) -> Observation:
    """Create a deterministic observation with a numeric session ID."""

    from penampakan.models import Box

    return Observation(
        id=f"obs_{sequence:06d}",
        asset_id=asset_id,
        payload=payload,
        region=region if isinstance(region, Box) else None,
        confidence=confidence,
        provenance=make_provenance(),
        contradicts=contradicts,
    )


def make_remaining_budget(**overrides: int | float) -> RemainingBudget:
    """Create a non-exhausted deterministic budget snapshot."""

    values: dict[str, int | float] = {
        "steps": 4,
        "llm_calls": 5,
        "tool_calls": 6,
        "backend_calls": 7,
        "derived_assets": 8,
        "derivation_depth": 2,
        "context_chars": 2_000,
        "remaining_time_s": 30.0,
    }
    values.update(overrides)
    return RemainingBudget.model_validate(values, strict=True)


def make_tool_spec(name: str = "read_text") -> ToolSpec:
    """Create one strict tool declaration for policy tests."""

    return ToolSpec(
        name=name,
        description="Read localized visible text from an asset.",
        arguments_json_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "asset_id": {"type": "string"},
            },
            "required": ["asset_id"],
        },
    )


def make_policy_input(
    *,
    answer_only: bool = False,
    tools: tuple[ToolSpec, ...] | None = None,
    validation_feedback: tuple[object, ...] = (),
    invalid_model_output: str | None = None,
) -> PolicyInput:
    """Create deterministic trusted and untrusted policy input sections."""

    from penampakan.models import WarningInfo

    feedback = tuple(item for item in validation_feedback if isinstance(item, WarningInfo))
    return PolicyInput(
        question="What is the receipt total?",
        context='{"id":"obs_000001","type":"text","text":"RM 42.50"}',
        tools=tools if tools is not None else (make_tool_spec(),),
        prior_actions=(),
        remaining=make_remaining_budget(),
        answer_only=answer_only,
        validation_feedback=feedback,
        invalid_model_output=invalid_model_output,
    )


def make_trace(
    events: tuple[TraceEvent, ...] = (),
    *,
    tool_calls: int = 0,
    backend_calls: int = 0,
) -> RunTrace:
    """Create a deterministic completed trace for answer and evaluation tests."""

    trace_id = UUID("00000000-0000-4000-8000-000000000001")
    normalized_events = tuple(
        event.model_copy(update={"trace_id": trace_id, "sequence": index})
        for index, event in enumerate(events, start=1)
    )
    return RunTrace(
        summary=TraceSummary(
            trace_id=trace_id,
            started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            duration_ms=10,
            llm_calls=1,
            tool_calls=tool_calls,
            backend_calls=backend_calls,
            cache_hits=0,
            derived_assets=0,
            stop_reason="completed",
        ),
        events=normalized_events,
    )


def make_trace_event(event_type: str, data: dict[str, object]) -> TraceEvent:
    """Create an event whose trace identity and sequence are normalized later."""

    trace_id = UUID("00000000-0000-4000-8000-000000000001")
    return TraceEvent.model_validate(
        {
            "trace_id": trace_id,
            "sequence": 1,
            "event_type": event_type,
            "occurred_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "data": data,
        },
        strict=True,
    )
