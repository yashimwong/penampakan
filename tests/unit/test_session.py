from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Awaitable, Callable, Mapping, Sequence
from inspect import isawaitable
from typing import cast

import pytest

from penampakan.backends.callable import CallableVisionBackend
from penampakan.client import AsyncPenampakan
from penampakan.config import AgentSettings, RunLimits, Settings
from penampakan.errors import (
    BackendCallLimitExceededError,
    BackendUnavailableError,
    ContextLimitExceededError,
    InspectionFailedError,
    PenampakanError,
    SessionClosedError,
)
from penampakan.models import (
    AnswerAction,
    AnswerStatus,
    BackendDescriptor,
    BackendImage,
    Box,
    Capability,
    CapabilityDescriptor,
    CaptionPayload,
    CaptionRequest,
    ColorsRequest,
    DetectionPayload,
    DetectionRequest,
    EvidenceRef,
    InspectionOperation,
    InspectionPlan,
    LLMResponse,
    ObservationDraft,
    OCRRequest,
    PolicyAction,
    PolicyInput,
    TextPayload,
    TokenUsage,
    ToolAction,
    VisionRequest,
    VisionResult,
    WarningInfo,
)
from tests.fixtures.images import encode_image, quadrants_image
from tests.unit.reasoning.helpers import ScriptedPolicy, ScriptedTextLLM

Analyze = Callable[
    [BackendImage, VisionRequest],
    VisionResult | Awaitable[VisionResult],
]
PolicyFactory = Callable[[PolicyInput, int], PolicyAction | Awaitable[PolicyAction]]


class FunctionalPolicy:
    def __init__(self, factory: PolicyFactory) -> None:
        self._factory = factory
        self.inputs: list[PolicyInput] = []

    async def next_action(self, input: PolicyInput) -> PolicyAction:
        self.inputs.append(input)
        result = self._factory(input, len(self.inputs) - 1)
        if isawaitable(result):
            return await result
        return result


def _settings(
    initial_capabilities: tuple[Capability, ...] = (Capability.CAPTION,),
    *,
    run: RunLimits | None = None,
    max_identical_actions: int = 2,
) -> Settings:
    return Settings(
        run=run or RunLimits(),
        agent=AgentSettings(
            initial_capabilities=initial_capabilities,
            max_identical_actions=max_identical_actions,
        ),
    )


def _backend(
    name: str,
    capabilities: Sequence[Capability],
    analyze: Analyze,
    *,
    features: Mapping[Capability, frozenset[str]] | None = None,
    max_concurrency: int = 4,
    close: Callable[[], Awaitable[None] | None] | None = None,
) -> CallableVisionBackend:
    selected_features = features or {}
    descriptor = BackendDescriptor(
        name=name,
        version="1.0",
        capabilities=tuple(
            CapabilityDescriptor(
                capability=capability,
                features=selected_features.get(capability, frozenset()),
            )
            for capability in capabilities
        ),
        max_concurrency=max_concurrency,
    )
    return CallableVisionBackend(descriptor, analyze, close=close)


def _caption_result(request: CaptionRequest, text: str) -> VisionResult:
    return VisionResult(
        observations=(
            ObservationDraft(
                payload=CaptionPayload(text=text, focus=request.focus),
                region=request.region,
            ),
        )
    )


def _text_result(request: OCRRequest, text: str) -> VisionResult:
    return VisionResult(
        observations=(
            ObservationDraft(
                payload=TextPayload(text=text, block_kind="line"),
                region=request.region,
            ),
        )
    )


def _answered(
    observation_id: str, answer: str = "The visual evidence supports this."
) -> AnswerAction:
    return AnswerAction(
        status="answered",
        answer=answer,
        evidence=(EvidenceRef(observation_id=observation_id, supports="Visible evidence"),),
    )


def _abstain(answer: str = "The available evidence is insufficient.") -> AnswerAction:
    return AnswerAction(status="insufficient_evidence", answer=answer)


def _tool(name: str, arguments: dict[str, object], purpose: str = "Acquire evidence") -> ToolAction:
    return ToolAction.model_validate(
        {
            "tool": name,
            "arguments": arguments,
            "purpose": purpose,
        },
        strict=True,
    )


@pytest.fixture
def image_bytes() -> bytes:
    image = quadrants_image(16)
    try:
        return encode_image(image)
    finally:
        image.close()


@pytest.fixture
def second_image_bytes() -> bytes:
    image = quadrants_image(20)
    try:
        return encode_image(image)
    finally:
        image.close()


@pytest.mark.asyncio
async def test_direct_initial_answer_avoids_unrequested_detection(image_bytes: bytes) -> None:
    requests: list[VisionRequest] = []

    async def analyze(image: BackendImage, request: VisionRequest) -> VisionResult:
        requests.append(request)
        if isinstance(request, CaptionRequest):
            return _caption_result(request, "A four-color image")
        if isinstance(request, DetectionRequest):
            return VisionResult(
                observations=(
                    ObservationDraft(
                        payload=DetectionPayload(label="square"),
                        region=Box(x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0),
                    ),
                )
            )
        raise AssertionError("unexpected request")

    backend = _backend(
        "tests.overview",
        (Capability.CAPTION, Capability.DETECT),
        analyze,
        features={Capability.CAPTION: frozenset({"caption.focus"})},
    )
    policy = ScriptedPolicy([_answered("obs_000001", "It contains four colored quadrants.")])

    async with AsyncPenampakan(
        backends=(backend,),
        policy=policy,
        settings=_settings(),
    ) as client:
        answer = await client.ask(image_bytes, "What is shown?")

    assert answer.status is AnswerStatus.ANSWERED
    assert answer.evidence[0].observation.id == "obs_000001"
    assert answer.evidence[0].observation.payload.type == "caption"
    assert [type(request) for request in requests] == [CaptionRequest]
    assert len(policy.inputs) == 1


@pytest.mark.asyncio
async def test_reusable_session_reuses_initial_observations_and_keeps_monotonic_ids(
    image_bytes: bytes,
) -> None:
    requests: list[VisionRequest] = []

    async def analyze(image: BackendImage, request: VisionRequest) -> VisionResult:
        requests.append(request)
        if isinstance(request, CaptionRequest):
            return _caption_result(request, "Colored quadrants")
        if isinstance(request, OCRRequest):
            return _text_result(request, "SALE")
        raise AssertionError("unexpected request")

    backend = _backend(
        "tests.reuse",
        (Capability.CAPTION, Capability.OCR),
        analyze,
        features={Capability.CAPTION: frozenset({"caption.focus"})},
    )
    policy = ScriptedPolicy([_answered("obs_000001"), _answered("obs_000001")])
    settings = _settings((Capability.CAPTION, Capability.OCR))

    async with AsyncPenampakan(backends=(backend,), policy=policy, settings=settings) as client:
        session = await client.open_image(image_bytes)
        first = await session.ask("What is visible?")
        inspected = await session.inspect(
            InspectionPlan(
                operations=(InspectionOperation(request=ColorsRequest()),),
                include_available_overview=False,
            )
        )
        second = await session.ask("What did you see before?")
        snapshots = session.observations
        await session.aclose()

    assert tuple(item.id for item in snapshots) == (
        "obs_000001",
        "obs_000002",
        "obs_000003",
    )
    assert tuple(item.id for item in inspected.observations) == ("obs_000003",)
    assert first.evidence[0].observation == second.evidence[0].observation
    assert sum(isinstance(request, CaptionRequest) for request in requests) == 1
    assert sum(isinstance(request, OCRRequest) for request in requests) == 1
    assert "obs_000001" in policy.inputs[1].context
    assert "obs_000002" in policy.inputs[1].context


@pytest.mark.asyncio
async def test_explicit_inspection_runs_concurrently_but_commits_in_plan_order(
    image_bytes: bytes,
) -> None:
    active = 0
    maximum_active = 0
    delays = {"first": 0.06, "second": 0.03, "third": 0.0}

    async def analyze(image: BackendImage, request: VisionRequest) -> VisionResult:
        nonlocal active, maximum_active
        assert isinstance(request, CaptionRequest)
        assert request.focus is not None
        active += 1
        maximum_active = max(maximum_active, active)
        try:
            await asyncio.sleep(delays[request.focus])
            return _caption_result(request, request.focus)
        finally:
            active -= 1

    backend = _backend(
        "tests.concurrent_inspect",
        (Capability.CAPTION,),
        analyze,
        features={Capability.CAPTION: frozenset({"caption.focus"})},
        max_concurrency=3,
    )
    operations = tuple(
        InspectionOperation(request=CaptionRequest(focus=focus))
        for focus in ("first", "second", "third")
    )

    async with AsyncPenampakan(backends=(backend,), settings=_settings(())) as client:
        result = await client.inspect(
            image_bytes,
            InspectionPlan(operations=operations, include_available_overview=False),
        )

    assert maximum_active == 3
    assert tuple(item.payload.text for item in result.observations) == (
        "first",
        "second",
        "third",
    )
    assert tuple(item.id for item in result.observations) == (
        "obs_000001",
        "obs_000002",
        "obs_000003",
    )


@pytest.mark.asyncio
async def test_fail_fast_stops_after_first_failed_operation(image_bytes: bytes) -> None:
    calls: list[str | None] = []

    async def analyze(image: BackendImage, request: VisionRequest) -> VisionResult:
        assert isinstance(request, CaptionRequest)
        calls.append(request.focus)
        if request.focus == "fail":
            raise RuntimeError("backend detail")
        return _caption_result(request, "unused")

    backend = _backend(
        "tests.fail_fast",
        (Capability.CAPTION,),
        analyze,
        features={Capability.CAPTION: frozenset({"caption.focus"})},
    )
    plan = InspectionPlan(
        operations=(
            InspectionOperation(request=CaptionRequest(focus="fail")),
            InspectionOperation(request=CaptionRequest(focus="unused")),
        ),
        include_available_overview=False,
        fail_fast=True,
    )

    async with AsyncPenampakan(backends=(backend,), settings=_settings(())) as client:
        with pytest.raises(InspectionFailedError) as captured:
            await client.inspect(image_bytes, plan)

    assert calls == ["fail"]
    assert captured.value.partial_result is not None
    assert captured.value.partial_result.observations == ()
    assert captured.value.partial_result.trace.summary.stop_reason == "error"


@pytest.mark.asyncio
async def test_required_failure_exposes_successful_partial_result(image_bytes: bytes) -> None:
    async def analyze(image: BackendImage, request: VisionRequest) -> VisionResult:
        if isinstance(request, CaptionRequest):
            await asyncio.sleep(0.02)
            return _caption_result(request, "usable partial caption")
        if isinstance(request, OCRRequest):
            raise RuntimeError("ocr unavailable")
        raise AssertionError("unexpected request")

    backend = _backend(
        "tests.required_partial",
        (Capability.CAPTION, Capability.OCR),
        analyze,
    )
    plan = InspectionPlan(
        operations=(
            InspectionOperation(request=CaptionRequest()),
            InspectionOperation(request=OCRRequest(), required=True),
        ),
        include_available_overview=False,
    )

    async with AsyncPenampakan(backends=(backend,), settings=_settings(())) as client:
        session = await client.open_image(image_bytes)
        with pytest.raises(InspectionFailedError) as captured:
            await session.inspect(plan)
        snapshots = session.observations
        await session.aclose()

    partial = captured.value.partial_result
    assert partial is not None
    assert tuple(item.payload.type for item in partial.observations) == ("caption",)
    assert tuple(item.id for item in partial.observations) == ("obs_000001",)
    assert snapshots == partial.observations


@pytest.mark.asyncio
async def test_detect_crop_focused_caption_and_cited_answer(image_bytes: bytes) -> None:
    state: dict[str, object] = {}
    calls: list[tuple[str, str, str | None]] = []

    async def analyze(image: BackendImage, request: VisionRequest) -> VisionResult:
        focus = request.focus if isinstance(request, CaptionRequest) else None
        calls.append((request.capability.value, image.asset.id, focus))
        if isinstance(request, CaptionRequest):
            text = "Focused logo detail" if image.asset.parent_id is not None else "Global overview"
            return _caption_result(request, text)
        if isinstance(request, DetectionRequest):
            return VisionResult(
                observations=(
                    ObservationDraft(
                        payload=DetectionPayload(label="logo"),
                        region=Box(x_min=0.0, y_min=0.0, x_max=0.5, y_max=0.5),
                        confidence=0.9,
                    ),
                )
            )
        raise AssertionError("unexpected request")

    def decide(input: PolicyInput, index: int) -> PolicyAction:
        session = state["session"]
        if index == 0:
            return _tool("detect_objects", {"asset_id": session.root_asset.id})
        if index == 1:
            return _tool(
                "crop",
                {
                    "asset_id": session.root_asset.id,
                    "box": {"x_min": 0.0, "y_min": 0.0, "x_max": 0.5, "y_max": 0.5},
                },
            )
        if index == 2:
            return _tool(
                "describe_image",
                {"asset_id": session.assets[-1].id, "focus": "logo"},
            )
        return _answered(session.observations[-1].id, "The logo is visible in the crop.")

    backend = _backend(
        "tests.visual_chain",
        (Capability.CAPTION, Capability.DETECT),
        analyze,
        features={Capability.CAPTION: frozenset({"caption.focus"})},
    )
    policy = FunctionalPolicy(decide)

    async with AsyncPenampakan(
        backends=(backend,),
        policy=policy,
        settings=_settings(),
    ) as client:
        session = await client.open_image(image_bytes)
        state["session"] = session
        answer = await session.ask("Where is the logo?")
        assets = session.assets
        observations = session.observations
        await session.aclose()

    assert tuple(item.payload.type for item in observations) == (
        "caption",
        "detection",
        "transform",
        "caption",
    )
    assert len(assets) == 2
    assert assets[1].parent_id == assets[0].id
    assert answer.evidence[0].observation.id == "obs_000004"
    assert answer.evidence[0].observation.asset_id == assets[1].id
    assert answer.evidence[0].observation.provenance.parent_observation_ids == ("obs_000003",)
    assert calls[-1] == (Capability.CAPTION.value, assets[1].id, "logo")


@pytest.mark.asyncio
async def test_empty_backend_result_is_visible_and_policy_abstains(image_bytes: bytes) -> None:
    async def analyze(image: BackendImage, request: VisionRequest) -> VisionResult:
        assert isinstance(request, OCRRequest)
        return VisionResult(observations=())

    backend = _backend("tests.empty", (Capability.OCR,), analyze)
    policy = ScriptedPolicy([_abstain()])

    async with AsyncPenampakan(
        backends=(backend,),
        policy=policy,
        settings=_settings((Capability.OCR,)),
    ) as client:
        session = await client.open_image(image_bytes)
        answer = await session.ask("What text is present?")
        observations = session.observations
        await session.aclose()

    assert answer.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert tuple(warning.code for warning in answer.warnings) == ("no_text_detected",)
    assert tuple(item.payload.type for item in observations) == ("warning",)
    assert observations[0].payload.code == "no_text_detected"
    assert "no_text_detected" in policy.inputs[0].context


@pytest.mark.asyncio
async def test_retryable_backend_failure_falls_back_and_records_both_attempts(
    image_bytes: bytes,
) -> None:
    first_calls = 0
    second_calls = 0

    async def unavailable(image: BackendImage, request: VisionRequest) -> VisionResult:
        nonlocal first_calls
        first_calls += 1
        raise BackendUnavailableError(code="scripted_unavailable")

    async def successful(image: BackendImage, request: VisionRequest) -> VisionResult:
        nonlocal second_calls
        second_calls += 1
        assert isinstance(request, CaptionRequest)
        return _caption_result(request, "Fallback caption")

    features = {Capability.CAPTION: frozenset({"caption.focus"})}
    first = _backend("tests.fallback_first", (Capability.CAPTION,), unavailable, features=features)
    second = _backend("tests.fallback_second", (Capability.CAPTION,), successful, features=features)
    policy = ScriptedPolicy([_answered("obs_000001")])

    async with AsyncPenampakan(
        backends=(first, second),
        policy=policy,
        settings=_settings(),
    ) as client:
        answer = await client.ask(image_bytes, "Describe it")

    finished = tuple(
        event for event in answer.trace.events if event.event_type == "backend_call_finished"
    )
    assert first_calls == 1
    assert second_calls == 1
    assert answer.trace.summary.backend_calls == 2
    assert tuple(event.data["outcome"] for event in finished) == ("error", "ok")
    assert answer.evidence[0].observation.provenance.backend_name == "tests.fallback_second"
    assert "backend_fallback" in {warning.code for warning in answer.warnings}


@pytest.mark.asyncio
async def test_third_identical_action_is_blocked_then_final_answer_only_runs(
    image_bytes: bytes,
) -> None:
    caption_calls = 0

    async def analyze(image: BackendImage, request: VisionRequest) -> VisionResult:
        nonlocal caption_calls
        caption_calls += 1
        assert isinstance(request, CaptionRequest)
        return _caption_result(request, f"caption {caption_calls}")

    backend = _backend(
        "tests.cycle",
        (Capability.CAPTION,),
        analyze,
        features={Capability.CAPTION: frozenset({"caption.focus"})},
    )
    repeated = _tool(
        "describe_image",
        {"asset_id": "img_aaaaaaaaaaaaaaaa", "focus": "repeat"},
    )
    policy = ScriptedPolicy([repeated, repeated, repeated, _answered("obs_000003")])
    run = RunLimits(
        max_steps=4,
        max_llm_calls=5,
        max_tool_calls=4,
        max_backend_calls=4,
        max_parallel_tools=4,
    )

    async with AsyncPenampakan(
        backends=(backend,),
        policy=policy,
        settings=_settings(run=run),
    ) as client:
        session = await client.open_image(image_bytes)
        root_id = session.root_asset.id
        repeated.arguments["asset_id"] = root_id
        answer = await session.ask("Repeat the description")
        observations = session.observations
        await session.aclose()

    assert caption_calls == 3
    assert tuple(item.payload.type for item in observations) == (
        "caption",
        "caption",
        "caption",
        "warning",
    )
    assert observations[-1].payload.code == "repeated_action_cycle"
    assert tuple(item.answer_only for item in policy.inputs) == (False, False, False, True)
    assert answer.trace.summary.llm_calls == 4
    assert answer.trace.summary.tool_calls == 3
    assert answer.trace.summary.backend_calls == 3


@pytest.mark.asyncio
async def test_invalid_action_repair_counts_llm_calls_but_not_an_extra_step(
    image_bytes: bytes,
) -> None:
    async def analyze(image: BackendImage, request: VisionRequest) -> VisionResult:
        assert isinstance(request, CaptionRequest)
        return _caption_result(request, "Repair evidence")

    backend = _backend("tests.action_repair", (Capability.CAPTION,), analyze)
    repaired_json = (
        '{"type":"answer","status":"answered","answer":"Repaired answer",'
        '"evidence":[{"observation_id":"obs_000001","supports":"Visible evidence"}]}'
    )
    llm = ScriptedTextLLM(["not valid JSON", repaired_json])
    run = RunLimits(
        max_steps=1,
        max_llm_calls=2,
        max_tool_calls=1,
        max_backend_calls=1,
        max_parallel_tools=1,
    )

    async with AsyncPenampakan(
        backends=(backend,),
        llm=llm,
        settings=_settings(run=run),
    ) as client:
        answer = await client.ask(image_bytes, "Can repair work at the exact limit?")

    assert answer.answer == "Repaired answer"
    assert answer.trace.summary.llm_calls == 2
    assert len(llm.requests) == 2
    assert llm.requests[1].metadata["repair"] == "true"
    assert llm.requests[1].metadata["answer_only"] == "true"


@pytest.mark.asyncio
async def test_evidence_repair_uses_llm_budget_without_consuming_an_extra_step(
    image_bytes: bytes,
) -> None:
    async def analyze(image: BackendImage, request: VisionRequest) -> VisionResult:
        assert isinstance(request, CaptionRequest)
        return _caption_result(request, "Evidence for repair")

    backend = _backend("tests.evidence_repair", (Capability.CAPTION,), analyze)
    invalid = _answered("obs_999999", "Invalid evidence")
    repaired = _answered("obs_000001", "Valid repaired evidence")
    policy = ScriptedPolicy([invalid, repaired])
    run = RunLimits(
        max_steps=1,
        max_llm_calls=2,
        max_tool_calls=1,
        max_backend_calls=1,
        max_parallel_tools=1,
    )

    async with AsyncPenampakan(
        backends=(backend,),
        policy=policy,
        settings=_settings(run=run),
    ) as client:
        answer = await client.ask(image_bytes, "Repair the evidence")

    assert answer.answer == "Valid repaired evidence"
    assert answer.trace.summary.llm_calls == 2
    assert tuple(item.answer_only for item in policy.inputs) == (True, True)
    assert policy.inputs[1].validation_feedback[0].code == "evidence_validation"
    assert tuple(item.remaining.steps for item in policy.inputs) == (1, 0)


@pytest.mark.asyncio
async def test_initial_plan_stops_at_exact_tool_limit(image_bytes: bytes) -> None:
    requests: list[VisionRequest] = []

    async def analyze(image: BackendImage, request: VisionRequest) -> VisionResult:
        requests.append(request)
        if isinstance(request, CaptionRequest):
            return _caption_result(request, "Only affordable call")
        if isinstance(request, OCRRequest):
            return _text_result(request, "UNUSED")
        raise AssertionError("unexpected request")

    backend = _backend(
        "tests.tool_limit",
        (Capability.CAPTION, Capability.OCR),
        analyze,
    )
    policy = ScriptedPolicy([_answered("obs_000001")])
    run = RunLimits(
        max_steps=1,
        max_llm_calls=2,
        max_tool_calls=1,
        max_backend_calls=2,
        max_parallel_tools=1,
    )

    async with AsyncPenampakan(
        backends=(backend,),
        policy=policy,
        settings=_settings((Capability.CAPTION, Capability.OCR), run=run),
    ) as client:
        answer = await client.ask(image_bytes, "Use only one tool")

    assert [type(request) for request in requests] == [CaptionRequest]
    assert answer.trace.summary.tool_calls == 1
    assert answer.trace.summary.backend_calls == 1
    assert "initial_plan_truncated" in {warning.code for warning in answer.warnings}


@pytest.mark.asyncio
async def test_fallback_stops_before_exceeding_exact_backend_limit(image_bytes: bytes) -> None:
    first_calls = 0
    second_calls = 0

    async def first_analyze(image: BackendImage, request: VisionRequest) -> VisionResult:
        nonlocal first_calls
        first_calls += 1
        raise BackendUnavailableError()

    async def second_analyze(image: BackendImage, request: VisionRequest) -> VisionResult:
        nonlocal second_calls
        second_calls += 1
        assert isinstance(request, CaptionRequest)
        return _caption_result(request, "Must not run")

    first = _backend("tests.backend_limit_first", (Capability.CAPTION,), first_analyze)
    second = _backend("tests.backend_limit_second", (Capability.CAPTION,), second_analyze)
    policy = ScriptedPolicy([_abstain()])
    run = RunLimits(
        max_steps=1,
        max_llm_calls=2,
        max_tool_calls=1,
        max_backend_calls=1,
        max_parallel_tools=1,
    )

    async with AsyncPenampakan(
        backends=(first, second),
        policy=policy,
        settings=_settings(run=run),
    ) as client:
        session = await client.open_image(image_bytes)
        with pytest.raises(BackendCallLimitExceededError):
            await session.ask("Respect backend limit")
        await session.aclose()

    assert first_calls == 1
    assert second_calls == 0
    assert policy.inputs == []


@pytest.mark.parametrize(
    ("max_assets", "max_depth", "reason"),
    [
        (1, 3, "asset_limit_exceeded"),
        (3, 1, "derivation_depth_limit_exceeded"),
    ],
)
@pytest.mark.asyncio
async def test_asset_and_depth_limits_stop_before_second_derivative(
    image_bytes: bytes,
    max_assets: int,
    max_depth: int,
    reason: str,
) -> None:
    state: dict[str, object] = {}

    async def analyze(image: BackendImage, request: VisionRequest) -> VisionResult:
        assert isinstance(request, CaptionRequest)
        return _caption_result(request, "Root evidence")

    def decide(input: PolicyInput, index: int) -> PolicyAction:
        session = state["session"]
        if index == 0:
            asset_id = session.root_asset.id
        elif index == 1:
            asset_id = session.assets[-1].id
        else:
            return _answered("obs_000002")
        return _tool(
            "crop",
            {
                "asset_id": asset_id,
                "box": {"x_min": 0.0, "y_min": 0.0, "x_max": 0.5, "y_max": 0.5},
            },
        )

    backend = _backend("tests.asset_limits", (Capability.CAPTION,), analyze)
    policy = FunctionalPolicy(decide)
    run = RunLimits(
        max_steps=3,
        max_llm_calls=4,
        max_tool_calls=3,
        max_backend_calls=1,
        max_derived_assets=max_assets,
        max_derivation_depth=max_depth,
        max_parallel_tools=3,
    )

    async with AsyncPenampakan(
        backends=(backend,),
        policy=policy,
        settings=_settings(run=run),
    ) as client:
        session = await client.open_image(image_bytes)
        state["session"] = session
        answer = await session.ask("Crop within exact limits")
        assets = session.assets
        await session.aclose()

    budget_warning = next(warning for warning in answer.warnings if warning.code == "budget_stop")
    assert budget_warning.details["reason"] == reason
    assert len(assets) == 2
    assert tuple(item.answer_only for item in policy.inputs) == (False, False, True)


@pytest.mark.asyncio
async def test_context_limit_fails_before_policy_call(image_bytes: bytes) -> None:
    async def analyze(image: BackendImage, request: VisionRequest) -> VisionResult:
        assert isinstance(request, CaptionRequest)
        return _caption_result(request, "Context cannot fit")

    backend = _backend("tests.context_limit", (Capability.CAPTION,), analyze)
    policy = ScriptedPolicy([_abstain()])
    run = RunLimits(max_context_chars=1)

    async with AsyncPenampakan(
        backends=(backend,),
        policy=policy,
        settings=_settings(run=run),
    ) as client:
        session = await client.open_image(image_bytes)
        with pytest.raises(ContextLimitExceededError):
            await session.ask("This context cannot fit")
        assert tuple(item.id for item in session.observations) == ("obs_000001",)
        await session.aclose()

    assert policy.inputs == []


@pytest.mark.asyncio
async def test_same_session_asks_serialize_and_reuse_initial_observation(
    image_bytes: bytes,
) -> None:
    backend_calls = 0
    active_policy = 0
    maximum_policy = 0

    async def analyze(image: BackendImage, request: VisionRequest) -> VisionResult:
        nonlocal backend_calls
        backend_calls += 1
        assert isinstance(request, CaptionRequest)
        return _caption_result(request, "Shared observation")

    async def decide(input: PolicyInput, index: int) -> PolicyAction:
        nonlocal active_policy, maximum_policy
        active_policy += 1
        maximum_policy = max(maximum_policy, active_policy)
        try:
            await asyncio.sleep(0.04)
            return _answered("obs_000001")
        finally:
            active_policy -= 1

    backend = _backend("tests.same_session", (Capability.CAPTION,), analyze)
    policy = FunctionalPolicy(decide)

    async with AsyncPenampakan(
        backends=(backend,),
        policy=policy,
        settings=_settings(),
    ) as client:
        session = await client.open_image(image_bytes)
        first, second = await asyncio.gather(
            session.ask("First question"),
            session.ask("Second question"),
        )
        snapshots = session.observations
        await session.aclose()

    assert first.evidence[0].observation == second.evidence[0].observation
    assert backend_calls == 1
    assert maximum_policy == 1
    assert tuple(item.id for item in snapshots) == ("obs_000001",)


@pytest.mark.asyncio
async def test_different_sessions_overlap(image_bytes: bytes, second_image_bytes: bytes) -> None:
    active_policy = 0
    maximum_policy = 0

    async def analyze(image: BackendImage, request: VisionRequest) -> VisionResult:
        assert isinstance(request, CaptionRequest)
        return _caption_result(request, f"Image width {image.asset.width}")

    async def decide(input: PolicyInput, index: int) -> PolicyAction:
        nonlocal active_policy, maximum_policy
        active_policy += 1
        maximum_policy = max(maximum_policy, active_policy)
        try:
            await asyncio.sleep(0.05)
            return _answered("obs_000001")
        finally:
            active_policy -= 1

    backend = _backend("tests.different_sessions", (Capability.CAPTION,), analyze)
    policy = FunctionalPolicy(decide)

    async with AsyncPenampakan(
        backends=(backend,),
        policy=policy,
        settings=_settings(),
    ) as client:
        first = await client.open_image(image_bytes)
        second = await client.open_image(second_image_bytes)
        answers = await asyncio.gather(first.ask("First"), second.ask("Second"))
        await asyncio.gather(first.aclose(), second.aclose())

    assert maximum_policy == 2
    assert tuple(answer.status for answer in answers) == (
        AnswerStatus.ANSWERED,
        AnswerStatus.ANSWERED,
    )


@pytest.mark.asyncio
async def test_cancelled_thread_result_never_commits_and_session_remains_reusable(
    image_bytes: bytes,
) -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    calls = 0

    def analyze(image: BackendImage, request: VisionRequest) -> VisionResult:
        nonlocal calls
        calls += 1
        assert isinstance(request, CaptionRequest)
        if calls == 1:
            started.set()
            release.wait(timeout=5.0)
            finished.set()
            return _caption_result(request, "Late result")
        return _caption_result(request, "Fresh result")

    backend = _backend("tests.late_thread", (Capability.CAPTION,), analyze)
    plan = InspectionPlan(
        operations=(InspectionOperation(request=CaptionRequest()),),
        include_available_overview=False,
    )

    async with AsyncPenampakan(backends=(backend,), settings=_settings(())) as client:
        session = await client.open_image(image_bytes)
        task = asyncio.create_task(session.inspect(plan))
        assert await asyncio.to_thread(started.wait, 2.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert session.observations == ()
        release.set()
        assert await asyncio.to_thread(finished.wait, 2.0)
        await asyncio.sleep(0)
        assert session.observations == ()
        result = await session.inspect(plan)
        snapshots = session.observations
        await asyncio.gather(session.aclose(), session.aclose())

    assert tuple(item.payload.text for item in result.observations) == ("Fresh result",)
    assert tuple(item.id for item in snapshots) == ("obs_000001",)
    assert calls == 2


@pytest.mark.asyncio
async def test_idle_session_close_is_idempotent_and_rejects_operations(image_bytes: bytes) -> None:
    async with AsyncPenampakan(settings=_settings(())) as client:
        session = await client.open_image(image_bytes)
        await asyncio.gather(session.aclose(), session.aclose(), session.aclose())
        await session.aclose()

        assert session.closed is True
        with pytest.raises(SessionClosedError):
            _ = session.root_asset
        with pytest.raises(SessionClosedError):
            await session.inspect()


@pytest.mark.asyncio
async def test_client_close_waits_for_active_session_without_cancelling_it(
    image_bytes: bytes,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    close_count = 0

    async def analyze(image: BackendImage, request: VisionRequest) -> VisionResult:
        assert isinstance(request, CaptionRequest)
        started.set()
        await release.wait()
        return _caption_result(request, "Completed before close")

    async def close_backend() -> None:
        nonlocal close_count
        close_count += 1

    backend = _backend(
        "tests.active_close",
        (Capability.CAPTION,),
        analyze,
        close=close_backend,
    )
    client = AsyncPenampakan(backends=(backend,), settings=_settings(()))
    active = await client.open_image(image_bytes)
    idle = await client.open_image(image_bytes)
    plan = InspectionPlan(
        operations=(InspectionOperation(request=CaptionRequest()),),
        include_available_overview=False,
    )
    operation = asyncio.create_task(active.inspect(plan))
    await started.wait()
    close_task = asyncio.create_task(client.aclose())
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert close_task.done() is False
    assert operation.cancelled() is False
    with pytest.raises(SessionClosedError):
        await client.open_image(image_bytes)

    release.set()
    result = await operation
    await close_task
    await client.aclose()

    assert tuple(item.payload.text for item in result.observations) == ("Completed before close",)
    assert active.closed is True
    assert idle.closed is True
    assert client.closed is True
    assert close_count == 1


class _RecordingCache:
    def __init__(self, *, durable: bool) -> None:
        self.durable = durable
        self.gets: list[str] = []
        self.sets: list[str] = []
        self._entries: dict[str, bytes] = {}

    async def get(self, key: str) -> bytes | None:
        self.gets.append(key)
        return self._entries.get(key)

    async def set(self, key: str, value: bytes, *, size: int) -> None:
        self.sets.append(key)
        self._entries[key] = value

    async def aclose(self) -> None:
        self._entries.clear()


class _ModelCaptionBackend:
    def __init__(
        self,
        *,
        model_revision: str | None,
        delay_s: float = 0.0,
        warnings: Sequence[WarningInfo] = (),
    ) -> None:
        self._descriptor = BackendDescriptor(
            name="tests.model_caption",
            version="1.0",
            model_id="org/caption",
            model_revision=model_revision,
            capabilities=(CapabilityDescriptor(capability=Capability.CAPTION),),
            max_concurrency=4,
        )
        self._delay_s = delay_s
        self._warnings = tuple(warnings)
        self.calls = 0

    @property
    def descriptor(self) -> BackendDescriptor:
        return self._descriptor

    def supports(self, request: VisionRequest) -> bool:
        return isinstance(request, CaptionRequest)

    async def analyze(self, image: BackendImage, request: VisionRequest) -> VisionResult:
        assert isinstance(request, CaptionRequest)
        self.calls += 1
        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        result = _caption_result(request, "A four-color image")
        return VisionResult(observations=result.observations, warnings=self._warnings)

    async def aclose(self) -> None:
        return None


def _duplicate_caption_plan(count: int = 2) -> InspectionPlan:
    return InspectionPlan(
        operations=tuple(InspectionOperation(request=CaptionRequest()) for _ in range(count)),
        include_available_overview=False,
    )


@pytest.mark.asyncio
async def test_durable_cache_is_bypassed_for_unresolved_model_weights(
    image_bytes: bytes,
) -> None:
    backend = _ModelCaptionBackend(model_revision=None, delay_s=0.05)
    cache = _RecordingCache(durable=True)

    async with AsyncPenampakan(
        backends=(backend,),
        cache=cache,
        settings=_settings(()),
    ) as client:
        first = await client.inspect(image_bytes, _duplicate_caption_plan())
        second = await client.inspect(image_bytes, _duplicate_caption_plan(1))

    assert cache.gets == []
    assert cache.sets == []
    assert backend.calls == 2
    assert len(first.observations) == 2
    assert sum(item.provenance.cache_hit for item in first.observations) == 1
    assert second.observations[0].provenance.cache_hit is False
    assert second.observations[0].provenance.model_revision is None


@pytest.mark.asyncio
async def test_durable_cache_is_used_for_resolved_model_weights(image_bytes: bytes) -> None:
    backend = _ModelCaptionBackend(model_revision="a" * 40, delay_s=0.05)
    cache = _RecordingCache(durable=True)

    async with AsyncPenampakan(
        backends=(backend,),
        cache=cache,
        settings=_settings(()),
    ) as client:
        await client.inspect(image_bytes, _duplicate_caption_plan())
        second = await client.inspect(image_bytes, _duplicate_caption_plan(1))

    assert len(cache.sets) == 1
    assert cache.gets[0] == cache.sets[0]
    assert backend.calls == 1
    assert second.observations[0].provenance.cache_hit is True
    assert second.observations[0].provenance.model_revision == "a" * 40


@pytest.mark.asyncio
async def test_non_durable_cache_serves_unresolved_model_weights(image_bytes: bytes) -> None:
    backend = _ModelCaptionBackend(model_revision=None, delay_s=0.05)
    cache = _RecordingCache(durable=False)

    async with AsyncPenampakan(
        backends=(backend,),
        cache=cache,
        settings=_settings(()),
    ) as client:
        await client.inspect(image_bytes, _duplicate_caption_plan())
        second = await client.inspect(image_bytes, _duplicate_caption_plan(1))

    assert len(cache.sets) == 1
    assert len(cache.gets) == 3
    assert backend.calls == 1
    assert second.observations[0].provenance.cache_hit is True
    assert second.observations[0].provenance.model_revision is None


@pytest.mark.asyncio
async def test_durable_cache_bypass_keeps_single_flight_deduplication(
    image_bytes: bytes,
) -> None:
    backend = _ModelCaptionBackend(model_revision=None, delay_s=0.1)
    cache = _RecordingCache(durable=True)

    async with AsyncPenampakan(
        backends=(backend,),
        cache=cache,
        settings=_settings(()),
    ) as client:
        result = await client.inspect(image_bytes, _duplicate_caption_plan(3))

    assert backend.calls == 1
    assert len(result.observations) == 3
    assert cache.gets == []
    assert cache.sets == []


_CACHE_FAILURE = "cache_operation_failed"
_CACHE_CANARY = "canary-cache-secret-8d41"
_RESOLVED_REVISION = "a" * 40


class _FailingCache(_RecordingCache):
    """A recording cache whose selected operations raise a leaky failure."""

    def __init__(
        self,
        *,
        fail_get: bool = False,
        fail_set: bool = False,
        error: BaseException | None = None,
    ) -> None:
        super().__init__(durable=False)
        self._fail_get = fail_get
        self._fail_set = fail_set
        self._error = error

    def _fail(self, key: str) -> None:
        if self._error is not None:
            raise self._error
        raise RuntimeError(f"{_CACHE_CANARY} while reaching /var/cache/{key}")

    async def get(self, key: str) -> bytes | None:
        if self._fail_get:
            self.gets.append(key)
            self._fail(key)
        return await super().get(key)

    async def set(self, key: str, value: bytes, *, size: int) -> None:
        if self._fail_set:
            self.sets.append(key)
            self._fail(key)
        await super().set(key, value, size=size)


@pytest.mark.asyncio
async def test_a_failing_cache_get_degrades_to_one_reported_miss(image_bytes: bytes) -> None:
    backend = _ModelCaptionBackend(model_revision=_RESOLVED_REVISION)
    cache = _FailingCache(fail_get=True)

    async with AsyncPenampakan(
        backends=(backend,),
        cache=cache,
        settings=_settings(()),
    ) as client:
        result = await client.inspect(image_bytes, _duplicate_caption_plan(1))

    assert backend.calls == 1
    assert len(result.observations) == 1
    assert result.observations[0].provenance.cache_hit is False
    assert [warning.code for warning in result.warnings] == [_CACHE_FAILURE]
    assert result.warnings[0].details == {
        "error_type": "RuntimeError",
        "failed_operations": 1,
    }


@pytest.mark.asyncio
async def test_a_failing_cache_set_degrades_to_one_reported_no_op(image_bytes: bytes) -> None:
    backend = _ModelCaptionBackend(model_revision=_RESOLVED_REVISION)
    cache = _FailingCache(fail_set=True)

    async with AsyncPenampakan(
        backends=(backend,),
        cache=cache,
        settings=_settings(()),
    ) as client:
        result = await client.inspect(image_bytes, _duplicate_caption_plan(1))

    assert backend.calls == 1
    assert len(cache.gets) == 1
    assert len(cache.sets) == 1
    assert len(result.observations) == 1
    assert [warning.code for warning in result.warnings] == [_CACHE_FAILURE]
    assert result.warnings[0].details["failed_operations"] == 1


@pytest.mark.asyncio
async def test_two_failing_cache_operations_report_exactly_one_warning(
    image_bytes: bytes,
) -> None:
    backend = _ModelCaptionBackend(model_revision=_RESOLVED_REVISION)
    cache = _FailingCache(fail_get=True, fail_set=True)

    async with AsyncPenampakan(
        backends=(backend,),
        cache=cache,
        settings=_settings(()),
    ) as client:
        result = await client.inspect(image_bytes, _duplicate_caption_plan(1))

    codes = [warning.code for warning in result.warnings]
    assert codes == [_CACHE_FAILURE]
    assert result.warnings[0].details["failed_operations"] == 2
    assert (
        result.warnings[0].message
        == "A cache operation failed; this perception ran without caching."
    )


@pytest.mark.asyncio
async def test_a_cache_failure_leaks_no_key_path_or_exception_text(image_bytes: bytes) -> None:
    backend = _ModelCaptionBackend(model_revision=_RESOLVED_REVISION)
    cache = _FailingCache(fail_get=True, fail_set=True)

    async with AsyncPenampakan(
        backends=(backend,),
        cache=cache,
        settings=_settings(()),
    ) as client:
        result = await client.inspect(image_bytes, _duplicate_caption_plan(1))

    serialized = result.model_dump_json()
    assert _CACHE_CANARY not in serialized
    assert "/var/cache/" not in serialized
    assert cache.gets and cache.sets
    for key in (*cache.gets, *cache.sets):
        assert key not in serialized


@pytest.mark.asyncio
async def test_cache_failures_are_counted_once_per_operation_in_the_trace(
    image_bytes: bytes,
) -> None:
    backend = _ModelCaptionBackend(model_revision=_RESOLVED_REVISION)
    cache = _FailingCache(
        fail_get=True,
        fail_set=True,
        error=PenampakanError(_CACHE_CANARY, code="cache_unwritable"),
    )

    async with AsyncPenampakan(
        backends=(backend,),
        cache=cache,
        settings=_settings(()),
    ) as client:
        result = await client.inspect(image_bytes, _duplicate_caption_plan(1))

    failures = tuple(event for event in result.trace.events if event.event_type == _CACHE_FAILURE)
    assert [event.data["operation"] for event in failures] == ["get", "set"]
    assert {event.data["error_type"] for event in failures} == {"PenampakanError"}
    assert {event.data["error_code"] for event in failures} == {"cache_unwritable"}
    assert result.trace.summary.cache_hits == 0
    assert result.warnings[0].details["error_code"] == "cache_unwritable"
    assert _CACHE_CANARY not in result.model_dump_json()


@pytest.mark.asyncio
async def test_cache_cancellation_still_propagates(image_bytes: bytes) -> None:
    backend = _ModelCaptionBackend(model_revision=_RESOLVED_REVISION)
    cache = _FailingCache(fail_get=True, error=asyncio.CancelledError())

    async with AsyncPenampakan(
        backends=(backend,),
        cache=cache,
        settings=_settings(()),
    ) as client:
        with pytest.raises(asyncio.CancelledError):
            await client.inspect(image_bytes, _duplicate_caption_plan(1))

    assert backend.calls == 0


@pytest.mark.asyncio
async def test_a_genuine_cache_hit_keeps_backend_attribution(image_bytes: bytes) -> None:
    backend = _ModelCaptionBackend(model_revision=_RESOLVED_REVISION)
    cache = _RecordingCache(durable=True)

    async with AsyncPenampakan(
        backends=(backend,),
        cache=cache,
        settings=_settings(()),
    ) as client:
        await client.inspect(image_bytes, _duplicate_caption_plan(1))
        second = await client.inspect(image_bytes, _duplicate_caption_plan(1))

    provenance = second.observations[0].provenance
    assert backend.calls == 1
    assert provenance.cache_hit is True
    assert provenance.backend_name == "tests.model_caption"
    assert provenance.model_id == "org/caption"
    assert provenance.model_revision == _RESOLVED_REVISION
    assert [warning.code for warning in second.warnings] == []


@pytest.mark.asyncio
async def test_a_shared_population_from_a_fallback_backend_is_not_a_cache_hit(
    image_bytes: bytes,
) -> None:
    unavailable_calls = 0
    successful_calls = 0

    async def unavailable(image: BackendImage, request: VisionRequest) -> VisionResult:
        nonlocal unavailable_calls
        unavailable_calls += 1
        await asyncio.sleep(0.05)
        raise BackendUnavailableError(code="scripted_unavailable")

    async def successful(image: BackendImage, request: VisionRequest) -> VisionResult:
        nonlocal successful_calls
        successful_calls += 1
        await asyncio.sleep(0.05)
        assert isinstance(request, CaptionRequest)
        return _caption_result(request, "Fallback caption")

    features = {Capability.CAPTION: frozenset({"caption.focus"})}
    primary = _backend(
        "tests.fallback_first", (Capability.CAPTION,), unavailable, features=features
    )
    secondary = _backend(
        "tests.fallback_second", (Capability.CAPTION,), successful, features=features
    )

    async with AsyncPenampakan(
        backends=(primary, secondary),
        cache=_RecordingCache(durable=False),
        settings=_settings(()),
    ) as client:
        result = await client.inspect(image_bytes, _duplicate_caption_plan(3))

    assert len(result.observations) == 3
    assert {item.provenance.backend_name for item in result.observations} == {
        "tests.fallback_second"
    }
    assert not any(item.provenance.cache_hit for item in result.observations)
    assert unavailable_calls == 3
    assert successful_calls == 3


@pytest.mark.asyncio
async def test_a_shared_population_deduplicates_across_separate_sessions(
    image_bytes: bytes,
) -> None:
    calls = 0

    async def slow_caption(image: BackendImage, request: VisionRequest) -> VisionResult:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        assert isinstance(request, CaptionRequest)
        return _caption_result(request, "A shared caption")

    backend = _backend("tests.shared", (Capability.CAPTION,), slow_caption)
    plan = _duplicate_caption_plan(1)

    async with AsyncPenampakan(
        backends=(backend,),
        cache=_RecordingCache(durable=False),
        settings=_settings(()),
    ) as client:
        # Two private sessions over identical bytes share the client's flight.
        first, second = await asyncio.gather(
            client.inspect(image_bytes, plan),
            client.inspect(image_bytes, plan),
        )

    assert calls == 1
    served = {item.provenance.backend_name for item in (*first.observations, *second.observations)}
    assert served == {"tests.shared"}


_UNRESOLVED_REVISION = "unresolved_model_revision"


def _adapter_unresolved_warning() -> WarningInfo:
    return WarningInfo(
        code=_UNRESOLVED_REVISION,
        message=(
            "The exact model weight revision is unresolved; pin an immutable "
            "commit revision for reproducible inference and durable caching."
        ),
    )


@pytest.mark.asyncio
async def test_unresolved_model_weights_warn_on_every_inspection(image_bytes: bytes) -> None:
    backend = _ModelCaptionBackend(model_revision=None)

    async with AsyncPenampakan(
        backends=(backend,),
        settings=_settings(()),
    ) as client:
        result = await client.inspect(image_bytes, _duplicate_caption_plan(1))

    assert [warning.code for warning in result.warnings] == [_UNRESOLVED_REVISION]
    assert result.warnings[0].message == _adapter_unresolved_warning().message
    assert result.observations[0].provenance.model_revision is None


@pytest.mark.asyncio
async def test_unresolved_model_weights_warn_on_every_run(image_bytes: bytes) -> None:
    backend = _ModelCaptionBackend(model_revision=None)
    policy = ScriptedPolicy([_answered("obs_000001")])

    async with AsyncPenampakan(
        backends=(backend,),
        policy=policy,
        settings=_settings(),
    ) as client:
        answer = await client.ask(image_bytes, "What is visible?")

    assert [warning.code for warning in answer.warnings] == [_UNRESOLVED_REVISION]


@pytest.mark.asyncio
async def test_a_backend_reported_unresolved_warning_is_not_duplicated(
    image_bytes: bytes,
) -> None:
    backend = _ModelCaptionBackend(
        model_revision=None,
        warnings=(_adapter_unresolved_warning(),),
    )

    async with AsyncPenampakan(
        backends=(backend,),
        settings=_settings(()),
    ) as client:
        result = await client.inspect(image_bytes, _duplicate_caption_plan(1))

    assert [warning.code for warning in result.warnings] == [_UNRESOLVED_REVISION]
    assert result.warnings[0] == _adapter_unresolved_warning()


@pytest.mark.asyncio
async def test_resolved_model_weights_report_no_unresolved_warning(image_bytes: bytes) -> None:
    backend = _ModelCaptionBackend(model_revision="a" * 40)

    async with AsyncPenampakan(
        backends=(backend,),
        settings=_settings(()),
    ) as client:
        result = await client.inspect(image_bytes, _duplicate_caption_plan(1))

    assert [warning.code for warning in result.warnings] == []
    assert result.observations[0].provenance.model_revision == "a" * 40


@pytest.mark.asyncio
async def test_a_non_model_backend_reports_no_unresolved_warning(image_bytes: bytes) -> None:
    plan = InspectionPlan(
        operations=(InspectionOperation(request=ColorsRequest()),),
        include_available_overview=False,
    )

    async with AsyncPenampakan(settings=_settings(())) as client:
        result = await client.inspect(image_bytes, plan)

    assert result.observations[0].provenance.model_id is None
    assert [warning.code for warning in result.warnings] == []


@pytest.mark.asyncio
async def test_unresolved_model_weights_warn_on_an_ephemeral_cache_hit(
    image_bytes: bytes,
) -> None:
    backend = _ModelCaptionBackend(model_revision=None)
    cache = _RecordingCache(durable=False)

    async with AsyncPenampakan(
        backends=(backend,),
        cache=cache,
        settings=_settings(()),
    ) as client:
        await client.inspect(image_bytes, _duplicate_caption_plan(1))
        second = await client.inspect(image_bytes, _duplicate_caption_plan(1))

    assert backend.calls == 1
    assert second.observations[0].provenance.cache_hit is True
    assert [warning.code for warning in second.warnings] == [_UNRESOLVED_REVISION]


class DegradingPolicy(ScriptedPolicy):
    """A policy that reports typed provider degradation for a run."""

    def __init__(self, actions: Sequence[PolicyAction], warnings: Sequence[WarningInfo]) -> None:
        super().__init__(actions)
        self._warnings = tuple(warnings)

    @property
    def degradations(self) -> tuple[WarningInfo, ...]:
        """Return the typed degradation this policy reports."""

        return self._warnings


def _degraded_warning() -> WarningInfo:
    return WarningInfo(
        code="degraded_schema_enforcement",
        message="The language model provider could not enforce the action schema strictly.",
        details={"schema_enforcement": "json_only"},
    )


@pytest.mark.asyncio
async def test_policy_degradation_is_attached_to_the_run_exactly_once(
    image_bytes: bytes,
) -> None:
    async def analyze(image: BackendImage, request: VisionRequest) -> VisionResult:
        return _caption_result(request, "Colored quadrants")

    backend = _backend("tests.degraded", (Capability.CAPTION,), analyze)
    policy = DegradingPolicy(
        [_answered("obs_000001"), _answered("obs_000001")],
        (_degraded_warning(), _degraded_warning()),
    )

    async with AsyncPenampakan(
        backends=(backend,),
        policy=policy,
        settings=_settings(),
    ) as client:
        session = await client.open_image(image_bytes)
        first = await session.ask("What is visible?")
        second = await session.ask("What is visible?")
        await session.aclose()

    for answer in (first, second):
        codes = [warning.code for warning in answer.warnings]
        assert codes.count("degraded_schema_enforcement") == 1
        warning = next(
            item
            for item in answer.warnings
            if item.code == codes[codes.index("degraded_schema_enforcement")]
        )
        assert warning.details == {"schema_enforcement": "json_only"}


@pytest.mark.asyncio
async def test_policy_degradation_is_ignored_when_malformed_or_excessive(
    image_bytes: bytes,
) -> None:
    async def analyze(image: BackendImage, request: VisionRequest) -> VisionResult:
        return _caption_result(request, "Colored quadrants")

    backend = _backend("tests.malformed", (Capability.CAPTION,), analyze)
    surplus = tuple(
        WarningInfo(code=f"degraded_{index}", message="Degraded.") for index in range(9)
    )
    policy = DegradingPolicy([_answered("obs_000001")], surplus)
    policy._warnings = cast(  # type: ignore[assignment]
        "tuple[WarningInfo, ...]",
        ("not-a-warning", *surplus),
    )

    async with AsyncPenampakan(
        backends=(backend,),
        policy=policy,
        settings=_settings(),
    ) as client:
        answer = await client.ask(image_bytes, "What is visible?")

    degraded = [warning.code for warning in answer.warnings if warning.code.startswith("degraded_")]
    assert len(degraded) <= 4
    assert all(isinstance(warning, WarningInfo) for warning in answer.warnings)


@pytest.mark.asyncio
async def test_a_policy_without_degradations_adds_no_warnings(image_bytes: bytes) -> None:
    async def analyze(image: BackendImage, request: VisionRequest) -> VisionResult:
        return _caption_result(request, "Colored quadrants")

    backend = _backend("tests.plain", (Capability.CAPTION,), analyze)
    policy = ScriptedPolicy([_answered("obs_000001")])

    async with AsyncPenampakan(
        backends=(backend,),
        policy=policy,
        settings=_settings(),
    ) as client:
        answer = await client.ask(image_bytes, "What is visible?")

    assert [warning.code for warning in answer.warnings] == []


def _answer_json(observation_id: str) -> str:
    return json.dumps(
        {
            "type": "answer",
            "status": "answered",
            "answer": "It contains four colored quadrants.",
            "evidence": [{"observation_id": observation_id, "supports": "The caption."}],
        }
    )


@pytest.mark.asyncio
async def test_provider_attempts_and_tokens_reach_the_run_trace(image_bytes: bytes) -> None:
    async def analyze(image: BackendImage, request: VisionRequest) -> VisionResult:
        return _caption_result(request, "Colored quadrants")

    backend = _backend("tests.metrics", (Capability.CAPTION,), analyze)
    llm = ScriptedTextLLM(
        [
            LLMResponse(
                text=_answer_json("obs_000001"),
                model_id="provider-model",
                usage=TokenUsage(input_tokens=11, output_tokens=7),
                provider="openai",
                attempts=3,
            )
        ]
    )

    async with AsyncPenampakan(
        backends=(backend,),
        llm=llm,
        settings=_settings(),
    ) as client:
        answer = await client.ask(image_bytes, "What is shown?")

    assert answer.trace.summary.input_tokens == 11
    assert answer.trace.summary.output_tokens == 7
    # One orchestrator reservation, three reported provider attempts.
    assert answer.trace.summary.llm_calls == 1
    finished = [
        event for event in answer.trace.events if event.event_type == "policy_call_finished"
    ]
    assert len(finished) == 1
    assert finished[0].data["provider_attempts"] == 3
    assert finished[0].data["input_tokens"] == 11
    assert finished[0].data["schema_enforcement"] == "strict"


@pytest.mark.asyncio
async def test_invalid_response_metrics_are_preserved_when_repair_succeeds(
    image_bytes: bytes,
) -> None:
    async def analyze(image: BackendImage, request: VisionRequest) -> VisionResult:
        return _caption_result(request, "Colored quadrants")

    backend = _backend("tests.repair_metrics", (Capability.CAPTION,), analyze)
    llm = ScriptedTextLLM(
        [
            LLMResponse(
                text="not valid JSON",
                usage=TokenUsage(input_tokens=5, output_tokens=2),
                attempts=2,
            ),
            LLMResponse(
                text=_answer_json("obs_000001"),
                usage=TokenUsage(input_tokens=11, output_tokens=7),
                attempts=3,
            ),
        ]
    )

    async with AsyncPenampakan(
        backends=(backend,),
        llm=llm,
        settings=_settings(),
    ) as client:
        answer = await client.ask(image_bytes, "What is shown?")

    assert answer.trace.summary.llm_calls == 2
    assert answer.trace.summary.input_tokens == 16
    assert answer.trace.summary.output_tokens == 9
    finished = [
        event for event in answer.trace.events if event.event_type == "policy_call_finished"
    ]
    assert [event.data["action_type"] for event in finished] == ["invalid", "answer"]
    assert [event.data["provider_attempts"] for event in finished] == [2, 3]


@pytest.mark.asyncio
async def test_a_response_without_usage_reports_no_token_counters(image_bytes: bytes) -> None:
    async def analyze(image: BackendImage, request: VisionRequest) -> VisionResult:
        return _caption_result(request, "Colored quadrants")

    backend = _backend("tests.no_usage", (Capability.CAPTION,), analyze)
    llm = ScriptedTextLLM([_answer_json("obs_000001")])

    async with AsyncPenampakan(
        backends=(backend,),
        llm=llm,
        settings=_settings(),
    ) as client:
        answer = await client.ask(image_bytes, "What is shown?")

    assert answer.trace.summary.input_tokens is None
    assert answer.trace.summary.output_tokens is None
    finished = next(
        event for event in answer.trace.events if event.event_type == "policy_call_finished"
    )
    assert "input_tokens" not in finished.data
    assert finished.data["provider_attempts"] == 1
