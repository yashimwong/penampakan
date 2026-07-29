from __future__ import annotations

import asyncio
import json
from collections import deque
from io import BytesIO

import pytest
from PIL import Image

from penampakan.backends.callable import CallableVisionBackend
from penampakan.client import AsyncPenampakan
from penampakan.config import AgentSettings, Settings
from penampakan.errors import SyncInAsyncContextError
from penampakan.llms.callable import CallableTextLLM
from penampakan.models import (
    AnswerAction,
    AnswerStatus,
    BackendDescriptor,
    BackendImage,
    Capability,
    CapabilityDescriptor,
    CaptionPayload,
    CaptionRequest,
    ColorsRequest,
    EvidenceRef,
    InspectionOperation,
    InspectionPlan,
    LLMRequest,
    MetadataRequest,
    ObservationDraft,
    PolicyAction,
    PolicyInput,
    VisionRequest,
    VisionResult,
)
from penampakan.sync import Penampakan


def _png_bytes() -> bytes:
    image = Image.new("RGB", (12, 8), (210, 30, 20))
    output = BytesIO()
    try:
        image.save(output, format="PNG")
        return output.getvalue()
    finally:
        output.close()
        image.close()


def _caption_backend() -> CallableVisionBackend:
    descriptor = BackendDescriptor(
        name="tests.caption",
        version="1.0",
        capabilities=(
            CapabilityDescriptor(
                capability=Capability.CAPTION,
                features=frozenset({"caption.focus"}),
            ),
        ),
    )

    def analyze(image: BackendImage, request: VisionRequest) -> VisionResult:
        assert image.asset.width == 12
        assert isinstance(request, CaptionRequest)
        return VisionResult(
            observations=(
                ObservationDraft(
                    payload=CaptionPayload(
                        text="A red rectangular image.",
                        focus=request.focus,
                    )
                ),
            )
        )

    return CallableVisionBackend(descriptor, analyze)


def _caption_settings() -> Settings:
    return Settings(
        agent=AgentSettings(
            initial_capabilities=(Capability.METADATA, Capability.CAPTION),
        )
    )


def _answer_json(observation_id: str, answer: str) -> str:
    return json.dumps(
        {
            "type": "answer",
            "status": "answered",
            "answer": answer,
            "evidence": [
                {
                    "observation_id": observation_id,
                    "supports": "The cited visual observation supports the answer.",
                }
            ],
            "uncertainties": [],
        },
        separators=(",", ":"),
    )


class MetadataPolicy:
    def __init__(self) -> None:
        self.inputs: list[PolicyInput] = []

    async def next_action(self, input: PolicyInput) -> PolicyAction:
        self.inputs.append(input)
        metadata_id = next(
            str(value["id"])
            for line in input.context.splitlines()
            if line.startswith("{")
            for value in (json.loads(line),)
            if isinstance(value, dict) and value.get("type") == "metadata"
        )
        return AnswerAction(
            status="answered",
            answer="The image is 12 by 8 pixels.",
            evidence=(
                EvidenceRef(
                    observation_id=metadata_id,
                    supports="The metadata gives the normalized dimensions.",
                ),
            ),
        )


@pytest.mark.asyncio
async def test_async_inspect_only_pillow_workflow() -> None:
    plan = InspectionPlan(
        operations=(
            InspectionOperation(request=MetadataRequest()),
            InspectionOperation(request=ColorsRequest(count=3)),
        ),
        include_available_overview=False,
    )
    async with AsyncPenampakan() as vision:
        result = await vision.inspect(_png_bytes(), plan)

    assert result.root_asset.width == 12
    assert result.root_asset.height == 8
    assert tuple(item.payload.type for item in result.observations) == (
        "metadata",
        "colors",
    )
    assert result.trace.summary.tool_calls == 2
    assert result.trace.summary.backend_calls == 2


def test_sync_inspect_only_pillow_workflow() -> None:
    plan = InspectionPlan(
        operations=(InspectionOperation(request=MetadataRequest()),),
        include_available_overview=False,
    )
    with Penampakan() as vision:
        result = vision.inspect(_png_bytes(), plan)

    assert result.root_asset.width == 12
    assert len(result.observations) == 1
    assert result.observations[0].payload.type == "metadata"


@pytest.mark.asyncio
async def test_callable_caption_can_be_cited_directly() -> None:
    requests: list[LLMRequest] = []

    def complete(request: LLMRequest) -> str:
        requests.append(request)
        return _answer_json("obs_000002", "The image is red and rectangular.")

    async with AsyncPenampakan(
        llm=CallableTextLLM(complete),
        backends=(_caption_backend(),),
        settings=_caption_settings(),
    ) as vision:
        answer = await vision.ask(_png_bytes(), "What does the image look like?")

    assert answer.status is AnswerStatus.ANSWERED
    assert answer.evidence[0].observation.id == "obs_000002"
    assert answer.evidence[0].observation.payload.type == "caption"
    assert len(requests) == 1
    assert answer.trace.summary.llm_calls == 1


@pytest.mark.asyncio
async def test_malformed_json_receives_one_successful_repair() -> None:
    responses = deque(
        (
            "this is not json",
            _answer_json("obs_000002", "The image is red and rectangular."),
        )
    )
    requests: list[LLMRequest] = []

    def complete(request: LLMRequest) -> str:
        requests.append(request)
        return responses.popleft()

    async with AsyncPenampakan(
        llm=CallableTextLLM(complete),
        backends=(_caption_backend(),),
        settings=_caption_settings(),
    ) as vision:
        answer = await vision.ask(_png_bytes(), "Describe the image.")

    assert answer.status is AnswerStatus.ANSWERED
    assert len(requests) == 2
    assert requests[0].metadata["repair"] == "false"
    assert requests[1].metadata["repair"] == "true"
    assert answer.trace.summary.llm_calls == 2


@pytest.mark.asyncio
async def test_custom_policy_answers_from_metadata() -> None:
    policy = MetadataPolicy()
    async with AsyncPenampakan(policy=policy) as vision:
        answer = await vision.ask(_png_bytes(), "What are the image dimensions?")

    assert answer.answer == "The image is 12 by 8 pixels."
    assert answer.evidence[0].observation.payload.type == "metadata"
    assert len(policy.inputs) == 1


@pytest.mark.asyncio
async def test_reusable_session_reuses_initial_observations_for_second_question() -> None:
    policy = MetadataPolicy()
    async with (
        AsyncPenampakan(policy=policy) as vision,
        await vision.open_image(_png_bytes()) as session,
    ):
        first = await session.ask("What are the dimensions?")
        first_observations = session.observations
        second = await session.ask("How wide is the image?")
        second_observations = session.observations

    assert tuple(item.id for item in first_observations) == ("obs_000001",)
    assert second_observations == first_observations
    assert first.evidence[0].observation.id == "obs_000001"
    assert second.evidence[0].observation.id == "obs_000001"
    assert len(policy.inputs) == 2


@pytest.mark.asyncio
async def test_one_shot_evidence_survives_session_and_client_cleanup() -> None:
    vision = AsyncPenampakan(policy=MetadataPolicy())
    answer = await vision.ask(_png_bytes(), "What are the image dimensions?")
    await vision.aclose()

    evidence = answer.evidence[0]
    assert evidence.observation.id == "obs_000001"
    assert evidence.observation.asset_id == answer.evidence[0].observation.asset_id
    assert evidence.observation.payload.type == "metadata"
    assert answer.answer == "The image is 12 by 8 pixels."


@pytest.mark.asyncio
async def test_sync_facade_rejects_calls_from_a_running_loop() -> None:
    vision = Penampakan()
    try:
        with pytest.raises(SyncInAsyncContextError):
            vision.inspect(_png_bytes())
    finally:
        await asyncio.to_thread(vision.close)
