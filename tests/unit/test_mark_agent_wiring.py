from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from penampakan.backends.callable import CallableVisionBackend
from penampakan.client import AsyncPenampakan
from penampakan.config import AgentSettings, Settings
from penampakan.models import (
    AnswerAction,
    BackendDescriptor,
    BackendImage,
    Box,
    Capability,
    CapabilityDescriptor,
    CaptionPayload,
    CaptionRequest,
    DetectionPayload,
    DetectionRequest,
    ObservationDraft,
    PolicyAction,
    PolicyInput,
    ToolAction,
    VisionRequest,
    VisionResult,
)
from penampakan.perception.registry import ToolRegistry
from penampakan.perception.router import BackendRouter
from penampakan.reasoning.prompts import (
    AGENT_V1_SYSTEM_PROMPT,
    build_policy_request,
    build_system_prompt,
)
from penampakan.tools.builtin import MarkRegionsArguments, register_mark_tool
from tests.fixtures.images import encode_image, quadrants_image
from tests.unit.reasoning.helpers import make_policy_input, make_tool_spec


def _descriptor(
    name: str,
    capabilities: tuple[CapabilityDescriptor, ...],
    *,
    pinned: bool = True,
) -> BackendDescriptor:
    return BackendDescriptor(
        name=name,
        version="1.0",
        model_id="tests/mark-aware-model",
        model_revision="a" * 40 if pinned else None,
        capabilities=capabilities,
    )


def _capability(
    capability: Capability,
    *features: str,
) -> CapabilityDescriptor:
    return CapabilityDescriptor(capability=capability, features=frozenset(features))


@pytest.mark.parametrize(
    ("descriptors", "expected"),
    [
        ((_descriptor("tests.detect_only", (_capability(Capability.DETECT),)),), False),
        (
            (
                _descriptor(
                    "tests.mark_only",
                    (_capability(Capability.CAPTION, "caption.mark_references"),),
                ),
            ),
            False,
        ),
        (
            (
                _descriptor(
                    "tests.near_match",
                    (
                        _capability(Capability.DETECT),
                        _capability(Capability.CAPTION, "caption.mark_reference"),
                    ),
                ),
            ),
            False,
        ),
        (
            (
                _descriptor(
                    "tests.unpinned",
                    (
                        _capability(Capability.DETECT),
                        _capability(Capability.CAPTION, "caption.mark_references"),
                    ),
                    pinned=False,
                ),
            ),
            False,
        ),
        (
            (
                _descriptor(
                    "tests.eligible",
                    (
                        _capability(Capability.DETECT),
                        _capability(Capability.CAPTION, "caption.mark_references"),
                    ),
                ),
            ),
            True,
        ),
    ],
    ids=("detection-only", "mark-feature-only", "near-match", "unpinned", "eligible"),
)
def test_mark_tools_require_localization_and_an_exact_pinned_backend_feature(
    descriptors: tuple[BackendDescriptor, ...],
    expected: bool,
) -> None:
    router = cast(BackendRouter, SimpleNamespace(descriptors=descriptors))

    tool_names = {spec.name for spec in AsyncPenampakan._build_tools(router).specs}

    assert ("mark_regions" in tool_names) is expected
    assert ("describe_marks" in tool_names) is expected


def test_mark_regions_policy_schema_accepts_only_asset_and_observation_ids() -> None:
    schema = MarkRegionsArguments.model_json_schema()

    assert schema["additionalProperties"] is False
    assert schema["required"] == ["asset_id", "source_observation_ids"]
    assert set(schema["properties"]) == {"asset_id", "source_observation_ids"}
    observation_ids = schema["properties"]["source_observation_ids"]
    assert observation_ids["minItems"] == 1
    assert observation_ids["maxItems"] == 99
    assert observation_ids["items"] == {
        "type": "string",
        "pattern": r"^obs_[0-9]{6,}$",
    }


def test_prompt_dispatch_preserves_v1_and_limits_v2_mark_guidance_to_mark_tool() -> None:
    registry = ToolRegistry()
    register_mark_tool(registry)
    mark_tool = registry.spec("mark_regions")
    unrelated_tool = make_tool_spec()

    assert build_system_prompt(prompt_version="agent-v1", tools=(mark_tool,)) == (
        AGENT_V1_SYSTEM_PROMPT
    )
    assert build_system_prompt(prompt_version="agent-v2", tools=(unrelated_tool,)) == (
        AGENT_V1_SYSTEM_PROMPT
    )

    request = build_policy_request(
        make_policy_input(tools=(mark_tool,)),
        prompt_version="agent-v2",
    )
    prompt = request.messages[0].content

    assert prompt.startswith(AGENT_V1_SYSTEM_PROMPT + "\n")
    assert "references derived from source detection or segmentation" in prompt
    assert "untrusted visual perception" in prompt
    assert "cite both its original detection or segmentation" in prompt
    assert request.metadata["prompt_version"] == "agent-v2"


class _VisibilityPolicy:
    def __init__(self) -> None:
        self.asset_id: str | None = None
        self.inputs: list[PolicyInput] = []

    async def next_action(self, input: PolicyInput) -> PolicyAction:
        self.inputs.append(input)
        names = {tool.name for tool in input.tools}
        if len(self.inputs) == 1:
            assert "detect_objects" in names
            assert "mark_regions" not in names
            assert "describe_marks" not in names
            assert self.asset_id is not None
            return ToolAction.model_validate(
                {
                    "tool": "detect_objects",
                    "arguments": {"asset_id": self.asset_id},
                    "purpose": "Locate the visible squares",
                },
                strict=True,
            )
        assert "mark_regions" in names
        assert "describe_marks" not in names
        return AnswerAction(
            status="insufficient_evidence",
            answer="The requested distinction is not established.",
        )


@pytest.mark.asyncio
async def test_mark_tool_is_omitted_until_a_localized_source_is_visible() -> None:
    async def analyze(image: BackendImage, request: VisionRequest) -> VisionResult:
        if isinstance(request, CaptionRequest):
            return VisionResult(
                observations=(ObservationDraft(payload=CaptionPayload(text="Colored squares.")),)
            )
        assert isinstance(request, DetectionRequest)
        return VisionResult(
            observations=(
                ObservationDraft(
                    payload=DetectionPayload(label="square"),
                    region=Box(x_min=0.1, y_min=0.1, x_max=0.6, y_max=0.6),
                ),
            )
        )

    descriptor = _descriptor(
        "tests.dynamic_mark_visibility",
        (
            _capability(Capability.DETECT),
            _capability(Capability.CAPTION, "caption.mark_references"),
        ),
    )
    backend = CallableVisionBackend(descriptor, analyze)
    policy = _VisibilityPolicy()
    settings = Settings(agent=AgentSettings(initial_capabilities=(Capability.CAPTION,)))
    image = quadrants_image(16)
    try:
        encoded = encode_image(image)
    finally:
        image.close()

    async with AsyncPenampakan(
        backends=(backend,),
        policy=policy,
        settings=settings,
    ) as client:
        session = await client.open_image(encoded)
        policy.asset_id = session.root_asset.id
        try:
            await session.ask("Which square is distinct?")
        finally:
            await session.aclose()

    assert len(policy.inputs) == 2
