import json
from collections.abc import Iterator
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from penampakan.models import (
    AnswerAction,
    CaptionRequest,
    ColorsRequest,
    DetectionRequest,
    MetadataRequest,
    OCRRequest,
    PolicyAction,
    SegmentationRequest,
    ToolAction,
    VisionRequest,
)

VISION_REQUEST_ADAPTER = TypeAdapter(VisionRequest)
POLICY_ACTION_ADAPTER = TypeAdapter(PolicyAction)

VISION_REQUEST_CASES = (
    (MetadataRequest, {"capability": "metadata"}),
    (
        ColorsRequest,
        {
            "capability": "colors",
            "region": {"x_min": 0.0, "y_min": 0.0, "x_max": 0.5, "y_max": 1.0},
            "count": 16,
        },
    ),
    (
        CaptionRequest,
        {
            "capability": "caption",
            "region": None,
            "focus": "Read the display",
            "max_sentences": 8,
        },
    ),
    (
        OCRRequest,
        {
            "capability": "ocr",
            "region": None,
            "languages": ["en", "ms-MY"],
            "mode": "single_line",
            "min_confidence": 0.0,
        },
    ),
    (
        DetectionRequest,
        {
            "capability": "detect",
            "region": None,
            "labels": ["Display", "Button"],
            "min_confidence": 1.0,
            "max_results": 1,
        },
    ),
    (
        SegmentationRequest,
        {
            "capability": "segment",
            "region": None,
            "labels": ["Screen"],
            "points": [{"x": 0.25, "y": 0.75}],
            "min_confidence": 0.5,
            "max_results": 32,
        },
    ),
)

POLICY_ACTION_CASES = (
    (
        ToolAction,
        {
            "type": "tool",
            "tool": "inspect_image",
            "arguments": {
                "asset_id": "img_0123456789abcdef",
                "include_regions": True,
                "thresholds": [0.25, 0.5],
            },
            "purpose": "Inspect the display for readable text.",
        },
    ),
    (
        AnswerAction,
        {
            "type": "answer",
            "status": "answered",
            "answer": "The display reads 42.",
            "evidence": [
                {
                    "observation_id": "obs_000001",
                    "supports": "The localized text reads 42.",
                }
            ],
            "uncertainties": [],
        },
    ),
    (
        AnswerAction,
        {
            "type": "answer",
            "status": "insufficient_evidence",
            "answer": "The serial number is not visible.",
            "evidence": [],
            "uncertainties": ["The image does not show the rear label."],
        },
    ),
)


def encode_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"))


def object_schemas(value: Any, path: str = "$") -> Iterator[tuple[str, dict[str, Any]]]:
    if isinstance(value, dict):
        if "properties" in value:
            yield path, value
        for key, child in value.items():
            yield from object_schemas(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from object_schemas(child, f"{path}[{index}]")


@pytest.mark.parametrize(("expected_type", "payload"), VISION_REQUEST_CASES)
def test_every_vision_request_branch_validates_from_json(
    expected_type: type[object],
    payload: dict[str, object],
) -> None:
    request = VISION_REQUEST_ADAPTER.validate_json(encode_json(payload))

    assert type(request) is expected_type
    assert request.capability.value == payload["capability"]


@pytest.mark.parametrize(("expected_type", "payload"), POLICY_ACTION_CASES)
def test_every_policy_action_branch_validates_from_json(
    expected_type: type[object],
    payload: dict[str, object],
) -> None:
    action = POLICY_ACTION_ADAPTER.validate_json(encode_json(payload))

    assert type(action) is expected_type
    assert action.type == payload["type"]


@pytest.mark.parametrize(("expected_type", "payload"), VISION_REQUEST_CASES)
def test_vision_request_json_round_trip_preserves_branch(
    expected_type: type[object],
    payload: dict[str, object],
) -> None:
    request = VISION_REQUEST_ADAPTER.validate_json(encode_json(payload))
    encoded = VISION_REQUEST_ADAPTER.dump_json(request)
    reparsed = VISION_REQUEST_ADAPTER.validate_json(encoded)

    assert type(reparsed) is expected_type
    assert reparsed == request


@pytest.mark.parametrize(("expected_type", "payload"), POLICY_ACTION_CASES)
def test_policy_action_json_round_trip_preserves_branch(
    expected_type: type[object],
    payload: dict[str, object],
) -> None:
    action = POLICY_ACTION_ADAPTER.validate_json(encode_json(payload))
    encoded = POLICY_ACTION_ADAPTER.dump_json(action)
    reparsed = POLICY_ACTION_ADAPTER.validate_json(encoded)

    assert type(reparsed) is expected_type
    assert reparsed == action


@pytest.mark.parametrize(("expected_type", "payload"), VISION_REQUEST_CASES)
def test_every_vision_request_branch_rejects_unknown_fields(
    expected_type: type[object],
    payload: dict[str, object],
) -> None:
    invalid = {**payload, "unexpected": expected_type.__name__}

    with pytest.raises(ValidationError):
        VISION_REQUEST_ADAPTER.validate_json(encode_json(invalid))


@pytest.mark.parametrize(("expected_type", "payload"), POLICY_ACTION_CASES)
def test_every_policy_action_branch_rejects_unknown_fields(
    expected_type: type[object],
    payload: dict[str, object],
) -> None:
    invalid = {**payload, "unexpected": expected_type.__name__}

    with pytest.raises(ValidationError):
        POLICY_ACTION_ADAPTER.validate_json(encode_json(invalid))


@pytest.mark.parametrize(
    "payload",
    (
        {},
        {"capability": "unknown"},
        {"capability": "metadata", "count": 5},
        {"capability": "colors", "count": "5"},
        {"capability": "colors", "count": True},
        {"capability": "colors", "count": 0},
        {"capability": "colors", "count": 17},
        {"capability": "caption", "focus": ""},
        {"capability": "caption", "max_sentences": 9},
        {"capability": "ocr", "languages": ["en"] * 9},
        {"capability": "ocr", "languages": ["not_a_language"]},
        {"capability": "ocr", "min_confidence": -0.01},
        {"capability": "detect", "labels": ["label"] * 65},
        {"capability": "detect", "min_confidence": 1.01},
        {"capability": "detect", "max_results": 0},
        {"capability": "segment", "points": [{"x": 0.0, "y": 1.01}]},
        {"capability": "segment", "max_results": "32"},
    ),
)
def test_vision_request_parser_rejects_invalid_boundaries(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        VISION_REQUEST_ADAPTER.validate_json(encode_json(payload))


@pytest.mark.parametrize(
    "payload",
    (
        {},
        {"type": "unknown"},
        {"type": "tool", "tool": "inspect", "arguments": {}, "purpose": ""},
        {"type": "tool", "tool": "Invalid Tool", "arguments": {}, "purpose": "Inspect"},
        {
            "type": "answer",
            "status": "unknown",
            "answer": "Unavailable",
            "evidence": [],
            "uncertainties": [],
        },
        {
            "type": "answer",
            "status": "answered",
            "answer": "Available",
            "evidence": [{"observation_id": "obs_1", "supports": "Visible"}],
            "uncertainties": [],
        },
        {
            "type": "answer",
            "status": "answered",
            "answer": 42,
            "evidence": [],
            "uncertainties": [],
        },
    ),
)
def test_policy_action_parser_rejects_invalid_boundaries(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        POLICY_ACTION_ADAPTER.validate_json(encode_json(payload))


@pytest.mark.parametrize(
    "document",
    (
        '```json\n{"type":"answer"}\n```',
        '{"type":"answer"} trailing',
        '{"type":"answer"}{"type":"tool"}',
        '[{"type":"answer"}]',
        "null",
        '{"type":"tool","tool":"inspect","arguments":{"score":NaN},"purpose":"Inspect"}',
    ),
)
def test_policy_action_parser_rejects_non_strict_json_documents(document: str) -> None:
    with pytest.raises(ValidationError):
        POLICY_ACTION_ADAPTER.validate_json(document)


def test_policy_action_parser_accepts_surrounding_ascii_whitespace() -> None:
    payload = POLICY_ACTION_CASES[2][1]
    document = f"\t\n {encode_json(payload)} \r\n"

    action = POLICY_ACTION_ADAPTER.validate_json(document)

    assert isinstance(action, AnswerAction)
    assert action.status == "insufficient_evidence"


@pytest.mark.parametrize(
    "adapter",
    (
        TypeAdapter(MetadataRequest),
        TypeAdapter(ColorsRequest),
        TypeAdapter(CaptionRequest),
        TypeAdapter(OCRRequest),
        TypeAdapter(DetectionRequest),
        TypeAdapter(SegmentationRequest),
        VISION_REQUEST_ADAPTER,
        POLICY_ACTION_ADAPTER,
    ),
)
def test_generated_model_object_schemas_forbid_additional_properties(
    adapter: TypeAdapter[Any],
) -> None:
    schema = adapter.json_schema()
    closed_objects = tuple(object_schemas(schema))

    assert closed_objects
    for path, object_schema in closed_objects:
        assert object_schema.get("type") == "object", path
        assert object_schema.get("additionalProperties") is False, path
