from datetime import datetime, timezone
from uuid import UUID

import pytest
from pydantic import ValidationError

from penampakan.models import (
    JSON_VALUE_ADAPTER,
    BackendImage,
    Box,
    CaptionRequest,
    DetectionRequest,
    ImageAsset,
    InspectionPlan,
    Point,
    TraceSummary,
    TransformDescriptor,
)


def root_asset() -> ImageAsset:
    return ImageAsset(
        id="img_0123456789abcdef",
        width=640,
        height=480,
        mode="RGB",
        mime_type="image/png",
        original_format="PNG",
        digest_sha256="0" * 64,
        parent_id=None,
        derivation_depth=0,
        transform=None,
    )


def test_json_value_accepts_strict_recursive_values() -> None:
    value = {"items": [None, True, 3, 4.5, "text"], "nested": {"ok": False}}

    assert JSON_VALUE_ADAPTER.validate_python(value) == value


@pytest.mark.parametrize("value", [float("nan"), float("inf"), b"secret", {1: "value"}])
def test_json_value_rejects_unsafe_values(value: object) -> None:
    with pytest.raises(ValidationError):
        JSON_VALUE_ADAPTER.validate_python(value)


def test_box_geometry_handles_overlap_and_containment() -> None:
    outer = Box(x_min=0.0, y_min=0.0, x_max=0.75, y_max=0.75)
    inner = Box(x_min=0.25, y_min=0.25, x_max=0.5, y_max=0.5)

    assert outer.area == pytest.approx(0.5625)
    assert outer.intersection(inner) == inner
    assert outer.iou(inner) == pytest.approx(inner.area / outer.area)
    assert outer.contains(inner)
    assert outer.contains(Point(x=0.75, y=0.75))


@pytest.mark.parametrize(
    "values",
    [
        (0.0, 0.0, 0.0, 1.0),
        (0.0, 1.0, 1.0, 1.0),
        (-0.01, 0.0, 1.0, 1.0),
        (0.0, 0.0, float("nan"), 1.0),
    ],
)
def test_box_rejects_invalid_extents(values: tuple[float, float, float, float]) -> None:
    with pytest.raises(ValidationError):
        Box(x_min=values[0], y_min=values[1], x_max=values[2], y_max=values[3])


def test_asset_lineage_distinguishes_root_and_derived_assets() -> None:
    parent = root_asset()
    transform = TransformDescriptor(name="crop", parameters={"padding_fraction": 0.0})
    derived = ImageAsset(
        id="img_fedcba9876543210",
        width=320,
        height=240,
        mode="RGB",
        mime_type="image/png",
        original_format=None,
        digest_sha256="f" * 64,
        parent_id=parent.id,
        derivation_depth=1,
        transform=transform,
    )

    assert derived.parent_id == parent.id
    assert derived.transform == transform


def test_asset_rejects_partial_lineage() -> None:
    data = root_asset().model_dump()
    data["parent_id"] = "img_fedcba9876543210"

    with pytest.raises(ValidationError):
        ImageAsset.model_validate(data)


def test_request_text_is_clean_and_labels_are_deduplicated() -> None:
    caption = CaptionRequest(focus="  total amount  ")
    detection = DetectionRequest(labels=("Car", "car", " Bicycle "))

    assert caption.focus == "total amount"
    assert detection.labels == ("Car", "Bicycle")


def test_models_reject_unknown_fields_and_assignment() -> None:
    with pytest.raises(ValidationError):
        CaptionRequest.model_validate({"focus": "subject", "unknown": True})

    asset = root_asset()
    with pytest.raises(ValidationError):
        asset.width = 10


def test_backend_image_hides_canonical_bytes() -> None:
    protected = BackendImage(asset=root_asset(), content=b"private-image-bytes")

    assert "private-image-bytes" not in repr(protected)
    assert "content" not in protected.model_dump()


def test_empty_explicit_inspection_plan_is_rejected() -> None:
    with pytest.raises(ValidationError):
        InspectionPlan(include_available_overview=False)


def test_trace_summary_requires_utc_time() -> None:
    values = {
        "trace_id": UUID("12345678-1234-5678-1234-567812345678"),
        "started_at": datetime(2026, 2, 10, tzinfo=timezone.utc),
        "duration_ms": 10,
        "llm_calls": 0,
        "tool_calls": 1,
        "backend_calls": 1,
        "cache_hits": 0,
        "derived_assets": 0,
        "stop_reason": "completed",
    }

    assert TraceSummary(**values).started_at.tzinfo == timezone.utc
    values["started_at"] = datetime(2026, 2, 10)
    with pytest.raises(ValidationError):
        TraceSummary(**values)
