from datetime import datetime, timezone
from uuid import UUID

import pytest
from pydantic import ValidationError

from penampakan.models import (
    JSON_VALUE_ADAPTER,
    BackendDescriptor,
    BackendImage,
    Box,
    Capability,
    CapabilityDescriptor,
    CaptionRequest,
    DetectionRequest,
    ImageAsset,
    InspectionPlan,
    MarkPayload,
    MarkRef,
    Observation,
    ObservationDraft,
    Point,
    Provenance,
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


def mark_ref(index: int = 1, observation_id: str = "obs_000001") -> MarkRef:
    return MarkRef(
        index=index,
        observation_id=observation_id,
        region=Box(x_min=0.1, y_min=0.2, x_max=0.3, y_max=0.4),
        source_label=" car ",
    )


def mark_payload(*marks: MarkRef) -> MarkPayload:
    return MarkPayload(
        derived_asset_id="img_fedcba9876543210",
        parent_asset_id="img_0123456789abcdef",
        marks=marks or (mark_ref(),),
    )


def test_mark_ref_enforces_index_id_region_and_label_contracts() -> None:
    ref = mark_ref()

    assert ref.index == 1
    assert ref.observation_id == "obs_000001"
    assert ref.source_label == "car"

    for index in (0, 100):
        with pytest.raises(ValidationError):
            mark_ref(index=index)
    with pytest.raises(ValidationError):
        MarkRef(
            index=1,
            observation_id="not-an-observation",
            region=ref.region,
        )
    with pytest.raises(ValidationError):
        MarkRef(
            index=1,
            observation_id="obs_000001",
            region={"x_min": 0.4, "y_min": 0.2, "x_max": 0.3, "y_max": 0.5},
        )
    with pytest.raises(ValidationError):
        MarkRef(
            index=1,
            observation_id="obs_000001",
            region=ref.region,
            source_label="unsafe\x00label",
        )


@pytest.mark.parametrize(
    "marks",
    [
        (),
        (mark_ref(), mark_ref(index=1, observation_id="obs_000002")),
        (mark_ref(), mark_ref(index=2)),
        (mark_ref(), mark_ref(index=3, observation_id="obs_000002")),
        (mark_ref(index=2), mark_ref(index=1, observation_id="obs_000002")),
    ],
)
def test_mark_payload_requires_unique_contiguous_indices_and_observation_ids(
    marks: tuple[MarkRef, ...],
) -> None:
    values = {
        "derived_asset_id": "img_fedcba9876543210",
        "parent_asset_id": "img_0123456789abcdef",
        "marks": marks,
    }

    with pytest.raises(ValidationError):
        MarkPayload(**values)


def test_mark_payload_has_no_region_and_decodes_as_observation_payload() -> None:
    payload = mark_payload()
    draft = ObservationDraft.model_validate({"payload": payload.model_dump()})

    assert draft.payload == payload
    assert draft.region is None
    with pytest.raises(ValidationError):
        MarkPayload.model_validate({**payload.model_dump(), "region": mark_ref().region})


def test_mark_observations_span_the_asset_while_refs_carry_regions() -> None:
    payload = mark_payload()
    provenance = Provenance(
        tool="mark_regions",
        capability=None,
        backend_name="penampakan.core",
        backend_version="1.0",
        request_hash="a" * 64,
        parent_observation_ids=("obs_000001",),
        duration_ms=1,
    )
    observation_values = {
        "id": "obs_000002",
        "asset_id": payload.derived_asset_id,
        "payload": payload,
        "provenance": provenance,
    }

    observation = Observation(**observation_values)

    assert observation.region is None
    assert observation.payload.marks[0].region == mark_ref().region
    with pytest.raises(ValidationError, match="cannot have a region"):
        Observation(**observation_values, region=mark_ref().region)
    with pytest.raises(ValidationError, match="cannot have a region"):
        ObservationDraft(payload=payload, region=mark_ref().region)


def _mark_capable_descriptor(
    *, model_id: str | None, model_revision: str | None, feature: bool = True
) -> BackendDescriptor:
    features = ("caption.mark_references",) if feature else ()
    return BackendDescriptor(
        name="tests.backend",
        version="1.0",
        model_id=model_id,
        model_revision=model_revision,
        capabilities=(
            CapabilityDescriptor(capability=Capability.CAPTION, features=frozenset(features)),
        ),
    )


@pytest.mark.parametrize(
    ("model_id", "model_revision", "feature", "expected"),
    [
        ("tests/model", "a" * 40, True, True),
        ("tests/model", None, True, False),
        (None, None, True, False),
        ("tests/model", "a" * 40, False, False),
    ],
    ids=("pinned-feature", "unpinned", "unidentified", "pinned-no-feature"),
)
def test_advertises_proven_mark_references_requires_pinned_revision_and_feature(
    model_id: str | None,
    model_revision: str | None,
    feature: bool,
    expected: bool,
) -> None:
    descriptor = _mark_capable_descriptor(
        model_id=model_id, model_revision=model_revision, feature=feature
    )

    assert descriptor.advertises_proven_mark_references is expected
