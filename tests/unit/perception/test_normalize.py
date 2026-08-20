import pytest

from penampakan.errors import InvalidBackendOutputError
from penampakan.image.geometry import NORMALIZED_TOLERANCE
from penampakan.models import (
    Box,
    CaptionPayload,
    CaptionRequest,
    ColorsPayload,
    ColorsRequest,
    DetectionPayload,
    DetectionRequest,
    MarkDescriptionPayload,
    MarkDescriptionRef,
    MetadataPayload,
    MetadataRequest,
    ObservationDraft,
    OCRRequest,
    SegmentationPayload,
    SegmentationRequest,
    TextPayload,
    VisionRequest,
    VisionResult,
    WarningInfo,
)
from penampakan.perception.normalize import NormalizationLimits, normalize_backend_result


def result_with(*drafts: ObservationDraft) -> VisionResult:
    return VisionResult(observations=drafts)


def raw_result(*drafts: object) -> dict[str, object]:
    return {"observations": drafts, "warnings": ()}


@pytest.mark.parametrize(
    ("vision_request", "draft"),
    [
        (
            MetadataRequest(),
            ObservationDraft(
                payload=MetadataPayload(
                    width=20,
                    height=10,
                    aspect_ratio=2.0,
                    has_alpha=False,
                )
            ),
        ),
        (ColorsRequest(), ObservationDraft(payload=ColorsPayload(swatches=()))),
        (
            CaptionRequest(),
            ObservationDraft(payload=CaptionPayload(text="A small blue square.")),
        ),
        (
            OCRRequest(),
            ObservationDraft(
                payload=TextPayload(text="TOTAL"),
                region=Box(x_min=0.1, y_min=0.1, x_max=0.3, y_max=0.2),
            ),
        ),
        (
            DetectionRequest(),
            ObservationDraft(
                payload=DetectionPayload(label="car"),
                region=Box(x_min=0.1, y_min=0.1, x_max=0.3, y_max=0.3),
            ),
        ),
        (
            SegmentationRequest(),
            ObservationDraft(
                payload=SegmentationPayload(label="road"),
                region=Box(x_min=0.1, y_min=0.1, x_max=0.9, y_max=0.8),
            ),
        ),
    ],
)
def test_request_capability_accepts_only_its_payload(
    vision_request: VisionRequest,
    draft: ObservationDraft,
) -> None:
    normalized = normalize_backend_result(result_with(draft), vision_request)

    assert normalized.observations == (draft,)


def test_request_capability_mismatch_invalidates_complete_call() -> None:
    result = result_with(ObservationDraft(payload=CaptionPayload(text="caption")))

    with pytest.raises(InvalidBackendOutputError):
        normalize_backend_result(result, OCRRequest())


def test_mark_reference_request_requires_structured_requested_indices() -> None:
    request = CaptionRequest(mark_indices=(1, 2))
    structured = ObservationDraft(
        payload=MarkDescriptionPayload(
            references=(
                MarkDescriptionRef(index=1, description="left red car"),
                MarkDescriptionRef(index=2, description="right blue car"),
            )
        )
    )

    assert normalize_backend_result(result_with(structured), request).observations == (structured,)
    with pytest.raises(InvalidBackendOutputError):
        normalize_backend_result(
            result_with(ObservationDraft(payload=CaptionPayload(text="Mark 1 is red."))),
            request,
        )
    with pytest.raises(InvalidBackendOutputError):
        normalize_backend_result(
            result_with(
                ObservationDraft(
                    payload=MarkDescriptionPayload(
                        references=(MarkDescriptionRef(index=3, description="spurious"),)
                    )
                )
            ),
            request,
        )


@pytest.mark.parametrize(
    "limits",
    [
        NormalizationLimits(max_ocr_chars_per_observation=3),
        NormalizationLimits(max_total_text_chars=1),
        NormalizationLimits(max_total_items=1),
        NormalizationLimits(max_observations=1),
    ],
)
def test_configured_per_result_and_aggregate_limits_invalidate_call(
    limits: NormalizationLimits,
) -> None:
    first = ObservationDraft(payload=TextPayload(text="four"))
    second = ObservationDraft(payload=TextPayload(text="more"))

    with pytest.raises(InvalidBackendOutputError):
        normalize_backend_result(result_with(first, second), OCRRequest(), limits=limits)


def test_request_result_limit_is_enforced_before_deduplication() -> None:
    region = Box(x_min=0.1, y_min=0.1, x_max=0.2, y_max=0.2)
    draft = ObservationDraft(payload=DetectionPayload(label="car"), region=region)

    with pytest.raises(InvalidBackendOutputError):
        normalize_backend_result(
            result_with(draft, draft),
            DetectionRequest(max_results=1),
        )


def test_color_item_limit_matches_requested_count() -> None:
    result = raw_result(
        {
            "payload": {
                "type": "colors",
                "swatches": (
                    {"rgb": (0, 0, 0), "hex": "#000000", "fraction": 0.5},
                    {"rgb": (255, 255, 255), "hex": "#FFFFFF", "fraction": 0.5},
                ),
            }
        }
    )

    with pytest.raises(InvalidBackendOutputError):
        normalize_backend_result(result, ColorsRequest(count=1))


def test_newlines_and_controls_are_cleaned_and_reported() -> None:
    result = raw_result(
        {
            "payload": {"type": "text", "text": "  A\r\nB\rC\x00\x01\tD  "},
            "region": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.9, "y_max": 0.9},
        }
    )

    normalized = normalize_backend_result(result, OCRRequest())

    draft = normalized.observations[0]
    assert isinstance(draft.payload, TextPayload)
    assert draft.payload.text == "A\nB\nC\tD"
    warning = next(item for item in draft.warnings if item.code == "control_chars_removed")
    assert warning.details == {"count": 2}


def test_cleaned_empty_text_is_discarded_with_result_warning() -> None:
    result = raw_result({"payload": {"type": "text", "text": " \x00\r\t "}})

    normalized = normalize_backend_result(result, OCRRequest())

    assert normalized.observations == ()
    assert normalized.warnings[-1].code == "empty_text_discarded"
    assert normalized.warnings[-1].details == {"count": 1}


def test_empty_text_does_not_hide_other_schema_errors() -> None:
    result = raw_result(
        {
            "payload": {"type": "text", "text": "\x00"},
            "unknown": "forbidden",
        }
    )

    with pytest.raises(InvalidBackendOutputError):
        normalize_backend_result(result, OCRRequest())


def test_small_box_overshoot_is_clamped() -> None:
    result = raw_result(
        {
            "payload": {"type": "detection", "label": "edge"},
            "region": {
                "x_min": -NORMALIZED_TOLERANCE,
                "y_min": 0.25,
                "x_max": 1.0 + NORMALIZED_TOLERANCE,
                "y_max": 0.75,
            },
        }
    )

    normalized = normalize_backend_result(result, DetectionRequest())

    assert normalized.observations[0].region == Box(
        x_min=0.0,
        y_min=0.25,
        x_max=1.0,
        y_max=0.75,
    )


def test_box_overshoot_beyond_tolerance_invalidates_call() -> None:
    result = raw_result(
        {
            "payload": {"type": "detection", "label": "edge"},
            "region": {
                "x_min": -2.0 * NORMALIZED_TOLERANCE,
                "y_min": 0.25,
                "x_max": 1.0,
                "y_max": 0.75,
            },
        }
    )

    with pytest.raises(InvalidBackendOutputError):
        normalize_backend_result(result, DetectionRequest())


def test_request_region_tolerance_is_clipped_to_scope() -> None:
    scope = Box(x_min=0.2, y_min=0.2, x_max=0.8, y_max=0.8)
    result = raw_result(
        {
            "payload": {"type": "detection", "label": "inside"},
            "region": {
                "x_min": scope.x_min - NORMALIZED_TOLERANCE / 2.0,
                "y_min": 0.3,
                "x_max": 0.7,
                "y_max": scope.y_max + NORMALIZED_TOLERANCE / 2.0,
            },
        }
    )

    normalized = normalize_backend_result(result, DetectionRequest(region=scope))

    assert normalized.observations[0].region == Box(
        x_min=scope.x_min,
        y_min=0.3,
        x_max=0.7,
        y_max=scope.y_max,
    )


def test_result_outside_request_region_invalidates_complete_call() -> None:
    scope = Box(x_min=0.2, y_min=0.2, x_max=0.8, y_max=0.8)
    result = raw_result(
        {
            "payload": {"type": "detection", "label": "outside"},
            "region": {
                "x_min": scope.x_min - 2.0 * NORMALIZED_TOLERANCE,
                "y_min": 0.3,
                "x_max": 0.7,
                "y_max": 0.7,
            },
        }
    )

    with pytest.raises(InvalidBackendOutputError):
        normalize_backend_result(result, DetectionRequest(region=scope))


def test_regional_caption_without_region_invalidates_call() -> None:
    scope = Box(x_min=0.2, y_min=0.2, x_max=0.8, y_max=0.8)
    result = result_with(ObservationDraft(payload=CaptionPayload(text="detail")))

    with pytest.raises(InvalidBackendOutputError):
        normalize_backend_result(result, CaptionRequest(region=scope))


def test_ocr_sort_is_spatial_and_stable_for_equal_regions() -> None:
    left = Box(x_min=0.1, y_min=0.1, x_max=0.2, y_max=0.2)
    right = Box(x_min=0.7, y_min=0.1, x_max=0.8, y_max=0.2)
    lower = Box(x_min=0.1, y_min=0.7, x_max=0.2, y_max=0.8)
    drafts = (
        ObservationDraft(payload=TextPayload(text="lower"), region=lower),
        ObservationDraft(payload=TextPayload(text="right"), region=right),
        ObservationDraft(payload=TextPayload(text="first"), region=left),
        ObservationDraft(payload=TextPayload(text="second"), region=left),
        ObservationDraft(payload=TextPayload(text="unlocated")),
    )

    normalized = normalize_backend_result(result_with(*drafts), OCRRequest())

    assert tuple(draft.payload.text for draft in normalized.observations) == (
        "first",
        "second",
        "right",
        "lower",
        "unlocated",
    )


def test_exact_duplicates_are_removed_without_merging_distinct_drafts() -> None:
    region = Box(x_min=0.1, y_min=0.1, x_max=0.2, y_max=0.2)
    duplicate = ObservationDraft(
        payload=DetectionPayload(label="car"),
        region=region,
        confidence=0.9,
    )
    distinct = ObservationDraft(
        payload=DetectionPayload(label="car"),
        region=region,
        confidence=0.8,
    )

    normalized = normalize_backend_result(
        result_with(duplicate, duplicate, distinct),
        DetectionRequest(),
    )

    assert normalized.observations == (duplicate, distinct)
    warning = next(
        item for item in normalized.warnings if item.code == "duplicate_observations_removed"
    )
    assert warning.details == {"count": 1}


def test_missing_localization_gets_one_location_warning() -> None:
    existing = WarningInfo(
        code="location_unavailable",
        message="No bounding box was returned.",
    )
    draft = ObservationDraft(
        payload=DetectionPayload(label="object"),
        warnings=(existing,),
    )

    normalized = normalize_backend_result(result_with(draft), DetectionRequest())

    assert (
        tuple(item.code for item in normalized.observations[0].warnings).count(
            "location_unavailable"
        )
        == 1
    )


def test_one_invalid_draft_invalidates_the_complete_response() -> None:
    valid = {
        "payload": {"type": "text", "text": "valid"},
        "region": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.2, "y_max": 0.2},
    }
    invalid = {
        "payload": {"type": "text", "text": "invalid", "extra": True},
        "region": {"x_min": 0.3, "y_min": 0.3, "x_max": 0.4, "y_max": 0.4},
    }

    with pytest.raises(InvalidBackendOutputError):
        normalize_backend_result(raw_result(valid, invalid), OCRRequest())
