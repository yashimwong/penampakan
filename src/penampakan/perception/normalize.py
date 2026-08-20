"""Strict normalization of untrusted vision backend results."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from pydantic import BaseModel

from penampakan.errors import InvalidBackendOutputError
from penampakan.image.geometry import NORMALIZED_TOLERANCE, box_contains, clamp_normalized_box
from penampakan.models import (
    Box,
    CaptionPayload,
    CaptionRequest,
    ColorsPayload,
    ColorsRequest,
    DetectionPayload,
    DetectionRequest,
    MarkDescriptionPayload,
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


@dataclass(frozen=True, slots=True)
class NormalizationLimits:
    """Aggregate safety limits applied to one backend response."""

    max_ocr_chars_per_observation: int = 8_000
    max_total_text_chars: int = 128_000
    max_total_items: int = 65_536
    max_observations: int = 4_096

    def __post_init__(self) -> None:
        values = (
            self.max_ocr_chars_per_observation,
            self.max_total_text_chars,
            self.max_total_items,
            self.max_observations,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("normalization limits must be integers")
        if any(value <= 0 for value in values):
            raise ValueError("normalization limits must be positive")


@dataclass(frozen=True, slots=True)
class _TextChange:
    removed_controls: int = 0
    discard: bool = False


@dataclass(frozen=True, slots=True)
class _PreparedResult:
    data: object
    text_changes: dict[int, _TextChange]


def _clean_text(value: str) -> tuple[str, int]:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    characters: list[str] = []
    removed = 0
    for character in normalized:
        if ord(character) < 32 and character not in {"\t", "\n"}:
            removed += 1
        else:
            characters.append(character)
    return "".join(characters).strip(), removed


def _model_mapping(value: object) -> dict[str, object] | None:
    if isinstance(value, BaseModel):
        return cast(dict[str, object], value.model_dump(mode="python"))
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return None


def _prepare_region(value: object) -> object:
    if value is None:
        return None
    mapping = _model_mapping(value)
    if mapping is None:
        return value
    names = ("x_min", "y_min", "x_max", "y_max")
    if not all(name in mapping for name in names):
        return mapping
    box = clamp_normalized_box(
        cast(float, mapping["x_min"]),
        cast(float, mapping["y_min"]),
        cast(float, mapping["x_max"]),
        cast(float, mapping["y_max"]),
        tolerance=NORMALIZED_TOLERANCE,
    )
    mapping.update(box.model_dump(mode="python"))
    return mapping


def _prepare_payload(value: object) -> tuple[object, _TextChange]:
    mapping = _model_mapping(value)
    if mapping is None:
        return value, _TextChange()
    payload_type = mapping.get("type")
    removed_controls = 0
    discard = False
    if payload_type in {"caption", "text"}:
        text = mapping.get("text")
        if isinstance(text, str):
            cleaned, removed = _clean_text(text)
            removed_controls += removed
            if cleaned:
                mapping["text"] = cleaned
            else:
                mapping["text"] = "discarded"
                discard = True
    if payload_type == "caption":
        focus = mapping.get("focus")
        if isinstance(focus, str):
            cleaned_focus, removed = _clean_text(focus)
            removed_controls += removed
            mapping["focus"] = cleaned_focus or None
    return mapping, _TextChange(removed_controls=removed_controls, discard=discard)


def _prepare_draft(value: object) -> tuple[object, _TextChange]:
    mapping = _model_mapping(value)
    if mapping is None:
        return value, _TextChange()
    payload, change = _prepare_payload(mapping.get("payload"))
    mapping["payload"] = payload
    if "region" in mapping:
        mapping["region"] = _prepare_region(mapping["region"])
    warnings = mapping.get("warnings")
    if isinstance(warnings, tuple):
        mapping["warnings"] = tuple(_model_mapping(warning) or warning for warning in warnings)
    return mapping, change


def _prepare_result(value: VisionResult | Mapping[str, object]) -> _PreparedResult:
    mapping = _model_mapping(value)
    if mapping is None:
        return _PreparedResult(data=value, text_changes={})
    observations = mapping.get("observations")
    changes: dict[int, _TextChange] = {}
    if isinstance(observations, tuple):
        prepared_observations: list[object] = []
        for index, observation in enumerate(observations):
            prepared, change = _prepare_draft(observation)
            prepared_observations.append(prepared)
            if change.removed_controls or change.discard:
                changes[index] = change
        mapping["observations"] = tuple(prepared_observations)
    warnings = mapping.get("warnings")
    if isinstance(warnings, tuple):
        mapping["warnings"] = tuple(_model_mapping(warning) or warning for warning in warnings)
    return _PreparedResult(data=mapping, text_changes=changes)


def _control_warning(count: int) -> WarningInfo:
    return WarningInfo(
        code="control_chars_removed",
        message="Disallowed control characters were removed from backend text.",
        details={"count": count},
    )


def _location_warning() -> WarningInfo:
    return WarningInfo(
        code="location_unavailable",
        message="The backend did not provide a location for this observation.",
    )


def _empty_text_warning(count: int) -> WarningInfo:
    return WarningInfo(
        code="empty_text_discarded",
        message="Empty backend text observations were discarded.",
        details={"count": count},
    )


def _duplicate_warning(count: int) -> WarningInfo:
    return WarningInfo(
        code="duplicate_observations_removed",
        message="Exact duplicate backend observations were removed.",
        details={"count": count},
    )


def _append_warning(
    warnings: tuple[WarningInfo, ...],
    warning: WarningInfo,
) -> tuple[WarningInfo, ...]:
    if any(existing.code == warning.code for existing in warnings):
        return warnings
    return (*warnings, warning)


def _compatible_payload(request: VisionRequest, draft: ObservationDraft) -> bool:
    payload = draft.payload
    if isinstance(request, MetadataRequest):
        return isinstance(payload, MetadataPayload)
    if isinstance(request, ColorsRequest):
        return isinstance(payload, ColorsPayload)
    if isinstance(request, CaptionRequest):
        if request.mark_indices:
            return isinstance(payload, MarkDescriptionPayload)
        return isinstance(payload, CaptionPayload)
    if isinstance(request, OCRRequest):
        return isinstance(payload, TextPayload)
    if isinstance(request, DetectionRequest):
        return isinstance(payload, DetectionPayload)
    if isinstance(request, SegmentationRequest):
        return isinstance(payload, SegmentationPayload)
    return False


def _request_result_limit(request: VisionRequest, limits: NormalizationLimits) -> int:
    if isinstance(request, (MetadataRequest, ColorsRequest, CaptionRequest)):
        return 1
    if isinstance(request, (DetectionRequest, SegmentationRequest)):
        return request.max_results
    return limits.max_observations


def _string_size(value: object) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, Mapping):
        return sum(len(str(key)) + _string_size(item) for key, item in value.items())
    if isinstance(value, (list, tuple, set, frozenset)):
        return sum(_string_size(item) for item in value)
    return 0


def _nested_items(value: object) -> int:
    if isinstance(value, Mapping):
        return len(value) + sum(_nested_items(item) for item in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return len(value) + sum(_nested_items(item) for item in value)
    return 0


def _enforce_limits(
    result: VisionResult,
    request: VisionRequest,
    limits: NormalizationLimits,
    changes: dict[int, _TextChange],
) -> None:
    observation_count = len(result.observations)
    if observation_count > limits.max_observations:
        raise ValueError("backend response exceeds the observation limit")
    if observation_count > _request_result_limit(request, limits):
        raise ValueError("backend response exceeds the request result limit")
    if isinstance(request, ColorsRequest):
        for draft in result.observations:
            if (
                isinstance(draft.payload, ColorsPayload)
                and len(draft.payload.swatches) > request.count
            ):
                raise ValueError("backend response exceeds the requested color count")
    if isinstance(request, CaptionRequest) and request.mark_indices:
        requested = frozenset(request.mark_indices)
        for draft in result.observations:
            if isinstance(draft.payload, MarkDescriptionPayload) and any(
                reference.index not in requested for reference in draft.payload.references
            ):
                raise ValueError("backend returned an unrequested mark index")
    for index, draft in enumerate(result.observations):
        if isinstance(draft.payload, TextPayload):
            change = changes.get(index, _TextChange())
            text_length = 0 if change.discard else len(draft.payload.text)
            if text_length > limits.max_ocr_chars_per_observation:
                raise ValueError("OCR observation exceeds the character limit")
    dumped = result.model_dump(mode="json", exclude_none=True)
    if _string_size(dumped) > limits.max_total_text_chars:
        raise ValueError("backend response exceeds the aggregate character limit")
    if _nested_items(dumped) > limits.max_total_items:
        raise ValueError("backend response exceeds the aggregate item limit")


def _scope_region(region: Box | None, request: VisionRequest, payload: object) -> Box | None:
    requested_region = _requested_region(request)
    if requested_region is None:
        return region
    if region is None:
        if isinstance(payload, (CaptionPayload, ColorsPayload)):
            raise ValueError("regional caption and color results must identify their region")
        return None
    if not box_contains(requested_region, region, tolerance=NORMALIZED_TOLERANCE):
        raise ValueError("backend result lies outside the requested region")
    return Box(
        x_min=max(requested_region.x_min, region.x_min),
        y_min=max(requested_region.y_min, region.y_min),
        x_max=min(requested_region.x_max, region.x_max),
        y_max=min(requested_region.y_max, region.y_max),
    )


def _requested_region(request: VisionRequest) -> Box | None:
    if isinstance(request, MetadataRequest):
        return None
    return request.region


def _normalized_draft(
    draft: ObservationDraft,
    request: VisionRequest,
    change: _TextChange,
) -> ObservationDraft:
    warnings = draft.warnings
    if change.removed_controls:
        warnings = _append_warning(warnings, _control_warning(change.removed_controls))
    region = _scope_region(draft.region, request, draft.payload)
    if (
        isinstance(draft.payload, (TextPayload, DetectionPayload, SegmentationPayload))
        and region is None
    ):
        warnings = _append_warning(warnings, _location_warning())
    return ObservationDraft(
        payload=draft.payload,
        region=region,
        confidence=draft.confidence,
        warnings=warnings,
    )


def _region_key(region: Box | None, *, none_first: bool) -> tuple[int, float, float, float, float]:
    if region is None:
        return (0 if none_first else 1, 0.0, 0.0, 0.0, 0.0)
    return (1 if none_first else 0, region.y_min, region.x_min, region.y_max, region.x_max)


def _sort_key(draft: ObservationDraft) -> tuple[int, float, float, float, float]:
    if isinstance(draft.payload, (CaptionPayload, ColorsPayload, MetadataPayload)):
        return _region_key(draft.region, none_first=True)
    return _region_key(draft.region, none_first=False)


def _deduplicate(
    drafts: tuple[ObservationDraft, ...],
) -> tuple[tuple[ObservationDraft, ...], int]:
    unique: list[ObservationDraft] = []
    keys: set[str] = set()
    duplicate_count = 0
    for draft in drafts:
        key = json.dumps(
            draft.model_dump(mode="json", exclude_none=True),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if key in keys:
            duplicate_count += 1
        else:
            keys.add(key)
            unique.append(draft)
    return tuple(unique), duplicate_count


def normalize_backend_result(
    result: VisionResult | Mapping[str, object],
    request: VisionRequest,
    *,
    limits: NormalizationLimits | None = None,
) -> VisionResult:
    """Validate and normalize one complete untrusted backend response."""
    active_limits = limits or NormalizationLimits()
    try:
        prepared = _prepare_result(result)
        validated = VisionResult.model_validate(prepared.data, strict=True)
        if any(not _compatible_payload(request, draft) for draft in validated.observations):
            raise ValueError("backend payload is incompatible with the request capability")
        _enforce_limits(validated, request, active_limits, prepared.text_changes)
        normalized: list[ObservationDraft] = []
        empty_count = 0
        for index, draft in enumerate(validated.observations):
            change = prepared.text_changes.get(index, _TextChange())
            if change.discard:
                empty_count += 1
                continue
            normalized.append(_normalized_draft(draft, request, change))
        ordered = tuple(sorted(normalized, key=_sort_key))
        unique, duplicate_count = _deduplicate(ordered)
        warnings = validated.warnings
        if empty_count:
            warnings = (*warnings, _empty_text_warning(empty_count))
        if duplicate_count:
            warnings = (*warnings, _duplicate_warning(duplicate_count))
        return VisionResult(observations=unique, warnings=warnings)
    except InvalidBackendOutputError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise InvalidBackendOutputError(cause=error) from error


def normalize_vision_result(
    result: VisionResult | Mapping[str, object],
    request: VisionRequest,
    *,
    limits: NormalizationLimits | None = None,
) -> VisionResult:
    """Validate and normalize one complete untrusted backend response."""
    return normalize_backend_result(result, request, limits=limits)


__all__ = [
    "NormalizationLimits",
    "normalize_backend_result",
    "normalize_vision_result",
]
