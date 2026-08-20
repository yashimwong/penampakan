"""Immutable public data contracts for Penampakan."""

from __future__ import annotations

import math
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from enum import Enum
from itertools import pairwise
from typing import TYPE_CHECKING, Annotated, BinaryIO, Literal, TypeAlias
from uuid import UUID

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    field_serializer,
    field_validator,
    model_validator,
)

if sys.version_info >= (3, 12):
    from typing import TypeAliasType
else:
    from typing_extensions import TypeAliasType

if TYPE_CHECKING:
    from PIL.Image import Image as _PillowImage
else:

    class _PillowImage:
        """Runtime placeholder that keeps Pillow out of the domain import graph."""


ImageSource: TypeAlias = (
    str | os.PathLike[str] | bytes | bytearray | memoryview | BinaryIO | _PillowImage
)


def _reject_nul(value: str) -> str:
    if "\x00" in value:
        raise ValueError("text must not contain NUL")
    return value


def _finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("value must be finite")
    return value


def _json_value(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON numbers must be finite")
    if isinstance(value, str):
        _reject_nul(value)
    if isinstance(value, dict):
        for key in value:
            _reject_nul(key)
    return value


def _json_mapping(value: dict[str, JsonValue]) -> dict[str, JsonValue]:
    for key in value:
        _reject_nul(key)
    return value


def _sorted_json(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return {key: _sorted_json(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sorted_json(item) for item in value]
    return value


if TYPE_CHECKING:
    JsonValue: TypeAlias = (
        bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"] | None
    )
else:
    JsonValue = TypeAliasType(
        "JsonValue",
        Annotated[
            bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"] | None,
            AfterValidator(_json_value),
        ],
    )

JSON_VALUE_ADAPTER: TypeAdapter[JsonValue] = TypeAdapter(
    JsonValue,
    config=ConfigDict(strict=True),
)


_CleanText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
    AfterValidator(_reject_nul),
]
_RawText = Annotated[str, AfterValidator(_reject_nul)]
_AssetId = Annotated[str, StringConstraints(pattern=r"^img_[0-9a-f]{16,64}$")]
_ObservationId = Annotated[str, StringConstraints(pattern=r"^obs_[0-9]{6,}$")]
_Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_BackendName = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z][a-z0-9_.-]{0,63}$"),
]
_SnakeName = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$")]
_FeatureName = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$"),
]
_LanguageTag = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$",
    ),
]
_Label = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=100),
    AfterValidator(_reject_nul),
]
_Confidence = Annotated[float, Field(ge=0.0, le=1.0), AfterValidator(_finite)]
_PositiveFinite = Annotated[float, Field(gt=0.0), AfterValidator(_finite)]
_NonNegativeFinite = Annotated[float, Field(ge=0.0), AfterValidator(_finite)]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )


class Capability(str, Enum):
    """A semantic image-analysis capability."""

    METADATA = "metadata"
    COLORS = "colors"
    CAPTION = "caption"
    OCR = "ocr"
    DETECT = "detect"
    SEGMENT = "segment"


class Point(_FrozenModel):
    """A normalized point on a post-orientation image asset."""

    x: Annotated[float, Field(ge=0.0, le=1.0), AfterValidator(_finite)]
    y: Annotated[float, Field(ge=0.0, le=1.0), AfterValidator(_finite)]


class Box(_FrozenModel):
    """A non-empty normalized rectangle on an image asset."""

    x_min: Annotated[float, Field(ge=0.0, le=1.0), AfterValidator(_finite)]
    y_min: Annotated[float, Field(ge=0.0, le=1.0), AfterValidator(_finite)]
    x_max: Annotated[float, Field(ge=0.0, le=1.0), AfterValidator(_finite)]
    y_max: Annotated[float, Field(ge=0.0, le=1.0), AfterValidator(_finite)]

    @model_validator(mode="after")
    def _validate_extents(self) -> Box:
        if self.x_min >= self.x_max:
            raise ValueError("x_min must be less than x_max")
        if self.y_min >= self.y_max:
            raise ValueError("y_min must be less than y_max")
        return self

    @property
    def area(self) -> float:
        """Return the normalized area of the box."""

        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def intersection(self, other: Box) -> Box | None:
        """Return the shared rectangle, or ``None`` when the boxes do not overlap."""

        x_min = max(self.x_min, other.x_min)
        y_min = max(self.y_min, other.y_min)
        x_max = min(self.x_max, other.x_max)
        y_max = min(self.y_max, other.y_max)
        if x_min >= x_max or y_min >= y_max:
            return None
        return Box(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)

    def iou(self, other: Box) -> float:
        """Return intersection over union with another box."""

        overlap = self.intersection(other)
        if overlap is None:
            return 0.0
        return overlap.area / (self.area + other.area - overlap.area)

    def contains(self, other: Box | Point) -> bool:
        """Return whether this box fully contains a box or point."""

        if isinstance(other, Point):
            return self.x_min <= other.x <= self.x_max and self.y_min <= other.y <= self.y_max
        return (
            self.x_min <= other.x_min
            and self.y_min <= other.y_min
            and self.x_max >= other.x_max
            and self.y_max >= other.y_max
        )


class TransformDescriptor(_FrozenModel):
    """A deterministic description of an applied image transform."""

    name: Literal[
        "crop",
        "tile",
        "rotate",
        "enhance_contrast",
        "grayscale",
        "coordinate_grid",
        "set_of_mark",
    ]
    parameters: dict[str, JsonValue]

    @field_validator("parameters")
    @classmethod
    def _validate_parameters(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return _json_mapping(value)

    @field_serializer("parameters")
    def _serialize_parameters(self, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return {key: _sorted_json(value[key]) for key in sorted(value)}


class ImageAsset(_FrozenModel):
    """Immutable metadata for a normalized root or derived image asset."""

    id: _AssetId
    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]
    mode: Literal["RGB", "RGBA"]
    mime_type: Literal["image/png"]
    original_format: Literal["PNG", "JPEG", "WEBP"] | None
    digest_sha256: _Sha256
    parent_id: _AssetId | None
    derivation_depth: Annotated[int, Field(ge=0)]
    transform: TransformDescriptor | None

    @model_validator(mode="after")
    def _validate_lineage(self) -> ImageAsset:
        is_root = self.parent_id is None and self.transform is None
        is_derived = self.parent_id is not None and self.transform is not None
        if not is_root and not is_derived:
            raise ValueError("parent_id and transform must either both be set or both be absent")
        if is_root and self.derivation_depth != 0:
            raise ValueError("root assets must have derivation_depth zero")
        if is_derived and self.derivation_depth == 0:
            raise ValueError("derived assets must have positive derivation_depth")
        if is_derived and self.original_format is not None:
            raise ValueError("derived assets must not have an original_format")
        return self


class CapabilityDescriptor(_FrozenModel):
    """A backend capability and its supported option features."""

    capability: Capability
    features: frozenset[_FeatureName] = Field(default_factory=frozenset)

    @field_serializer("features")
    def _serialize_features(self, value: frozenset[str]) -> tuple[str, ...]:
        return tuple(sorted(value))


class BackendDescriptor(_FrozenModel):
    """Stable identity, capability, and concurrency metadata for a backend."""

    name: _BackendName
    version: _CleanText
    model_id: _CleanText | None = None
    model_revision: _CleanText | None = None
    capabilities: tuple[CapabilityDescriptor, ...]
    is_remote: bool = False
    max_concurrency: Annotated[int, Field(gt=0)] = 1

    @property
    def durable_cache_eligible(self) -> bool:
        """Return whether this identity is exact enough for durable cross-process caching."""
        return self.model_id is None or self.model_revision is not None

    @field_validator("capabilities")
    @classmethod
    def _unique_capabilities(
        cls,
        value: tuple[CapabilityDescriptor, ...],
    ) -> tuple[CapabilityDescriptor, ...]:
        capabilities = tuple(item.capability for item in value)
        if len(capabilities) != len(set(capabilities)):
            raise ValueError("backend capabilities must be unique")
        return value


class Provenance(_FrozenModel):
    """Attribution for the operation that produced an observation."""

    tool: _SnakeName
    capability: Capability | None
    backend_name: _BackendName
    backend_version: _CleanText
    model_id: _CleanText | None = None
    model_revision: _CleanText | None = None
    request_hash: _Sha256
    parent_observation_ids: tuple[_ObservationId, ...] = Field(default_factory=tuple)
    cache_hit: bool = False
    duration_ms: Annotated[int, Field(ge=0)]


class MetadataRequest(_FrozenModel):
    """Request normalized image metadata."""

    capability: Literal[Capability.METADATA] = Capability.METADATA


class ColorsRequest(_FrozenModel):
    """Request dominant colors from an image or region."""

    capability: Literal[Capability.COLORS] = Capability.COLORS
    region: Box | None = None
    count: Annotated[int, Field(ge=1, le=16)] = 5


class CaptionRequest(_FrozenModel):
    """Request a bounded natural-language image description."""

    capability: Literal[Capability.CAPTION] = Capability.CAPTION
    region: Box | None = None
    focus: (
        Annotated[
            str,
            StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
            AfterValidator(_reject_nul),
        ]
        | None
    ) = None
    max_sentences: Annotated[int, Field(ge=1, le=8)] = 3
    mark_indices: Annotated[tuple[int, ...], Field(max_length=99)] = Field(default_factory=tuple)

    @field_validator("mark_indices")
    @classmethod
    def _validate_mark_indices(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(isinstance(index, bool) or not 1 <= index <= 99 for index in value):
            raise ValueError("mark indices must be integers between 1 and 99")
        if len(value) != len(set(value)):
            raise ValueError("mark indices must be unique")
        return value


class OCRRequest(_FrozenModel):
    """Request localized optical character recognition output."""

    capability: Literal[Capability.OCR] = Capability.OCR
    region: Box | None = None
    languages: Annotated[tuple[_LanguageTag, ...], Field(max_length=8)] = Field(
        default_factory=tuple
    )
    mode: Literal["auto", "sparse", "dense", "single_line"] = "auto"
    min_confidence: _Confidence | None = None


def _deduplicate_labels(value: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for label in value:
        key = label.casefold()
        if key not in seen:
            seen.add(key)
            result.append(label)
    return tuple(result)


class DetectionRequest(_FrozenModel):
    """Request bounded object detections, optionally constrained by labels."""

    capability: Literal[Capability.DETECT] = Capability.DETECT
    region: Box | None = None
    labels: Annotated[tuple[_Label, ...], Field(max_length=64)] = Field(default_factory=tuple)
    min_confidence: _Confidence = 0.25
    max_results: Annotated[int, Field(gt=0)] = 100

    @field_validator("labels")
    @classmethod
    def _normalize_labels(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _deduplicate_labels(value)


class SegmentationRequest(_FrozenModel):
    """Request object segmentations from labels or normalized point prompts."""

    capability: Literal[Capability.SEGMENT] = Capability.SEGMENT
    region: Box | None = None
    labels: Annotated[tuple[_Label, ...], Field(max_length=64)] = Field(default_factory=tuple)
    points: tuple[Point, ...] = Field(default_factory=tuple)
    min_confidence: _Confidence = 0.25
    max_results: Annotated[int, Field(gt=0)] = 32

    @field_validator("labels")
    @classmethod
    def _normalize_labels(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _deduplicate_labels(value)


VisionRequest: TypeAlias = Annotated[
    MetadataRequest
    | ColorsRequest
    | CaptionRequest
    | OCRRequest
    | DetectionRequest
    | SegmentationRequest,
    Field(discriminator="capability"),
]


class MetadataPayload(_FrozenModel):
    """Normalized dimensions and alpha information for an asset."""

    type: Literal["metadata"] = "metadata"
    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]
    aspect_ratio: _PositiveFinite
    has_alpha: bool


class ColorSwatch(_FrozenModel):
    """A normalized dominant-color estimate."""

    rgb: tuple[
        Annotated[int, Field(ge=0, le=255)],
        Annotated[int, Field(ge=0, le=255)],
        Annotated[int, Field(ge=0, le=255)],
    ]
    hex: Annotated[str, StringConstraints(pattern=r"^#[0-9A-F]{6}$")]
    fraction: _Confidence
    name: _CleanText | None = None


class ColorsPayload(_FrozenModel):
    """An ordered collection of dominant color swatches."""

    type: Literal["colors"] = "colors"
    swatches: tuple[ColorSwatch, ...]


class CaptionPayload(_FrozenModel):
    """A bounded semantic caption from a vision backend."""

    type: Literal["caption"] = "caption"
    text: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=8_000),
        AfterValidator(_reject_nul),
    ]
    focus: (
        Annotated[
            str,
            StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
            AfterValidator(_reject_nul),
        ]
        | None
    ) = None


class MarkDescriptionRef(_FrozenModel):
    """One structured description returned for a visible numeric mark."""

    index: Annotated[int, Field(ge=1, le=99)]
    description: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=1_000),
        AfterValidator(_reject_nul),
    ]


class MarkDescriptionPayload(_FrozenModel):
    """Bounded structured descriptions from a proven mark-aware backend."""

    type: Literal["mark_description"] = "mark_description"
    references: Annotated[
        tuple[MarkDescriptionRef, ...],
        Field(min_length=1, max_length=99),
    ]

    @field_validator("references")
    @classmethod
    def _unique_reference_indices(
        cls, value: tuple[MarkDescriptionRef, ...]
    ) -> tuple[MarkDescriptionRef, ...]:
        indices = tuple(reference.index for reference in value)
        if len(indices) != len(set(indices)):
            raise ValueError("mark description indices must be unique")
        return value


class TextPayload(_FrozenModel):
    """A bounded OCR text block with optional language information."""

    type: Literal["text"] = "text"
    text: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=32_000),
        AfterValidator(_reject_nul),
    ]
    language: _LanguageTag | None = None
    block_kind: Literal["word", "line", "paragraph", "unknown"] = "unknown"


class DetectionPayload(_FrozenModel):
    """A localized object label and optional backend attributes."""

    type: Literal["detection"] = "detection"
    label: _Label
    attributes: tuple[_Label, ...] = Field(default_factory=tuple)


class SegmentationPayload(_FrozenModel):
    """An object label and optional normalized segmentation polygon."""

    type: Literal["segmentation"] = "segmentation"
    label: _Label
    polygon: tuple[Point, ...] = Field(default_factory=tuple)

    @field_validator("polygon")
    @classmethod
    def _validate_polygon(cls, value: tuple[Point, ...]) -> tuple[Point, ...]:
        if value and not 3 <= len(value) <= 1_024:
            raise ValueError("polygon must be empty or contain between 3 and 1024 points")
        return value


class TransformPayload(_FrozenModel):
    """The lineage record for a newly created or reused derived asset."""

    type: Literal["transform"] = "transform"
    derived_asset_id: _AssetId
    parent_asset_id: _AssetId
    transform: TransformDescriptor


class MarkRef(_FrozenModel):
    """One deterministic numeric reference to a source observation region."""

    index: Annotated[int, Field(ge=1, le=99)]
    observation_id: _ObservationId
    region: Box
    source_label: _Label | None = None


class MarkPayload(_FrozenModel):
    """A transform-fact mapping from rendered indices to source observations."""

    type: Literal["mark"] = "mark"
    derived_asset_id: _AssetId
    parent_asset_id: _AssetId
    marks: Annotated[tuple[MarkRef, ...], Field(min_length=1, max_length=99)]

    @field_validator("marks")
    @classmethod
    def _validate_marks(cls, value: tuple[MarkRef, ...]) -> tuple[MarkRef, ...]:
        indices = tuple(mark.index for mark in value)
        observation_ids = tuple(mark.observation_id for mark in value)
        if len(indices) != len(set(indices)):
            raise ValueError("mark indices must be unique")
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("mark observation IDs must be unique")
        if indices != tuple(range(1, len(value) + 1)):
            raise ValueError("mark indices must be contiguous and start at one")
        return value


class WarningPayload(_FrozenModel):
    """A stable warning represented as an observation payload."""

    type: Literal["warning"] = "warning"
    code: _SnakeName
    message: _CleanText


ObservationPayload: TypeAlias = Annotated[
    MetadataPayload
    | ColorsPayload
    | CaptionPayload
    | MarkDescriptionPayload
    | TextPayload
    | DetectionPayload
    | SegmentationPayload
    | TransformPayload
    | MarkPayload
    | WarningPayload,
    Field(discriminator="type"),
]


class WarningInfo(_FrozenModel):
    """A stable warning code with a safe summary and structured details."""

    code: _SnakeName
    message: _CleanText
    details: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("details")
    @classmethod
    def _validate_details(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return _json_mapping(value)


class CacheStats(_FrozenModel):
    """A transactional snapshot of one managed cache's retained content.

    ``entry_count`` and ``total_bytes`` are derived inside the same transaction
    that produced them, from verified value sizes rather than caller-supplied
    accounting. ``removed_entries`` and ``removed_bytes`` report what a
    ``prune`` call discarded; ``stats`` leaves them at zero.
    """

    entry_count: Annotated[int, Field(ge=0)]
    total_bytes: Annotated[int, Field(ge=0)]
    max_entries: Annotated[int, Field(gt=0)]
    max_bytes: Annotated[int, Field(gt=0)]
    removed_entries: Annotated[int, Field(ge=0)] = 0
    removed_bytes: Annotated[int, Field(ge=0)] = 0


class Observation(_FrozenModel):
    """An immutable, attributable observation attached to a session asset."""

    id: _ObservationId
    asset_id: _AssetId
    payload: ObservationPayload
    region: Box | None = None
    confidence: _Confidence | None = None
    provenance: Provenance
    supersedes: tuple[_ObservationId, ...] = Field(default_factory=tuple)
    contradicts: tuple[_ObservationId, ...] = Field(default_factory=tuple)
    warnings: tuple[WarningInfo, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _validate_region_semantics(self) -> Observation:
        if (
            isinstance(self.payload, (MetadataPayload, TransformPayload, MarkPayload))
            and self.region is not None
        ):
            raise ValueError("metadata, transform, and mark observations cannot have a region")
        return self


class ObservationDraft(_FrozenModel):
    """Untrusted backend output before session attribution and ID assignment."""

    payload: ObservationPayload
    region: Box | None = None
    confidence: _Confidence | None = None
    warnings: tuple[WarningInfo, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _validate_region_semantics(self) -> ObservationDraft:
        if (
            isinstance(self.payload, (MetadataPayload, TransformPayload, MarkPayload))
            and self.region is not None
        ):
            raise ValueError("metadata, transform, and mark drafts cannot have a region")
        return self


class VisionResult(_FrozenModel):
    """A complete backend response awaiting core normalization."""

    observations: tuple[ObservationDraft, ...]
    warnings: tuple[WarningInfo, ...] = Field(default_factory=tuple)


class TraceSummary(_FrozenModel):
    """Aggregate counters and termination information for a completed run."""

    trace_id: UUID
    started_at: datetime
    duration_ms: Annotated[int, Field(ge=0)]
    llm_calls: Annotated[int, Field(ge=0)]
    tool_calls: Annotated[int, Field(ge=0)]
    backend_calls: Annotated[int, Field(ge=0)]
    cache_hits: Annotated[int, Field(ge=0)]
    cache_failures: Annotated[int, Field(ge=0)] = 0
    derived_assets: Annotated[int, Field(ge=0)]
    input_tokens: Annotated[int, Field(ge=0)] | None = None
    output_tokens: Annotated[int, Field(ge=0)] | None = None
    stop_reason: Literal[
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

    @field_validator("started_at")
    @classmethod
    def _utc_started_at(cls, value: datetime) -> datetime:
        return _validate_utc(value)


class TraceEvent(_FrozenModel):
    """One immutable, already-redacted event in a run trace."""

    model_config = ConfigDict(
        json_schema_extra={
            "allOf": [
                {
                    "if": {
                        "properties": {"schema_version": {"const": 2}},
                        "required": ["schema_version"],
                    },
                    "then": {"properties": {"event_type": {"pattern": r"^[a-z][a-z0-9_]*$"}}},
                }
            ]
        }
    )

    # The default exists for backward parsing and direct legacy construction.
    # TraceBuilder explicitly stamps every newly emitted event as schema v2.
    schema_version: Literal[1, 2] = 1
    trace_id: UUID
    sequence: Annotated[int, Field(ge=0)]
    event_type: _CleanText
    occurred_at: datetime
    duration_ms: Annotated[int, Field(ge=0)] | None = None
    invocation_id: _CleanText | None = None
    parent_invocation_id: _CleanText | None = None
    data: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("occurred_at")
    @classmethod
    def _utc_occurred_at(cls, value: datetime) -> datetime:
        return _validate_utc(value)

    @field_validator("data")
    @classmethod
    def _validate_data(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return _json_mapping(value)

    @model_validator(mode="after")
    def _validate_versioned_event_type(self) -> TraceEvent:
        if self.schema_version == 2 and not re.fullmatch(r"[a-z][a-z0-9_]*", self.event_type):
            raise ValueError("schema-v2 event_type must use lower snake case")
        return self


def _validate_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware UTC")
    if value.utcoffset() != timedelta(0):
        raise ValueError("datetime must use UTC")
    return value.astimezone(timezone.utc)


class RunTrace(_FrozenModel):
    """A completed run summary and its ordered redacted events."""

    summary: TraceSummary
    events: tuple[TraceEvent, ...]

    @model_validator(mode="after")
    def _validate_events(self) -> RunTrace:
        if any(event.trace_id != self.summary.trace_id for event in self.events):
            raise ValueError("all trace events must use the summary trace_id")
        sequences = tuple(event.sequence for event in self.events)
        if any(current <= previous for previous, current in pairwise(sequences)):
            raise ValueError("trace event sequences must be strictly increasing")
        return self


class InspectionOperation(_FrozenModel):
    """One caller-directed perception request in an inspection plan."""

    request: VisionRequest
    asset_id: _AssetId | None = None
    required: bool = False
    backend: _BackendName | None = None


class InspectionPlan(_FrozenModel):
    """An ordered set of perception operations and overview behavior."""

    operations: tuple[InspectionOperation, ...] = Field(default_factory=tuple)
    include_available_overview: bool = True
    fail_fast: bool = False

    @model_validator(mode="after")
    def _require_work(self) -> InspectionPlan:
        if not self.include_available_overview and not self.operations:
            raise ValueError("an inspection plan must request at least one operation")
        return self


class InspectionResult(_FrozenModel):
    """New observations, warnings, and trace produced by an inspection call."""

    root_asset: ImageAsset
    observations: tuple[Observation, ...]
    warnings: tuple[WarningInfo, ...]
    trace: RunTrace


class MessageRole(str, Enum):
    """The role of one text-LLM request message."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(_FrozenModel):
    """A text-only message sent to a language model."""

    role: MessageRole
    content: Annotated[str, StringConstraints(min_length=1), AfterValidator(_reject_nul)]


class TokenUsage(_FrozenModel):
    """Optional token counters reported by a language-model provider."""

    input_tokens: Annotated[int, Field(ge=0)] | None = None
    output_tokens: Annotated[int, Field(ge=0)] | None = None


class SchemaEnforcement(str, Enum):
    """How strongly a provider enforced the compiled action schema."""

    STRICT = "strict"
    JSON_ONLY = "json_only"


class RetryPolicy(_FrozenModel):
    """Bounded provider retry budget with capped exponential backoff.

    The policy is a pure wire contract: it never carries callable state. A
    deterministic random source is supplied privately by adapter tests.
    """

    max_attempts: Annotated[int, Field(ge=1, le=6)] = 3
    base_delay_s: Annotated[float, Field(gt=0), AfterValidator(_finite)] = 0.25
    max_delay_s: Annotated[float, Field(gt=0), AfterValidator(_finite)] = 4.0

    @model_validator(mode="after")
    def _validate_delays(self) -> RetryPolicy:
        if self.max_delay_s < self.base_delay_s:
            raise ValueError("max_delay_s must be greater than or equal to base_delay_s")
        return self


class LLMRequest(_FrozenModel):
    """A provider-neutral text-only language-model request."""

    messages: Annotated[tuple[Message, ...], Field(min_length=1)]
    response_json_schema: dict[str, JsonValue]
    temperature: Annotated[float, AfterValidator(_finite)] = 0.0
    max_output_tokens: Annotated[int, Field(gt=0)] = 800
    timeout_s: _PositiveFinite | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def _safe_metadata(cls, value: dict[str, str]) -> dict[str, str]:
        for key, item in value.items():
            _reject_nul(key)
            _reject_nul(item)
        return value

    @field_validator("response_json_schema")
    @classmethod
    def _validate_response_schema(
        cls,
        value: dict[str, JsonValue],
    ) -> dict[str, JsonValue]:
        return _json_mapping(value)


class LLMResponse(_FrozenModel):
    """Provider-neutral textual output from a language model.

    Provider-specific metadata that is not represented by a field here MUST NOT
    be smuggled into logs or exceptions; new metadata requires a contract-schema
    update.
    """

    text: _RawText
    model_id: _CleanText | None = None
    usage: TokenUsage | None = None
    finish_reason: _CleanText | None = None
    provider: _CleanText | None = None
    request_id: _CleanText | None = None
    backend_fingerprint: _CleanText | None = None
    attempts: Annotated[int, Field(ge=1)] = 1
    schema_enforcement: SchemaEnforcement = SchemaEnforcement.STRICT


class BackendImage(_FrozenModel):
    """A normalized asset and its private canonical PNG content for a backend."""

    asset: ImageAsset
    content: bytes = Field(repr=False, exclude=True)


class ToolAction(_FrozenModel):
    """A validated request by a policy to invoke one declared tool."""

    type: Literal["tool"] = "tool"
    tool: _SnakeName
    arguments: dict[str, JsonValue]
    purpose: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=300),
        AfterValidator(_reject_nul),
    ]

    @field_validator("arguments")
    @classmethod
    def _validate_arguments(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return _json_mapping(value)


class EvidenceRef(_FrozenModel):
    """A policy-emitted reference to an observation supporting a concise claim."""

    observation_id: _ObservationId
    supports: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=300),
        AfterValidator(_reject_nul),
    ]


class AnswerAction(_FrozenModel):
    """A policy-emitted final answer or explicit evidence abstention."""

    type: Literal["answer"] = "answer"
    status: Literal["answered", "insufficient_evidence"]
    answer: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=8_000),
        AfterValidator(_reject_nul),
    ]
    evidence: tuple[EvidenceRef, ...] = Field(default_factory=tuple)
    uncertainties: tuple[_CleanText, ...] = Field(default_factory=tuple)


PolicyAction: TypeAlias = Annotated[
    ToolAction | AnswerAction,
    Field(discriminator="type"),
]


class AnswerStatus(str, Enum):
    """The outcome category of a completed visual question."""

    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class Evidence(_FrozenModel):
    """A snapshot of one observation and the material claim it supports."""

    observation: Observation
    supports: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=300),
        AfterValidator(_reject_nul),
    ]


class VisionAnswer(_FrozenModel):
    """A completed answer with evidence snapshots, warnings, and trace."""

    status: AnswerStatus
    answer: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=8_000),
        AfterValidator(_reject_nul),
    ]
    evidence: tuple[Evidence, ...]
    uncertainties: tuple[_CleanText, ...]
    warnings: tuple[WarningInfo, ...]
    trace: RunTrace


class ToolSpec(_FrozenModel):
    """An LLM-visible tool declaration with a strict JSON argument schema."""

    name: _SnakeName
    description: _CleanText
    arguments_json_schema: dict[str, JsonValue]
    creates_assets: bool = False
    cost_hint: Annotated[int, Field(ge=1, le=100)] = 1

    @field_validator("arguments_json_schema")
    @classmethod
    def _validate_argument_schema(
        cls,
        value: dict[str, JsonValue],
    ) -> dict[str, JsonValue]:
        return _json_mapping(value)


class RemainingBudget(_FrozenModel):
    """A non-negative snapshot of resources available to a policy call."""

    steps: Annotated[int, Field(ge=0)]
    llm_calls: Annotated[int, Field(ge=0)]
    tool_calls: Annotated[int, Field(ge=0)]
    backend_calls: Annotated[int, Field(ge=0)]
    derived_assets: Annotated[int, Field(ge=0)]
    derivation_depth: Annotated[int, Field(ge=0)]
    context_chars: Annotated[int, Field(ge=0)]
    remaining_time_s: _NonNegativeFinite

    @field_validator("remaining_time_s", mode="before")
    @classmethod
    def _clamp_remaining_time(cls, value: object) -> object:
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value < 0.0
        ):
            return 0.0
        return value


class PolicyInput(_FrozenModel):
    """Trusted policy instructions, untrusted context, and remaining resources."""

    question: _CleanText
    context: Annotated[str, StringConstraints(min_length=1), AfterValidator(_reject_nul)]
    tools: tuple[ToolSpec, ...]
    prior_actions: tuple[PolicyAction, ...]
    remaining: RemainingBudget
    answer_only: bool = False
    validation_feedback: tuple[WarningInfo, ...] = Field(default_factory=tuple)
    invalid_model_output: _RawText | None = Field(default=None, repr=False)


__all__ = [
    "JSON_VALUE_ADAPTER",
    "AnswerAction",
    "AnswerStatus",
    "BackendDescriptor",
    "BackendImage",
    "Box",
    "CacheStats",
    "Capability",
    "CapabilityDescriptor",
    "CaptionPayload",
    "CaptionRequest",
    "ColorSwatch",
    "ColorsPayload",
    "ColorsRequest",
    "DetectionPayload",
    "DetectionRequest",
    "Evidence",
    "EvidenceRef",
    "ImageAsset",
    "ImageSource",
    "InspectionOperation",
    "InspectionPlan",
    "InspectionResult",
    "JsonValue",
    "LLMRequest",
    "LLMResponse",
    "MarkDescriptionPayload",
    "MarkDescriptionRef",
    "MarkPayload",
    "MarkRef",
    "Message",
    "MessageRole",
    "MetadataPayload",
    "MetadataRequest",
    "OCRRequest",
    "Observation",
    "ObservationDraft",
    "ObservationPayload",
    "Point",
    "PolicyAction",
    "PolicyInput",
    "Provenance",
    "RemainingBudget",
    "RetryPolicy",
    "RunTrace",
    "SchemaEnforcement",
    "SegmentationPayload",
    "SegmentationRequest",
    "TextPayload",
    "TokenUsage",
    "ToolAction",
    "ToolSpec",
    "TraceEvent",
    "TraceSummary",
    "TransformDescriptor",
    "TransformPayload",
    "VisionAnswer",
    "VisionRequest",
    "VisionResult",
    "WarningInfo",
    "WarningPayload",
]
