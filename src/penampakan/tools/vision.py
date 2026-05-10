"""Typed perception tool declarations and request translation."""

from __future__ import annotations

from collections.abc import Collection
from typing import Annotated, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from penampakan.models import (
    Box,
    Capability,
    CaptionRequest,
    ColorsRequest,
    DetectionRequest,
    MetadataRequest,
    OCRRequest,
    Point,
    SegmentationRequest,
)
from penampakan.perception.registry import ToolExecutionContext, ToolRegistry, ToolResult

AssetId = Annotated[str, StringConstraints(pattern=r"^img_[0-9a-f]{16,64}$")]
Label = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
LanguageTag = Annotated[
    str,
    StringConstraints(strip_whitespace=True, pattern=r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$"),
]
ArgumentsT = TypeVar("ArgumentsT", bound=BaseModel)


class _Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class MetadataArguments(_Arguments):
    """Arguments for authoritative normalized image metadata."""

    asset_id: AssetId


class ColorsArguments(_Arguments):
    """Arguments for dominant colors on an image or region."""

    asset_id: AssetId
    region: Box | None = None
    count: int = Field(default=5, ge=1, le=16)


class CaptionArguments(_Arguments):
    """Arguments for a bounded image description."""

    asset_id: AssetId
    region: Box | None = None
    focus: (
        Annotated[
            str,
            StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
        ]
        | None
    ) = None
    max_sentences: int = Field(default=3, ge=1, le=8)


class OCRArguments(_Arguments):
    """Arguments for localized optical character recognition."""

    asset_id: AssetId
    region: Box | None = None
    languages: tuple[LanguageTag, ...] = Field(default_factory=tuple, max_length=8)
    mode: Literal["auto", "sparse", "dense", "single_line"] = "auto"
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)


class DetectionArguments(_Arguments):
    """Arguments for bounded object detection."""

    asset_id: AssetId
    region: Box | None = None
    labels: tuple[Label, ...] = Field(default_factory=tuple, max_length=64)
    min_confidence: float = Field(default=0.25, ge=0.0, le=1.0, allow_inf_nan=False)
    max_results: int = Field(default=100, ge=1, le=100)


class SegmentationArguments(_Arguments):
    """Arguments for bounded optional object segmentation."""

    asset_id: AssetId
    region: Box | None = None
    labels: tuple[Label, ...] = Field(default_factory=tuple, max_length=64)
    points: tuple[Point, ...] = Field(default_factory=tuple, max_length=1_024)
    min_confidence: float = Field(default=0.25, ge=0.0, le=1.0, allow_inf_nan=False)
    max_results: int = Field(default=32, ge=1, le=32)


async def _metadata(context: ToolExecutionContext, arguments: BaseModel) -> ToolResult:
    values = _typed(arguments, MetadataArguments)
    return await context.perceive(values.asset_id, MetadataRequest())


async def _colors(context: ToolExecutionContext, arguments: BaseModel) -> ToolResult:
    values = _typed(arguments, ColorsArguments)
    return await context.perceive(
        values.asset_id,
        ColorsRequest(region=values.region, count=values.count),
    )


async def _caption(context: ToolExecutionContext, arguments: BaseModel) -> ToolResult:
    values = _typed(arguments, CaptionArguments)
    return await context.perceive(
        values.asset_id,
        CaptionRequest(
            region=values.region,
            focus=values.focus,
            max_sentences=values.max_sentences,
        ),
    )


async def _ocr(context: ToolExecutionContext, arguments: BaseModel) -> ToolResult:
    values = _typed(arguments, OCRArguments)
    return await context.perceive(
        values.asset_id,
        OCRRequest(
            region=values.region,
            languages=values.languages,
            mode=values.mode,
            min_confidence=values.min_confidence,
        ),
    )


async def _detect(context: ToolExecutionContext, arguments: BaseModel) -> ToolResult:
    values = _typed(arguments, DetectionArguments)
    return await context.perceive(
        values.asset_id,
        DetectionRequest(
            region=values.region,
            labels=values.labels,
            min_confidence=values.min_confidence,
            max_results=values.max_results,
        ),
    )


async def _segment(context: ToolExecutionContext, arguments: BaseModel) -> ToolResult:
    values = _typed(arguments, SegmentationArguments)
    return await context.perceive(
        values.asset_id,
        SegmentationRequest(
            region=values.region,
            labels=values.labels,
            points=values.points,
            min_confidence=values.min_confidence,
            max_results=values.max_results,
        ),
    )


def register_vision_tools(registry: ToolRegistry, capabilities: Collection[Capability]) -> None:
    """Register only perception tools backed by current capabilities."""

    available = frozenset(capabilities)
    registry.register(
        name="get_metadata",
        description="Return authoritative normalized dimensions and alpha presence for an asset.",
        arguments_model=MetadataArguments,
        executor=_metadata,
    )
    registry.register(
        name="get_colors",
        description="Estimate dominant colors for an asset or normalized region.",
        arguments_model=ColorsArguments,
        executor=_colors,
    )
    if Capability.CAPTION in available:
        registry.register(
            name="describe_image",
            description=("Describe visible content; focus requires a compatible caption backend."),
            arguments_model=CaptionArguments,
            executor=_caption,
            cost_hint=3,
        )
    if Capability.OCR in available:
        registry.register(
            name="read_text",
            description="Read localized visible text; requested languages require backend support.",
            arguments_model=OCRArguments,
            executor=_ocr,
            cost_hint=3,
        )
    if Capability.DETECT in available:
        registry.register(
            name="detect_objects",
            description="Locate visible objects; open-vocabulary labels require backend support.",
            arguments_model=DetectionArguments,
            executor=_detect,
            cost_hint=4,
        )
    if Capability.SEGMENT in available:
        registry.register(
            name="segment_objects",
            description="Segment visible objects when a compatible optional backend is registered.",
            arguments_model=SegmentationArguments,
            executor=_segment,
            cost_hint=5,
        )


def _typed(value: BaseModel, expected: type[ArgumentsT]) -> ArgumentsT:
    if not isinstance(value, expected):
        raise TypeError("validated tool arguments have an unexpected type")
    return value


__all__ = [
    "CaptionArguments",
    "ColorsArguments",
    "DetectionArguments",
    "MetadataArguments",
    "OCRArguments",
    "SegmentationArguments",
    "register_vision_tools",
]
