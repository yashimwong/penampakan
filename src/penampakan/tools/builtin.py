"""Typed built-in image transform tool declarations and execution."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Annotated, Literal, TypeVar

from PIL.Image import Image as PillowImage
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from penampakan.image import transforms
from penampakan.image.assets import PendingAsset
from penampakan.models import Box
from penampakan.perception.registry import ToolExecutionContext, ToolRegistry, ToolResult

AssetId = Annotated[str, StringConstraints(pattern=r"^img_[0-9a-f]{16,64}$")]
ArgumentsT = TypeVar("ArgumentsT", bound=BaseModel)


class _Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class CropArguments(_Arguments):
    """Arguments for a normalized pixel-aligned crop."""

    asset_id: AssetId
    box: Box
    padding_fraction: float = Field(default=0.0, ge=0.0, le=0.5, allow_inf_nan=False)


class TileArguments(_Arguments):
    """Arguments for complete row-major image tiling."""

    asset_id: AssetId
    rows: int = Field(ge=1, le=8)
    columns: int = Field(ge=1, le=8)
    overlap_fraction: float = Field(default=0.0, ge=0.0, le=0.5, allow_inf_nan=False)

    @model_validator(mode="after")
    def _validate_fanout(self) -> TileArguments:
        if self.rows == 1 and self.columns == 1:
            raise ValueError("at least one tile dimension must exceed one")
        if self.rows * self.columns > 16:
            raise ValueError("tile fanout cannot exceed sixteen assets")
        return self


class RotateArguments(_Arguments):
    """Arguments for a non-clipping clockwise right-angle rotation."""

    asset_id: AssetId
    degrees: Literal[90, 180, 270]


class ContrastArguments(_Arguments):
    """Arguments for bounded contrast enhancement."""

    asset_id: AssetId
    factor: float = Field(default=2.0, ge=0.25, le=4.0, allow_inf_nan=False)


class GrayscaleArguments(_Arguments):
    """Arguments for alpha-preserving RGB luminance conversion."""

    asset_id: AssetId


class CoordinateGridArguments(_Arguments):
    """Arguments for a labeled high-contrast coordinate grid."""

    asset_id: AssetId
    rows: int = Field(default=4, ge=2, le=20)
    columns: int = Field(default=4, ge=2, le=20)
    labels: bool = True


async def _crop(context: ToolExecutionContext, arguments: BaseModel) -> ToolResult:
    values = _typed(arguments, CropArguments)
    context.ensure_asset_capacity(values.asset_id, 1)
    source = context.image(values.asset_id)
    pending = await _render(
        source,
        lambda: (transforms.crop(source, values.box, values.padding_fraction),),
    )
    return ToolResult(assets=pending)


async def _tile(context: ToolExecutionContext, arguments: BaseModel) -> ToolResult:
    values = _typed(arguments, TileArguments)
    count = values.rows * values.columns
    context.ensure_asset_capacity(values.asset_id, count)
    source = context.image(values.asset_id)
    pending = await _render(
        source,
        lambda: transforms.tile(
            source,
            rows=values.rows,
            columns=values.columns,
            overlap_fraction=values.overlap_fraction,
        ),
    )
    return ToolResult(assets=pending)


async def _rotate(context: ToolExecutionContext, arguments: BaseModel) -> ToolResult:
    values = _typed(arguments, RotateArguments)
    context.ensure_asset_capacity(values.asset_id, 1)
    source = context.image(values.asset_id)
    pending = await _render(
        source,
        lambda: (transforms.rotate(source, values.degrees),),
    )
    return ToolResult(assets=pending)


async def _contrast(context: ToolExecutionContext, arguments: BaseModel) -> ToolResult:
    values = _typed(arguments, ContrastArguments)
    context.ensure_asset_capacity(values.asset_id, 1)
    source = context.image(values.asset_id)
    pending = await _render(
        source,
        lambda: (transforms.enhance_contrast(source, values.factor),),
    )
    return ToolResult(assets=pending)


async def _grayscale(context: ToolExecutionContext, arguments: BaseModel) -> ToolResult:
    values = _typed(arguments, GrayscaleArguments)
    context.ensure_asset_capacity(values.asset_id, 1)
    source = context.image(values.asset_id)
    pending = await _render(source, lambda: (transforms.to_grayscale(source),))
    return ToolResult(assets=pending)


async def _coordinate_grid(context: ToolExecutionContext, arguments: BaseModel) -> ToolResult:
    values = _typed(arguments, CoordinateGridArguments)
    context.ensure_asset_capacity(values.asset_id, 1)
    source = context.image(values.asset_id)
    pending = await _render(
        source,
        lambda: (
            transforms.add_coordinate_grid(
                source,
                rows=values.rows,
                columns=values.columns,
                labels=values.labels,
            ),
        ),
    )
    return ToolResult(assets=pending)


async def _render(
    source: PillowImage,
    render: Callable[[], tuple[PendingAsset, ...]],
) -> tuple[PendingAsset, ...]:
    task = asyncio.create_task(asyncio.to_thread(_render_owned, source, render))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        task.add_done_callback(_close_late_render)
        raise


def _render_owned(
    source: PillowImage,
    render: Callable[[], tuple[PendingAsset, ...]],
) -> tuple[PendingAsset, ...]:
    try:
        return render()
    finally:
        source.close()


def _close_late_render(task: asyncio.Task[tuple[PendingAsset, ...]]) -> None:
    try:
        pending = task.result()
    except (asyncio.CancelledError, Exception):
        return
    for asset in pending:
        asset.close()


def register_transform_tools(registry: ToolRegistry) -> None:
    """Register every safe deterministic image transform."""

    registry.register(
        name="crop",
        description=(
            "Create a pixel-aligned crop from a normalized box with optional bounded padding."
        ),
        arguments_model=CropArguments,
        executor=_crop,
        creates_assets=True,
    )
    registry.register(
        name="tile",
        description=(
            "Create complete row-major tiles with bounded overlap and at most sixteen assets."
        ),
        arguments_model=TileArguments,
        executor=_tile,
        creates_assets=True,
        cost_hint=2,
    )
    registry.register(
        name="rotate",
        description="Create a clockwise 90, 180, or 270 degree rotation without clipping.",
        arguments_model=RotateArguments,
        executor=_rotate,
        creates_assets=True,
    )
    registry.register(
        name="enhance_contrast",
        description="Create a bounded contrast-enhanced asset while preserving alpha.",
        arguments_model=ContrastArguments,
        executor=_contrast,
        creates_assets=True,
    )
    registry.register(
        name="to_grayscale",
        description="Create an RGB or RGBA luminance asset while preserving alpha.",
        arguments_model=GrayscaleArguments,
        executor=_grayscale,
        creates_assets=True,
    )
    registry.register(
        name="add_coordinate_grid",
        description=(
            "Create a visual coordinate grid with optional cell labels for later perception."
        ),
        arguments_model=CoordinateGridArguments,
        executor=_coordinate_grid,
        creates_assets=True,
    )


def _typed(value: BaseModel, expected: type[ArgumentsT]) -> ArgumentsT:
    if not isinstance(value, expected):
        raise TypeError("validated tool arguments have an unexpected type")
    return value


__all__ = [
    "ContrastArguments",
    "CoordinateGridArguments",
    "CropArguments",
    "GrayscaleArguments",
    "RotateArguments",
    "TileArguments",
    "register_transform_tools",
]
