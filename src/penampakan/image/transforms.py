"""Deterministic image transforms that produce transaction-ready assets."""

from __future__ import annotations

import math
from typing import Literal, cast

from PIL import ImageDraw, ImageEnhance
from PIL.Image import Image as PillowImage
from PIL.Image import Transpose

from penampakan.image.assets import PendingAsset
from penampakan.image.geometry import PixelBox, build_crop_geometry, pixels_to_box
from penampakan.models import Box, JsonValue, TransformDescriptor


def _finite_number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _bounded_number(value: float, name: str, minimum: float, maximum: float) -> float:
    result = _finite_number(value, name)
    if result < minimum or result > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result


def _bounded_integer(value: int, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _normalized_image(image: PillowImage) -> PillowImage:
    if not isinstance(image, PillowImage):
        raise TypeError("image must be a Pillow image")
    if image.mode not in {"RGB", "RGBA"}:
        raise ValueError("image mode must be RGB or RGBA")
    if image.width <= 0 or image.height <= 0:
        raise ValueError("image dimensions must be positive")
    return image


def _box_parameters(box: Box) -> dict[str, JsonValue]:
    return {
        "x_min": box.x_min,
        "y_min": box.y_min,
        "x_max": box.x_max,
        "y_max": box.y_max,
    }


def _descriptor(
    name: Literal[
        "crop",
        "tile",
        "rotate",
        "enhance_contrast",
        "grayscale",
        "coordinate_grid",
    ],
    parameters: dict[str, JsonValue],
) -> TransformDescriptor:
    return TransformDescriptor(name=name, parameters=parameters)


def crop(
    image: PillowImage,
    box: Box,
    padding_fraction: float = 0.0,
) -> PendingAsset:
    """Render an exact pixel-aligned crop with bounded normalized padding."""
    source = _normalized_image(image)
    geometry = build_crop_geometry(box, source.width, source.height, padding_fraction)
    result = source.crop(geometry.pixel_box.as_tuple())
    result.info.clear()
    descriptor = _descriptor(
        "crop",
        {
            "requested_box": _box_parameters(geometry.requested_box),
            "expanded_box": _box_parameters(geometry.expanded_box),
            "applied_box": _box_parameters(geometry.applied_box),
            "padding_fraction": geometry.padding_fraction,
        },
    )
    return PendingAsset(image=result, transform=descriptor)


def _axis_ranges(length: int, parts: int, overlap_fraction: float) -> tuple[tuple[int, int], ...]:
    if parts > length:
        raise ValueError("tile count cannot exceed pixels along an axis")
    boundaries = tuple(index * length // parts for index in range(parts + 1))
    ranges: list[tuple[int, int]] = []
    for index in range(parts):
        base_start = boundaries[index]
        base_end = boundaries[index + 1]
        base_length = base_end - base_start
        padding = math.ceil(base_length * overlap_fraction / 2.0)
        start = max(0, base_start - padding)
        end = min(length, base_end + padding)
        ranges.append((start, end))
    return tuple(ranges)


def tile(
    image: PillowImage,
    rows: int,
    columns: int,
    overlap_fraction: float = 0.0,
) -> tuple[PendingAsset, ...]:
    """Render complete row-major tiled coverage of a normalized image."""
    source = _normalized_image(image)
    row_count = _bounded_integer(rows, "rows", 1, 8)
    column_count = _bounded_integer(columns, "columns", 1, 8)
    if row_count * column_count > 16:
        raise ValueError("rows multiplied by columns cannot exceed 16")
    if row_count == 1 and column_count == 1:
        raise ValueError("at least one tile dimension must exceed one")
    overlap = _bounded_number(overlap_fraction, "overlap_fraction", 0.0, 0.5)
    y_ranges = _axis_ranges(source.height, row_count, overlap)
    x_ranges = _axis_ranges(source.width, column_count, overlap)
    pending: list[PendingAsset] = []
    try:
        for row, (top, bottom) in enumerate(y_ranges):
            for column, (left, right) in enumerate(x_ranges):
                pixel_box = PixelBox(left=left, top=top, right=right, bottom=bottom)
                applied_box = pixels_to_box(pixel_box, source.width, source.height)
                result = source.crop(pixel_box.as_tuple())
                result.info.clear()
                descriptor = _descriptor(
                    "tile",
                    {
                        "rows": row_count,
                        "columns": column_count,
                        "row": row,
                        "column": column,
                        "overlap_fraction": overlap,
                        "applied_box": _box_parameters(applied_box),
                    },
                )
                pending.append(PendingAsset(image=result, transform=descriptor))
        return tuple(pending)
    except BaseException:
        for item in pending:
            item.close()
        raise


def rotate(image: PillowImage, degrees: int) -> PendingAsset:
    """Rotate a normalized image clockwise without clipping."""
    source = _normalized_image(image)
    if isinstance(degrees, bool) or not isinstance(degrees, int):
        raise TypeError("degrees must be an integer")
    operations = {
        90: Transpose.ROTATE_270,
        180: Transpose.ROTATE_180,
        270: Transpose.ROTATE_90,
    }
    try:
        operation = operations[degrees]
    except KeyError as error:
        raise ValueError("degrees must be 90, 180, or 270") from error
    result = source.transpose(operation)
    result.info.clear()
    return PendingAsset(
        image=result,
        transform=_descriptor("rotate", {"degrees": degrees}),
    )


def enhance_contrast(image: PillowImage, factor: float = 2.0) -> PendingAsset:
    """Enhance RGB luminance contrast while preserving an RGBA alpha channel."""
    source = _normalized_image(image)
    applied_factor = _bounded_number(factor, "factor", 0.25, 4.0)
    alpha = source.getchannel("A") if source.mode == "RGBA" else None
    rgb = source.convert("RGB") if source.mode == "RGBA" else source.copy()
    try:
        result = ImageEnhance.Contrast(rgb).enhance(applied_factor)
    finally:
        rgb.close()
    if alpha is not None:
        try:
            result.putalpha(alpha)
        finally:
            alpha.close()
    result.info.clear()
    return PendingAsset(
        image=result,
        transform=_descriptor("enhance_contrast", {"factor": applied_factor}),
    )


def to_grayscale(image: PillowImage) -> PendingAsset:
    """Render luminance as RGB while preserving an RGBA alpha channel."""
    source = _normalized_image(image)
    alpha = source.getchannel("A") if source.mode == "RGBA" else None
    luminance = source.convert("L")
    try:
        result = luminance.convert("RGB")
    finally:
        luminance.close()
    if alpha is not None:
        try:
            result.putalpha(alpha)
        finally:
            alpha.close()
    result.info.clear()
    return PendingAsset(image=result, transform=_descriptor("grayscale", {}))


def _local_colors(image: PillowImage, x: int, y: int) -> tuple[tuple[int, int, int], ...]:
    pixel = image.getpixel((min(image.width - 1, max(0, x)), min(image.height - 1, max(0, y))))
    red, green, blue = cast(tuple[int, int, int], pixel)[:3]
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    foreground = (0, 0, 0) if luminance >= 128.0 else (255, 255, 255)
    outline = (255, 255, 255) if foreground == (0, 0, 0) else (0, 0, 0)
    return foreground, outline


def add_coordinate_grid(
    image: PillowImage,
    rows: int = 4,
    columns: int = 4,
    labels: bool = True,
) -> PendingAsset:
    """Overlay a locally contrasting labeled coordinate grid."""
    source = _normalized_image(image)
    row_count = _bounded_integer(rows, "rows", 2, 20)
    column_count = _bounded_integer(columns, "columns", 2, 20)
    if not isinstance(labels, bool):
        raise TypeError("labels must be a boolean")
    alpha = source.getchannel("A") if source.mode == "RGBA" else None
    result = source.convert("RGB") if source.mode == "RGBA" else source.copy()
    draw = ImageDraw.Draw(result)
    for column in range(1, column_count):
        x = column * source.width // column_count
        foreground, outline = _local_colors(result, x, source.height // 2)
        draw.line((x, 0, x, source.height - 1), fill=outline, width=3)
        draw.line((x, 0, x, source.height - 1), fill=foreground, width=1)
    for row in range(1, row_count):
        y = row * source.height // row_count
        foreground, outline = _local_colors(result, source.width // 2, y)
        draw.line((0, y, source.width - 1, y), fill=outline, width=3)
        draw.line((0, y, source.width - 1, y), fill=foreground, width=1)
    if labels:
        for row in range(row_count):
            top = row * source.height // row_count
            for column in range(column_count):
                left = column * source.width // column_count
                text = f"{chr(ord('A') + column)}{row + 1}"
                x = min(source.width - 1, left + 2)
                y = min(source.height - 1, top + 2)
                foreground, outline = _local_colors(result, x, y)
                draw.text(
                    (x, y),
                    text,
                    fill=foreground,
                    stroke_width=1,
                    stroke_fill=outline,
                )
    if alpha is not None:
        try:
            result.putalpha(alpha)
        finally:
            alpha.close()
    result.info.clear()
    return PendingAsset(
        image=result,
        transform=_descriptor(
            "coordinate_grid",
            {"rows": row_count, "columns": column_count, "labels": labels},
        ),
    )


def grayscale(image: PillowImage) -> PendingAsset:
    """Return the RGB or RGBA luminance transform."""
    return to_grayscale(image)


def coordinate_grid(
    image: PillowImage,
    rows: int = 4,
    columns: int = 4,
    labels: bool = True,
) -> PendingAsset:
    """Return the labeled coordinate-grid transform."""
    return add_coordinate_grid(image, rows=rows, columns=columns, labels=labels)


__all__ = [
    "add_coordinate_grid",
    "coordinate_grid",
    "crop",
    "enhance_contrast",
    "grayscale",
    "rotate",
    "tile",
    "to_grayscale",
]
