"""Deterministic image transforms that produce transaction-ready assets."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

from PIL import ImageDraw, ImageEnhance, ImageFont
from PIL.Image import Image as PillowImage
from PIL.Image import Transpose

from penampakan.image.assets import PendingAsset
from penampakan.image.geometry import PixelBox, build_crop_geometry, pixels_to_box
from penampakan.models import Box, JsonValue, MarkRef, TransformDescriptor, WarningInfo


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
        "set_of_mark",
    ],
    parameters: dict[str, JsonValue],
) -> TransformDescriptor:
    return TransformDescriptor(name=name, parameters=parameters)


@dataclass(frozen=True, slots=True)
class _MarkCandidate:
    region: Box
    observation_id: str | None = None
    source_label: str | None = None
    priority: float = 0.0


@dataclass(frozen=True, slots=True)
class _MarkRenderResult:
    pending: PendingAsset
    marks: tuple[MarkRef, ...]
    warnings: tuple[WarningInfo, ...]


_DIGITS: dict[str, tuple[str, ...]] = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "001", "001", "001"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
}
_LABEL_WHITESPACE = re.compile(r"\s+")


def _sanitize_mark_label(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("mark labels must be strings or None")
    safe = "".join(character if ord(character) >= 32 else " " for character in value)
    safe = _LABEL_WHITESPACE.sub(" ", safe).strip()
    return safe[:24] or None


def _mark_sort_key(candidate: _MarkCandidate) -> tuple[float, float, float, float, str]:
    region = candidate.region
    return (
        region.y_min,
        region.x_min,
        region.y_max,
        region.x_max,
        candidate.observation_id or "",
    )


def _normalize_mark_candidates(
    candidates: Sequence[_MarkCandidate],
    near_duplicate_iou: float | None,
) -> tuple[tuple[_MarkCandidate, ...], int]:
    selected = tuple(candidates)
    if not 1 <= len(selected) <= 99:
        raise ValueError("mark regions must contain between 1 and 99 items")
    if near_duplicate_iou is None:
        threshold = None
    else:
        threshold = _bounded_number(
            near_duplicate_iou,
            "near_duplicate_iou",
            0.0,
            1.0,
        )
    by_id: dict[str, _MarkCandidate] = {}
    idless: list[_MarkCandidate] = []
    for candidate in selected:
        if not isinstance(candidate, _MarkCandidate):
            raise TypeError("mark candidates must be resolved mark values")
        if not math.isfinite(candidate.priority):
            raise ValueError("mark priority must be finite")
        if candidate.observation_id is None:
            idless.append(candidate)
            continue
        previous = by_id.get(candidate.observation_id)
        if previous is not None and previous != candidate:
            raise ValueError("duplicate observation IDs must resolve to the same mark")
        by_id[candidate.observation_id] = candidate
    ordered = sorted((*by_id.values(), *idless), key=_mark_sort_key)
    retained: list[_MarkCandidate] = []
    dropped = len(selected) - len(ordered)
    for candidate in ordered:
        if threshold is not None and any(
            candidate.region.iou(existing.region) >= threshold for existing in retained
        ):
            dropped += 1
            continue
        retained.append(candidate)
    if not retained:
        raise ValueError("mark deduplication removed every region")
    return tuple(retained), dropped


def _pixel_region(box: Box, width: int, height: int) -> tuple[int, int, int, int]:
    left = min(width - 1, max(0, math.floor(box.x_min * width)))
    top = min(height - 1, max(0, math.floor(box.y_min * height)))
    right = min(width - 1, max(left, math.ceil(box.x_max * width) - 1))
    bottom = min(height - 1, max(top, math.ceil(box.y_max * height) - 1))
    return left, top, right, bottom


def _badge_size(index: int, width: int, height: int) -> tuple[int, int, int]:
    target_height = max(9, min(32, round(min(width, height) * 0.075)))
    target_height = min(target_height, height)
    unit = max(1, (target_height - 4) // 5)
    digits = len(str(index))
    badge_width = 4 + digits * 3 * unit + (digits - 1) * unit
    badge_height = 4 + 5 * unit
    while (badge_width > width or badge_height > height) and unit > 1:
        unit -= 1
        badge_width = 4 + digits * 3 * unit + (digits - 1) * unit
        badge_height = 4 + 5 * unit
    return min(width, badge_width), min(height, badge_height), unit


def _intersects(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> bool:
    return not (
        first[2] < second[0] or second[2] < first[0] or first[3] < second[1] or second[3] < first[1]
    )


def _badge_candidates(
    region: tuple[int, int, int, int],
    badge_width: int,
    badge_height: int,
    image_width: int,
    image_height: int,
) -> tuple[tuple[int, int, int, int], ...]:
    left, top, right, bottom = region
    positions = (
        (left, top),
        (right - badge_width + 1, top),
        (left, top - badge_height),
        (left, bottom + 1),
        (right - badge_width + 1, top - badge_height),
        (right - badge_width + 1, bottom + 1),
        ((left + right - badge_width + 1) // 2, (top + bottom - badge_height + 1) // 2),
    )
    bounded: list[tuple[int, int, int, int]] = []
    for x, y in positions:
        x = min(image_width - badge_width, max(0, x))
        y = min(image_height - badge_height, max(0, y))
        box = (x, y, x + badge_width - 1, y + badge_height - 1)
        if box not in bounded:
            bounded.append(box)
    return tuple(bounded)


def _draw_vector_index(
    draw: ImageDraw.ImageDraw,
    index: int,
    badge: tuple[int, int, int, int],
    unit: int,
    fill: tuple[int, int, int],
) -> None:
    x = badge[0] + 2
    y = badge[1] + 2
    for digit in str(index):
        for row, pattern in enumerate(_DIGITS[digit]):
            for column, enabled in enumerate(pattern):
                if enabled == "1":
                    draw.rectangle(
                        (
                            x + column * unit,
                            y + row * unit,
                            x + (column + 1) * unit - 1,
                            y + (row + 1) * unit - 1,
                        ),
                        fill=fill,
                    )
        x += 4 * unit


def _render_mark_candidates(
    image: PillowImage,
    candidates: Sequence[_MarkCandidate],
    *,
    include_labels: bool,
    near_duplicate_iou: float | None,
    source: Literal["caller_supplied", "observation"],
) -> _MarkRenderResult:
    source_image = _normalized_image(image)
    if not isinstance(include_labels, bool):
        raise TypeError("include_labels must be a boolean")
    normalized, deduplicated = _normalize_mark_candidates(candidates, near_duplicate_iou)
    alpha = source_image.getchannel("A") if source_image.mode == "RGBA" else None
    result = source_image.convert("RGB") if source_image.mode == "RGBA" else source_image.copy()
    draw = ImageDraw.Draw(result)
    occupied: list[tuple[int, int, int, int]] = []
    placements: dict[_MarkCandidate, tuple[int, int, int, int]] = {}
    # Higher-priority observations claim scarce badge space first; all ties are
    # resolved by the canonical spatial key, never by caller ordering.
    for provisional_index, candidate in sorted(
        enumerate(normalized, start=1),
        key=lambda item: (-item[1].priority, _mark_sort_key(item[1])),
    ):
        badge_width, badge_height, _ = _badge_size(
            provisional_index,
            source_image.width,
            source_image.height,
        )
        region = _pixel_region(candidate.region, source_image.width, source_image.height)
        placement = next(
            (
                box
                for box in _badge_candidates(
                    region,
                    badge_width,
                    badge_height,
                    source_image.width,
                    source_image.height,
                )
                if not any(_intersects(box, existing) for existing in occupied)
            ),
            None,
        )
        if placement is not None:
            placements[candidate] = placement
            occupied.append(placement)
    retained = tuple(candidate for candidate in normalized if candidate in placements)
    warnings: list[WarningInfo] = []
    if deduplicated:
        warnings.append(
            WarningInfo(
                code="duplicate_marks_removed",
                message="Duplicate or near-duplicate mark regions were removed.",
                details={"count": deduplicated},
            )
        )
    crowded = len(normalized) - len(retained)
    if crowded:
        warnings.append(
            WarningInfo(
                code="mark_crowding",
                message="Crowded marks without a legible badge placement were dropped.",
                details={"count": crowded},
            )
        )
    if not retained:
        result.close()
        if alpha is not None:
            alpha.close()
        raise ValueError("no mark badge can be placed legibly")
    refs: list[MarkRef] = []
    for index, candidate in enumerate(retained, start=1):
        region = _pixel_region(candidate.region, source_image.width, source_image.height)
        badge = placements[candidate]
        foreground, outline = _local_colors(result, badge[0], badge[1])
        line_width = max(1, min(3, round(min(source_image.size) * 0.006)))
        draw.rectangle(region, outline=outline, width=min(3, line_width + 2))
        draw.rectangle(region, outline=foreground, width=line_width)
        draw.rounded_rectangle(badge, radius=max(1, line_width), fill=foreground, outline=outline)
        _, _, unit = _badge_size(index, source_image.width, source_image.height)
        _draw_vector_index(draw, index, badge, unit, outline)
        label = _sanitize_mark_label(candidate.source_label) if include_labels else None
        if label:
            label_y = min(source_image.height - 1, badge[3] + 1)
            draw.text(
                (badge[0], label_y),
                label,
                font=ImageFont.load_default(),
                fill=foreground,
                stroke_width=1,
                stroke_fill=outline,
            )
        if candidate.observation_id is not None:
            refs.append(
                MarkRef(
                    index=index,
                    observation_id=candidate.observation_id,
                    region=candidate.region,
                    source_label=candidate.source_label,
                )
            )
    if alpha is not None:
        try:
            result.putalpha(alpha)
        finally:
            alpha.close()
    result.info.clear()
    descriptor = _descriptor(
        "set_of_mark",
        {
            "include_labels": include_labels,
            "near_duplicate_iou": near_duplicate_iou,
            "source": source,
            "mark_count": len(retained),
            "dropped_count": deduplicated + crowded,
        },
    )
    return _MarkRenderResult(
        pending=PendingAsset(image=result, transform=descriptor),
        marks=tuple(refs),
        warnings=tuple(warnings),
    )


def mark_regions(
    image: PillowImage,
    regions: Sequence[Box],
    *,
    labels: Sequence[str | None] | None = None,
    include_labels: bool = False,
    near_duplicate_iou: float | None = 0.98,
) -> PendingAsset:
    """Render deterministic caller-supplied regions without creating evidence.

    Raw boxes are explicitly recorded as ``caller_supplied`` transform input.
    They are pixels-only annotations and cannot produce a :class:`MarkPayload`.
    """

    selected_regions = tuple(regions)
    selected_labels = tuple(labels) if labels is not None else (None,) * len(selected_regions)
    if len(selected_labels) != len(selected_regions):
        raise ValueError("labels must contain exactly one value per region")
    candidates = tuple(
        _MarkCandidate(region=region, source_label=_sanitize_mark_label(label))
        for region, label in zip(selected_regions, selected_labels, strict=True)
    )
    return _render_mark_candidates(
        image,
        candidates,
        include_labels=include_labels,
        near_duplicate_iou=near_duplicate_iou,
        source="caller_supplied",
    ).pending


def _mark_observation_regions(
    image: PillowImage,
    candidates: Sequence[_MarkCandidate],
    *,
    near_duplicate_iou: float | None = 0.98,
) -> _MarkRenderResult:
    return _render_mark_candidates(
        image,
        candidates,
        include_labels=False,
        near_duplicate_iou=near_duplicate_iou,
        source="observation",
    )


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
    "mark_regions",
    "rotate",
    "tile",
    "to_grayscale",
]
