"""Deterministic normalized and pixel geometry operations."""

from __future__ import annotations

import math
from dataclasses import dataclass

from penampakan.models import Box, Point

NORMALIZED_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class PixelBox:
    """A non-empty rectangle using exclusive right and bottom pixel bounds."""

    left: int
    top: int
    right: int
    bottom: int

    def __post_init__(self) -> None:
        values = (self.left, self.top, self.right, self.bottom)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("pixel coordinates must be integers")
        if self.left < 0 or self.top < 0:
            raise ValueError("pixel coordinates cannot be negative")
        if self.left >= self.right or self.top >= self.bottom:
            raise ValueError("pixel bounds must have positive width and height")

    @property
    def width(self) -> int:
        """Return the pixel width."""
        return self.right - self.left

    @property
    def height(self) -> int:
        """Return the pixel height."""
        return self.bottom - self.top

    @property
    def area(self) -> int:
        """Return the pixel area."""
        return self.width * self.height

    def as_tuple(self) -> tuple[int, int, int, int]:
        """Return bounds in Pillow crop order."""
        return self.left, self.top, self.right, self.bottom

    def intersection(self, other: PixelBox) -> PixelBox | None:
        """Return the shared pixels, or none when the rectangles do not overlap."""
        left = max(self.left, other.left)
        top = max(self.top, other.top)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)
        if left >= right or top >= bottom:
            return None
        return PixelBox(left=left, top=top, right=right, bottom=bottom)

    def contains(self, other: PixelBox) -> bool:
        """Return whether this rectangle fully contains another rectangle."""
        return (
            self.left <= other.left
            and self.top <= other.top
            and self.right >= other.right
            and self.bottom >= other.bottom
        )

    def translated(self, x: int, y: int) -> PixelBox:
        """Return a rectangle translated by an integer pixel offset."""
        if isinstance(x, bool) or isinstance(y, bool):
            raise TypeError("pixel offsets must be integers")
        if not isinstance(x, int) or not isinstance(y, int):
            raise TypeError("pixel offsets must be integers")
        return PixelBox(
            left=self.left + x,
            top=self.top + y,
            right=self.right + x,
            bottom=self.bottom + y,
        )


@dataclass(frozen=True, slots=True)
class CropGeometry:
    """The requested, expanded, and pixel-aligned geometry of a crop."""

    requested_box: Box
    expanded_box: Box
    applied_box: Box
    pixel_box: PixelBox
    padding_fraction: float


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _tolerance(value: float) -> float:
    tolerance = _finite(value, "tolerance")
    if tolerance < 0.0:
        raise ValueError("tolerance cannot be negative")
    return tolerance


def _dimension(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def clamp_normalized_coordinate(
    value: float,
    *,
    tolerance: float = NORMALIZED_TOLERANCE,
) -> float:
    """Clamp a normalized coordinate whose overshoot is within tolerance."""
    coordinate = _finite(value, "coordinate")
    allowed = _tolerance(tolerance)
    if coordinate < -allowed or coordinate > 1.0 + allowed:
        raise ValueError("normalized coordinate exceeds the allowed tolerance")
    return min(1.0, max(0.0, coordinate))


def clamp_normalized_box(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    *,
    tolerance: float = NORMALIZED_TOLERANCE,
) -> Box:
    """Construct a strict box after tolerance-limited boundary clamping."""
    return Box(
        x_min=clamp_normalized_coordinate(x_min, tolerance=tolerance),
        y_min=clamp_normalized_coordinate(y_min, tolerance=tolerance),
        x_max=clamp_normalized_coordinate(x_max, tolerance=tolerance),
        y_max=clamp_normalized_coordinate(y_max, tolerance=tolerance),
    )


def clamp_normalized_point(
    x: float,
    y: float,
    *,
    tolerance: float = NORMALIZED_TOLERANCE,
) -> Point:
    """Construct a strict point after tolerance-limited boundary clamping."""
    return Point(
        x=clamp_normalized_coordinate(x, tolerance=tolerance),
        y=clamp_normalized_coordinate(y, tolerance=tolerance),
    )


def box_to_pixels(box: Box, width: int, height: int) -> PixelBox:
    """Convert a normalized box to deterministic Pillow crop bounds."""
    image_width = _dimension(width, "width")
    image_height = _dimension(height, "height")
    left = math.floor(box.x_min * image_width)
    top = math.floor(box.y_min * image_height)
    right = math.ceil(box.x_max * image_width)
    bottom = math.ceil(box.y_max * image_height)
    left = min(image_width, max(0, left))
    top = min(image_height, max(0, top))
    right = min(image_width, max(0, right))
    bottom = min(image_height, max(0, bottom))
    return PixelBox(left=left, top=top, right=right, bottom=bottom)


def normalized_box_to_pixel_box(box: Box, width: int, height: int) -> PixelBox:
    """Convert a normalized box to deterministic exclusive pixel bounds."""
    return box_to_pixels(box, width, height)


def normalized_box_to_pixels(box: Box, width: int, height: int) -> PixelBox:
    """Convert a normalized box to deterministic exclusive pixel bounds."""
    return box_to_pixels(box, width, height)


def pixels_to_box(pixel_box: PixelBox, width: int, height: int) -> Box:
    """Convert valid exclusive pixel bounds to a normalized box."""
    image_width = _dimension(width, "width")
    image_height = _dimension(height, "height")
    if pixel_box.right > image_width or pixel_box.bottom > image_height:
        raise ValueError("pixel box lies outside the image")
    return Box(
        x_min=pixel_box.left / image_width,
        y_min=pixel_box.top / image_height,
        x_max=pixel_box.right / image_width,
        y_max=pixel_box.bottom / image_height,
    )


def pixel_box_to_normalized_box(pixel_box: PixelBox, width: int, height: int) -> Box:
    """Convert exclusive pixel bounds to normalized full-image coordinates."""
    return pixels_to_box(pixel_box, width, height)


def pixel_bounds_to_box(
    left: int,
    top: int,
    right: int,
    bottom: int,
    width: int,
    height: int,
) -> Box:
    """Validate raw pixel bounds and convert them to normalized coordinates."""
    return pixels_to_box(
        PixelBox(left=left, top=top, right=right, bottom=bottom),
        width,
        height,
    )


def boxes_intersect(first: Box, second: Box, *, tolerance: float = 0.0) -> bool:
    """Return whether two normalized boxes overlap beyond a tolerance."""
    allowed = _tolerance(tolerance)
    return (
        min(first.x_max, second.x_max) - max(first.x_min, second.x_min) > allowed
        and min(first.y_max, second.y_max) - max(first.y_min, second.y_min) > allowed
    )


def box_contains(
    outer: Box,
    inner: Box,
    *,
    tolerance: float = NORMALIZED_TOLERANCE,
) -> bool:
    """Return whether a box lies within another box under boundary tolerance."""
    allowed = _tolerance(tolerance)
    return (
        inner.x_min >= outer.x_min - allowed
        and inner.y_min >= outer.y_min - allowed
        and inner.x_max <= outer.x_max + allowed
        and inner.y_max <= outer.y_max + allowed
    )


def point_is_within(
    outer: Box,
    point: Point,
    *,
    tolerance: float = NORMALIZED_TOLERANCE,
) -> bool:
    """Return whether a point lies within a box under boundary tolerance."""
    allowed = _tolerance(tolerance)
    return (
        outer.x_min - allowed <= point.x <= outer.x_max + allowed
        and outer.y_min - allowed <= point.y <= outer.y_max + allowed
    )


def remap_box_from_region(box: Box, region: Box) -> Box:
    """Map region-local normalized coordinates into full-asset coordinates."""
    region_width = region.x_max - region.x_min
    region_height = region.y_max - region.y_min
    return clamp_normalized_box(
        region.x_min + box.x_min * region_width,
        region.y_min + box.y_min * region_height,
        region.x_min + box.x_max * region_width,
        region.y_min + box.y_max * region_height,
    )


def remap_point_from_region(point: Point, region: Box) -> Point:
    """Map a region-local normalized point into full-asset coordinates."""
    region_width = region.x_max - region.x_min
    region_height = region.y_max - region.y_min
    return clamp_normalized_point(
        region.x_min + point.x * region_width,
        region.y_min + point.y * region_height,
    )


def remap_box_to_region(
    box: Box,
    region: Box,
    *,
    tolerance: float = NORMALIZED_TOLERANCE,
) -> Box:
    """Map a contained full-asset box into region-local coordinates."""
    allowed = _tolerance(tolerance)
    if not box_contains(region, box, tolerance=allowed):
        raise ValueError("box lies outside the requested region")
    x_min = min(region.x_max, max(region.x_min, box.x_min))
    y_min = min(region.y_max, max(region.y_min, box.y_min))
    x_max = min(region.x_max, max(region.x_min, box.x_max))
    y_max = min(region.y_max, max(region.y_min, box.y_max))
    region_width = region.x_max - region.x_min
    region_height = region.y_max - region.y_min
    return clamp_normalized_box(
        (x_min - region.x_min) / region_width,
        (y_min - region.y_min) / region_height,
        (x_max - region.x_min) / region_width,
        (y_max - region.y_min) / region_height,
        tolerance=allowed,
    )


def remap_point_to_region(
    point: Point,
    region: Box,
    *,
    tolerance: float = NORMALIZED_TOLERANCE,
) -> Point:
    """Map a contained full-asset point into region-local coordinates."""
    allowed = _tolerance(tolerance)
    if not point_is_within(region, point, tolerance=allowed):
        raise ValueError("point lies outside the requested region")
    x = min(region.x_max, max(region.x_min, point.x))
    y = min(region.y_max, max(region.y_min, point.y))
    return clamp_normalized_point(
        (x - region.x_min) / (region.x_max - region.x_min),
        (y - region.y_min) / (region.y_max - region.y_min),
        tolerance=allowed,
    )


def remap_pixel_box_from_region(
    pixel_box: PixelBox,
    region_pixels: PixelBox,
    asset_width: int,
    asset_height: int,
) -> Box:
    """Map crop-local pixel bounds directly into full-asset coordinates."""
    if pixel_box.right > region_pixels.width or pixel_box.bottom > region_pixels.height:
        raise ValueError("pixel box lies outside the cropped region")
    translated = pixel_box.translated(region_pixels.left, region_pixels.top)
    return pixels_to_box(translated, asset_width, asset_height)


def expand_box(box: Box, padding_fraction: float = 0.0) -> Box:
    """Expand every box side by a fraction of its width or height."""
    padding = _finite(padding_fraction, "padding_fraction")
    if padding < 0.0 or padding > 0.5:
        raise ValueError("padding_fraction must be between 0 and 0.5")
    x_padding = (box.x_max - box.x_min) * padding
    y_padding = (box.y_max - box.y_min) * padding
    return Box(
        x_min=max(0.0, box.x_min - x_padding),
        y_min=max(0.0, box.y_min - y_padding),
        x_max=min(1.0, box.x_max + x_padding),
        y_max=min(1.0, box.y_max + y_padding),
    )


def build_crop_geometry(
    box: Box,
    width: int,
    height: int,
    padding_fraction: float = 0.0,
) -> CropGeometry:
    """Resolve requested crop geometry to the exact pixels that will be applied."""
    expanded = expand_box(box, padding_fraction)
    pixels = box_to_pixels(expanded, width, height)
    applied = pixels_to_box(pixels, width, height)
    return CropGeometry(
        requested_box=box,
        expanded_box=expanded,
        applied_box=applied,
        pixel_box=pixels,
        padding_fraction=float(padding_fraction),
    )


def calculate_crop_geometry(
    box: Box,
    width: int,
    height: int,
    padding_fraction: float = 0.0,
) -> CropGeometry:
    """Resolve a padded normalized crop into exact applied geometry."""
    return build_crop_geometry(box, width, height, padding_fraction)


__all__ = [
    "NORMALIZED_TOLERANCE",
    "CropGeometry",
    "PixelBox",
    "box_contains",
    "box_to_pixels",
    "boxes_intersect",
    "build_crop_geometry",
    "calculate_crop_geometry",
    "clamp_normalized_box",
    "clamp_normalized_coordinate",
    "clamp_normalized_point",
    "expand_box",
    "normalized_box_to_pixel_box",
    "normalized_box_to_pixels",
    "pixel_bounds_to_box",
    "pixel_box_to_normalized_box",
    "pixels_to_box",
    "point_is_within",
    "remap_box_from_region",
    "remap_box_to_region",
    "remap_pixel_box_from_region",
    "remap_point_from_region",
    "remap_point_to_region",
]
