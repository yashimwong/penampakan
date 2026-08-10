import math
from typing import cast

import pytest

from penampakan.image.geometry import (
    PixelBox,
    box_contains,
    boxes_intersect,
    calculate_crop_geometry,
    clamp_normalized_coordinate,
    expand_box,
    normalized_box_to_pixel_box,
    normalized_box_to_pixels,
    pixel_bounds_to_box,
    pixel_box_to_normalized_box,
    point_is_within,
    remap_box_to_region,
    remap_pixel_box_from_region,
    remap_point_from_region,
    remap_point_to_region,
)
from penampakan.models import Box, Point


def test_pixel_box_value_operations() -> None:
    outer = PixelBox(1, 2, 9, 10)
    inner = PixelBox(3, 4, 7, 8)

    assert outer.width == 8
    assert outer.height == 8
    assert outer.area == 64
    assert outer.as_tuple() == (1, 2, 9, 10)
    assert outer.contains(inner)
    assert outer.intersection(inner) == inner
    assert outer.intersection(PixelBox(9, 2, 10, 3)) is None
    assert inner.translated(1, 2) == PixelBox(4, 6, 8, 10)


@pytest.mark.parametrize(
    ("values", "error"),
    [
        ((True, 0, 1, 1), TypeError),
        ((-1, 0, 1, 1), ValueError),
        ((0, 0, 0, 1), ValueError),
    ],
)
def test_pixel_box_rejects_invalid_bounds(
    values: tuple[int, int, int, int],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        PixelBox(*values)


@pytest.mark.parametrize("offset", [True, 1.5])
def test_pixel_box_rejects_invalid_translation(offset: object) -> None:
    with pytest.raises(TypeError):
        PixelBox(0, 0, 1, 1).translated(cast(int, offset), 0)


def test_normalized_coordinate_and_dimension_validation() -> None:
    with pytest.raises(TypeError):
        clamp_normalized_coordinate(True)
    with pytest.raises(ValueError):
        clamp_normalized_coordinate(math.inf)
    with pytest.raises(ValueError):
        clamp_normalized_coordinate(0.5, tolerance=-1.0)
    with pytest.raises(TypeError):
        normalized_box_to_pixels(Box(x_min=0, y_min=0, x_max=1, y_max=1), True, 1)
    with pytest.raises(ValueError):
        normalized_box_to_pixels(Box(x_min=0, y_min=0, x_max=1, y_max=1), 1, 0)


def test_pixel_conversion_aliases_and_bounds() -> None:
    box = Box(x_min=0.1, y_min=0.2, x_max=0.7, y_max=0.8)
    pixels = PixelBox(1, 2, 7, 8)

    assert normalized_box_to_pixel_box(box, 10, 10) == pixels
    assert normalized_box_to_pixels(box, 10, 10) == pixels
    assert pixel_box_to_normalized_box(pixels, 10, 10) == box
    assert pixel_bounds_to_box(1, 2, 7, 8, 10, 10) == box
    with pytest.raises(ValueError):
        pixel_box_to_normalized_box(PixelBox(0, 0, 11, 1), 10, 10)


def test_normalized_containment_and_intersection() -> None:
    outer = Box(x_min=0.1, y_min=0.1, x_max=0.9, y_max=0.9)
    inner = Box(x_min=0.2, y_min=0.2, x_max=0.8, y_max=0.8)
    touching = Box(x_min=0.9, y_min=0.2, x_max=1.0, y_max=0.8)

    assert box_contains(outer, inner)
    assert boxes_intersect(outer, inner)
    assert not boxes_intersect(outer, touching)
    assert point_is_within(outer, Point(x=0.5, y=0.5))
    assert not point_is_within(outer, Point(x=0.0, y=0.0))
    with pytest.raises(ValueError):
        boxes_intersect(outer, inner, tolerance=-1.0)


def test_point_region_remapping_round_trip_and_rejection() -> None:
    region = Box(x_min=0.2, y_min=0.1, x_max=0.8, y_max=0.9)
    local = Point(x=0.25, y=0.75)

    full = remap_point_from_region(local, region)

    restored = remap_point_to_region(full, region)

    assert restored.x == pytest.approx(local.x)
    assert restored.y == pytest.approx(local.y)
    with pytest.raises(ValueError):
        remap_point_to_region(Point(x=0.0, y=0.0), region)
    with pytest.raises(ValueError):
        remap_box_to_region(Box(x_min=0.0, y_min=0.0, x_max=0.1, y_max=0.1), region)


def test_crop_and_pixel_region_helpers() -> None:
    requested = Box(x_min=0.25, y_min=0.25, x_max=0.75, y_max=0.75)
    expanded = expand_box(requested, 0.5)
    geometry = calculate_crop_geometry(requested, 8, 8, 0.0)

    assert expanded == Box(x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0)
    assert geometry.pixel_box == PixelBox(2, 2, 6, 6)
    assert remap_pixel_box_from_region(PixelBox(1, 1, 3, 3), PixelBox(2, 2, 6, 6), 8, 8) == Box(
        x_min=0.375,
        y_min=0.375,
        x_max=0.625,
        y_max=0.625,
    )
    with pytest.raises(ValueError):
        remap_pixel_box_from_region(PixelBox(0, 0, 5, 1), PixelBox(2, 2, 6, 6), 8, 8)
    with pytest.raises(ValueError):
        expand_box(requested, 0.6)
