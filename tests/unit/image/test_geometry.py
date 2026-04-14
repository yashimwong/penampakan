import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from penampakan.image.geometry import (
    NORMALIZED_TOLERANCE,
    PixelBox,
    box_to_pixels,
    build_crop_geometry,
    clamp_normalized_box,
    pixels_to_box,
    remap_box_from_region,
    remap_box_to_region,
)
from penampakan.models import Box


def test_normalized_box_uses_floor_and_ceil_pixel_bounds() -> None:
    box = Box(x_min=0.11, y_min=0.21, x_max=0.61, y_max=0.81)

    pixels = box_to_pixels(box, width=10, height=10)

    assert pixels == PixelBox(left=1, top=2, right=7, bottom=9)


def test_boundary_boxes_clamp_to_image_extent() -> None:
    box = Box(x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0)

    assert box_to_pixels(box, width=7, height=5) == PixelBox(0, 0, 7, 5)


def test_tolerance_clamps_only_small_coordinate_overshoot() -> None:
    clamped = clamp_normalized_box(
        -NORMALIZED_TOLERANCE,
        0.0,
        1.0 + NORMALIZED_TOLERANCE,
        1.0,
    )

    assert clamped == Box(x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0)
    with pytest.raises(ValueError):
        clamp_normalized_box(-2 * NORMALIZED_TOLERANCE, 0.0, 1.0, 1.0)


def test_crop_geometry_records_requested_expanded_and_applied_boxes() -> None:
    requested = Box(x_min=0.25, y_min=0.25, x_max=0.5, y_max=0.5)

    geometry = build_crop_geometry(requested, width=10, height=10, padding_fraction=0.5)

    assert geometry.requested_box == requested
    assert geometry.expanded_box == Box(x_min=0.125, y_min=0.125, x_max=0.625, y_max=0.625)
    assert geometry.pixel_box == PixelBox(1, 1, 7, 7)
    assert geometry.applied_box == Box(x_min=0.1, y_min=0.1, x_max=0.7, y_max=0.7)


def test_region_remapping_round_trip() -> None:
    region = Box(x_min=0.2, y_min=0.1, x_max=0.8, y_max=0.9)
    local = Box(x_min=0.25, y_min=0.25, x_max=0.75, y_max=0.75)

    full = remap_box_from_region(local, region)
    restored = remap_box_to_region(full, region)

    assert restored.x_min == pytest.approx(local.x_min)
    assert restored.y_min == pytest.approx(local.y_min)
    assert restored.x_max == pytest.approx(local.x_max)
    assert restored.y_max == pytest.approx(local.y_max)


@given(
    width=st.integers(min_value=1, max_value=10_000),
    height=st.integers(min_value=1, max_value=10_000),
    x_min=st.floats(min_value=0.0, max_value=0.49, allow_nan=False, allow_infinity=False),
    y_min=st.floats(min_value=0.0, max_value=0.49, allow_nan=False, allow_infinity=False),
    x_max=st.floats(min_value=0.51, max_value=1.0, allow_nan=False, allow_infinity=False),
    y_max=st.floats(min_value=0.51, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_pixel_conversion_preserves_non_empty_in_bounds_geometry(
    width: int,
    height: int,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> None:
    box = Box(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)

    pixels = box_to_pixels(box, width, height)
    restored = pixels_to_box(pixels, width, height)

    assert 0 <= pixels.left < pixels.right <= width
    assert 0 <= pixels.top < pixels.bottom <= height
    assert restored.x_min <= box.x_min or math.isclose(restored.x_min, box.x_min)
    assert restored.y_min <= box.y_min or math.isclose(restored.y_min, box.y_min)
    assert restored.x_max >= box.x_max or math.isclose(restored.x_max, box.x_max)
    assert restored.y_max >= box.y_max or math.isclose(restored.y_max, box.y_max)
