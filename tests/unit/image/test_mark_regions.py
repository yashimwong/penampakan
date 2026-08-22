from io import BytesIO

import pytest
from PIL import Image, ImageFont

from penampakan.image import transforms
from penampakan.image.assets import canonical_png_bytes
from penampakan.image.transforms import (
    _mark_observation_regions,
    _MarkCandidate,
    mark_regions,
)
from penampakan.models import Box


def _box(x_min: float, y_min: float, x_max: float, y_max: float) -> Box:
    return Box(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def _textured_image(width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height))
    image.putdata(
        tuple(
            ((x * 13 + y * 7) % 256, (x * 5 + y * 17) % 256, (x * 11 + y * 3) % 256)
            for y in range(height)
            for x in range(width)
        )
    )
    return image


def _grid_boxes(columns: int, rows: int, count: int) -> tuple[Box, ...]:
    boxes: list[Box] = []
    for index in range(count):
        column = index % columns
        row = index // columns
        x_min = (column + 0.15) / columns
        y_min = (row + 0.15) / rows
        x_max = (column + 0.85) / columns
        y_max = (row + 0.85) / rows
        boxes.append(_box(x_min, y_min, x_max, y_max))
    return tuple(boxes)


def test_mark_regions_is_independent_of_caller_order() -> None:
    source = Image.new("RGB", (160, 120), (96, 128, 160))
    regions = (
        _box(0.55, 0.55, 0.9, 0.9),
        _box(0.05, 0.05, 0.35, 0.35),
        _box(0.55, 0.05, 0.9, 0.35),
    )

    first = mark_regions(source, regions)
    second = mark_regions(source, tuple(reversed(regions)))

    assert first.transform == second.transform
    assert first.image.tobytes() == second.image.tobytes()
    assert canonical_png_bytes(first.image) == canonical_png_bytes(second.image)
    first.close()
    second.close()
    source.close()


def test_mark_regions_deduplicates_exact_and_near_duplicate_boxes() -> None:
    source = Image.new("RGB", (160, 120), "white")
    region = _box(0.1, 0.1, 0.4, 0.4)
    near_duplicate = _box(0.11, 0.11, 0.41, 0.41)

    pending = mark_regions(
        source,
        (near_duplicate, region, region),
        near_duplicate_iou=0.85,
    )

    assert pending.transform.parameters["mark_count"] == 1
    assert pending.transform.parameters["dropped_count"] == 2
    assert pending.transform.parameters["near_duplicate_iou"] == 0.85
    pending.close()
    source.close()


def test_mark_regions_preserves_rgba_alpha_and_canonical_png_mode() -> None:
    source = Image.new("RGBA", (64, 48), (40, 80, 120, 0))
    alpha = Image.new("L", source.size)
    alpha.putdata(tuple((x * 7 + y * 11) % 256 for y in range(48) for x in range(64)))
    source.putalpha(alpha)
    expected_alpha = source.getchannel("A").tobytes()

    pending = mark_regions(source, (_box(0.0, 0.0, 1.0, 1.0),))
    encoded = canonical_png_bytes(pending.image)
    decoded = Image.open(BytesIO(encoded))
    decoded.load()

    assert pending.image.mode == "RGBA"
    assert pending.image.getchannel("A").tobytes() == expected_alpha
    assert decoded.mode == "RGBA"
    assert decoded.getchannel("A").tobytes() == expected_alpha
    decoded.close()
    pending.close()
    alpha.close()
    source.close()


def test_mark_regions_records_caller_supplied_descriptor() -> None:
    source = Image.new("RGB", (80, 80), "black")

    pending = mark_regions(
        source,
        (_box(0.2, 0.2, 0.8, 0.8),),
        labels=("untrusted label",),
    )

    assert pending.transform.name == "set_of_mark"
    assert pending.transform.parameters == {
        "include_labels": False,
        "near_duplicate_iou": 0.98,
        "source": "caller_supplied",
        "mark_count": 1,
        "dropped_count": 0,
    }
    pending.close()
    source.close()


def test_mark_regions_drops_marks_when_badges_cannot_fit_without_overlap() -> None:
    source = Image.new("RGB", (9, 9), "white")

    pending = mark_regions(
        source,
        (
            _box(0.0, 0.0, 0.2, 0.2),
            _box(0.8, 0.8, 1.0, 1.0),
        ),
        near_duplicate_iou=None,
    )

    assert pending.transform.parameters["mark_count"] == 1
    assert pending.transform.parameters["dropped_count"] == 1
    pending.close()
    source.close()


def test_mark_region_labels_are_sanitized_and_truncated_before_rendering() -> None:
    source = Image.new("RGB", (240, 120), "white")
    region = (_box(0.1, 0.1, 0.8, 0.8),)

    unsafe = mark_regions(
        source,
        region,
        labels=("\t cat\nname\x01 ",),
        include_labels=True,
    )
    sanitized = mark_regions(source, region, labels=("cat name",), include_labels=True)
    overlong = mark_regions(source, region, labels=("x" * 40,), include_labels=True)
    truncated = mark_regions(source, region, labels=("x" * 24,), include_labels=True)

    assert unsafe.image.tobytes() == sanitized.image.tobytes()
    assert overlong.image.tobytes() == truncated.image.tobytes()
    unsafe.close()
    sanitized.close()
    overlong.close()
    truncated.close()
    source.close()


def test_mark_region_labels_are_not_rendered_by_default() -> None:
    source = Image.new("RGB", (160, 100), "white")
    region = (_box(0.1, 0.1, 0.8, 0.8),)

    first = mark_regions(source, region, labels=("first label",))
    second = mark_regions(source, region, labels=("different label",))

    assert first.image.tobytes() == second.image.tobytes()
    first.close()
    second.close()
    source.close()


@pytest.mark.parametrize(
    ("regions", "kwargs"),
    [
        ((), {}),
        ((_box(0.1, 0.1, 0.9, 0.9),) * 100, {}),
        ((_box(0.1, 0.1, 0.9, 0.9),), {"near_duplicate_iou": -0.01}),
        ((_box(0.1, 0.1, 0.9, 0.9),), {"near_duplicate_iou": 1.01}),
        ((_box(0.1, 0.1, 0.9, 0.9),), {"near_duplicate_iou": float("nan")}),
        ((_box(0.1, 0.1, 0.9, 0.9),), {"include_labels": 1}),
    ],
)
def test_mark_regions_enforces_public_bounds(
    regions: tuple[Box, ...], kwargs: dict[str, object]
) -> None:
    source = Image.new("RGB", (40, 40), "white")

    with pytest.raises((TypeError, ValueError)):
        mark_regions(source, regions, **kwargs)

    source.close()


def test_mark_regions_requires_exactly_one_label_per_region() -> None:
    source = Image.new("RGB", (40, 40), "white")

    with pytest.raises(ValueError, match="exactly one"):
        mark_regions(source, (_box(0.1, 0.1, 0.9, 0.9),), labels=())

    source.close()


@pytest.mark.parametrize("color", [(8, 8, 8), (247, 247, 247)])
def test_mark_regions_render_on_dark_and_light_backgrounds(color: tuple[int, int, int]) -> None:
    source = Image.new("RGB", (160, 120), color)
    regions = (_box(0.1, 0.1, 0.45, 0.45), _box(0.55, 0.55, 0.9, 0.9))

    first = mark_regions(source, regions)
    second = mark_regions(source, regions)

    assert first.transform.parameters["mark_count"] == 2
    # Auto-contrast means the rendered badges differ from the flat background.
    assert first.image.convert("RGB").tobytes() != source.tobytes()
    assert canonical_png_bytes(first.image) == canonical_png_bytes(second.image)
    first.close()
    second.close()
    source.close()


def test_mark_regions_render_deterministically_on_a_textured_background() -> None:
    source = _textured_image(160, 120)
    regions = (_box(0.2, 0.2, 0.5, 0.5), _box(0.55, 0.1, 0.95, 0.6))

    first = mark_regions(source, regions)
    second = mark_regions(source, tuple(reversed(regions)))
    encoded = canonical_png_bytes(first.image)
    decoded = Image.open(BytesIO(encoded))
    decoded.load()

    assert first.transform.parameters["mark_count"] == 2
    assert canonical_png_bytes(first.image) == canonical_png_bytes(second.image)
    assert decoded.mode == "RGB"
    assert canonical_png_bytes(decoded) == encoded
    decoded.close()
    first.close()
    second.close()
    source.close()


def test_mark_regions_place_badges_in_bounds_for_every_edge_and_corner() -> None:
    source = Image.new("RGB", (200, 200), (120, 120, 120))
    regions = (
        _box(0.0, 0.0, 0.14, 0.14),
        _box(0.86, 0.0, 1.0, 0.14),
        _box(0.0, 0.86, 0.14, 1.0),
        _box(0.86, 0.86, 1.0, 1.0),
        _box(0.43, 0.0, 0.57, 0.1),
        _box(0.43, 0.9, 0.57, 1.0),
        _box(0.0, 0.43, 0.1, 0.57),
        _box(0.9, 0.43, 1.0, 0.57),
    )

    pending = mark_regions(source, regions, near_duplicate_iou=None)

    # Every edge-hugging region is numbered; nothing is dropped for going out of bounds.
    assert pending.transform.parameters["mark_count"] == len(regions)
    assert pending.transform.parameters["dropped_count"] == 0
    assert pending.image.size == source.size
    pending.close()
    source.close()


def test_mark_regions_keep_distinct_overlapping_boxes() -> None:
    source = Image.new("RGB", (200, 160), "white")
    regions = (_box(0.1, 0.1, 0.6, 0.6), _box(0.35, 0.35, 0.85, 0.85))

    pending = mark_regions(source, regions)

    # Overlap alone is not near-duplication: both boxes survive with distinct marks.
    assert pending.transform.parameters["mark_count"] == 2
    assert pending.transform.parameters["dropped_count"] == 0
    pending.close()
    source.close()


def test_mark_regions_render_tiny_regions_with_the_badge_size_floor() -> None:
    source = Image.new("RGB", (300, 300), "white")
    regions = (_box(0.1, 0.1, 0.12, 0.12), _box(0.8, 0.8, 0.82, 0.82))

    pending = mark_regions(source, regions, near_duplicate_iou=None)

    assert pending.transform.parameters["mark_count"] == 2
    pending.close()
    source.close()


def test_mark_regions_number_ninety_nine_regions_when_space_allows() -> None:
    source = Image.new("RGB", (1000, 1000), (200, 200, 200))
    regions = _grid_boxes(10, 10, 99)

    pending = mark_regions(source, regions, near_duplicate_iou=None)

    assert pending.transform.parameters["mark_count"] == 99
    assert pending.transform.parameters["dropped_count"] == 0
    pending.close()
    source.close()


def test_mark_regions_drop_crowded_marks_deterministically_at_ninety_nine() -> None:
    source = Image.new("RGB", (70, 70), "white")
    regions = _grid_boxes(10, 10, 99)

    first = mark_regions(source, regions, near_duplicate_iou=None)
    second = mark_regions(source, regions, near_duplicate_iou=None)

    mark_count = first.transform.parameters["mark_count"]
    dropped = first.transform.parameters["dropped_count"]
    assert 1 <= mark_count < 99
    assert dropped == 99 - mark_count
    assert canonical_png_bytes(first.image) == canonical_png_bytes(second.image)
    first.close()
    second.close()
    source.close()


def test_mark_regions_render_indices_without_a_font(monkeypatch: pytest.MonkeyPatch) -> None:
    def _no_font(*args: object, **kwargs: object) -> ImageFont.ImageFont:
        raise OSError("no font available")

    monkeypatch.setattr(transforms.ImageFont, "load_default", _no_font)
    source = Image.new("RGB", (160, 120), "white")

    # Indices are drawn from bundled vector digits, so default rendering never
    # touches the font loader even when it is unavailable.
    pending = mark_regions(source, (_box(0.2, 0.2, 0.8, 0.8),))

    assert pending.transform.parameters["mark_count"] == 1
    pending.close()
    source.close()


def test_mark_observation_regions_drop_lowest_priority_when_crowded() -> None:
    source = Image.new("RGB", (9, 9), "white")
    keep = _MarkCandidate(
        region=_box(0.0, 0.0, 0.2, 0.2),
        observation_id="obs_000001",
        priority=0.9,
    )
    drop = _MarkCandidate(
        region=_box(0.8, 0.8, 1.0, 1.0),
        observation_id="obs_000002",
        priority=0.1,
    )

    result = _mark_observation_regions(source, (keep, drop), near_duplicate_iou=None)

    assert result.pending.transform.parameters["mark_count"] == 1
    assert result.pending.transform.parameters["dropped_count"] == 1
    # The surviving mark is the higher-priority observation, and the drop is warned.
    assert tuple(ref.observation_id for ref in result.marks) == ("obs_000001",)
    assert any(warning.code == "mark_crowding" for warning in result.warnings)
    result.pending.close()
    source.close()
