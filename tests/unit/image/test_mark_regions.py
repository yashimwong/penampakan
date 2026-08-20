from io import BytesIO

import pytest
from PIL import Image

from penampakan.image.assets import canonical_png_bytes
from penampakan.image.transforms import mark_regions
from penampakan.models import Box


def _box(x_min: float, y_min: float, x_max: float, y_max: float) -> Box:
    return Box(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


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
