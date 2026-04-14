import pytest
from PIL import Image

from penampakan.image.transforms import (
    add_coordinate_grid,
    crop,
    enhance_contrast,
    rotate,
    tile,
    to_grayscale,
)
from penampakan.models import Box
from tests.fixtures.images import quadrants_image, transparent_image


def close_all(*images: Image.Image) -> None:
    for image in images:
        image.close()


def test_crop_uses_pixel_aligned_box_and_records_geometry() -> None:
    source = quadrants_image()
    pending = crop(source, Box(x_min=0.0, y_min=0.0, x_max=0.5, y_max=0.5))

    assert pending.image.size == (4, 4)
    assert pending.image.getcolors() == [(16, (255, 0, 0))]
    assert pending.transform.name == "crop"
    assert pending.transform.parameters["requested_box"] == {
        "x_min": 0.0,
        "y_min": 0.0,
        "x_max": 0.5,
        "y_max": 0.5,
    }
    pending.close()
    source.close()


def test_tiles_cover_image_in_row_major_order() -> None:
    source = quadrants_image()
    pending = tile(source, rows=2, columns=2)

    colors = tuple(item.image.getpixel((0, 0)) for item in pending)

    assert colors == ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0))
    assert tuple(item.transform.parameters["row"] for item in pending) == (0, 0, 1, 1)
    assert tuple(item.transform.parameters["column"] for item in pending) == (0, 1, 0, 1)
    for item in pending:
        item.close()
    source.close()


def test_tile_rounding_remainder_goes_to_right_and_bottom() -> None:
    source = Image.new("RGB", (5, 5), "red")
    pending = tile(source, rows=2, columns=2)

    assert tuple(item.image.size for item in pending) == ((2, 2), (3, 2), (2, 3), (3, 3))
    for item in pending:
        item.close()
    source.close()


def test_rotate_is_clockwise_and_expands_dimensions() -> None:
    source = Image.new("RGB", (2, 3), "black")
    source.putpixel((0, 0), (255, 0, 0))

    pending = rotate(source, 90)

    assert pending.image.size == (3, 2)
    assert pending.image.getpixel((2, 0)) == (255, 0, 0)
    pending.close()
    source.close()


def test_contrast_and_grayscale_preserve_alpha() -> None:
    source = transparent_image()
    contrasted = enhance_contrast(source, factor=2.0)
    grayscale = to_grayscale(source)

    assert contrasted.image.mode == "RGBA"
    assert grayscale.image.mode == "RGBA"
    assert contrasted.image.getchannel("A").tobytes() == source.getchannel("A").tobytes()
    assert grayscale.image.getchannel("A").tobytes() == source.getchannel("A").tobytes()
    red, green, blue, alpha = grayscale.image.getpixel((1, 1))
    assert red == green == blue
    assert alpha == 255
    contrasted.close()
    grayscale.close()
    source.close()


def test_coordinate_grid_preserves_mode_and_records_arguments() -> None:
    source = Image.new("RGB", (80, 80), "white")

    pending = add_coordinate_grid(source, rows=4, columns=4, labels=True)

    assert pending.image.mode == "RGB"
    assert pending.image.tobytes() != source.tobytes()
    assert pending.transform.name == "coordinate_grid"
    assert pending.transform.parameters == {"rows": 4, "columns": 4, "labels": True}
    pending.close()
    source.close()


@pytest.mark.parametrize(
    ("operation", "arguments"),
    [
        (tile, {"rows": 1, "columns": 1}),
        (tile, {"rows": 8, "columns": 8}),
        (tile, {"rows": 2, "columns": 2, "overlap_fraction": 0.51}),
        (rotate, {"degrees": 45}),
        (enhance_contrast, {"factor": 0.24}),
        (add_coordinate_grid, {"rows": 1}),
    ],
)
def test_transform_arguments_are_bounded(operation: object, arguments: dict[str, object]) -> None:
    source = Image.new("RGB", (16, 16), "white")

    with pytest.raises((TypeError, ValueError)):
        operation(source, **arguments)

    source.close()
