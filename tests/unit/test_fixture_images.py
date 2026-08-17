from io import BytesIO

from PIL import Image, ImageOps

from tests.fixtures.images import (
    GEOMETRIC_BOXES,
    NON_LATIN_TEXT,
    TEXT_GRID_WORDS,
    blank_image,
    dense_text_image,
    geometric_boxes_image,
    non_latin_text_image,
    oriented_text_grid_jpeg,
    text_grid_image,
)


def test_text_grid_and_orientation_are_deterministic() -> None:
    first = text_grid_image()
    second = text_grid_image()
    assert first.tobytes() == second.tobytes()
    assert TEXT_GRID_WORDS == ("NORTHWEST", "NORTHEAST", "SOUTHWEST", "SOUTHEAST")

    with Image.open(BytesIO(oriented_text_grid_jpeg())) as encoded:
        assert encoded.getexif()[274] == 6
        assert ImageOps.exif_transpose(encoded).size == (640, 400)


def test_dense_blank_and_non_latin_fixtures() -> None:
    dense = dense_text_image(lines=5)
    blank = blank_image()
    non_latin = non_latin_text_image()
    assert dense.size == (900, 164)
    assert blank.getbbox() == (0, 0, 640, 400)
    assert non_latin.getbbox() == (0, 0, 640, 180)
    assert any(ord(character) > 127 for character in NON_LATIN_TEXT)


def test_geometric_boxes_match_drawn_pixels() -> None:
    image, boxes = geometric_boxes_image()
    assert boxes == GEOMETRIC_BOXES
    assert image.getpixel((20, 15)) == (229, 57, 53)
    assert image.getpixel((130, 30)) == (67, 160, 71)
    assert image.getpixel((55, 120)) == (30, 136, 229)
