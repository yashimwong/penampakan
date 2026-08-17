from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, NamedTuple

from PIL import Image, ImageDraw, ImageFont


class GeometricBox(NamedTuple):
    label: str
    xyxy: tuple[int, int, int, int]


TEXT_GRID_WORDS = ("NORTHWEST", "NORTHEAST", "SOUTHWEST", "SOUTHEAST")
NON_LATIN_TEXT = "مرحبا بالعالم"
GEOMETRIC_BOXES = (
    GeometricBox("red", (20, 15, 100, 75)),
    GeometricBox("green", (130, 30, 225, 105)),
    GeometricBox("blue", (55, 120, 180, 185)),
)


def encode_image(image: Image.Image, format_name: str = "PNG", **options: Any) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format=format_name, **options)
    return buffer.getvalue()


def quadrants_image(size: int = 8) -> Image.Image:
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    midpoint = size // 2
    draw.rectangle((0, 0, midpoint - 1, midpoint - 1), fill=(255, 0, 0))
    draw.rectangle((midpoint, 0, size - 1, midpoint - 1), fill=(0, 255, 0))
    draw.rectangle((0, midpoint, midpoint - 1, size - 1), fill=(0, 0, 255))
    draw.rectangle((midpoint, midpoint, size - 1, size - 1), fill=(255, 255, 0))
    return image


def transparent_image() -> Image.Image:
    image = Image.new("RGBA", (4, 4), (255, 0, 0, 255))
    image.putpixel((0, 0), (0, 0, 255, 0))
    return image


def oriented_jpeg(orientation: int) -> bytes:
    image = Image.new("RGB", (3, 2), "black")
    image.putpixel((0, 0), (255, 0, 0))
    image.putpixel((2, 1), (0, 255, 0))
    exif = Image.Exif()
    exif[274] = orientation
    return encode_image(image, "JPEG", exif=exif, quality=100, subsampling=0)


def animated_webp() -> bytes:
    first = Image.new("RGB", (2, 2), "red")
    second = Image.new("RGB", (2, 2), "blue")
    return encode_image(
        first,
        "WEBP",
        save_all=True,
        append_images=[second],
        duration=100,
        loop=0,
        lossless=True,
    )


def fixture_font(size: int = 28, path: str | Path | None = None) -> ImageFont.FreeTypeFont:
    if path is None:
        return ImageFont.load_default(size=size)
    return ImageFont.truetype(str(path), size=size)


def text_grid_image(
    size: tuple[int, int] = (640, 400),
    *,
    font_path: str | Path | None = None,
) -> Image.Image:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    font = fixture_font(28, font_path)
    positions = (
        (24, 28),
        (size[0] // 2 + 24, 28),
        (24, size[1] // 2 + 28),
        (size[0] // 2 + 24, size[1] // 2 + 28),
    )
    for word, position in zip(TEXT_GRID_WORDS, positions, strict=True):
        draw.text(position, word, fill="black", font=font)
    return image


def oriented_text_grid_jpeg(
    *,
    font_path: str | Path | None = None,
) -> bytes:
    source = text_grid_image(font_path=font_path)
    image = source.transpose(Image.Transpose.ROTATE_90)
    source.close()
    exif = Image.Exif()
    exif[274] = 6
    try:
        return encode_image(image, "JPEG", exif=exif, quality=95, subsampling=0)
    finally:
        image.close()


def dense_text_image(
    lines: int = 36,
    *,
    width: int = 900,
    font_path: str | Path | None = None,
) -> Image.Image:
    font = fixture_font(20, font_path)
    line_height = 28
    image = Image.new("RGB", (width, lines * line_height + 24), "white")
    draw = ImageDraw.Draw(image)
    for index in range(lines):
        text = f"LINE {index:02d} DETERMINISTIC DENSE TEXT FOR TRUNCATION"
        draw.text((18, 12 + index * line_height), text, fill="black", font=font)
    return image


def blank_image(size: tuple[int, int] = (640, 400)) -> Image.Image:
    return Image.new("RGB", size, "white")


def non_latin_text_image(
    text: str = NON_LATIN_TEXT,
    *,
    font_path: str | Path | None = None,
) -> Image.Image:
    font = fixture_font(42, font_path)
    image = Image.new("RGB", (640, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.text((600, 55), text, fill="black", font=font, anchor="ra")
    return image


def geometric_boxes_image(
    size: tuple[int, int] = (260, 210),
) -> tuple[Image.Image, tuple[GeometricBox, ...]]:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    colors = {"red": "#e53935", "green": "#43a047", "blue": "#1e88e5"}
    for box in GEOMETRIC_BOXES:
        draw.rectangle(box.xyxy, fill=colors[box.label])
    return image, GEOMETRIC_BOXES
