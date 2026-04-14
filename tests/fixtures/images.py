from __future__ import annotations

from io import BytesIO
from typing import Any

from PIL import Image, ImageDraw


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
