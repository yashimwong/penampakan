from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from penampakan.config import ImageLimits
from penampakan.errors import (
    ImageLimitExceededError,
    InvalidImageError,
    RemoteSourceDisabledError,
    UnsupportedImageError,
)
from penampakan.image.loader import load_image
from tests.fixtures.images import animated_webp, encode_image, oriented_jpeg, transparent_image


class NonSeekableStream:
    def __init__(self, content: bytes) -> None:
        self._content = BytesIO(content)

    def read(self, size: int = -1) -> bytes:
        return self._content.read(size)


@pytest.mark.parametrize("format_name", ["PNG", "JPEG", "WEBP"])
def test_loads_supported_encoded_formats(format_name: str) -> None:
    source = Image.new("RGB", (6, 4), "purple")
    encoded = (
        encode_image(source, format_name, lossless=True)
        if format_name == "WEBP"
        else encode_image(source, format_name)
    )

    loaded = load_image(encoded)

    assert loaded.original_format == format_name
    assert loaded.image.size == (6, 4)
    assert loaded.mode == "RGB"
    assert loaded.canonical_png.startswith(b"\x89PNG")
    assert len(loaded.digest_sha256) == 64
    loaded.image.close()
    source.close()


@pytest.mark.parametrize("wrapper", [bytes, bytearray, memoryview])
def test_loads_every_bytes_like_source(wrapper: type[object]) -> None:
    source = Image.new("RGB", (2, 2), "red")
    encoded = encode_image(source)

    loaded = load_image(wrapper(encoded))

    assert loaded.image.size == (2, 2)
    loaded.image.close()
    source.close()


def test_loads_literal_path_and_pathlike(tmp_path: Path) -> None:
    source = Image.new("RGB", (3, 2), "green")
    image_path = tmp_path / "source.png"
    image_path.write_bytes(encode_image(source))

    from_string = load_image(str(image_path))
    from_path = load_image(image_path)

    assert from_string.digest_sha256 == from_path.digest_sha256
    from_string.image.close()
    from_path.image.close()
    source.close()


def test_seekable_stream_position_is_restored() -> None:
    source = Image.new("RGB", (2, 2), "blue")
    stream = BytesIO(encode_image(source))
    position = stream.tell()

    loaded = load_image(stream)

    assert stream.tell() == position
    assert not stream.closed
    loaded.image.close()
    source.close()


def test_non_seekable_stream_is_consumed() -> None:
    source = Image.new("RGB", (2, 2), "blue")
    stream = NonSeekableStream(encode_image(source))

    loaded = load_image(stream)

    assert stream.read() == b""
    loaded.image.close()
    source.close()


@pytest.mark.parametrize(
    "source", ["http://example.test/image.png", "HTTPS://example.test/image.png"]
)
def test_remote_source_is_rejected_without_fetching(source: str) -> None:
    with pytest.raises(RemoteSourceDisabledError):
        load_image(source)


def test_byte_limit_is_enforced_before_decoding() -> None:
    source = Image.new("RGB", (3, 3), "red")
    encoded = encode_image(source)

    loaded = load_image(encoded, ImageLimits(max_input_bytes=len(encoded)))
    with pytest.raises(ImageLimitExceededError):
        load_image(encoded, ImageLimits(max_input_bytes=len(encoded) - 1))

    loaded.image.close()
    source.close()


@pytest.mark.parametrize(
    "limits",
    [
        ImageLimits(max_width=3),
        ImageLimits(max_height=2),
        ImageLimits(max_pixels=11),
    ],
)
def test_dimension_limits_are_enforced(limits: ImageLimits) -> None:
    source = Image.new("RGB", (4, 3), "red")

    with pytest.raises(ImageLimitExceededError):
        load_image(source, limits)

    source.close()


def test_animated_and_unsupported_images_are_rejected() -> None:
    gif = Image.new("RGB", (2, 2), "red")

    with pytest.raises(UnsupportedImageError):
        load_image(animated_webp())
    with pytest.raises(UnsupportedImageError):
        load_image(encode_image(gif, "GIF"))

    gif.close()


@pytest.mark.parametrize(
    ("orientation", "expected_size"),
    [(3, (3, 2)), (6, (2, 3)), (8, (2, 3))],
)
def test_exif_orientation_is_applied(orientation: int, expected_size: tuple[int, int]) -> None:
    loaded = load_image(oriented_jpeg(orientation))

    assert loaded.image.size == expected_size
    loaded.image.close()


def test_alpha_is_preserved_only_when_non_opaque() -> None:
    transparent = transparent_image()
    opaque = Image.new("RGBA", (2, 2), (255, 0, 0, 255))

    transparent_loaded = load_image(transparent)
    opaque_loaded = load_image(opaque)

    assert transparent_loaded.mode == "RGBA"
    assert opaque_loaded.mode == "RGB"
    transparent_loaded.image.close()
    opaque_loaded.image.close()
    transparent.close()
    opaque.close()


def test_caller_pillow_image_is_unchanged_and_metadata_is_removed() -> None:
    source = Image.new("RGB", (2, 2), "red")
    source.info["icc_profile"] = b"profile"
    source.info["private"] = "value"

    loaded = load_image(source)

    assert source.info["private"] == "value"
    assert loaded.image.info == {}
    assert tuple(item.code for item in loaded.warnings) == ("icc_profile_discarded",)
    loaded.image.close()
    source.close()


def test_invalid_or_truncated_input_raises_documented_error() -> None:
    source = Image.new("RGB", (8, 8), "red")
    encoded = encode_image(source)

    with pytest.raises(InvalidImageError):
        load_image(b"not-an-image")
    with pytest.raises(InvalidImageError):
        load_image(encoded[: len(encoded) // 2])

    source.close()
