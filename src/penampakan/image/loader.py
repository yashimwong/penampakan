"""Safe loading and canonical normalization of caller-provided images."""

from __future__ import annotations

import os
import warnings
from contextlib import suppress
from dataclasses import dataclass, field
from io import BytesIO
from typing import Literal, cast

from PIL import Image, ImageOps, UnidentifiedImageError

from ..config import ImageLimits
from ..errors import (
    ImageLimitExceededError,
    InvalidImageError,
    PenampakanError,
    RemoteSourceDisabledError,
    UnsupportedImageError,
)
from ..models import ImageSource, WarningInfo
from .canonical import canonical_digest, encode_canonical_png

_SUPPORTED_FORMATS = frozenset({"PNG", "JPEG", "WEBP"})
_READ_CHUNK_SIZE = 64 * 1024


@dataclass(frozen=True, slots=True)
class LoadedImage:
    """An owned normalized image and its deterministic canonical representation."""

    image: Image.Image = field(repr=False, compare=False)
    canonical_png: bytes = field(repr=False)
    digest_sha256: str
    original_format: Literal["PNG", "JPEG", "WEBP"] | None
    warnings: tuple[WarningInfo, ...] = ()

    @property
    def width(self) -> int:
        """Return the post-orientation image width."""

        return self.image.width

    @property
    def height(self) -> int:
        """Return the post-orientation image height."""

        return self.image.height

    @property
    def mode(self) -> Literal["RGB", "RGBA"]:
        """Return the canonical image mode."""

        return cast(Literal["RGB", "RGBA"], self.image.mode)

    def close(self) -> None:
        """Release the owned normalized image idempotently."""

        self.image.close()


def load_image(source: ImageSource, limits: ImageLimits | None = None) -> LoadedImage:
    """Load an image source into an owned canonical RGB or RGBA representation."""

    active_limits = limits if limits is not None else ImageLimits()
    source_kind = _validate_source(source)
    if source_kind == "pillow":
        image = cast(Image.Image, source)
        return _load_pillow_image(image, active_limits)
    encoded = _read_encoded_source(source, source_kind, active_limits.max_input_bytes)
    return _load_encoded_image(encoded, active_limits)


def _validate_source(
    source: object,
) -> Literal["pillow", "bytes", "path", "stream"]:
    if isinstance(source, Image.Image):
        return "pillow"
    if isinstance(source, (bytes, bytearray, memoryview)):
        return "bytes"
    if isinstance(source, str):
        lowered = source.casefold()
        if lowered.startswith("http://") or lowered.startswith("https://"):
            raise RemoteSourceDisabledError(code="remote_source_disabled")
        return "path"
    if isinstance(source, os.PathLike):
        try:
            path_value = os.fspath(source)
        except (TypeError, ValueError, OSError) as error:
            raise InvalidImageError(code="invalid_image_source", cause=error) from error
        if not isinstance(path_value, str):
            raise InvalidImageError(code="invalid_image_source")
        return "path"
    if callable(getattr(source, "read", None)):
        return "stream"
    raise InvalidImageError(code="invalid_image_source")


def _read_encoded_source(
    source: object,
    source_kind: Literal["bytes", "path", "stream"],
    max_input_bytes: int,
) -> bytes:
    if source_kind == "bytes":
        return _copy_bounded_bytes(source, max_input_bytes)
    if source_kind == "path":
        return _read_path(source, max_input_bytes)
    return _read_stream(source, max_input_bytes)


def _copy_bounded_bytes(source: object, max_input_bytes: int) -> bytes:
    size = source.nbytes if isinstance(source, memoryview) else len(cast(bytes | bytearray, source))
    if size > max_input_bytes:
        raise ImageLimitExceededError(code="image_input_bytes_exceeded")
    if isinstance(source, memoryview):
        return source.tobytes()
    return bytes(cast(bytes | bytearray, source))


def _read_path(source: object, max_input_bytes: int) -> bytes:
    try:
        path_value = os.fspath(cast(str | os.PathLike[str], source))
        with open(path_value, "rb") as stream:
            return _read_bounded(stream, max_input_bytes)
    except PenampakanError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise InvalidImageError(code="image_source_unreadable", cause=error) from error


def _read_stream(source: object, max_input_bytes: int) -> bytes:
    stream = source
    position = _stream_position(stream)
    try:
        return _read_bounded(stream, max_input_bytes)
    except PenampakanError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise InvalidImageError(code="image_source_unreadable", cause=error) from error
    finally:
        if position is not None:
            seek = getattr(stream, "seek", None)
            if callable(seek):
                with suppress(OSError, TypeError, ValueError, AttributeError):
                    seek(position)


def _stream_position(stream: object) -> int | None:
    tell = getattr(stream, "tell", None)
    seek = getattr(stream, "seek", None)
    if not callable(tell) or not callable(seek):
        return None
    try:
        position = tell()
    except (OSError, TypeError, ValueError):
        return None
    if not isinstance(position, int) or position < 0:
        return None
    return position


def _read_bounded(stream: object, max_input_bytes: int) -> bytes:
    read = getattr(stream, "read", None)
    if not callable(read):
        raise InvalidImageError(code="invalid_image_source")
    remaining = max_input_bytes + 1
    chunks: list[bytes] = []
    while remaining > 0:
        chunk = read(min(_READ_CHUNK_SIZE, remaining))
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise InvalidImageError(code="invalid_image_stream")
        if not chunk:
            break
        copied = bytes(chunk)
        if len(copied) > remaining:
            copied = copied[:remaining]
        chunks.append(copied)
        remaining -= len(copied)
    encoded = b"".join(chunks)
    if len(encoded) > max_input_bytes:
        raise ImageLimitExceededError(code="image_input_bytes_exceeded")
    return encoded


def _load_encoded_image(encoded: bytes, limits: ImageLimits) -> LoadedImage:
    encoded_stream = BytesIO(encoded)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", Image.DecompressionBombWarning)
            image = Image.open(encoded_stream)
    except Image.DecompressionBombError as error:
        raise ImageLimitExceededError(code="image_decompression_bomb", cause=error) from error
    except UnidentifiedImageError as error:
        raise InvalidImageError(code="invalid_encoded_image", cause=error) from error
    except (OSError, SyntaxError, ValueError) as error:
        raise InvalidImageError(code="invalid_encoded_image", cause=error) from error

    try:
        original_format = _validated_format(image.format)
        _validate_single_frame(image)
        _validate_dimensions(image.width, image.height, limits)
        if any(isinstance(item.message, Image.DecompressionBombWarning) for item in caught):
            raise ImageLimitExceededError(code="image_decompression_bomb")
        _decode_pixels(image)
        return _normalize_image(image, original_format, limits)
    finally:
        image.close()


def _load_pillow_image(image: Image.Image, limits: ImageLimits) -> LoadedImage:
    original_format = _validated_format(image.format, allow_missing=True)
    _validate_single_frame(image)
    _validate_dimensions(image.width, image.height, limits)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            copied = image.copy()
            copied.load()
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
        raise ImageLimitExceededError(code="image_decompression_bomb", cause=error) from error
    except (OSError, SyntaxError, ValueError) as error:
        raise InvalidImageError(code="image_decode_failed", cause=error) from error
    try:
        return _normalize_image(copied, original_format, limits)
    finally:
        copied.close()


def _validated_format(
    image_format: str | None,
    allow_missing: bool = False,
) -> Literal["PNG", "JPEG", "WEBP"] | None:
    if image_format is None and allow_missing:
        return None
    if not isinstance(image_format, str):
        raise UnsupportedImageError(code="unsupported_image_format")
    normalized = image_format.upper()
    if normalized not in _SUPPORTED_FORMATS:
        raise UnsupportedImageError(code="unsupported_image_format")
    return cast(Literal["PNG", "JPEG", "WEBP"], normalized)


def _validate_single_frame(image: Image.Image) -> None:
    try:
        frame_count = getattr(image, "n_frames", 1)
    except (OSError, SyntaxError, ValueError) as error:
        raise InvalidImageError(code="invalid_frame_count", cause=error) from error
    if not isinstance(frame_count, int) or frame_count < 1:
        raise InvalidImageError(code="invalid_frame_count")
    if frame_count != 1:
        raise UnsupportedImageError(code="multiple_frames_unsupported")


def _validate_dimensions(width: int, height: int, limits: ImageLimits) -> None:
    if width <= 0 or height <= 0:
        raise InvalidImageError(code="invalid_image_dimensions")
    if width > limits.max_width:
        raise ImageLimitExceededError(code="image_width_exceeded")
    if height > limits.max_height:
        raise ImageLimitExceededError(code="image_height_exceeded")
    if width * height > limits.max_pixels:
        raise ImageLimitExceededError(code="image_pixels_exceeded")


def _decode_pixels(image: Image.Image) -> None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image.load()
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
        raise ImageLimitExceededError(code="image_decompression_bomb", cause=error) from error
    except (OSError, SyntaxError, ValueError) as error:
        raise InvalidImageError(code="image_decode_failed", cause=error) from error


def _normalize_image(
    image: Image.Image,
    original_format: Literal["PNG", "JPEG", "WEBP"] | None,
    limits: ImageLimits,
) -> LoadedImage:
    has_icc_profile = "icc_profile" in image.info
    try:
        oriented = ImageOps.exif_transpose(image)
        normalized = _normalize_mode(oriented)
        _validate_dimensions(normalized.width, normalized.height, limits)
        canonical_image = normalized.copy()
        canonical_image.info.clear()
        _drop_image_attributes(canonical_image)
        canonical_png = _encode_canonical_png(canonical_image)
    except PenampakanError:
        raise
    except (OSError, SyntaxError, ValueError) as error:
        raise InvalidImageError(code="image_normalization_failed", cause=error) from error
    warnings_result = _metadata_warnings(has_icc_profile)
    return LoadedImage(
        image=canonical_image,
        canonical_png=canonical_png,
        digest_sha256=canonical_digest(canonical_png),
        original_format=original_format,
        warnings=warnings_result,
    )


def _normalize_mode(image: Image.Image) -> Image.Image:
    if image.mode in {"P", "PA", "RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        alpha_extrema = cast(tuple[float, float] | None, rgba.getchannel("A").getextrema())
        if alpha_extrema is not None and alpha_extrema[0] < 255:
            return rgba
        rgb = rgba.convert("RGB")
        rgba.close()
        return rgb
    return image.convert("RGB")


def _drop_image_attributes(image: Image.Image) -> None:
    image.format = None
    if hasattr(image, "filename"):
        try:
            delattr(image, "filename")
        except AttributeError:
            image.filename = ""


def _encode_canonical_png(image: Image.Image) -> bytes:
    try:
        return encode_canonical_png(image)
    except (OSError, SyntaxError, ValueError) as error:
        raise InvalidImageError(code="canonical_image_encoding_failed", cause=error) from error


def _metadata_warnings(has_icc_profile: bool) -> tuple[WarningInfo, ...]:
    if not has_icc_profile:
        return ()
    return (
        WarningInfo(
            code="icc_profile_discarded",
            message="The embedded ICC color profile was discarded.",
        ),
    )


__all__ = ["LoadedImage", "load_image"]
