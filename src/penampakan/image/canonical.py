"""Shared deterministic encoding for normalized image assets."""

from __future__ import annotations

import hashlib
from io import BytesIO

from PIL.Image import Image as PillowImage

CANONICAL_PNG_COMPRESSION_LEVEL = 6


def encode_canonical_png(image: PillowImage) -> bytes:
    """Encode caller-owned normalized pixels with the stable core settings."""

    output = BytesIO()
    try:
        image.save(
            output,
            format="PNG",
            optimize=False,
            compress_level=CANONICAL_PNG_COMPRESSION_LEVEL,
        )
        return output.getvalue()
    finally:
        output.close()


def canonical_digest(canonical_png: bytes) -> str:
    """Return the lowercase SHA-256 digest of canonical PNG bytes."""

    return hashlib.sha256(canonical_png).hexdigest()


__all__ = [
    "CANONICAL_PNG_COMPRESSION_LEVEL",
    "canonical_digest",
    "encode_canonical_png",
]
