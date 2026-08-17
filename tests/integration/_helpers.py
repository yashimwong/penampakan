from __future__ import annotations

import hashlib
from io import BytesIO

from PIL import Image

from penampakan.image.loader import load_image
from penampakan.models import BackendImage, ImageAsset


def backend_image(image: Image.Image, *, original_format: str | None = "PNG") -> BackendImage:
    output = BytesIO()
    image.save(output, format="PNG")
    content = output.getvalue()
    digest = hashlib.sha256(content).hexdigest()
    return BackendImage(
        asset=ImageAsset.model_validate(
            {
                "id": f"img_{digest[:16]}",
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "mime_type": "image/png",
                "original_format": original_format,
                "digest_sha256": digest,
                "parent_id": None,
                "derivation_depth": 0,
                "transform": None,
            },
            strict=True,
        ),
        content=content,
    )


def backend_image_bytes(content: bytes) -> BackendImage:
    loaded = load_image(content)
    try:
        return backend_image(loaded.image, original_format=loaded.original_format)
    finally:
        loaded.close()
