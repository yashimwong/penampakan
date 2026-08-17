from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import pytest

from penampakan.backends.pillow import PillowBackend
from penampakan.backends.tesseract import TesseractBackend
from penampakan.errors import BackendUnavailableError
from penampakan.models import MetadataPayload, MetadataRequest, OCRRequest, TextPayload
from tests.fixtures.images import TEXT_GRID_WORDS, oriented_text_grid_jpeg
from tests.integration._helpers import backend_image_bytes

pytestmark = [
    pytest.mark.integration,
    pytest.mark.integration_category("e2e"),
    pytest.mark.ocr,
]

_FONT = Path(os.getenv("PENAMPAKAN_LATIN_FONT", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))


@pytest.mark.asyncio
async def test_orientation_metadata_and_ocr_share_the_normalized_frame() -> None:
    pytest.importorskip("pytesseract", reason="install penampakan[ocr]")
    if not _FONT.is_file():
        pytest.skip(f"pinned Latin font is unavailable: {_FONT}")
    image = backend_image_bytes(oriented_text_grid_jpeg(font_path=_FONT))
    metadata_backend = PillowBackend()
    ocr_backend = TesseractBackend(languages=("eng",))
    metadata = await metadata_backend.analyze(image, MetadataRequest())
    payload = cast(MetadataPayload, metadata.observations[0].payload)
    assert (payload.width, payload.height) == (640, 400)
    try:
        result = await ocr_backend.analyze(image, OCRRequest(mode="sparse"))
    except BackendUnavailableError as error:
        await ocr_backend.aclose()
        await metadata_backend.aclose()
        pytest.skip(f"Tesseract English runtime is unavailable: {error}")
    text = " ".join(cast(TextPayload, item.payload).text.casefold() for item in result.observations)
    assert sum(word.casefold() in "".join(text.split()) for word in TEXT_GRID_WORDS) >= 3
    assert all(item.region is not None for item in result.observations)
    await ocr_backend.aclose()
    await metadata_backend.aclose()
