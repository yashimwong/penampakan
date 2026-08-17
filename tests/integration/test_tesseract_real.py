from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import pytest

from penampakan.backends.tesseract import TesseractBackend
from penampakan.errors import InvalidBackendOutputError
from penampakan.models import Box, OCRRequest, TextPayload
from penampakan.perception.normalize import NormalizationLimits, normalize_backend_result
from tests.fixtures.images import (
    NON_LATIN_TEXT,
    TEXT_GRID_WORDS,
    blank_image,
    dense_text_image,
    non_latin_text_image,
    text_grid_image,
)
from tests.integration._helpers import backend_image

pytestmark = [
    pytest.mark.integration,
    pytest.mark.integration_category("ocr"),
    pytest.mark.ocr,
]

_LATIN_FONT = Path(
    os.getenv("PENAMPAKAN_LATIN_FONT", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
)
_ARABIC_FONT = Path(
    os.getenv(
        "PENAMPAKAN_ARABIC_FONT",
        "/usr/share/fonts/opentype/fonts-hosny-amiri/Amiri-Regular.ttf",
    )
)


@pytest.fixture(scope="module")
def tesseract_runtime() -> tuple[str, frozenset[str]]:
    pytesseract = pytest.importorskip(
        "pytesseract", reason="install penampakan[ocr] for real Tesseract integration tests"
    )
    try:
        version = str(pytesseract.get_tesseract_version())
        languages = frozenset(pytesseract.get_languages())
    except Exception as error:
        pytest.skip(f"Tesseract binary or tessdata is unavailable: {error}")
    return version, languages


def _tokens(result: object) -> tuple[tuple[str, Box | None], ...]:
    observations = result.observations
    return tuple(
        (cast(TextPayload, item.payload).text.casefold(), item.region) for item in observations
    )


@pytest.mark.asyncio
async def test_quadrant_words_and_region_mapping(
    tesseract_runtime: tuple[str, frozenset[str]],
) -> None:
    if not _LATIN_FONT.is_file():
        pytest.skip(f"pinned Latin font is unavailable: {_LATIN_FONT}")
    backend = TesseractBackend(languages=("eng",))
    image = backend_image(text_grid_image(font_path=_LATIN_FONT))
    result = await backend.analyze(image, OCRRequest(mode="sparse"))
    tokens = _tokens(result)
    expected = {
        TEXT_GRID_WORDS[0].casefold(): Box(x_min=0, y_min=0, x_max=0.5, y_max=0.5),
        TEXT_GRID_WORDS[1].casefold(): Box(x_min=0.5, y_min=0, x_max=1, y_max=0.5),
        TEXT_GRID_WORDS[2].casefold(): Box(x_min=0, y_min=0.5, x_max=0.5, y_max=1),
        TEXT_GRID_WORDS[3].casefold(): Box(x_min=0.5, y_min=0.5, x_max=1, y_max=1),
    }
    for word, quadrant in expected.items():
        matches = [box for text, box in tokens if word in "".join(text.split())]
        assert matches and matches[0] is not None and quadrant.contains(matches[0])
    scope = Box(x_min=0.5, y_min=0.5, x_max=1, y_max=1)
    scoped = await backend.analyze(image, OCRRequest(region=scope, mode="sparse"))
    assert scoped.observations
    assert all(item.region is None or scope.contains(item.region) for item in scoped.observations)
    await backend.aclose()
    await backend.aclose()


@pytest.mark.asyncio
async def test_blank_dense_and_non_latin_paths(
    tesseract_runtime: tuple[str, frozenset[str]],
) -> None:
    _, languages = tesseract_runtime
    backend = TesseractBackend(languages=("eng",))
    blank = await backend.analyze(backend_image(blank_image()), OCRRequest())
    assert blank.observations == ()
    assert "no_text_detected" in {warning.code for warning in blank.warnings}
    if _LATIN_FONT.is_file():
        dense = await backend.analyze(
            backend_image(dense_text_image(font_path=_LATIN_FONT)), OCRRequest(mode="dense")
        )
        assert dense.observations
        normalize_backend_result(
            dense,
            OCRRequest(mode="dense"),
            limits=NormalizationLimits(),
        )
        with pytest.raises(InvalidBackendOutputError):
            normalize_backend_result(
                dense,
                OCRRequest(mode="dense"),
                limits=NormalizationLimits(
                    max_ocr_chars_per_observation=10,
                    max_observations=1,
                ),
            )
        assert all(
            len(cast(TextPayload, item.payload).text) <= 8_000 for item in dense.observations
        )
        assert len(dense.observations) <= 4_096
    if "ara" not in languages or not _ARABIC_FONT.is_file():
        pytest.skip("Arabic tessdata and the pinned Amiri font are required for non-Latin OCR")
    arabic = TesseractBackend(languages=("ara",))
    result = await arabic.analyze(
        backend_image(non_latin_text_image(font_path=_ARABIC_FONT)), OCRRequest(languages=("ara",))
    )
    assert result.observations
    assert any(cast(TextPayload, item.payload).language == "ara" for item in result.observations)
    recognized = "".join(text for text, _ in _tokens(result))
    assert (
        sum(character in recognized for character in set(NON_LATIN_TEXT) if not character.isspace())
        >= 2
    )
    await arabic.aclose()
    await backend.aclose()
