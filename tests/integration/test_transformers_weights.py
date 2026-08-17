from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest
from PIL import Image

from penampakan.backends.transformers import (
    TransformersCaptionBackend,
    TransformersDetectionBackend,
)
from penampakan.models import (
    BackendImage,
    Box,
    CaptionPayload,
    CaptionRequest,
    DetectionPayload,
    DetectionRequest,
)
from tests.integration._helpers import backend_image

_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
_CAPTION_REVISION = "82a37760796d32b1411fe092ab5d4e227313294b"
_DETECTION_MODEL = "IDEA-Research/grounding-dino-tiny"
_DETECTION_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
_FIXTURE = Path(__file__).parents[1] / "fixtures" / "natural" / "astronaut.png"
_METADATA = _FIXTURE.with_suffix(".json")


def _require_snapshot(model_id: str, revision: str) -> None:
    hub = pytest.importorskip("huggingface_hub", reason="install penampakan[transformers]")
    try:
        hub.snapshot_download(model_id, revision=revision, local_files_only=True)
    except Exception as error:
        pytest.skip(f"prefetch the pinned model snapshot {model_id}@{revision}: {error}")


def _natural_image() -> tuple[BackendImage, dict[str, object]]:
    metadata = json.loads(_METADATA.read_text(encoding="utf-8"))
    content = _FIXTURE.read_bytes()
    assert hashlib.sha256(content).hexdigest() == metadata["sha256"]
    with Image.open(_FIXTURE) as source:
        source.load()
        image = source.convert("RGB")
    try:
        return backend_image(image), metadata
    finally:
        image.close()


@pytest.mark.integration
@pytest.mark.integration_category("caption")
@pytest.mark.models
@pytest.mark.asyncio
async def test_pinned_caption_weight_smoke() -> None:
    pytest.importorskip("transformers", reason="install penampakan[transformers]")
    _require_snapshot(_CAPTION_MODEL, _CAPTION_REVISION)
    image, _ = _natural_image()
    backend = TransformersCaptionBackend(
        _CAPTION_MODEL, revision=_CAPTION_REVISION, local_files_only=True
    )
    result = await backend.analyze(image, CaptionRequest(max_sentences=2))
    assert backend.descriptor.model_id == _CAPTION_MODEL
    assert backend.descriptor.model_revision == _CAPTION_REVISION
    assert len(result.observations) == 1
    text = cast(CaptionPayload, result.observations[0].payload).text
    assert 1 <= len(text) <= 500
    await backend.aclose()
    await backend.aclose()


@pytest.mark.integration
@pytest.mark.integration_category("detection")
@pytest.mark.models
@pytest.mark.asyncio
async def test_pinned_detection_weight_golden_and_blank_smoke() -> None:
    pytest.importorskip("transformers", reason="install penampakan[transformers]")
    _require_snapshot(_DETECTION_MODEL, _DETECTION_REVISION)
    image, metadata = _natural_image()
    golden = cast(dict[str, object], cast(list[object], metadata["objects"])[0])
    threshold = cast(float, golden["min_confidence"])
    backend = TransformersDetectionBackend(
        _DETECTION_MODEL, revision=_DETECTION_REVISION, local_files_only=True
    )
    result = await backend.analyze(
        image,
        DetectionRequest(labels=(cast(str, golden["label"]),), min_confidence=threshold),
    )
    coordinates = cast(list[float], golden["box"])
    expected = Box(
        x_min=coordinates[0],
        y_min=coordinates[1],
        x_max=coordinates[2],
        y_max=coordinates[3],
    )
    matches = [
        item
        for item in result.observations
        if cast(DetectionPayload, item.payload).label == golden["label"]
    ]
    assert matches
    assert max(cast(Box, item.region).iou(expected) for item in matches) >= golden["min_iou"]
    assert backend.descriptor.model_id == _DETECTION_MODEL
    assert backend.descriptor.model_revision == _DETECTION_REVISION
    blank = Image.new("RGB", (512, 512), "white")
    try:
        blank_result = await backend.analyze(
            backend_image(blank), DetectionRequest(labels=("person",), min_confidence=threshold)
        )
    finally:
        blank.close()
    assert len(blank_result.observations) <= 100
    await backend.aclose()
    await backend.aclose()
