from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast

import pytest
from PIL import Image

import penampakan.backends.transformers as adapter
from penampakan.backends.transformers import TransformersDetectionBackend
from penampakan.models import Box, DetectionRequest
from tests.integration._helpers import backend_image

pytestmark = [
    pytest.mark.integration,
    pytest.mark.integration_category("geometry"),
    pytest.mark.models,
]

_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"


class _Pipeline:
    def __init__(self, output: object) -> None:
        self.output = output

    def __call__(self, image: Image.Image, **kwargs: object) -> object:
        return self.output(image) if callable(self.output) else self.output


def _install(monkeypatch: pytest.MonkeyPatch, output: object) -> None:
    def factory(*args: object, **kwargs: object) -> tuple[object, object]:
        return _Pipeline(output), nullcontext

    monkeypatch.setattr(adapter, "_create_pipeline", factory)


def _processor_output(raw_box: list[float]) -> object:
    transformers = pytest.importorskip("transformers", reason="install penampakan[transformers]")
    torch = pytest.importorskip("torch", reason="install penampakan[transformers]")
    processor = transformers.GroundingDinoImageProcessor()

    def output(image: Image.Image) -> list[dict[str, object]]:
        model_output = SimpleNamespace(
            logits=torch.tensor([[[8.0]]]),
            pred_boxes=torch.tensor([[raw_box]]),
        )
        processed = processor.post_process_object_detection(
            model_output,
            threshold=0.0,
            target_sizes=torch.tensor([[image.height, image.width]]),
        )[0]
        box = processed["boxes"][0].tolist()
        return [
            {
                "label": "cat",
                "score": float(processed["scores"][0]),
                "box": {"xmin": box[0], "ymin": box[1], "xmax": box[2], "ymax": box[3]},
            }
        ]

    return output


@pytest.mark.asyncio
async def test_real_processor_cxcywh_to_absolute_xyxy_and_root_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _processor_output([0.5, 0.5, 0.5, 0.4]))
    source = Image.new("RGB", (200, 100), "white")
    try:
        image = backend_image(source)
    finally:
        source.close()
    backend = TransformersDetectionBackend(revision=_REVISION)
    result = await backend.analyze(image, DetectionRequest(labels=("cat",)))
    box = cast(Box, result.observations[0].region)
    actual = (box.x_min, box.y_min, box.x_max, box.y_max)
    assert actual == pytest.approx((0.25, 0.3, 0.75, 0.7), abs=1e-6)
    await backend.aclose()


@pytest.mark.asyncio
async def test_region_clipping_filtering_and_result_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("transformers", reason="install penampakan[transformers]")
    pytest.importorskip("torch", reason="install penampakan[transformers]")
    output = [
        {"label": "cat", "score": 0.8, "box": {"xmin": 10, "ymin": 10, "xmax": 40, "ymax": 30}},
        {"label": "cat", "score": 0.9, "box": {"xmin": -10, "ymin": 50, "xmax": 210, "ymax": 110}},
        {"label": "cat", "score": 0.7, "box": {"xmin": 20, "ymin": 10, "xmax": 20, "ymax": 30}},
        {"label": "cat", "score": 0.6, "box": {"xmin": 80, "ymin": 10, "xmax": 100, "ymax": 30}},
    ]
    _install(monkeypatch, output)
    source = Image.new("RGB", (400, 200), "white")
    try:
        image = backend_image(source)
    finally:
        source.close()
    scope = Box(x_min=0.25, y_min=0.25, x_max=0.75, y_max=0.75)
    backend = TransformersDetectionBackend(revision=_REVISION)
    result = await backend.analyze(
        image,
        DetectionRequest(
            region=scope,
            labels=("cat",),
            min_confidence=0.65,
            max_results=2,
        ),
    )
    assert tuple(item.confidence for item in result.observations) == (0.9, 0.8)
    assert all(scope.contains(cast(Box, item.region)) for item in result.observations)
    assert "zero_area_detection_discarded" in {item.code for item in result.warnings}
    await backend.aclose()


@pytest.mark.parametrize(
    "mutated",
    (
        [0.25, 0.5, 0.5, 0.4],
        [0.5, 0.25, 0.5, 0.4],
        [0.5, 0.5, 0.4, 0.5],
        [0.5, 0.5, 1.0, 0.8],
    ),
)
def test_geometry_oracle_rejects_axis_order_and_scale_mutations(mutated: list[float]) -> None:
    output = cast(object, _processor_output(mutated))
    source = Image.new("RGB", (200, 100), "white")
    try:
        item = cast(list[dict[str, object]], output(source))[0]
    finally:
        source.close()
    box = cast(dict[str, float], item["box"])
    actual = (box["xmin"] / 200, box["ymin"] / 100, box["xmax"] / 200, box["ymax"] / 100)
    with pytest.raises(AssertionError):
        assert actual == pytest.approx((0.25, 0.3, 0.75, 0.7), abs=1e-6)
