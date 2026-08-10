from __future__ import annotations

from collections.abc import Mapping

import pytest
from PIL import Image

from penampakan.errors import ToolExecutionError
from penampakan.models import JsonValue, VisionRequest
from penampakan.perception.registry import ToolRegistry, ToolResult
from penampakan.tools.builtin import register_transform_tools

ASSET_ID = "img_0123456789abcdef"


class TransformContext:
    def __init__(self) -> None:
        self.source = Image.new("RGB", (12, 8), "navy")
        self.reservations: list[tuple[str, int]] = []

    def image(self, asset_id: str) -> Image.Image:
        if asset_id != ASSET_ID:
            raise KeyError(asset_id)
        return self.source.copy()

    def ensure_asset_capacity(self, parent_id: str, count: int) -> None:
        self.reservations.append((parent_id, count))

    async def perceive(self, asset_id: str, request: VisionRequest) -> ToolResult:
        raise AssertionError((asset_id, request))

    def close(self) -> None:
        self.source.close()


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    register_transform_tools(registry)
    return registry


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "arguments", "count", "transform"),
    [
        (
            "crop",
            {
                "asset_id": ASSET_ID,
                "box": {"x_min": 0.0, "y_min": 0.0, "x_max": 0.5, "y_max": 0.5},
            },
            1,
            "crop",
        ),
        (
            "tile",
            {"asset_id": ASSET_ID, "rows": 2, "columns": 2, "overlap_fraction": 0.1},
            4,
            "tile",
        ),
        ("rotate", {"asset_id": ASSET_ID, "degrees": 90}, 1, "rotate"),
        (
            "enhance_contrast",
            {"asset_id": ASSET_ID, "factor": 1.5},
            1,
            "enhance_contrast",
        ),
        ("to_grayscale", {"asset_id": ASSET_ID}, 1, "grayscale"),
        (
            "add_coordinate_grid",
            {"asset_id": ASSET_ID, "rows": 3, "columns": 4, "labels": False},
            1,
            "coordinate_grid",
        ),
    ],
)
async def test_registered_transform_executes_owned_render(
    name: str,
    arguments: Mapping[str, JsonValue],
    count: int,
    transform: str,
) -> None:
    context = TransformContext()
    try:
        result = await _registry().execute(context, name, dict(arguments))

        assert context.reservations == [(ASSET_ID, count)]
        assert len(result.assets) == count
        assert {asset.transform.name for asset in result.assets} == {transform}
        for asset in result.assets:
            asset.close()
    finally:
        context.close()


@pytest.mark.parametrize(
    "arguments",
    [
        {"asset_id": ASSET_ID, "rows": 1, "columns": 1},
        {"asset_id": ASSET_ID, "rows": 5, "columns": 4},
    ],
)
def test_tile_arguments_reject_invalid_fanout(arguments: dict[str, JsonValue]) -> None:
    with pytest.raises(ToolExecutionError) as raised:
        _registry().validate_arguments("tile", arguments)

    assert raised.value.code == "invalid_tool_arguments"
