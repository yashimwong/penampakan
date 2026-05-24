import pytest
from PIL import Image

from penampakan.errors import ConfigurationError, ToolExecutionError
from penampakan.models import Capability, CaptionRequest, VisionRequest
from penampakan.perception.registry import ToolExecutionContext, ToolRegistry, ToolResult
from penampakan.tools.builtin import register_transform_tools
from penampakan.tools.vision import register_vision_tools

ASSET_ID = "img_0123456789abcdef"


class RecordingContext:
    def __init__(self) -> None:
        self.source = Image.new("RGB", (8, 8), "red")
        self.reservations: list[tuple[str, int]] = []
        self.perceptions: list[tuple[str, VisionRequest]] = []

    def image(self, asset_id: str) -> Image.Image:
        if asset_id != ASSET_ID:
            raise KeyError(asset_id)
        return self.source.copy()

    def ensure_asset_capacity(self, parent_id: str, count: int) -> None:
        self.reservations.append((parent_id, count))

    async def perceive(self, asset_id: str, request: VisionRequest) -> ToolResult:
        self.perceptions.append((asset_id, request))
        return ToolResult()

    def close(self) -> None:
        self.source.close()


def registry() -> ToolRegistry:
    value = ToolRegistry()
    register_vision_tools(
        value,
        (Capability.METADATA, Capability.COLORS, Capability.CAPTION),
    )
    register_transform_tools(value)
    return value


def test_registry_exposes_only_available_perception_capabilities() -> None:
    value = registry()

    assert value.names[:3] == ("get_metadata", "get_colors", "describe_image")
    assert "read_text" not in value.names
    assert "detect_objects" not in value.names
    assert "segment_objects" not in value.names
    assert "crop" in value.names


def test_registry_rejects_duplicate_names() -> None:
    value = registry()

    with pytest.raises(ConfigurationError):
        register_transform_tools(value)


def test_tool_schemas_are_strict_and_do_not_select_backends() -> None:
    value = registry()

    for spec in value.specs:
        assert spec.arguments_json_schema["additionalProperties"] is False
        assert "backend" not in spec.arguments_json_schema.get("properties", {})


async def test_transform_execution_reserves_before_returning_assets() -> None:
    value = registry()
    context = RecordingContext()

    result = await value.execute(
        context,
        "crop",
        {
            "asset_id": ASSET_ID,
            "box": {"x_min": 0.0, "y_min": 0.0, "x_max": 0.5, "y_max": 0.5},
        },
    )

    assert context.reservations == [(ASSET_ID, 1)]
    assert len(result.assets) == 1
    assert result.assets[0].image.size == (4, 4)
    result.assets[0].close()
    context.close()


async def test_perception_execution_translates_typed_request() -> None:
    value = registry()
    context = RecordingContext()

    await value.execute(
        context,
        "describe_image",
        {"asset_id": ASSET_ID, "focus": "display", "max_sentences": 2},
    )

    assert len(context.perceptions) == 1
    asset_id, request = context.perceptions[0]
    assert asset_id == ASSET_ID
    assert isinstance(request, CaptionRequest)
    assert request.focus == "display"
    context.close()


async def test_arguments_reject_unknown_backend_selection() -> None:
    value = registry()
    context = RecordingContext()

    with pytest.raises(ToolExecutionError):
        await value.execute(
            context,
            "get_metadata",
            {"asset_id": ASSET_ID, "backend": "remote"},
        )

    context.close()


def test_recording_context_satisfies_tool_protocol() -> None:
    recording = RecordingContext()
    context: ToolExecutionContext = recording
    image = context.image(ASSET_ID)

    assert image.size == (8, 8)
    image.close()
    recording.close()
