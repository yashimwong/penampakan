from PIL import Image

from penampakan.backends.pillow import PillowBackend
from penampakan.image.assets import AssetStore
from penampakan.models import Box, ColorsPayload, ColorsRequest, MetadataPayload, MetadataRequest
from tests.fixtures.images import quadrants_image, transparent_image


async def test_metadata_uses_authoritative_normalized_asset() -> None:
    source = Image.new("RGBA", (4, 2), (255, 0, 0, 128))
    store = AssetStore.create(source)
    backend = PillowBackend()

    result = await backend.analyze(store.backend_image(store.root_id), MetadataRequest())

    payload = result.observations[0].payload
    assert isinstance(payload, MetadataPayload)
    assert payload.width == 4
    assert payload.height == 2
    assert payload.aspect_ratio == 2.0
    assert payload.has_alpha is True
    await backend.aclose()
    store.close()
    source.close()


async def test_solid_color_is_exact_and_normalized() -> None:
    source = Image.new("RGB", (8, 8), "red")
    store = AssetStore.create(source)
    backend = PillowBackend()

    result = await backend.analyze(store.backend_image(store.root_id), ColorsRequest(count=5))

    payload = result.observations[0].payload
    assert isinstance(payload, ColorsPayload)
    assert payload.swatches[0].rgb == (255, 0, 0)
    assert payload.swatches[0].hex == "#FF0000"
    assert payload.swatches[0].fraction == 1.0
    assert payload.swatches[0].name == "red"
    await backend.aclose()
    store.close()
    source.close()


async def test_split_colors_have_deterministic_tie_order() -> None:
    source = Image.new("RGB", (8, 4), "red")
    for x in range(4, 8):
        for y in range(4):
            source.putpixel((x, y), (0, 0, 255))
    store = AssetStore.create(source)
    backend = PillowBackend()

    result = await backend.analyze(store.backend_image(store.root_id), ColorsRequest(count=2))

    payload = result.observations[0].payload
    assert isinstance(payload, ColorsPayload)
    assert tuple(item.rgb for item in payload.swatches) == ((0, 0, 255), (255, 0, 0))
    assert sum(item.fraction for item in payload.swatches) == 1.0
    await backend.aclose()
    store.close()
    source.close()


async def test_color_region_uses_canonical_box_conversion() -> None:
    source = quadrants_image()
    store = AssetStore.create(source)
    backend = PillowBackend()
    region = Box(x_min=0.0, y_min=0.0, x_max=0.5, y_max=0.5)

    result = await backend.analyze(
        store.backend_image(store.root_id),
        ColorsRequest(region=region, count=1),
    )

    payload = result.observations[0].payload
    assert isinstance(payload, ColorsPayload)
    assert payload.swatches[0].rgb == (255, 0, 0)
    assert result.observations[0].region == region
    await backend.aclose()
    store.close()
    source.close()


async def test_transparency_emits_estimate_warning() -> None:
    source = transparent_image()
    store = AssetStore.create(source)
    backend = PillowBackend()

    result = await backend.analyze(store.backend_image(store.root_id), ColorsRequest(count=2))

    assert tuple(item.code for item in result.warnings) == ("transparent_color_estimate",)
    await backend.aclose()
    store.close()
    source.close()
