import asyncio
from collections.abc import Callable, Sequence

from PIL import Image

from penampakan.backends.callable import CallableVisionBackend
from penampakan.backends.pillow import PillowBackend
from penampakan.image.assets import AssetStore
from penampakan.models import (
    BackendDescriptor,
    BackendImage,
    Capability,
    CapabilityDescriptor,
    CaptionPayload,
    CaptionRequest,
    ColorsRequest,
    DetectionRequest,
    MetadataRequest,
    ObservationDraft,
    VisionRequest,
    VisionResult,
)
from penampakan.protocols import VisionBackend


async def assert_backend_contract(
    factory: Callable[[], VisionBackend],
    supported_requests: Sequence[VisionRequest],
    unsupported_requests: Sequence[VisionRequest],
) -> None:
    source = Image.new("RGB", (4, 4), "red")
    store = AssetStore.create(source)
    backend = factory()
    first_descriptor = backend.descriptor
    image = store.backend_image(store.root_id)
    original_content = image.content

    assert isinstance(first_descriptor, BackendDescriptor)
    assert backend.descriptor == first_descriptor
    for request in supported_requests:
        assert backend.supports(request)
        assert backend.supports(request)
        response = await backend.analyze(image, request)
        assert isinstance(response, VisionResult)
        assert image.content == original_content
    for request in unsupported_requests:
        assert not backend.supports(request)

    await asyncio.gather(backend.aclose(), backend.aclose())
    store.close()
    source.close()


async def test_pillow_backend_contract() -> None:
    await assert_backend_contract(
        PillowBackend,
        (MetadataRequest(), ColorsRequest()),
        (CaptionRequest(), DetectionRequest()),
    )


async def test_callable_backend_contract() -> None:
    descriptor = BackendDescriptor(
        name="contract.caption",
        version="1.0",
        capabilities=(CapabilityDescriptor(capability=Capability.CAPTION),),
    )

    async def analyze(image: BackendImage, request: VisionRequest) -> VisionResult:
        return VisionResult(
            observations=(ObservationDraft(payload=CaptionPayload(text="A red square.")),)
        )

    def factory() -> CallableVisionBackend:
        return CallableVisionBackend(descriptor, analyze)

    await assert_backend_contract(
        factory,
        (CaptionRequest(),),
        (MetadataRequest(), CaptionRequest(focus="display")),
    )
