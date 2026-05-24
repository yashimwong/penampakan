import asyncio
import threading

from PIL import Image

from penampakan.backends.callable import CallableVisionBackend
from penampakan.image.assets import AssetStore
from penampakan.models import (
    BackendDescriptor,
    Capability,
    CapabilityDescriptor,
    CaptionPayload,
    CaptionRequest,
    ColorsRequest,
    DetectionRequest,
    MetadataRequest,
    ObservationDraft,
    VisionResult,
)


def descriptor(*features: str) -> BackendDescriptor:
    return BackendDescriptor(
        name="application.caption",
        version="1.0",
        capabilities=(
            CapabilityDescriptor(
                capability=Capability.CAPTION,
                features=frozenset(features),
            ),
        ),
        max_concurrency=2,
    )


def result() -> VisionResult:
    return VisionResult(
        observations=(ObservationDraft(payload=CaptionPayload(text="A red square.")),)
    )


async def test_sync_analyzer_runs_in_worker_thread() -> None:
    source = Image.new("RGB", (2, 2), "red")
    store = AssetStore.create(source)
    caller_thread = threading.get_ident()
    analyzer_threads: list[int] = []

    def analyze(image: object, request: object) -> VisionResult:
        analyzer_threads.append(threading.get_ident())
        return result()

    backend = CallableVisionBackend(descriptor(), analyze)

    response = await backend.analyze(store.backend_image(store.root_id), CaptionRequest())

    assert response == result()
    assert analyzer_threads and analyzer_threads[0] != caller_thread
    await backend.aclose()
    store.close()
    source.close()


async def test_async_analyzer_is_awaited() -> None:
    source = Image.new("RGB", (2, 2), "red")
    store = AssetStore.create(source)
    calls = 0

    async def analyze(image: object, request: object) -> VisionResult:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        return result()

    backend = CallableVisionBackend(descriptor(), analyze)

    assert await backend.analyze(store.backend_image(store.root_id), CaptionRequest()) == result()
    assert calls == 1
    await backend.aclose()
    store.close()
    source.close()


def test_descriptor_derived_support_rejects_unsupported_options() -> None:
    backend = CallableVisionBackend(descriptor(), lambda image, request: result())
    focused = CallableVisionBackend(
        descriptor("caption.focus"),
        lambda image, request: result(),
    )

    assert backend.supports(CaptionRequest())
    assert not backend.supports(CaptionRequest(focus="display"))
    assert focused.supports(CaptionRequest(focus="display"))
    assert not backend.supports(MetadataRequest())
    assert not backend.supports(ColorsRequest())
    assert not backend.supports(DetectionRequest(labels=("car",)))


async def test_close_callable_runs_once_for_concurrent_close() -> None:
    close_calls = 0

    async def close() -> None:
        nonlocal close_calls
        close_calls += 1
        await asyncio.sleep(0)

    backend = CallableVisionBackend(
        descriptor(),
        lambda image, request: result(),
        close=close,
    )

    await asyncio.gather(backend.aclose(), backend.aclose(), backend.aclose())

    assert close_calls == 1
