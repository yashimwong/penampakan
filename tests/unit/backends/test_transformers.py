from __future__ import annotations

import asyncio
import importlib
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import pytest
from PIL import Image

from penampakan.backends.transformers import (
    TransformersCaptionBackend,
    TransformersDetectionBackend,
)
from penampakan.errors import BackendUnavailableError
from penampakan.models import (
    BackendImage,
    Box,
    Capability,
    CaptionPayload,
    CaptionRequest,
    DetectionPayload,
    DetectionRequest,
    ImageAsset,
    MetadataRequest,
)

_CAPTION_COMMIT = "a1b2c3d4" * 5
_DETECTION_COMMIT = "0f1e2d3c" * 5


def _backend_image(width: int = 100, height: int = 80) -> BackendImage:
    source = Image.new("RGB", (width, height), "white")
    output = BytesIO()
    try:
        source.save(output, format="PNG")
        content = output.getvalue()
    finally:
        output.close()
        source.close()
    return BackendImage(
        asset=ImageAsset(
            id="img_bbbbbbbbbbbbbbbb",
            width=width,
            height=height,
            mode="RGB",
            mime_type="image/png",
            original_format="PNG",
            digest_sha256="b" * 64,
            parent_id=None,
            derivation_depth=0,
            transform=None,
        ),
        content=content,
    )


class FakePipeline:
    def __init__(self, outputs: object) -> None:
        self.outputs = outputs
        self.calls: list[tuple[tuple[int, int], str, dict[str, object], int]] = []
        self.close_calls = 0

    def __call__(self, image: Image.Image, **kwargs: object) -> object:
        self.calls.append((image.size, image.mode, kwargs, threading.get_ident()))
        return self.outputs

    def close(self) -> None:
        self.close_calls += 1


class FakeRuntime:
    def __init__(self, pipeline: FakePipeline) -> None:
        self.pipeline_instance = pipeline
        self.pipeline_calls: list[tuple[str, dict[str, object]]] = []
        self.inference_entries = 0
        self.transformers = SimpleNamespace(pipeline=self.create_pipeline)
        self.torch = SimpleNamespace(inference_mode=self.inference_mode)

    def create_pipeline(self, task: str, **kwargs: object) -> FakePipeline:
        self.pipeline_calls.append((task, kwargs))
        return self.pipeline_instance

    @contextmanager
    def inference_mode(self) -> Iterator[None]:
        self.inference_entries += 1
        yield


def _install(monkeypatch: pytest.MonkeyPatch, runtime: FakeRuntime) -> list[str]:
    imports: list[str] = []

    def load(name: str) -> object:
        imports.append(name)
        if name == "transformers":
            return runtime.transformers
        if name == "torch":
            return runtime.torch
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", load)
    return imports


def test_construction_and_discovery_are_lazy_and_descriptors_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imports: list[str] = []
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: imports.append(name),
    )

    caption = TransformersCaptionBackend(
        "org/caption",
        revision=_CAPTION_COMMIT,
        device="cpu",
        local_files_only=True,
        generation_kwargs={"num_beams": 2},
    )
    detection = TransformersDetectionBackend(
        "org/detection",
        revision=_DETECTION_COMMIT,
        device=-1,
        local_files_only=True,
    )

    assert imports == []
    assert caption.descriptor.name == "transformers.caption"
    assert caption.descriptor.model_id == "org/caption"
    assert caption.descriptor.model_revision == _CAPTION_COMMIT
    assert caption.descriptor.durable_cache_eligible
    assert caption.descriptor.capabilities[0].capability is Capability.CAPTION
    assert caption.supports(CaptionRequest())
    assert not caption.supports(CaptionRequest(focus="question"))
    assert not caption.supports(MetadataRequest())
    assert detection.descriptor.name == "transformers.detection"
    assert detection.descriptor.model_id == "org/detection"
    assert detection.descriptor.model_revision == _DETECTION_COMMIT
    assert detection.descriptor.durable_cache_eligible
    assert detection.descriptor.capabilities[0].features == frozenset({"detect.open_vocabulary"})
    assert detection.supports(DetectionRequest(labels=("cat",)))
    assert not detection.supports(DetectionRequest())


@pytest.mark.asyncio
async def test_caption_loads_once_with_secure_pinned_local_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = FakePipeline([{"generated_text": "A red square."}])
    runtime = FakeRuntime(pipeline)
    imports = _install(monkeypatch, runtime)
    backend = TransformersCaptionBackend(
        "org/caption",
        revision=_CAPTION_COMMIT,
        device="cpu",
        local_files_only=True,
        generation_kwargs={"num_beams": 3, "do_sample": False},
    )

    first, second = await asyncio.gather(
        backend.analyze(_backend_image(), CaptionRequest()),
        backend.analyze(_backend_image(), CaptionRequest()),
    )

    assert first == second
    assert first.warnings == ()
    assert imports == ["transformers", "torch"]
    assert runtime.pipeline_calls == [
        (
            "image-to-text",
            {
                "model": "org/caption",
                "revision": _CAPTION_COMMIT,
                "device": "cpu",
                "trust_remote_code": False,
                "model_kwargs": {"local_files_only": True},
            },
        )
    ]
    assert tuple(call[2] for call in pipeline.calls) == (
        {"num_beams": 3, "do_sample": False},
        {"num_beams": 3, "do_sample": False},
    )
    assert runtime.inference_entries == 2


@pytest.mark.asyncio
async def test_caption_crops_normalizes_and_truncates_at_sentence_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = FakePipeline(
        [
            {"generated_text": "   First sentence.   Second sentence! Third sentence?   "},
        ]
    )
    runtime = FakeRuntime(pipeline)
    _install(monkeypatch, runtime)
    backend = TransformersCaptionBackend(revision=_CAPTION_COMMIT)
    region = Box(x_min=0.25, y_min=0.25, x_max=0.75, y_max=0.75)

    result = await backend.analyze(
        _backend_image(),
        CaptionRequest(region=region, max_sentences=2),
    )

    payload = cast(CaptionPayload, result.observations[0].payload)
    assert pipeline.calls[0][:2] == ((50, 40), "RGB")
    assert payload.text == "First sentence. Second sentence!"
    assert result.observations[0].region == region
    assert result.warnings == ()


@pytest.mark.asyncio
async def test_caption_reports_unresolved_revision_and_empty_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = FakePipeline([{"generated_text": " \x00 "}, {"unexpected": "value"}])
    runtime = FakeRuntime(pipeline)
    _install(monkeypatch, runtime)
    backend = TransformersCaptionBackend()

    result = await backend.analyze(_backend_image(), CaptionRequest())

    assert backend.descriptor.model_revision is None
    assert not backend.descriptor.durable_cache_eligible
    assert result.observations == ()
    assert tuple(item.code for item in result.warnings) == (
        "unresolved_model_revision",
        "no_caption_generated",
    )


@pytest.mark.asyncio
async def test_mutable_revision_is_unresolved_but_still_reaches_the_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = FakePipeline([{"generated_text": "A red square."}])
    runtime = FakeRuntime(pipeline)
    _install(monkeypatch, runtime)
    backend = TransformersCaptionBackend("org/caption", revision="main")

    result = await backend.analyze(_backend_image(), CaptionRequest())

    assert runtime.pipeline_calls[0][1]["revision"] == "main"
    assert backend.descriptor.model_revision is None
    assert not backend.descriptor.durable_cache_eligible
    assert tuple(item.code for item in result.warnings) == ("unresolved_model_revision",)
    assert backend.descriptor == backend.descriptor


@pytest.mark.asyncio
async def test_detection_reports_unresolved_revision_on_every_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = FakePipeline(
        [
            {
                "label": "cat",
                "score": 0.9,
                "box": {"xmin": 10, "ymin": 10, "xmax": 50, "ymax": 50},
            }
        ]
    )
    runtime = FakeRuntime(pipeline)
    _install(monkeypatch, runtime)
    backend = TransformersDetectionBackend("org/detection", revision="v1.0")

    populated = await backend.analyze(
        _backend_image(),
        DetectionRequest(labels=("cat",)),
    )
    empty = await backend.analyze(
        _backend_image(),
        DetectionRequest(labels=("dog",)),
    )

    assert runtime.pipeline_calls[0][1]["revision"] == "v1.0"
    assert len(populated.observations) == 1
    assert tuple(item.code for item in populated.warnings) == ("unresolved_model_revision",)
    assert empty.observations == ()
    assert tuple(item.code for item in empty.warnings) == (
        "unresolved_model_revision",
        "unmatched_detection_label",
    )


def test_hub_cache_metadata_resolves_the_weight_snapshot_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookups: list[tuple[str, str, str]] = []
    snapshot = (
        f"/home/user/.cache/huggingface/hub/models--org--caption/snapshots/"
        f"{_CAPTION_COMMIT}/model.safetensors"
    )

    def try_to_load_from_cache(*, repo_id: str, filename: str, revision: str) -> object:
        lookups.append((repo_id, filename, revision))
        if filename != "model.safetensors":
            return None
        return snapshot

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        cast(
            ModuleType,
            SimpleNamespace(try_to_load_from_cache=try_to_load_from_cache),
        ),
    )

    backend = TransformersCaptionBackend(
        "org/caption",
        revision="main",
        local_files_only=True,
    )

    assert lookups == [("org/caption", "model.safetensors", "main")]
    assert backend.descriptor.model_revision == _CAPTION_COMMIT
    assert backend.descriptor.durable_cache_eligible


def test_hub_cache_binary_weights_resolve_and_config_paths_do_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = "/cache/models--org--detection/snapshots"

    def binary_weights(*, repo_id: str, filename: str, revision: str) -> object:
        if filename == "pytorch_model.bin":
            return Path(f"{prefix}/{_DETECTION_COMMIT}/pytorch_model.bin")
        return None

    def config_only(*, repo_id: str, filename: str, revision: str) -> object:
        if filename == "config.json":
            return f"{prefix}/{_DETECTION_COMMIT}/config.json"
        return None

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        cast(ModuleType, SimpleNamespace(try_to_load_from_cache=binary_weights)),
    )
    resolved = TransformersDetectionBackend("org/detection", local_files_only=True)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        cast(ModuleType, SimpleNamespace(try_to_load_from_cache=config_only)),
    )
    unresolved = TransformersDetectionBackend("org/detection", local_files_only=True)

    assert resolved.descriptor.model_revision == _DETECTION_COMMIT
    assert unresolved.descriptor.model_revision is None
    assert not unresolved.descriptor.durable_cache_eligible


def test_hub_cache_metadata_resolves_sharded_weight_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cached = SimpleNamespace(
        commit_hash=_DETECTION_COMMIT,
        refs=frozenset({"release"}),
        files=(SimpleNamespace(file_name="model-00001-of-00002.safetensors"),),
    )
    repository = SimpleNamespace(repo_id="org/detection", revisions=(cached,))
    hub = SimpleNamespace(
        scan_cache_dir=lambda: SimpleNamespace(repos=(repository,)),
        try_to_load_from_cache=lambda **kwargs: pytest.fail("legacy lookup used"),
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", cast(ModuleType, hub))

    backend = TransformersDetectionBackend(
        "org/detection",
        revision="release",
        local_files_only=True,
    )

    assert backend.descriptor.model_revision == _DETECTION_COMMIT
    assert backend.descriptor.durable_cache_eligible


@pytest.mark.parametrize(
    "located",
    [
        None,
        object(),
        b"/cache/models--org--caption/snapshots/" + b"a" * 40 + b"/model.safetensors",
        "/cache/models--org--caption/snapshots/not-a-commit/model.safetensors",
        "/cache/models--org--caption/blobs/" + "a" * 40,
    ],
)
def test_unusable_hub_cache_results_stay_unresolved(
    monkeypatch: pytest.MonkeyPatch,
    located: object,
) -> None:
    def try_to_load_from_cache(*, repo_id: str, filename: str, revision: str) -> object:
        return located

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        cast(ModuleType, SimpleNamespace(try_to_load_from_cache=try_to_load_from_cache)),
    )

    backend = TransformersCaptionBackend("org/caption", local_files_only=True)

    assert backend.descriptor.model_revision is None


def test_hub_lookup_failures_do_not_break_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def exploding(*, repo_id: str, filename: str, revision: str) -> object:
        raise OSError("cache directory is unreadable")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        cast(ModuleType, SimpleNamespace(try_to_load_from_cache=exploding)),
    )
    exploding_backend = TransformersCaptionBackend("org/caption", local_files_only=True)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        cast(ModuleType, SimpleNamespace()),
    )
    incompatible_backend = TransformersCaptionBackend("org/caption", local_files_only=True)

    assert exploding_backend.descriptor.model_revision is None
    assert incompatible_backend.descriptor.model_revision is None


def test_resolved_revision_survives_hub_changes_after_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = f"/cache/models--org--caption/snapshots/{_CAPTION_COMMIT}/model.safetensors"

    def try_to_load_from_cache(*, repo_id: str, filename: str, revision: str) -> object:
        return snapshot if filename == "model.safetensors" else None

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        cast(ModuleType, SimpleNamespace(try_to_load_from_cache=try_to_load_from_cache)),
    )
    backend = TransformersCaptionBackend("org/caption", local_files_only=True)
    monkeypatch.delitem(sys.modules, "huggingface_hub")

    assert backend.descriptor == backend.descriptor
    assert backend.descriptor.model_revision == _CAPTION_COMMIT


@pytest.mark.asyncio
async def test_missing_extra_and_model_load_failures_are_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imports: list[str] = []

    def missing(name: str) -> object:
        imports.append(name)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", missing)
    backend = TransformersCaptionBackend()

    for _ in range(2):
        with pytest.raises(BackendUnavailableError) as raised:
            await backend.analyze(_backend_image(), CaptionRequest())
        assert raised.value.code == "transformers_extra_missing"

    assert imports == ["transformers"]

    pipeline = FakePipeline([])
    runtime = FakeRuntime(pipeline)

    def fail_pipeline(task: str, **kwargs: object) -> object:
        raise OSError

    monkeypatch.setattr(runtime.transformers, "pipeline", fail_pipeline)
    _install(monkeypatch, runtime)
    failed_load = TransformersDetectionBackend()
    request = DetectionRequest(labels=("cat",))

    for _ in range(2):
        with pytest.raises(BackendUnavailableError) as raised:
            await failed_load.analyze(_backend_image(), request)
        assert raised.value.code == "transformers_model_load_failed"


@pytest.mark.asyncio
async def test_partial_pipeline_setup_releases_created_resource(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NonCallablePipeline:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    created = NonCallablePipeline()
    runtime = SimpleNamespace(
        transformers=SimpleNamespace(pipeline=lambda *args, **kwargs: created),
        torch=SimpleNamespace(inference_mode=lambda: contextmanager(lambda: iter((None,)))()),
    )

    def load(name: str) -> object:
        return getattr(runtime, name)

    monkeypatch.setattr(importlib, "import_module", load)
    backend = TransformersCaptionBackend(revision=_CAPTION_COMMIT)

    for _ in range(2):
        with pytest.raises(BackendUnavailableError) as raised:
            await backend.analyze(_backend_image(), CaptionRequest())
        assert raised.value.code == "transformers_model_load_failed"

    assert created.close_calls == 1


@pytest.mark.asyncio
async def test_detection_maps_labels_clamps_region_filters_and_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = FakePipeline(
        [
            {
                "label": "cat.",
                "score": 0.9,
                "box": {"xmin": -5, "ymin": -4, "xmax": 60, "ymax": 50},
            },
            {
                "label": "dog",
                "score": 0.2,
                "box": {"xmin": 1, "ymin": 1, "xmax": 10, "ymax": 10},
            },
            {
                "label": "horse",
                "score": 0.95,
                "box": {"xmin": 1, "ymin": 1, "xmax": 10, "ymax": 10},
            },
            {
                "label": "Cat",
                "score": 0.8,
                "box": {"xmin": 70, "ymin": 20, "xmax": 80, "ymax": 30},
            },
        ]
    )
    runtime = FakeRuntime(pipeline)
    _install(monkeypatch, runtime)
    backend = TransformersDetectionBackend(revision=_DETECTION_COMMIT)
    region = Box(x_min=0.25, y_min=0.25, x_max=0.75, y_max=0.75)

    result = await backend.analyze(
        _backend_image(),
        DetectionRequest(
            region=region,
            labels=("Cat", "dog"),
            min_confidence=0.25,
        ),
    )

    assert pipeline.calls[0][0] == (50, 40)
    assert pipeline.calls[0][2] == {"candidate_labels": ["Cat", "dog"]}
    assert len(result.observations) == 1
    payload = cast(DetectionPayload, result.observations[0].payload)
    assert payload.label == "Cat"
    assert payload.attributes == ("cat.",)
    assert result.observations[0].confidence == 0.9
    assert result.observations[0].region == region
    assert tuple(item.code for item in result.warnings) == (
        "zero_area_detection_discarded",
        "unmatched_detection_label",
    )


@pytest.mark.asyncio
async def test_detection_nms_sorting_and_result_cap_are_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = FakePipeline(
        [
            {
                "label": "cat",
                "score": 0.8,
                "box": {"xmin": 11, "ymin": 11, "xmax": 51, "ymax": 51},
            },
            {
                "label": "dog",
                "score": 0.8,
                "box": {"xmin": 5, "ymin": 5, "xmax": 25, "ymax": 25},
            },
            {
                "label": "Cat",
                "score": 0.95,
                "box": {"xmin": 10, "ymin": 10, "xmax": 50, "ymax": 50},
            },
            {
                "label": "cat",
                "score": 0.7,
                "box": {"xmin": 60, "ymin": 10, "xmax": 80, "ymax": 30},
            },
        ]
    )
    runtime = FakeRuntime(pipeline)
    _install(monkeypatch, runtime)
    backend = TransformersDetectionBackend(revision=_DETECTION_COMMIT)

    result = await backend.analyze(
        _backend_image(100, 100),
        DetectionRequest(labels=("cat", "dog"), max_results=2),
    )

    assert tuple(item.confidence for item in result.observations) == (0.95, 0.8)
    assert tuple(cast(DetectionPayload, item.payload).label for item in result.observations) == (
        "cat",
        "dog",
    )
    assert result.warnings == ()


@pytest.mark.asyncio
async def test_detection_discards_zero_area_boxes_with_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = FakePipeline(
        [
            {
                "label": "cat",
                "score": 0.9,
                "box": {"x_min": 120, "y_min": 5, "x_max": 130, "y_max": 20},
            }
        ]
    )
    runtime = FakeRuntime(pipeline)
    _install(monkeypatch, runtime)
    backend = TransformersDetectionBackend(revision=_DETECTION_COMMIT)

    result = await backend.analyze(
        _backend_image(),
        DetectionRequest(labels=("cat",)),
    )

    assert result.observations == ()
    assert tuple(item.code for item in result.warnings) == ("zero_area_detection_discarded",)


@pytest.mark.asyncio
async def test_cancellation_finishes_worker_and_close_releases_pipeline_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingPipeline(FakePipeline):
        def __call__(self, image: Image.Image, **kwargs: object) -> object:
            started.set()
            if not release.wait(2.0):
                raise TimeoutError
            return super().__call__(image, **kwargs)

    pipeline = BlockingPipeline([{"generated_text": "Finished."}])
    runtime = FakeRuntime(pipeline)
    _install(monkeypatch, runtime)
    backend = TransformersCaptionBackend(revision=_CAPTION_COMMIT)
    task = asyncio.create_task(backend.analyze(_backend_image(), CaptionRequest()))
    assert await asyncio.to_thread(started.wait, 1.0)
    task.cancel()
    await asyncio.sleep(0)

    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.gather(backend.aclose(), backend.aclose())
    assert pipeline.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        await backend.analyze(_backend_image(), CaptionRequest())


def test_hub_cache_snapshot_is_not_trusted_when_the_loader_may_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookups: list[str] = []
    snapshot = f"/cache/models--org--caption/snapshots/{_CAPTION_COMMIT}/model.safetensors"

    def try_to_load_from_cache(*, repo_id: str, filename: str, revision: str) -> object:
        lookups.append(filename)
        return snapshot

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        cast(ModuleType, SimpleNamespace(try_to_load_from_cache=try_to_load_from_cache)),
    )
    # A mutable reference with downloads enabled may load a newer commit than the
    # one already cached, so the cached snapshot is never reported as the identity.
    mutable = TransformersCaptionBackend("org/caption", revision="main")
    default = TransformersCaptionBackend("org/caption")
    pinned = TransformersCaptionBackend("org/caption", revision=_CAPTION_COMMIT)

    assert lookups == []
    assert mutable.descriptor.model_revision is None
    assert default.descriptor.model_revision is None
    assert not mutable.descriptor.durable_cache_eligible
    # An explicit commit is exact without consulting the cache at all.
    assert pinned.descriptor.model_revision == _CAPTION_COMMIT
    assert pinned.descriptor.durable_cache_eligible
