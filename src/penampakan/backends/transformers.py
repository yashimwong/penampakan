"""Lazy local Transformers caption and zero-shot detection backends."""

from __future__ import annotations

import asyncio
import importlib
import math
import os
import re
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, suppress
from io import BytesIO
from pathlib import PurePath
from typing import Protocol, TypeVar, cast

from PIL import Image
from PIL.Image import Image as PillowImage

from penampakan.backends._optional import require_extra
from penampakan.errors import (
    BackendError,
    BackendUnavailableError,
    InvalidBackendOutputError,
)
from penampakan.image.geometry import box_to_pixels
from penampakan.models import (
    JSON_VALUE_ADAPTER,
    BackendDescriptor,
    BackendImage,
    Box,
    Capability,
    CapabilityDescriptor,
    CaptionPayload,
    CaptionRequest,
    DetectionPayload,
    DetectionRequest,
    JsonValue,
    ObservationDraft,
    VisionRequest,
    VisionResult,
    WarningInfo,
)

_T = TypeVar("_T")
_SENTENCE_BOUNDARY = re.compile(r"[.!?]+(?:[\"')\]]*)")
_LABEL_TRAILING_PUNCTUATION = " \t\r\n.,:;!?"
_NMS_IOU = 0.8
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
_SNAPSHOT_DIRECTORY = "snapshots"
_WEIGHT_FILENAMES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)
_WEIGHT_SHARD = re.compile(r"(?:model|pytorch_model)-\d{5}-of-\d{5}\.(?:safetensors|bin)")


class _Pipeline(Protocol):
    def __call__(self, image: PillowImage, **kwargs: object) -> object: ...


class _TransformersModule(Protocol):
    pipeline: Callable[..., object]


class _TorchModule(Protocol):
    inference_mode: Callable[[], AbstractContextManager[object]]


class _Detection:
    __slots__ = ("attributes", "box", "label", "score")

    def __init__(
        self,
        *,
        label: str,
        attributes: tuple[str, ...],
        score: float,
        box: Box,
    ) -> None:
        self.label = label
        self.attributes = attributes
        self.score = score
        self.box = box


def _clean_identifier(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    if not value or value != value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be non-blank, unpadded, and NUL-free")
    return value


def _clean_optional_identifier(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _clean_identifier(value, name)


def _validate_device(value: object) -> str | int | None:
    if value is None or isinstance(value, str):
        return None if value is None else _clean_identifier(value, "device")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise TypeError("device must be text, an integer, or None")


def _strict_json_mapping(
    value: Mapping[str, JsonValue] | None,
) -> dict[str, JsonValue]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("generation_kwargs must be a mapping")
    result: dict[str, JsonValue] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key or "\x00" in key:
            raise ValueError("generation_kwargs keys must be non-empty NUL-free text")
        if key == "trust_remote_code":
            raise ValueError("trust_remote_code is not a generation option")
        try:
            result[key] = JSON_VALUE_ADAPTER.validate_python(item, strict=True)
        except Exception as error:
            raise ValueError("generation_kwargs must contain strict JSON values") from error
    return result


def _snapshot_commit(located: object) -> str | None:
    if not isinstance(located, (str, os.PathLike)):
        return None
    try:
        path = os.fspath(located)
    except (TypeError, ValueError):
        return None
    if not isinstance(path, str) or not path:
        return None
    parts = PurePath(path).parts
    for index in range(len(parts) - 1, 0, -1):
        candidate = parts[index]
        if parts[index - 1] == _SNAPSHOT_DIRECTORY and _COMMIT_SHA.fullmatch(candidate):
            return candidate
    return None


def _cached_revision_from_metadata(
    hub: object,
    model_id: str,
    revision: str | None,
) -> str | None:
    scan = getattr(hub, "scan_cache_dir", None)
    if not callable(scan):
        return None
    try:
        repositories = scan().repos
    except Exception:
        return None
    selected = revision if revision is not None else "main"
    try:
        iterator = iter(repositories)
    except TypeError:
        return None
    for repository in iterator:
        if getattr(repository, "repo_id", None) != model_id:
            continue
        cached_revisions = getattr(repository, "revisions", ())
        try:
            revision_iterator = iter(cached_revisions)
        except TypeError:
            continue
        for cached in revision_iterator:
            commit = getattr(cached, "commit_hash", None)
            refs = getattr(cached, "refs", ())
            if not isinstance(commit, str) or _COMMIT_SHA.fullmatch(commit) is None:
                continue
            if selected != commit and selected not in refs:
                continue
            files = getattr(cached, "files", ())
            try:
                names = (getattr(item, "file_name", None) for item in files)
                if any(
                    isinstance(name, str)
                    and (name in _WEIGHT_FILENAMES or _WEIGHT_SHARD.fullmatch(name))
                    for name in names
                ):
                    return commit
            except TypeError:
                continue
    return None


def _resolve_model_revision(
    model_id: str,
    revision: str | None,
    *,
    local_files_only: bool,
) -> str | None:
    """Resolve the immutable weight snapshot commit without contacting the network.

    An explicit commit revision is exact on its own. A mutable reference resolves
    only through the local Hub cache, and only when the loader is restricted to
    local files: otherwise the loader may fetch a newer commit for that reference
    and the cached snapshot would not be the weight identity actually loaded.
    """
    if revision is not None and _COMMIT_SHA.fullmatch(revision):
        return revision
    if not local_files_only:
        return None
    try:
        hub = importlib.import_module("huggingface_hub")
    except ImportError:
        return None
    try:
        commit = _cached_revision_from_metadata(hub, model_id, revision)
    except Exception:
        commit = None
    if commit is not None:
        return commit
    lookup = getattr(hub, "try_to_load_from_cache", None)
    if not callable(lookup):
        return None
    for filename in _WEIGHT_FILENAMES:
        try:
            located = lookup(
                repo_id=model_id,
                filename=filename,
                revision=revision if revision is not None else "main",
            )
        except Exception:
            continue
        commit = _snapshot_commit(located)
        if commit is not None:
            return commit
    return None


def _unresolved_warning(resolved_revision: str | None) -> tuple[WarningInfo, ...]:
    if resolved_revision is not None:
        return ()
    return (
        WarningInfo(
            code="unresolved_model_revision",
            message=(
                "The exact model weight revision is unresolved; pin an immutable "
                "commit revision for reproducible inference and durable caching."
            ),
        ),
    )


def _normalize_caption(value: str) -> str:
    return " ".join(value.replace("\x00", "").split())


def _truncate_sentences(value: str, maximum: int) -> str:
    boundaries = tuple(_SENTENCE_BOUNDARY.finditer(value))
    if len(boundaries) < maximum:
        return value
    boundary = boundaries[maximum - 1]
    if boundary.end() == len(value) or value[boundary.end() :].strip():
        return value[: boundary.end()].strip()
    return value


def _decode_crop(
    image: BackendImage,
    region: Box | None,
) -> tuple[PillowImage, int, int]:
    with Image.open(BytesIO(image.content)) as decoded:
        decoded.load()
        working = decoded.convert("RGB")
    left = 0
    top = 0
    if region is not None:
        pixels = box_to_pixels(region, image.asset.width, image.asset.height)
        cropped = working.crop(pixels.as_tuple())
        working.close()
        working = cropped
        left = pixels.left
        top = pixels.top
    return working, left, top


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidBackendOutputError(code=f"invalid_detection_{field}")
    result = float(value)
    if not math.isfinite(result):
        raise InvalidBackendOutputError(code=f"invalid_detection_{field}")
    return result


def _mapping(value: object, code: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise InvalidBackendOutputError(code=code)
    return cast(Mapping[object, object], value)


def _box_coordinate(box: Mapping[object, object], primary: str, alternate: str) -> float:
    if primary in box:
        value = box[primary]
    elif alternate in box:
        value = box[alternate]
    else:
        raise InvalidBackendOutputError(code="invalid_detection_box")
    return _number(value, "box")


def _label_key(value: str) -> str:
    return value.casefold().rstrip(_LABEL_TRAILING_PUNCTUATION)


def _pipeline_items(value: object) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise InvalidBackendOutputError(code="invalid_transformers_output")
    return tuple(value)


def _create_pipeline(
    transformers: _TransformersModule,
    torch: _TorchModule,
    *,
    task: str,
    model_id: str,
    revision: str | None,
    device: str | int | None,
    local_files_only: bool,
) -> tuple[object, object]:
    created = transformers.pipeline(
        task,
        model=model_id,
        revision=revision,
        device=device,
        trust_remote_code=False,
        model_kwargs={"local_files_only": local_files_only},
    )
    return created, torch.inference_mode


class _TransformersBackend:
    def __init__(
        self,
        *,
        descriptor: BackendDescriptor,
        task: str,
        model_id: str,
        revision: str | None,
        device: str | int | None,
        local_files_only: bool,
    ) -> None:
        if not isinstance(local_files_only, bool):
            raise TypeError("local_files_only must be a boolean")
        self._descriptor = descriptor
        self._task = task
        self._model_id = model_id
        self._revision = revision
        self._device = device
        self._local_files_only = local_files_only
        self._revision_warnings = _unresolved_warning(descriptor.model_revision)
        self._pipeline: _Pipeline | None = None
        self._inference_mode: Callable[[], AbstractContextManager[object]] | None = None
        self._load_error: BackendUnavailableError | None = None
        self._semaphore = asyncio.Semaphore(1)
        self._close_task: asyncio.Task[None] | None = None

    @property
    def descriptor(self) -> BackendDescriptor:
        """Return stable adapter and configured model identity."""
        return self._descriptor

    async def _serialized(self, operation: Callable[[], _T]) -> _T:
        if self._close_task is not None:
            raise RuntimeError("backend is closed")
        async with self._semaphore:
            if self._close_task is not None:
                raise RuntimeError("backend is closed")
            worker = asyncio.create_task(asyncio.to_thread(operation))
            try:
                return await asyncio.shield(worker)
            except asyncio.CancelledError:
                while not worker.done():
                    try:
                        await asyncio.shield(worker)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                with suppress(Exception):
                    worker.result()
                raise

    def _infer(self, image: PillowImage, **kwargs: object) -> object:
        pipeline, inference_mode = self._ensure_pipeline()
        try:
            with inference_mode():
                return pipeline(image, **kwargs)
        except InvalidBackendOutputError:
            raise
        except Exception as error:
            raise BackendError(code="transformers_inference_failed", cause=error) from error

    def _ensure_pipeline(
        self,
    ) -> tuple[_Pipeline, Callable[[], AbstractContextManager[object]]]:
        if self._load_error is not None:
            raise self._load_error
        if self._pipeline is None or self._inference_mode is None:
            try:
                transformers = importlib.import_module("transformers")
                torch = importlib.import_module("torch")
            except (ImportError, ModuleNotFoundError) as error:
                unavailable = BackendUnavailableError(
                    code="transformers_extra_missing",
                    retryable=False,
                    cause=error,
                )
                self._load_error = unavailable
                raise unavailable from error
            created: object | None = None
            try:
                created, raw_inference_mode = _create_pipeline(
                    cast(_TransformersModule, transformers),
                    cast(_TorchModule, torch),
                    task=self._task,
                    model_id=self._model_id,
                    revision=self._revision,
                    device=self._device,
                    local_files_only=self._local_files_only,
                )
                inference_mode = cast(
                    Callable[[], AbstractContextManager[object]],
                    raw_inference_mode,
                )
                if not callable(created) or not callable(inference_mode):
                    raise TypeError("Transformers pipeline boundaries are not callable")
            except Exception as error:
                close = getattr(created, "close", None)
                if callable(close):
                    with suppress(Exception):
                        close()
                unavailable = BackendUnavailableError(
                    code="transformers_model_load_failed",
                    retryable=False,
                    cause=error,
                )
                self._load_error = unavailable
                raise unavailable from error
            self._pipeline = cast(_Pipeline, created)
            self._inference_mode = inference_mode
        return self._pipeline, self._inference_mode

    async def aclose(self) -> None:
        """Wait for active inference and release loaded pipeline resources once."""
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_owned())
        await asyncio.shield(self._close_task)

    async def _close_owned(self) -> None:
        async with self._semaphore:
            pipeline = self._pipeline
            self._pipeline = None
            self._inference_mode = None
            if pipeline is None:
                return
            close = getattr(pipeline, "close", None)
            if callable(close):
                try:
                    await asyncio.to_thread(close)
                except Exception as error:
                    raise BackendError(
                        code="transformers_close_failed",
                        cause=error,
                    ) from error


class TransformersCaptionBackend(_TransformersBackend):
    """Lazy local image captioning through a Transformers v4 pipeline."""

    def __init__(
        self,
        model_id: str = "Salesforce/blip-image-captioning-base",
        *,
        revision: str | None = None,
        device: str | int | None = None,
        local_files_only: bool = False,
        generation_kwargs: Mapping[str, JsonValue] | None = None,
    ) -> None:
        require_extra("transformers", "transformers", "torch")
        selected_model = _clean_identifier(model_id, "model_id")
        selected_revision = _clean_optional_identifier(revision, "revision")
        selected_device = _validate_device(device)
        self._generation_kwargs = _strict_json_mapping(generation_kwargs)
        descriptor = BackendDescriptor(
            name="transformers.caption",
            version="caption-v1",
            model_id=selected_model,
            model_revision=_resolve_model_revision(
                selected_model,
                selected_revision,
                local_files_only=local_files_only,
            ),
            capabilities=(CapabilityDescriptor(capability=Capability.CAPTION),),
            max_concurrency=1,
        )
        super().__init__(
            descriptor=descriptor,
            task="image-to-text",
            model_id=selected_model,
            revision=selected_revision,
            device=selected_device,
            local_files_only=local_files_only,
        )

    def supports(self, request: VisionRequest) -> bool:
        """Return whether this backend supports the complete caption request."""
        return isinstance(request, CaptionRequest) and request.focus is None

    async def analyze(self, image: BackendImage, request: VisionRequest) -> VisionResult:
        """Generate the first non-empty bounded caption for an image or crop."""
        if not self.supports(request) or not isinstance(request, CaptionRequest):
            raise ValueError("request is unsupported")
        return await self._serialized(lambda: self._analyze_sync(image, request))

    def _analyze_sync(self, image: BackendImage, request: CaptionRequest) -> VisionResult:
        working, _, _ = _decode_crop(image, request.region)
        try:
            output = self._infer(working, **cast(dict[str, object], self._generation_kwargs))
        finally:
            working.close()
        caption = ""
        for item in _pipeline_items(output):
            mapping = _mapping(item, "invalid_caption_output")
            generated = mapping.get("generated_text")
            if not isinstance(generated, str):
                continue
            normalized = _normalize_caption(generated)
            if normalized:
                caption = _truncate_sentences(normalized, request.max_sentences)
                break
        warnings = list(self._revision_warnings)
        if not caption:
            warnings.append(
                WarningInfo(
                    code="no_caption_generated",
                    message="The caption model returned no non-empty caption.",
                )
            )
            return VisionResult(observations=(), warnings=tuple(warnings))
        return VisionResult(
            observations=(
                ObservationDraft(
                    payload=CaptionPayload(text=caption),
                    region=request.region,
                ),
            ),
            warnings=tuple(warnings),
        )


class TransformersDetectionBackend(_TransformersBackend):
    """Lazy local open-vocabulary detection through a Transformers pipeline."""

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        *,
        revision: str | None = None,
        device: str | int | None = None,
        local_files_only: bool = False,
    ) -> None:
        require_extra("transformers", "transformers", "torch")
        selected_model = _clean_identifier(model_id, "model_id")
        selected_revision = _clean_optional_identifier(revision, "revision")
        selected_device = _validate_device(device)
        descriptor = BackendDescriptor(
            name="transformers.detection",
            version="detection-v1",
            model_id=selected_model,
            model_revision=_resolve_model_revision(
                selected_model,
                selected_revision,
                local_files_only=local_files_only,
            ),
            capabilities=(
                CapabilityDescriptor(
                    capability=Capability.DETECT,
                    features=frozenset({"detect.open_vocabulary"}),
                ),
            ),
            max_concurrency=1,
        )
        super().__init__(
            descriptor=descriptor,
            task="zero-shot-object-detection",
            model_id=selected_model,
            revision=selected_revision,
            device=selected_device,
            local_files_only=local_files_only,
        )

    def supports(self, request: VisionRequest) -> bool:
        """Return whether open-vocabulary labels make this request supported."""
        return isinstance(request, DetectionRequest) and bool(request.labels)

    async def analyze(self, image: BackendImage, request: VisionRequest) -> VisionResult:
        """Detect requested labels and normalize boxes to full-asset coordinates."""
        if not self.supports(request) or not isinstance(request, DetectionRequest):
            raise ValueError("request is unsupported")
        return await self._serialized(lambda: self._analyze_sync(image, request))

    def _analyze_sync(self, image: BackendImage, request: DetectionRequest) -> VisionResult:
        working, offset_x, offset_y = _decode_crop(image, request.region)
        try:
            output = self._infer(working, candidate_labels=list(request.labels))
            detections, warnings = self._detections(
                output,
                request,
                working.width,
                working.height,
                offset_x,
                offset_y,
                image.asset.width,
                image.asset.height,
            )
        finally:
            working.close()
        selected = self._nms(detections)[: request.max_results]
        drafts = tuple(
            ObservationDraft(
                payload=DetectionPayload(
                    label=item.label,
                    attributes=item.attributes,
                ),
                region=item.box,
                confidence=item.score,
            )
            for item in selected
        )
        return VisionResult(
            observations=drafts,
            warnings=(*self._revision_warnings, *warnings),
        )

    @staticmethod
    def _detections(
        output: object,
        request: DetectionRequest,
        crop_width: int,
        crop_height: int,
        offset_x: int,
        offset_y: int,
        full_width: int,
        full_height: int,
    ) -> tuple[list[_Detection], tuple[WarningInfo, ...]]:
        labels = {_label_key(label): label for label in request.labels}
        detections: list[_Detection] = []
        zero_area = 0
        unmatched = 0
        for item in _pipeline_items(output):
            mapping = _mapping(item, "invalid_detection_output")
            score = _number(mapping.get("score"), "score")
            if not 0.0 <= score <= 1.0:
                raise InvalidBackendOutputError(code="invalid_detection_score")
            if score < request.min_confidence:
                continue
            reported = mapping.get("label")
            if not isinstance(reported, str) or not reported.strip() or "\x00" in reported:
                raise InvalidBackendOutputError(code="invalid_detection_label")
            normalized_reported = reported.strip()
            label = labels.get(_label_key(normalized_reported))
            if label is None:
                unmatched += 1
                continue
            if len(normalized_reported) > 100:
                raise InvalidBackendOutputError(code="invalid_detection_label")
            raw_box = _mapping(mapping.get("box"), "invalid_detection_box")
            left = max(0.0, min(float(crop_width), _box_coordinate(raw_box, "xmin", "x_min")))
            top = max(0.0, min(float(crop_height), _box_coordinate(raw_box, "ymin", "y_min")))
            right = max(
                0.0,
                min(float(crop_width), _box_coordinate(raw_box, "xmax", "x_max")),
            )
            bottom = max(
                0.0,
                min(float(crop_height), _box_coordinate(raw_box, "ymax", "y_max")),
            )
            if right <= left or bottom <= top:
                zero_area += 1
                continue
            x_min = (offset_x + left) / full_width
            y_min = (offset_y + top) / full_height
            x_max = (offset_x + right) / full_width
            y_max = (offset_y + bottom) / full_height
            if request.region is not None:
                x_min = max(request.region.x_min, x_min)
                y_min = max(request.region.y_min, y_min)
                x_max = min(request.region.x_max, x_max)
                y_max = min(request.region.y_max, y_max)
            if x_max <= x_min or y_max <= y_min:
                zero_area += 1
                continue
            attributes = () if normalized_reported == label else (normalized_reported,)
            detections.append(
                _Detection(
                    label=label,
                    attributes=attributes,
                    score=score,
                    box=Box(
                        x_min=x_min,
                        y_min=y_min,
                        x_max=x_max,
                        y_max=y_max,
                    ),
                )
            )
        warnings: list[WarningInfo] = []
        if zero_area:
            warnings.append(
                WarningInfo(
                    code="zero_area_detection_discarded",
                    message="Zero-area detection boxes were discarded after clamping.",
                    details={"count": zero_area},
                )
            )
        if unmatched:
            warnings.append(
                WarningInfo(
                    code="unmatched_detection_label",
                    message="Detections outside the requested label set were discarded.",
                    details={"count": unmatched},
                )
            )
        return detections, tuple(warnings)

    @staticmethod
    def _nms(detections: list[_Detection]) -> list[_Detection]:
        ordered = sorted(
            detections,
            key=lambda item: (
                -item.score,
                item.box.x_min,
                item.box.y_min,
                item.box.x_max,
                item.box.y_max,
                item.label.casefold(),
                item.label,
            ),
        )
        retained: list[_Detection] = []
        for candidate in ordered:
            if all(
                existing.label.casefold() != candidate.label.casefold()
                or candidate.box.iou(existing.box) < _NMS_IOU
                for existing in retained
            ):
                retained.append(candidate)
        return retained


__all__ = ["TransformersCaptionBackend", "TransformersDetectionBackend"]
