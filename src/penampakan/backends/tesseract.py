"""Optional structured OCR backend powered by a caller-installed Tesseract runtime."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import math
import os
import re
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from typing import Protocol, cast

from PIL import Image
from PIL.Image import Image as PillowImage

from penampakan.backends._optional import require_extra
from penampakan.errors import (
    BackendUnavailableError,
    InvalidBackendOutputError,
    SessionClosedError,
)
from penampakan.image.geometry import (
    PixelBox,
    box_to_pixels,
    pixels_to_box,
    remap_pixel_box_from_region,
)
from penampakan.models import (
    BackendDescriptor,
    BackendImage,
    Box,
    Capability,
    CapabilityDescriptor,
    ObservationDraft,
    OCRRequest,
    TextPayload,
    VisionRequest,
    VisionResult,
    WarningInfo,
)

_ADAPTER_VERSION = "1.0"
_BACKEND_NAME = "tesseract"
_ENGINE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+:~-]{0,63}$")
_LANGUAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_PAYLOAD_LANGUAGE = re.compile(r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$")
_MODE_PSM = {"auto": 3, "sparse": 11, "dense": 6, "single_line": 7}
_PYTESSERACT_LOCK = threading.Lock()


class _OutputNamespace(Protocol):
    DICT: object


class _PytesseractCommand(Protocol):
    tesseract_cmd: str


class _PytesseractModule(Protocol):
    Output: _OutputNamespace
    pytesseract: _PytesseractCommand

    def get_tesseract_version(self) -> object: ...

    def get_languages(self, config: str = "") -> Sequence[str]: ...

    def image_to_data(
        self,
        image: PillowImage,
        *,
        lang: str,
        config: str,
        output_type: object,
    ) -> Mapping[str, Sequence[object]]: ...


@dataclass(frozen=True, slots=True)
class _Diagnostics:
    module: _PytesseractModule
    runtime_version: str
    available_languages: frozenset[str]


@dataclass(frozen=True, slots=True)
class _DiagnosticFailure:
    code: str
    cause: BaseException


@dataclass(frozen=True, slots=True)
class _Word:
    page: int
    block: int
    paragraph: int
    line: int
    word: int
    index: int
    text: str
    confidence: float | None
    box: PixelBox | None

    @property
    def line_key(self) -> tuple[int, int, int, int]:
        return self.page, self.block, self.paragraph, self.line

    @property
    def reading_key(self) -> tuple[int, int, int, int, int, int]:
        return (*self.line_key, self.word, self.index)


class _InvalidTSVError(ValueError):
    pass


class TesseractBackend:
    """Provide structured line OCR through a lazily diagnosed Tesseract runtime.

    ``engine_version`` pins the concrete Tesseract engine build this backend is
    configured against. Executing the binary is the only way to learn that
    version, and construction stays free of subprocesses, so the value is
    supplied by the caller and verified against the running binary on first
    analysis; a mismatch fails as ``BackendUnavailableError``. When it is
    pinned, the engine version becomes part of ``BackendDescriptor.version``,
    which is the identity that reaches provenance and the perception cache key,
    so results from different engine builds never share a durable cache entry.
    When it is not pinned, every result carries
    ``WarningInfo(code="unpinned_engine_version")`` reporting the engine version
    that actually produced it.
    """

    def __init__(
        self,
        *,
        executable: str | os.PathLike[str] | None = None,
        engine_version: str | None = None,
        languages: Sequence[str] = ("eng",),
        config: str = "",
        max_concurrency: int = 2,
    ) -> None:
        require_extra("ocr", "pytesseract")
        self._executable = self._validate_executable(executable)
        self._engine_version = self._validate_engine_version(engine_version)
        self._languages = self._validate_languages(languages)
        self._config = self._validate_config(config)
        concurrency = self._validate_concurrency(max_concurrency)
        self._descriptor = BackendDescriptor(
            name=_BACKEND_NAME,
            version=self._descriptor_version(
                self._engine_version,
                self._languages,
                self._config,
            ),
            capabilities=(
                CapabilityDescriptor(
                    capability=Capability.OCR,
                    features=frozenset({"ocr.languages", "ocr.word_boxes"}),
                ),
            ),
            max_concurrency=concurrency,
        )
        self._diagnostics: _Diagnostics | None = None
        self._diagnostic_failure: _DiagnosticFailure | None = None
        self._diagnostic_task: asyncio.Task[_Diagnostics] | None = None
        self._diagnostic_lock = asyncio.Lock()
        self._workers: set[asyncio.Task[VisionResult]] = set()
        self._state_lock = asyncio.Lock()
        self._idle = asyncio.Event()
        self._idle.set()
        self._active_calls = 0
        self._closing = False
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    @property
    def descriptor(self) -> BackendDescriptor:
        """Return the stable adapter and selected-language descriptor."""
        return self._descriptor

    def supports(self, request: VisionRequest) -> bool:
        """Return whether this adapter supports the complete OCR request."""
        if not isinstance(request, OCRRequest):
            return False
        return not request.languages or set(request.languages).issubset(self._languages)

    async def analyze(self, image: BackendImage, request: VisionRequest) -> VisionResult:
        """Run bounded structured OCR without blocking the event loop."""
        if not isinstance(image, BackendImage):
            raise TypeError("image must be a BackendImage")
        if not isinstance(request, OCRRequest) or not self.supports(request):
            raise ValueError("request is unsupported")
        await self._begin_call()
        worker_owned_call = False
        try:
            diagnostics = await self._ensure_diagnostics()
            worker = asyncio.create_task(
                asyncio.to_thread(self._analyze_sync, diagnostics, image, request)
            )
            self._workers.add(worker)
            worker.add_done_callback(self._finish_worker)
            worker_owned_call = True
            try:
                return await asyncio.shield(worker)
            except _InvalidTSVError as error:
                raise InvalidBackendOutputError(
                    code="tesseract_invalid_output",
                    backend_name=_BACKEND_NAME,
                    cause=error,
                ) from error
            except asyncio.CancelledError:
                raise
            except Exception as error:
                raise BackendUnavailableError(
                    code="tesseract_inference_failed",
                    backend_name=_BACKEND_NAME,
                    cause=error,
                ) from error
        finally:
            if not worker_owned_call:
                self._end_call()

    async def aclose(self) -> None:
        """Wait for active inference and close the stateless adapter idempotently."""
        async with self._state_lock:
            if self._close_task is None:
                self._closing = True
                self._close_task = asyncio.create_task(self._close_owned())
            close_task = self._close_task
        await asyncio.shield(close_task)

    async def _begin_call(self) -> None:
        async with self._state_lock:
            if self._closing or self._closed:
                raise SessionClosedError()
            self._active_calls += 1
            self._idle.clear()

    def _end_call(self) -> None:
        self._active_calls -= 1
        if self._active_calls == 0:
            self._idle.set()

    async def _close_owned(self) -> None:
        await self._idle.wait()
        diagnostic_task = self._diagnostic_task
        if diagnostic_task is not None and not diagnostic_task.done():
            await asyncio.gather(diagnostic_task, return_exceptions=True)
        self._closed = True

    def _finish_worker(self, task: asyncio.Task[VisionResult]) -> None:
        self._workers.discard(task)
        if not task.cancelled():
            task.exception()
        self._end_call()

    async def _ensure_diagnostics(self) -> _Diagnostics:
        async with self._diagnostic_lock:
            if self._diagnostics is not None:
                return self._diagnostics
            if self._diagnostic_failure is not None:
                raise self._unavailable(self._diagnostic_failure)
            if self._diagnostic_task is None:
                self._diagnostic_task = asyncio.create_task(asyncio.to_thread(self._probe_runtime))
                self._diagnostic_task.add_done_callback(self._consume_diagnostic_failure)
            diagnostic_task = self._diagnostic_task
            try:
                diagnostics = await asyncio.shield(diagnostic_task)
            except asyncio.CancelledError:
                raise
            except ImportError as error:
                failure = _DiagnosticFailure("ocr_extra_missing", error)
                self._diagnostic_failure = failure
                raise self._unavailable(failure) from error
            except Exception as error:
                failure = _DiagnosticFailure("tesseract_binary_unavailable", error)
                self._diagnostic_failure = failure
                raise self._unavailable(failure) from error
            if (
                self._engine_version is not None
                and diagnostics.runtime_version != self._engine_version
            ):
                mismatch = RuntimeError(
                    "the running Tesseract engine version differs from the pinned version"
                )
                failure = _DiagnosticFailure("tesseract_engine_version_mismatch", mismatch)
                self._diagnostic_failure = failure
                raise self._unavailable(failure) from mismatch
            missing = tuple(
                language
                for language in self._languages
                if language not in diagnostics.available_languages
            )
            if missing:
                language_error = RuntimeError("configured Tesseract language data is unavailable")
                failure = _DiagnosticFailure("tesseract_language_unavailable", language_error)
                self._diagnostic_failure = failure
                raise self._unavailable(failure) from language_error
            self._diagnostics = diagnostics
            return diagnostics

    def _probe_runtime(self) -> _Diagnostics:
        imported = importlib.import_module("pytesseract")
        module = cast(_PytesseractModule, imported)
        with self._configured_executable(module):
            version = self._safe_runtime_version(module.get_tesseract_version())
            available = module.get_languages(config=self._config)
        if isinstance(available, (str, bytes, bytearray)) or not isinstance(available, Sequence):
            raise _InvalidTSVError("Tesseract language diagnostics are invalid")
        if any(not isinstance(item, str) for item in available):
            raise _InvalidTSVError("Tesseract language diagnostics are invalid")
        return _Diagnostics(
            module=module,
            runtime_version=version,
            available_languages=frozenset(available),
        )

    def _analyze_sync(
        self,
        diagnostics: _Diagnostics,
        image: BackendImage,
        request: OCRRequest,
    ) -> VisionResult:
        with Image.open(BytesIO(image.content)) as decoded:
            decoded.load()
            working = decoded.copy()
        region_pixels: PixelBox | None = None
        try:
            if request.region is not None:
                region_pixels = box_to_pixels(
                    request.region,
                    image.asset.width,
                    image.asset.height,
                )
                cropped = working.crop(region_pixels.as_tuple())
                working.close()
                working = cropped
            selected_languages = request.languages or self._languages
            with self._configured_executable(diagnostics.module):
                data = diagnostics.module.image_to_data(
                    working,
                    lang="+".join(selected_languages),
                    config=self._effective_config(request),
                    output_type=diagnostics.module.Output.DICT,
                )
            return self._build_result(
                data,
                diagnostics.runtime_version,
                request,
                selected_languages,
                working.width,
                working.height,
                image.asset.width,
                image.asset.height,
                region_pixels,
            )
        finally:
            working.close()

    def _build_result(
        self,
        data: Mapping[str, Sequence[object]],
        runtime_version: str,
        request: OCRRequest,
        selected_languages: tuple[str, ...],
        working_width: int,
        working_height: int,
        asset_width: int,
        asset_height: int,
        region_pixels: PixelBox | None,
    ) -> VisionResult:
        words = self._parse_words(data, working_width, working_height)
        grouped: dict[tuple[int, int, int, int], list[_Word]] = {}
        for word in sorted(words, key=lambda item: item.reading_key):
            grouped.setdefault(word.line_key, []).append(word)
        language = self._payload_language(selected_languages)
        drafts: list[ObservationDraft] = []
        unfiltered_count = 0
        for line_words in grouped.values():
            text = " ".join(word.text for word in line_words).strip()
            if not text:
                continue
            unfiltered_count += 1
            confidence = self._line_confidence(line_words)
            if request.min_confidence is not None and (
                confidence is None or confidence < request.min_confidence
            ):
                continue
            local_box = self._line_box(line_words)
            region = self._full_asset_box(
                local_box,
                request.region,
                region_pixels,
                working_width,
                working_height,
                asset_width,
                asset_height,
            )
            drafts.append(
                ObservationDraft(
                    payload=TextPayload(text=text, language=language, block_kind="line"),
                    region=region,
                    confidence=confidence,
                )
            )
        warnings: tuple[WarningInfo, ...] = ()
        if not unfiltered_count:
            warnings = (
                WarningInfo(
                    code="no_text_detected",
                    message="Tesseract detected no text in the requested image region.",
                ),
            )
        elif request.min_confidence is not None and not drafts:
            warnings = (
                WarningInfo(
                    code="no_text_above_threshold",
                    message="No OCR text met the requested confidence threshold.",
                ),
            )
        return VisionResult(
            observations=tuple(drafts),
            warnings=(*warnings, *self._identity_warnings(runtime_version)),
        )

    def _identity_warnings(self, runtime_version: str) -> tuple[WarningInfo, ...]:
        """Report the engine build when it is absent from the backend identity."""
        if self._engine_version is not None:
            return ()
        return (
            WarningInfo(
                code="unpinned_engine_version",
                message=(
                    "The Tesseract engine version is not part of this backend identity, "
                    "so durable cache entries cannot be attributed to one engine build; "
                    "pass engine_version to pin it."
                ),
                details={"engine_version": runtime_version},
            ),
        )

    @classmethod
    def _parse_words(
        cls,
        data: Mapping[str, Sequence[object]],
        width: int,
        height: int,
    ) -> tuple[_Word, ...]:
        if not isinstance(data, Mapping):
            raise _InvalidTSVError("Tesseract data must be a mapping")
        required = (
            "page_num",
            "block_num",
            "par_num",
            "line_num",
            "word_num",
            "left",
            "top",
            "width",
            "height",
            "conf",
            "text",
        )
        columns: dict[str, Sequence[object]] = {}
        for name in required:
            column = data.get(name)
            if not isinstance(column, Sequence) or isinstance(column, (str, bytes, bytearray)):
                raise _InvalidTSVError("Tesseract data has an invalid column")
            columns[name] = column
        lengths = {len(column) for column in columns.values()}
        if len(lengths) != 1:
            raise _InvalidTSVError("Tesseract data columns have inconsistent lengths")
        count = next(iter(lengths), 0)
        words: list[_Word] = []
        for index in range(count):
            text_value = columns["text"][index]
            if not isinstance(text_value, str):
                raise _InvalidTSVError("Tesseract text must be a string")
            text = text_value.strip()
            if not text:
                continue
            left = cls._integer(columns["left"][index])
            top = cls._integer(columns["top"][index])
            box_width = cls._integer(columns["width"][index])
            box_height = cls._integer(columns["height"][index])
            words.append(
                _Word(
                    page=cls._integer(columns["page_num"][index]),
                    block=cls._integer(columns["block_num"][index]),
                    paragraph=cls._integer(columns["par_num"][index]),
                    line=cls._integer(columns["line_num"][index]),
                    word=cls._integer(columns["word_num"][index]),
                    index=index,
                    text=text,
                    confidence=cls._confidence(columns["conf"][index]),
                    box=cls._pixel_box(left, top, box_width, box_height, width, height),
                )
            )
        return tuple(words)

    @staticmethod
    def _integer(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise _InvalidTSVError("Tesseract integer field is invalid")
        try:
            parsed = int(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise _InvalidTSVError("Tesseract integer field is invalid") from error
        return parsed

    @staticmethod
    def _confidence(value: object) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(parsed) or parsed < 0.0 or parsed > 100.0:
            return None
        return parsed / 100.0

    @staticmethod
    def _pixel_box(
        left: int,
        top: int,
        width: int,
        height: int,
        image_width: int,
        image_height: int,
    ) -> PixelBox | None:
        if width <= 0 or height <= 0:
            return None
        clipped_left = max(0, min(image_width, left))
        clipped_top = max(0, min(image_height, top))
        clipped_right = max(0, min(image_width, left + width))
        clipped_bottom = max(0, min(image_height, top + height))
        if clipped_left >= clipped_right or clipped_top >= clipped_bottom:
            return None
        return PixelBox(clipped_left, clipped_top, clipped_right, clipped_bottom)

    @staticmethod
    def _line_confidence(words: Sequence[_Word]) -> float | None:
        weighted = 0.0
        characters = 0
        for word in words:
            if word.confidence is None:
                continue
            count = len(word.text)
            weighted += word.confidence * count
            characters += count
        return weighted / characters if characters else None

    @staticmethod
    def _line_box(words: Sequence[_Word]) -> PixelBox | None:
        boxes = tuple(word.box for word in words if word.box is not None)
        if not boxes:
            return None
        return PixelBox(
            left=min(box.left for box in boxes),
            top=min(box.top for box in boxes),
            right=max(box.right for box in boxes),
            bottom=max(box.bottom for box in boxes),
        )

    @classmethod
    def _full_asset_box(
        cls,
        local_box: PixelBox | None,
        requested_region: Box | None,
        region_pixels: PixelBox | None,
        working_width: int,
        working_height: int,
        asset_width: int,
        asset_height: int,
    ) -> Box | None:
        if local_box is None:
            return None
        if region_pixels is None:
            return pixels_to_box(local_box, working_width, working_height)
        mapped = remap_pixel_box_from_region(
            local_box,
            region_pixels,
            asset_width,
            asset_height,
        )
        if requested_region is None:
            return mapped
        return cls._intersect_boxes(mapped, requested_region)

    @staticmethod
    def _intersect_boxes(first: Box, second: Box) -> Box | None:
        x_min = max(first.x_min, second.x_min)
        y_min = max(first.y_min, second.y_min)
        x_max = min(first.x_max, second.x_max)
        y_max = min(first.y_max, second.y_max)
        if x_min >= x_max or y_min >= y_max:
            return None
        return Box(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)

    def _effective_config(self, request: OCRRequest) -> str:
        if self._config:
            return self._config
        return f"--psm {_MODE_PSM[request.mode]}"

    @contextmanager
    def _configured_executable(self, module: _PytesseractModule) -> Iterator[None]:
        if self._executable is None:
            yield
            return
        with _PYTESSERACT_LOCK:
            previous = module.pytesseract.tesseract_cmd
            module.pytesseract.tesseract_cmd = self._executable
            try:
                yield
            finally:
                module.pytesseract.tesseract_cmd = previous

    @staticmethod
    def _payload_language(languages: tuple[str, ...]) -> str | None:
        if len(languages) != 1 or _PAYLOAD_LANGUAGE.fullmatch(languages[0]) is None:
            return None
        return languages[0]

    @staticmethod
    def _safe_runtime_version(value: object) -> str:
        version = str(value).strip()
        if not version or len(version) > 100 or any(ord(character) < 32 for character in version):
            return "unknown"
        return version

    @staticmethod
    def _consume_diagnostic_failure(task: asyncio.Task[_Diagnostics]) -> None:
        if task.cancelled():
            return
        task.exception()

    @staticmethod
    def _unavailable(failure: _DiagnosticFailure) -> BackendUnavailableError:
        return BackendUnavailableError(
            code=failure.code,
            backend_name=_BACKEND_NAME,
            cause=failure.cause,
        )

    @staticmethod
    def _validate_executable(value: str | os.PathLike[str] | None) -> str | None:
        if value is None:
            return None
        try:
            executable = os.fspath(value)
        except (TypeError, ValueError, OSError) as error:
            raise TypeError("executable must be a text path") from error
        if not isinstance(executable, str):
            raise TypeError("executable must be a text path")
        if not executable or "\x00" in executable:
            raise ValueError("executable must be a non-empty NUL-free text path")
        return executable

    @staticmethod
    def _validate_engine_version(value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("engine_version must be text")
        engine_version = value.strip()
        if _ENGINE_VERSION.fullmatch(engine_version) is None:
            raise ValueError("engine_version must be a compact version token")
        return engine_version

    @staticmethod
    def _validate_languages(value: Sequence[str]) -> tuple[str, ...]:
        if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
            raise TypeError("languages must be a sequence of identifiers")
        languages = tuple(value)
        if not languages:
            raise ValueError("languages must not be empty")
        if any(not isinstance(language, str) for language in languages):
            raise TypeError("languages must contain text identifiers")
        if any(_LANGUAGE.fullmatch(language) is None for language in languages):
            raise ValueError("languages contain an invalid identifier")
        if len(languages) != len(set(languages)):
            raise ValueError("languages must be unique")
        return languages

    @classmethod
    def _descriptor_version(
        cls,
        engine_version: str | None,
        languages: tuple[str, ...],
        config: str,
    ) -> str:
        """Return the descriptor identity, leading with a pinned engine version.

        A pinned engine build is reported the way the Pillow backend reports its
        library version: the concrete engine version first, then the adapter
        marker and the preprocessing selections. An unpinned backend keeps the
        preprocessing version alone, because the router snapshots the descriptor
        once at registration and a descriptor that gained the engine version
        after the first analysis would never reach a cache key or provenance.
        """
        preprocessing = cls._preprocessing_version(languages, config)
        if engine_version is None:
            return preprocessing
        return f"{engine_version}+penampakan.{preprocessing}"

    @staticmethod
    def _preprocessing_version(languages: tuple[str, ...], config: str) -> str:
        """Return the adapter version with its construction-time OCR selections.

        Tesseract is an engine rather than a model, so it reports no model
        identity and never claims an unresolved model revision. Its default
        language set and configuration string are not part of a normalized
        ``OCRRequest`` yet still change results, so they belong in the
        preprocessing version that the perception cache key covers. The
        Tesseract runtime version itself is only knowable by executing the
        binary, which the lazy diagnosis defers to the first analysis, so it
        joins this version only when the caller pins it at construction.
        """
        version = f"{_ADAPTER_VERSION}+lang.{'+'.join(languages)}"
        if not config:
            return version
        digest = hashlib.sha256(config.encode("utf-8")).hexdigest()[:16]
        return f"{version}+config.{digest}"

    @staticmethod
    def _validate_config(value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("config must be text")
        if "\x00" in value:
            raise ValueError("config must not contain NUL")
        return value.strip()

    @staticmethod
    def _validate_concurrency(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("max_concurrency must be an integer")
        if value <= 0:
            raise ValueError("max_concurrency must be positive")
        return value


__all__ = ["TesseractBackend"]
