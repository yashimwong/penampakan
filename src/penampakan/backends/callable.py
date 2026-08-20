"""Vision backend adapter for application callables."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from penampakan._callables import call_async_or_thread
from penampakan.models import (
    BackendDescriptor,
    BackendImage,
    CaptionRequest,
    DetectionRequest,
    OCRRequest,
    SegmentationRequest,
    VisionRequest,
    VisionResult,
)

AnalyzeCallable = Callable[
    [BackendImage, VisionRequest],
    VisionResult | Awaitable[VisionResult],
]
SupportsCallable = Callable[[VisionRequest], bool]
CloseCallable = Callable[[], Awaitable[None] | None]


class CallableVisionBackend:
    """Adapt a synchronous or asynchronous application function as a backend."""

    def __init__(
        self,
        descriptor: BackendDescriptor,
        analyze: AnalyzeCallable,
        *,
        supports: SupportsCallable | None = None,
        close: CloseCallable | None = None,
    ) -> None:
        if not callable(analyze):
            raise TypeError("analyze must be callable")
        if supports is not None and not callable(supports):
            raise TypeError("supports must be callable")
        if close is not None and not callable(close):
            raise TypeError("close must be callable")
        self._descriptor = descriptor
        self._analyze = analyze
        self._supports = supports
        self._close = close
        self._close_task: asyncio.Task[None] | None = None

    @property
    def descriptor(self) -> BackendDescriptor:
        """Return the stable caller-provided descriptor."""

        return self._descriptor

    def supports(self, request: VisionRequest) -> bool:
        """Evaluate the pure support callback or descriptor capabilities."""

        if self._supports is not None:
            result = self._supports(request)
            if not isinstance(result, bool):
                raise TypeError("supports must return a boolean")
            return result
        capabilities = {item.capability: item.features for item in self._descriptor.capabilities}
        features = capabilities.get(request.capability)
        if features is None:
            return False
        if isinstance(request, CaptionRequest):
            if request.mark_indices and "caption.mark_references" not in features:
                return False
            if request.focus is not None and "caption.focus" not in features:
                return False
        if isinstance(request, OCRRequest) and request.languages:
            return "ocr.languages" in features
        if isinstance(request, DetectionRequest) and request.labels:
            return "detect.open_vocabulary" in features
        if isinstance(request, SegmentationRequest) and request.points:
            return "segment.point_prompt" in features
        return True

    async def analyze(self, image: BackendImage, request: VisionRequest) -> VisionResult:
        """Invoke the application analyzer without blocking the event loop."""

        if self._close_task is not None:
            raise RuntimeError("backend is closed")
        if not self.supports(request):
            raise ValueError("request is unsupported")
        result = await call_async_or_thread(self._analyze, image, request)
        return VisionResult.model_validate(result, strict=True)

    async def aclose(self) -> None:
        """Run the optional close callable exactly once."""

        if self._close_task is None:
            self._close_task = asyncio.create_task(self._run_close())
        await asyncio.shield(self._close_task)

    async def _run_close(self) -> None:
        if self._close is None:
            return
        await call_async_or_thread(self._close)


__all__ = ["AnalyzeCallable", "CallableVisionBackend", "CloseCallable", "SupportsCallable"]
