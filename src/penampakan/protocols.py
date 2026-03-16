from typing import Protocol, runtime_checkable

from penampakan.models import (
    BackendDescriptor,
    BackendImage,
    LLMRequest,
    LLMResponse,
    PolicyAction,
    PolicyInput,
    TraceEvent,
    VisionRequest,
    VisionResult,
)

__all__ = [
    "ActionPolicy",
    "Cache",
    "TextLLM",
    "TraceSink",
    "VisionBackend",
]


@runtime_checkable
class TextLLM(Protocol):
    """Language model adapter that returns textual structured output."""

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Complete a validated language model request."""
        ...


@runtime_checkable
class VisionBackend(Protocol):
    """Perception backend for validated image analysis requests."""

    @property
    def descriptor(self) -> BackendDescriptor:
        """Return the backend's stable capability descriptor."""
        ...

    def supports(self, request: VisionRequest) -> bool:
        """Return whether the complete request is supported."""
        ...

    async def analyze(self, image: BackendImage, request: VisionRequest) -> VisionResult:
        """Analyze an image without mutating or retaining its content."""
        ...

    async def aclose(self) -> None:
        """Release backend resources idempotently."""
        ...


class ActionPolicy(Protocol):
    """Decision policy for selecting the next validated action."""

    async def next_action(self, input: PolicyInput) -> PolicyAction:
        """Choose the next action from the compiled policy input."""
        ...


class Cache(Protocol):
    """Asynchronous byte cache for versioned validated JSON values."""

    async def get(self, key: str) -> bytes | None:
        """Return the cached bytes for a key or a cache miss."""
        ...

    async def set(self, key: str, value: bytes, *, size: int) -> None:
        """Store a byte value with its accounted size."""
        ...

    async def aclose(self) -> None:
        """Release cache resources idempotently."""
        ...


class TraceSink(Protocol):
    """Asynchronous destination for redacted immutable trace events."""

    async def emit(self, event: TraceEvent) -> None:
        """Receive one already-redacted trace event."""
        ...

    async def aclose(self) -> None:
        """Release trace sink resources idempotently."""
        ...
