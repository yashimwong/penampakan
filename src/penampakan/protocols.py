from typing import Protocol, runtime_checkable

from penampakan.models import (
    BackendDescriptor,
    BackendImage,
    CacheStats,
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
    "ManagedCache",
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
    """Decision policy for selecting the next validated action.

    An implementation MAY expose a ``degradations: tuple[WarningInfo, ...]``
    attribute reporting typed provider degradation, such as JSON-only schema
    enforcement. The set only grows, and each reported code is attached to a run
    exactly once. An implementation MAY also expose ``aclose`` so a caller that
    owns the policy can release it.
    """

    async def next_action(self, input: PolicyInput) -> PolicyAction:
        """Choose the next action from the compiled policy input."""
        ...


class Cache(Protocol):
    """Asynchronous byte cache for versioned validated JSON values.

    An implementation MAY expose an optional ``durable: bool`` attribute.
    Declaring ``durable = False`` asserts that no entry can outlive the current
    process. The attribute is not a required protocol member: when it is absent,
    or set to anything other than the exact value ``False``, the library assumes
    the cache is durable. Durable is therefore the default.

    A durable cache is bypassed for backends whose descriptor is not
    ``durable_cache_eligible``, because their results cannot be attributed to an
    exact weight identity across processes.
    """

    async def get(self, key: str) -> bytes | None:
        """Return the cached bytes for a key or a cache miss."""
        ...

    async def set(self, key: str, value: bytes, *, size: int) -> None:
        """Store a byte value with its accounted size."""
        ...

    async def aclose(self) -> None:
        """Release cache resources idempotently."""
        ...


@runtime_checkable
class ManagedCache(Cache, Protocol):
    """A cache that also exposes administration to an operator.

    A session only ever uses the :class:`Cache` surface, where a failure
    degrades to a miss or a no-op. These administrative operations instead
    raise typed errors, because silently doing nothing would mislead the
    operator who called them.
    """

    async def stats(self) -> CacheStats:
        """Return a transactional snapshot of retained content."""
        ...

    async def clear(self) -> None:
        """Remove every logical entry transactionally.

        This is reclamation, not secure erasure: it promises nothing about
        database pages, write-ahead logs, backups, or filesystem snapshots.
        """
        ...

    async def prune(self) -> CacheStats:
        """Drop expired and over-watermark entries and report the result."""
        ...


class TraceSink(Protocol):
    """Asynchronous destination for redacted immutable trace events."""

    async def emit(self, event: TraceEvent) -> None:
        """Receive one already-redacted trace event."""
        ...

    async def aclose(self) -> None:
        """Release trace sink resources idempotently."""
        ...
