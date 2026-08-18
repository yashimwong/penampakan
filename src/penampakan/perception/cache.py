"""Deterministic process-local perception caching primitives."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import ClassVar, Generic, TypeVar

from ..errors import PenampakanError
from ..models import JSON_VALUE_ADAPTER, BackendDescriptor, CacheStats, VisionRequest

CACHE_SCHEMA_VERSION = "perception-cache-v1"

# ``CacheStats`` limits must be positive, so a cache that can never retain
# anything reports the smallest limit it is allowed to declare rather than a
# capacity it would never honour.
_EMPTY_CACHE_LIMIT = 1


def canonical_request_json(request: VisionRequest) -> bytes:
    """Serialize a validated vision request into deterministic UTF-8 JSON bytes."""

    value = request.model_dump(mode="json", exclude_none=True)
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def build_perception_cache_key(
    *,
    asset_digest_sha256: str,
    request: VisionRequest,
    backend: BackendDescriptor,
    preprocessing_version: str,
    schema_version: str = CACHE_SCHEMA_VERSION,
) -> str:
    """Hash every contract dimension that can change a perception result."""

    _validate_digest(asset_digest_sha256)
    _validate_component("schema_version", schema_version)
    _validate_component("preprocessing_version", preprocessing_version)
    key_material = {
        "asset_digest_sha256": asset_digest_sha256,
        "backend_name": backend.name,
        "backend_version": backend.version,
        "capability": request.capability.value,
        "model_id": backend.model_id,
        "model_revision": backend.model_revision,
        "preprocessing_version": preprocessing_version,
        "request": request.model_dump(mode="json", exclude_none=True),
        "schema_version": schema_version,
    }
    canonical = json.dumps(
        key_material,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def is_durable_cache(cache: object) -> bool:
    """Return whether a cache must be treated as retaining values beyond this process.

    Only an explicit ``durable = False`` opts a cache out. An absent or
    non-``False`` declaration is treated as durable, because a missed cache hit
    is cheaper than a false claim of cross-process reproducibility.
    """

    return getattr(cache, "durable", True) is not False


class NullCache:
    """A disabled cache implementation that never retains values."""

    __slots__ = ("_closed",)

    durable: ClassVar[bool] = False

    def __init__(self) -> None:
        self._closed = False

    async def get(self, key: str) -> bytes | None:
        """Always return a cache miss."""

        validate_key(key)
        return None

    async def set(self, key: str, value: bytes, *, size: int) -> None:
        """Validate and discard a cache value."""

        validate_key(key)
        validate_json_bytes(value)
        validate_accounted_size(size, value)

    async def aclose(self) -> None:
        """Close the disabled cache idempotently."""

        self._closed = True

    async def stats(self) -> CacheStats:
        """Return the always-empty snapshot of a cache that never retains values.

        ``CacheStats`` requires positive limits, so a capacity of zero cannot be
        reported. The smallest positive limits are used instead: claiming a
        larger capacity would advertise headroom this cache does not have, while
        one entry and one byte describe the least a cache may declare and are
        already unreachable here, because every accepted value is discarded.
        """

        self._require_open()
        return CacheStats(
            entry_count=0,
            total_bytes=0,
            max_entries=_EMPTY_CACHE_LIMIT,
            max_bytes=_EMPTY_CACHE_LIMIT,
        )

    async def clear(self) -> None:
        """Remove every retained entry, of which there are never any."""

        self._require_open()

    async def prune(self) -> CacheStats:
        """Report that a cache retaining nothing has nothing to discard."""

        self._require_open()
        return await self.stats()

    @property
    def closed(self) -> bool:
        """Return whether close has been requested."""

        return self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise _closed_cache_error()


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    value: bytes
    size: int


class MemoryLRUCache:
    """A concurrency-safe in-memory LRU bounded by entries and accounted bytes."""

    __slots__ = (
        "_closed",
        "_entries",
        "_lock",
        "_max_bytes",
        "_max_entries",
        "_total_bytes",
    )

    durable: ClassVar[bool] = False

    def __init__(
        self,
        *,
        max_entries: int = 256,
        max_bytes: int = 128 * 1024 * 1024,
    ) -> None:
        self._max_entries = validate_positive_limit("max_entries", max_entries)
        self._max_bytes = validate_positive_limit("max_bytes", max_bytes)
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._total_bytes = 0
        self._lock = asyncio.Lock()
        self._closed = False

    async def get(self, key: str) -> bytes | None:
        """Return a value and promote it to most recently used."""

        validate_key(key)
        async with self._lock:
            if self._closed:
                return None
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.move_to_end(key)
            return entry.value

    async def set(self, key: str, value: bytes, *, size: int) -> None:
        """Store validated JSON bytes and evict least-recently-used entries."""

        validate_key(key)
        copied_value = validate_json_bytes(value)
        accounted_size = validate_accounted_size(size, copied_value)
        async with self._lock:
            if self._closed:
                return
            if accounted_size > self._max_bytes:
                # A value that can never fit is a no-op, so it must not discard
                # the entry already stored under this key.
                return
            replaced = self._entries.pop(key, None)
            if replaced is not None:
                self._total_bytes -= replaced.size
            self._entries[key] = _CacheEntry(copied_value, accounted_size)
            self._total_bytes += accounted_size
            self._evict()

    async def aclose(self) -> None:
        """Clear all retained data and close the cache idempotently."""

        async with self._lock:
            if self._closed:
                return
            self._closed = True
            self._entries.clear()
            self._total_bytes = 0

    async def stats(self) -> CacheStats:
        """Return a snapshot of the accounting this cache verified itself."""

        async with self._lock:
            if self._closed:
                raise _closed_cache_error()
            return self._snapshot()

    async def clear(self) -> None:
        """Remove every entry while holding the lock, so no reader sees a partial clear."""

        async with self._lock:
            if self._closed:
                raise _closed_cache_error()
            self._entries.clear()
            self._total_bytes = 0

    async def prune(self) -> CacheStats:
        """Evict down to the configured limits and report what was discarded.

        Every accepted write already evicts to the same limits, so a healthy
        cache reports no removals here; ``prune`` exists so an operator can
        confirm that rather than assume it.
        """

        async with self._lock:
            if self._closed:
                raise _closed_cache_error()
            removed_entries, removed_bytes = self._evict()
            return self._snapshot(
                removed_entries=removed_entries,
                removed_bytes=removed_bytes,
            )

    @property
    def max_entries(self) -> int:
        """Return the configured maximum entry count."""

        return self._max_entries

    @property
    def max_bytes(self) -> int:
        """Return the configured maximum accounted byte size."""

        return self._max_bytes

    @property
    def entry_count(self) -> int:
        """Return the current number of retained entries."""

        return len(self._entries)

    @property
    def current_bytes(self) -> int:
        """Return the current accounted byte size."""

        return self._total_bytes

    @property
    def closed(self) -> bool:
        """Return whether the cache has been closed."""

        return self._closed

    def _snapshot(self, *, removed_entries: int = 0, removed_bytes: int = 0) -> CacheStats:
        return CacheStats(
            entry_count=len(self._entries),
            total_bytes=self._total_bytes,
            max_entries=self._max_entries,
            max_bytes=self._max_bytes,
            removed_entries=removed_entries,
            removed_bytes=removed_bytes,
        )

    def _evict(self) -> tuple[int, int]:
        removed_entries = 0
        removed_bytes = 0
        while len(self._entries) > self._max_entries or self._total_bytes > self._max_bytes:
            _, removed = self._entries.popitem(last=False)
            self._total_bytes -= removed.size
            removed_entries += 1
            removed_bytes += removed.size
        return removed_entries, removed_bytes


_ValueT = TypeVar("_ValueT")


@dataclass(slots=True)
class _Flight(Generic[_ValueT]):
    future: asyncio.Future[_ValueT]
    waiters: int = 0


class SingleFlightCoordinator(Generic[_ValueT]):
    """Share one asynchronous population per key across concurrent waiters."""

    __slots__ = ("_closed", "_flights", "_lock")

    def __init__(self) -> None:
        self._flights: dict[str, _Flight[_ValueT]] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    async def run(
        self,
        key: str,
        populate: Callable[[], Awaitable[_ValueT]],
    ) -> _ValueT:
        """Return the shared population result without propagating waiter cancellation."""

        validate_key(key)
        async with self._lock:
            if self._closed:
                raise RuntimeError("single-flight coordinator is closed")
            flight = self._flights.get(key)
            if flight is None:
                future = asyncio.ensure_future(populate())
                flight = _Flight(future=future)
                self._flights[key] = flight
            flight.waiters += 1
        try:
            return await asyncio.shield(flight.future)
        finally:
            release_task = asyncio.create_task(self._release(key, flight))
            try:
                await asyncio.shield(release_task)
            except asyncio.CancelledError:
                release_task.add_done_callback(_consume_future)
                raise

    async def aclose(self) -> None:
        """Cancel orphaned populations and close the coordinator idempotently."""

        async with self._lock:
            if self._closed:
                return
            self._closed = True
            futures = tuple(flight.future for flight in self._flights.values())
            self._flights.clear()
        for future in futures:
            future.cancel()
        if futures:
            await asyncio.gather(*futures, return_exceptions=True)

    @property
    def active_keys(self) -> tuple[str, ...]:
        """Return a deterministic snapshot of keys with shared work in flight."""

        return tuple(sorted(self._flights))

    @property
    def closed(self) -> bool:
        """Return whether the coordinator has been closed."""

        return self._closed

    async def _release(self, key: str, flight: _Flight[_ValueT]) -> None:
        async with self._lock:
            current = self._flights.get(key)
            if current is not flight:
                return
            flight.waiters -= 1
            if flight.waiters != 0:
                return
            self._flights.pop(key, None)
            if not flight.future.done():
                flight.future.cancel()
                flight.future.add_done_callback(_consume_future)


def _consume_future(future: asyncio.Future[_ValueT]) -> None:
    with suppress(asyncio.CancelledError):
        future.exception()


def _closed_cache_error() -> PenampakanError:
    # Administration is operator-facing, so a closed cache says so instead of
    # answering with an empty snapshot that a live but empty cache would also
    # produce. No cache-specific error class exists yet; the safe base error
    # carries the distinguishing code.
    return PenampakanError(code="cache_closed")


def validate_positive_limit(name: str, value: int) -> int:
    """Return a strictly positive integer bound, rejecting booleans and zero."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_accounted_size(size: int, value: bytes) -> int:
    """Return the verified byte size, rejecting caller accounting that disagrees.

    An implementation persists the size it computed here, never the number the
    caller passed, so retained byte totals cannot be skewed by a wrong count.
    """

    accounted = validate_positive_limit("size", size)
    if accounted != len(value):
        raise ValueError("size must equal the byte length of value")
    return len(value)


def validate_key(key: str) -> str:
    """Return a canonical cache key, rejecting empty or NUL-bearing text."""

    if not isinstance(key, str) or not key or "\x00" in key:
        raise ValueError("cache key must be a non-empty NUL-free string")
    return key


def _validate_digest(value: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("asset_digest_sha256 must be a lowercase SHA-256 digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError("asset_digest_sha256 must be a lowercase SHA-256 digest")
    return value


def _validate_component(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty NUL-free string")
    return value


def validate_json_bytes(value: bytes) -> bytes:
    """Return a private copy of strict UTF-8 JSON bytes a cache may retain."""

    if not isinstance(value, bytes):
        raise TypeError("cache value must be bytes")
    try:
        decoded = value.decode("utf-8")
        parsed = json.loads(
            decoded,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
        JSON_VALUE_ADAPTER.validate_python(parsed)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise ValueError("cache value must contain strict UTF-8 JSON") from error
    return bytes(value)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON objects must not contain duplicate keys")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"JSON constant {value!r} is not finite")


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "MemoryLRUCache",
    "NullCache",
    "SingleFlightCoordinator",
    "build_perception_cache_key",
    "canonical_request_json",
    "is_durable_cache",
    "validate_accounted_size",
    "validate_json_bytes",
    "validate_key",
    "validate_positive_limit",
]
