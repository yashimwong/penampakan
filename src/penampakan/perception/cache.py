"""Deterministic process-local perception caching primitives."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Generic, TypeVar

from ..models import JSON_VALUE_ADAPTER, BackendDescriptor, VisionRequest

CACHE_SCHEMA_VERSION = "perception-cache-v1"


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


class NullCache:
    """A disabled cache implementation that never retains values."""

    __slots__ = ("_closed",)

    def __init__(self) -> None:
        self._closed = False

    async def get(self, key: str) -> bytes | None:
        """Always return a cache miss."""

        _validate_key(key)
        return None

    async def set(self, key: str, value: bytes, *, size: int) -> None:
        """Validate and discard a cache value."""

        _validate_key(key)
        _validate_size(size)
        _validate_json_bytes(value)

    async def aclose(self) -> None:
        """Close the disabled cache idempotently."""

        self._closed = True

    @property
    def closed(self) -> bool:
        """Return whether close has been requested."""

        return self._closed


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

    def __init__(
        self,
        *,
        max_entries: int = 256,
        max_bytes: int = 128 * 1024 * 1024,
    ) -> None:
        self._max_entries = _validate_positive_limit("max_entries", max_entries)
        self._max_bytes = _validate_positive_limit("max_bytes", max_bytes)
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._total_bytes = 0
        self._lock = asyncio.Lock()
        self._closed = False

    async def get(self, key: str) -> bytes | None:
        """Return a value and promote it to most recently used."""

        _validate_key(key)
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

        _validate_key(key)
        accounted_size = _validate_size(size)
        copied_value = _validate_json_bytes(value)
        async with self._lock:
            if self._closed:
                return
            replaced = self._entries.pop(key, None)
            if replaced is not None:
                self._total_bytes -= replaced.size
            if accounted_size > self._max_bytes:
                return
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

    def _evict(self) -> None:
        while len(self._entries) > self._max_entries or self._total_bytes > self._max_bytes:
            _, removed = self._entries.popitem(last=False)
            self._total_bytes -= removed.size


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

        _validate_key(key)
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


def _validate_positive_limit(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_size(size: int) -> int:
    return _validate_positive_limit("size", size)


def _validate_key(key: str) -> str:
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


def _validate_json_bytes(value: bytes) -> bytes:
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
]
