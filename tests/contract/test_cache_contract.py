"""The shared ``Cache`` contract, run over every shipped implementation.

Specification 07 D8 requires one contract suite that covers the null, memory,
and SQLite caches rather than three private suites that could drift apart. Each
implementation is described once, together with the two promises it is allowed
to make differently: whether it retains a value at all, and whether that value
can outlive the process. Every other assertion here holds for all three, so a
new implementation only has to be added to ``_IMPLEMENTATIONS`` to be held to
the same behavior.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from penampakan.errors import PenampakanError
from penampakan.models import CacheStats
from penampakan.perception.cache import MemoryLRUCache, NullCache, is_durable_cache
from penampakan.perception.sqlite_cache import MINIMUM_SQLITE_VERSION, SQLiteCache
from penampakan.protocols import Cache, ManagedCache

_MAX_ENTRIES = 8
_MAX_BYTES = 4096

# ``CacheStats`` limits must be positive, so a cache that never retains anything
# reports the smallest limits it is allowed to declare.
_EMPTY_CACHE_LIMIT = 1

_KEY = "0" * 64
_OTHER_KEY = "1" * 64
_VALUE = b'{"observations":[]}'
_OTHER_VALUE = b'{"observations":[],"warnings":[]}'


@dataclass(frozen=True)
class _Implementation:
    """One shipped cache and the two promises implementations may differ on."""

    name: str
    factory: Callable[[Path], Cache]
    retains: bool
    durable: bool
    max_entries: int
    max_bytes: int


def _sqlite(directory: Path) -> SQLiteCache:
    # The parent directory is deliberately absent so the cache creates and owns
    # it, which is how a caller-configured durable cache is normally opened.
    return SQLiteCache(
        directory / "cache" / "perception.db",
        max_entries=_MAX_ENTRIES,
        max_bytes=_MAX_BYTES,
    )


_IMPLEMENTATIONS = (
    _Implementation(
        name="null",
        factory=lambda _: NullCache(),
        retains=False,
        durable=False,
        max_entries=_EMPTY_CACHE_LIMIT,
        max_bytes=_EMPTY_CACHE_LIMIT,
    ),
    _Implementation(
        name="memory",
        factory=lambda _: MemoryLRUCache(max_entries=_MAX_ENTRIES, max_bytes=_MAX_BYTES),
        retains=True,
        durable=False,
        max_entries=_MAX_ENTRIES,
        max_bytes=_MAX_BYTES,
    ),
    _Implementation(
        name="sqlite",
        factory=_sqlite,
        retains=True,
        durable=True,
        max_entries=_MAX_ENTRIES,
        max_bytes=_MAX_BYTES,
    ),
)


@pytest.fixture(params=_IMPLEMENTATIONS, ids=lambda implementation: implementation.name)
def implementation(request: pytest.FixtureRequest) -> _Implementation:
    """Return one shipped cache description per parametrized run."""

    return cast(_Implementation, request.param)


@pytest.fixture
async def cache(implementation: _Implementation, tmp_path: Path) -> AsyncIterator[Cache]:
    """Yield a live cache and close it however the test left it."""

    instance = implementation.factory(tmp_path)
    try:
        yield instance
    finally:
        # Closing twice is part of the contract, so a test that already closed
        # the cache is not a teardown failure.
        await instance.aclose()


def _managed(cache: Cache) -> ManagedCache:
    assert isinstance(cache, ManagedCache)
    return cache


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["", "before\x00after", "\x00"])
async def test_a_cache_rejects_keys_it_cannot_canonicalize(cache: Cache, key: str) -> None:
    with pytest.raises(ValueError, match="cache key"):
        await cache.get(key)
    with pytest.raises(ValueError, match="cache key"):
        await cache.set(key, _VALUE, size=len(_VALUE))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(b'"\xff"', id="not-utf-8"),
        pytest.param(b"{not json}", id="not-json"),
        pytest.param(b'{"duplicate":1,"duplicate":2}', id="duplicate-object-keys"),
        pytest.param(b"NaN", id="nan"),
        pytest.param(b"Infinity", id="infinity"),
        pytest.param(b"-Infinity", id="negative-infinity"),
    ],
)
async def test_a_cache_rejects_a_value_that_is_not_strict_json(cache: Cache, value: bytes) -> None:
    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        await cache.set(_KEY, value, size=len(value))

    assert await cache.get(_KEY) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("offset", [-1, 1])
async def test_a_cache_rejects_caller_accounting_that_disagrees(cache: Cache, offset: int) -> None:
    with pytest.raises(ValueError, match="size must equal"):
        await cache.set(_KEY, _VALUE, size=len(_VALUE) + offset)

    assert await cache.get(_KEY) is None


def test_the_sqlite_runtime_version_decides_durable_cache_availability(
    tmp_path: Path,
) -> None:
    # Spec 07 D4: STRICT tables need SQLite 3.37+, and CI records and tests the
    # runtime it actually ran on. Recording it in the failure message means a
    # job on an older runtime reports the version rather than a bare assertion.
    runtime = sqlite3.sqlite_version_info
    supported = runtime >= MINIMUM_SQLITE_VERSION
    cache = SQLiteCache(tmp_path / "runtime.sqlite3")
    try:
        assert cache.available is supported, (
            f"SQLite runtime {sqlite3.sqlite_version} vs minimum "
            f"{'.'.join(str(part) for part in MINIMUM_SQLITE_VERSION)}"
        )
        if supported:
            assert cache.status is None
        else:
            assert cache.status is not None
            assert cache.status.code == "sqlite_version_unsupported"
    finally:
        asyncio.run(cache.aclose())


@pytest.mark.asyncio
async def test_an_absent_key_is_a_miss(cache: Cache) -> None:
    assert await cache.get(_KEY) is None


@pytest.mark.asyncio
async def test_retention_and_durability_match_the_declared_expectation(
    cache: Cache,
    implementation: _Implementation,
) -> None:
    await cache.set(_KEY, _VALUE, size=len(_VALUE))

    assert await cache.get(_KEY) == (_VALUE if implementation.retains else None)
    assert is_durable_cache(cache) is implementation.durable


@pytest.mark.asyncio
async def test_a_value_larger_than_the_budget_leaves_the_stored_value_alone(
    cache: Cache,
    implementation: _Implementation,
) -> None:
    # Spec 07 D6 rejects an individual value larger than ``max_bytes`` as a
    # no-op. A no-op must not discard the good value already under that key.
    oversized = b'{"observations":[],"note":"' + b"x" * implementation.max_bytes + b'"}'
    await cache.set(_KEY, _VALUE, size=len(_VALUE))

    await cache.set(_KEY, oversized, size=len(oversized))

    assert await cache.get(_KEY) == (_VALUE if implementation.retains else None)


@pytest.mark.asyncio
async def test_close_is_idempotent_and_the_post_close_contract_holds(cache: Cache) -> None:
    await cache.set(_KEY, _VALUE, size=len(_VALUE))

    await cache.aclose()
    await cache.aclose()

    # A session keeps using its cache through a shutdown race, so the session
    # surface must degrade to a miss and a no-op rather than raise.
    assert await cache.get(_KEY) is None
    await cache.set(_OTHER_KEY, _OTHER_VALUE, size=len(_OTHER_VALUE))
    assert await cache.get(_OTHER_KEY) is None

    managed = _managed(cache)
    for administration in (managed.stats(), managed.clear(), managed.prune()):
        with pytest.raises(PenampakanError) as raised:
            await administration
        assert raised.value.code == "cache_closed"


@pytest.mark.asyncio
async def test_a_managed_cache_reports_a_coherent_snapshot(
    cache: Cache,
    implementation: _Implementation,
) -> None:
    managed = _managed(cache)
    limits = {
        "max_entries": implementation.max_entries,
        "max_bytes": implementation.max_bytes,
    }

    assert await managed.stats() == CacheStats(entry_count=0, total_bytes=0, **limits)

    await cache.set(_KEY, _VALUE, size=len(_VALUE))
    filled = await managed.stats()

    assert filled == CacheStats(
        entry_count=1 if implementation.retains else 0,
        total_bytes=len(_VALUE) if implementation.retains else 0,
        **limits,
    )
    # ``stats`` reports the snapshot, never a removal: only ``prune`` does that.
    assert (filled.removed_entries, filled.removed_bytes) == (0, 0)
    assert filled.entry_count <= filled.max_entries
    assert filled.total_bytes <= filled.max_bytes


@pytest.mark.asyncio
async def test_clear_empties_the_cache_and_prune_reports_honestly(
    cache: Cache,
    implementation: _Implementation,
) -> None:
    managed = _managed(cache)
    await cache.set(_KEY, _VALUE, size=len(_VALUE))
    await cache.set(_OTHER_KEY, _OTHER_VALUE, size=len(_OTHER_VALUE))
    retained = 2 if implementation.retains else 0

    pruned = await managed.prune()

    # Every accepted write already evicted down to the configured limits, so an
    # honest prune of a healthy cache reports nothing discarded.
    assert (pruned.removed_entries, pruned.removed_bytes) == (0, 0)
    assert pruned.entry_count == retained

    await managed.clear()
    cleared = await managed.stats()

    assert (cleared.entry_count, cleared.total_bytes) == (0, 0)
    assert await cache.get(_KEY) is None
    assert await cache.get(_OTHER_KEY) is None
