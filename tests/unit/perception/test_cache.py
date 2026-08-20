from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import cast

import pytest

from penampakan.errors import PenampakanError
from penampakan.models import (
    BackendDescriptor,
    CacheStats,
    Capability,
    CapabilityDescriptor,
    CaptionRequest,
    OCRRequest,
    VisionRequest,
)
from penampakan.perception.cache import (
    MemoryLRUCache,
    NullCache,
    SingleFlightCoordinator,
    build_perception_cache_key,
    canonical_request_json,
    is_durable_cache,
)
from penampakan.protocols import ManagedCache


def _backend(
    *,
    name: str = "custom.caption",
    version: str = "1.0",
    model_id: str | None = "caption-model",
    model_revision: str | None = "revision-a",
) -> BackendDescriptor:
    return BackendDescriptor(
        name=name,
        version=version,
        model_id=model_id,
        model_revision=model_revision,
        capabilities=(CapabilityDescriptor(capability=Capability.CAPTION),),
    )


class _ProtocolOnlyCache:
    """A caller-supplied cache implementing exactly the documented ``Cache`` members."""

    def __init__(self) -> None:
        self._values: dict[str, bytes] = {}

    async def get(self, key: str) -> bytes | None:
        return self._values.get(key)

    async def set(self, key: str, value: bytes, *, size: int) -> None:
        self._values[key] = value

    async def aclose(self) -> None:
        self._values.clear()


def _cache_key(
    *,
    asset_digest_sha256: str = "a" * 64,
    request: VisionRequest | None = None,
    backend: BackendDescriptor | None = None,
    preprocessing_version: str = "preprocessing-a",
    schema_version: str = "schema-a",
) -> str:
    return build_perception_cache_key(
        asset_digest_sha256=asset_digest_sha256,
        request=request if request is not None else CaptionRequest(focus="receipt total"),
        backend=backend if backend is not None else _backend(),
        preprocessing_version=preprocessing_version,
        schema_version=schema_version,
    )


@pytest.mark.asyncio
async def test_null_cache_never_retains_values_and_closes_idempotently() -> None:
    cache = NullCache()

    await cache.set("answer", b'{"schema_version":1}', size=20)

    assert await cache.get("answer") is None
    assert not cache.closed

    await cache.aclose()
    await cache.aclose()
    await cache.set("answer", b'{"schema_version":2}', size=20)

    assert cache.closed
    assert await cache.get("answer") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value",
    [
        b"",
        b"not-json",
        b"NaN",
        b"Infinity",
        b'{"duplicate":1,"duplicate":2}',
        b'"\\u0000"',
        b"\xff",
    ],
)
async def test_caches_reject_invalid_json_bytes(value: bytes) -> None:
    memory = MemoryLRUCache(max_entries=4, max_bytes=100)
    null = NullCache()

    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        await memory.set("invalid", value, size=1)
    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        await null.set("invalid", value, size=1)


@pytest.mark.asyncio
async def test_caches_require_bytes_and_strict_positive_sizes() -> None:
    cache = MemoryLRUCache(max_entries=4, max_bytes=100)

    with pytest.raises(TypeError, match="must be bytes"):
        await cache.set("invalid", cast(bytes, bytearray(b"{}")), size=2)
    for size in (0, -1, cast(int, True), cast(int, 1.5)):
        with pytest.raises(ValueError, match="positive integer"):
            await cache.set("invalid", b"{}", size=size)

    for name, value in (("max_entries", 0), ("max_bytes", -1)):
        arguments = {name: value}
        with pytest.raises(ValueError, match="positive integer"):
            MemoryLRUCache(**arguments)


@pytest.mark.asyncio
async def test_entry_eviction_uses_get_promotion_order() -> None:
    cache = MemoryLRUCache(max_entries=2, max_bytes=100)

    await cache.set("first", b'{"value":1}', size=11)
    await cache.set("second", b'{"value":2}', size=11)
    assert await cache.get("first") == b'{"value":1}'
    await cache.set("third", b'{"value":3}', size=11)

    assert await cache.get("second") is None
    assert await cache.get("first") == b'{"value":1}'
    assert await cache.get("third") == b'{"value":3}'
    assert cache.entry_count == 2
    assert cache.current_bytes == 22


@pytest.mark.asyncio
async def test_replacement_updates_size_and_promotes_entry() -> None:
    # Every value below is sized by its own byte length, so the byte totals are
    # driven by the payloads rather than by accounting the cache would reject.
    cache = MemoryLRUCache(max_entries=2, max_bytes=10)

    await cache.set("first", b'"ab"', size=4)
    await cache.set("second", b'"cd"', size=4)
    await cache.set("first", b'"efgh"', size=6)

    assert cache.entry_count == 2
    assert cache.current_bytes == 10
    assert await cache.get("first") == b'"efgh"'

    await cache.set("third", b"1", size=1)

    assert await cache.get("second") is None
    assert await cache.get("first") == b'"efgh"'
    assert await cache.get("third") == b"1"
    assert cache.current_bytes == 7


@pytest.mark.asyncio
async def test_byte_eviction_and_oversized_replacement_are_bounded() -> None:
    # The byte limit is crossed by the third honestly sized value, not by a
    # caller claiming a size its payload does not have.
    cache = MemoryLRUCache(max_entries=10, max_bytes=7)

    await cache.set("first", b'"a"', size=3)
    await cache.set("second", b'"b"', size=3)
    await cache.set("third", b'"c"', size=3)

    assert await cache.get("first") is None
    assert await cache.get("second") == b'"b"'
    assert await cache.get("third") == b'"c"'
    assert cache.current_bytes == 6

    # A value larger than the whole budget is a no-op: it neither displaces the
    # good value already stored under its key nor evicts anything else.
    await cache.set("second", b'"abcdef"', size=8)

    assert await cache.get("second") == b'"b"'
    assert await cache.get("third") == b'"c"'
    assert cache.entry_count == 2
    assert cache.current_bytes == 6


@pytest.mark.asyncio
async def test_memory_cache_close_clears_and_disables_storage() -> None:
    cache = MemoryLRUCache(max_entries=2, max_bytes=20)
    await cache.set("first", b'{"a":1234}', size=10)

    await cache.aclose()
    await cache.aclose()
    await cache.set("second", b'{"b":5678}', size=10)

    assert cache.closed
    assert cache.entry_count == 0
    assert cache.current_bytes == 0
    assert await cache.get("first") is None
    assert await cache.get("second") is None


@pytest.mark.asyncio
async def test_caches_reject_caller_size_that_disagrees_with_the_value() -> None:
    memory = MemoryLRUCache(max_entries=4, max_bytes=100)
    null = NullCache()
    value = b'{"value":1}'

    for size in (len(value) - 1, len(value) + 1):
        with pytest.raises(ValueError, match="byte length"):
            await memory.set("mismatch", value, size=size)
        with pytest.raises(ValueError, match="byte length"):
            await null.set("mismatch", value, size=size)

    assert await memory.get("mismatch") is None
    assert memory.entry_count == 0
    assert memory.current_bytes == 0
    assert (await memory.stats()).total_bytes == 0


def test_shipped_caches_are_discoverable_as_managed_caches() -> None:
    assert isinstance(NullCache(), ManagedCache)
    assert isinstance(MemoryLRUCache(), ManagedCache)
    assert not isinstance(_ProtocolOnlyCache(), ManagedCache)


@pytest.mark.asyncio
async def test_null_cache_administration_reports_an_always_empty_snapshot() -> None:
    cache = NullCache()
    await cache.set("answer", b'{"value":1}', size=11)

    snapshot = await cache.stats()

    # A cache that never retains anything still has to declare positive limits,
    # so it declares the smallest ones it is allowed to.
    assert snapshot == CacheStats(entry_count=0, total_bytes=0, max_entries=1, max_bytes=1)

    await cache.clear()
    pruned = await cache.prune()

    assert pruned == snapshot
    assert (pruned.removed_entries, pruned.removed_bytes) == (0, 0)


@pytest.mark.asyncio
async def test_memory_cache_administration_reports_verified_accounting() -> None:
    cache = MemoryLRUCache(max_entries=4, max_bytes=64)
    await cache.set("first", b'"ab"', size=4)
    await cache.set("second", b'"cdef"', size=6)
    populated = CacheStats(entry_count=2, total_bytes=10, max_entries=4, max_bytes=64)

    assert await cache.stats() == populated

    # Accepted writes already evict to the limits, so pruning a healthy cache
    # discards nothing and keeps every entry readable.
    assert await cache.prune() == populated
    assert await cache.get("first") == b'"ab"'

    await cache.clear()

    assert await cache.stats() == CacheStats(
        entry_count=0,
        total_bytes=0,
        max_entries=4,
        max_bytes=64,
    )
    assert await cache.get("second") is None
    assert not cache.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("build_cache", [NullCache, MemoryLRUCache])
async def test_closed_caches_refuse_administration(
    build_cache: Callable[[], ManagedCache],
) -> None:
    cache = build_cache()
    await cache.aclose()

    for administer in (cache.stats, cache.clear, cache.prune):
        with pytest.raises(PenampakanError) as failure:
            await administer()
        assert failure.value.code == "cache_closed"

    assert await cache.get("first") is None


def test_canonical_request_json_is_compact_sorted_and_unicode_preserving() -> None:
    request = CaptionRequest(region=None, focus="jumlah RՄ", max_sentences=5)

    assert canonical_request_json(request) == (
        '{"capability":"caption","focus":"jumlah RՄ","mark_indices":[],"max_sentences":5}'.encode()
    )


def test_cache_key_changes_for_every_perception_dimension() -> None:
    baseline = _cache_key()
    variants = (
        _cache_key(schema_version="schema-b"),
        _cache_key(asset_digest_sha256="b" * 64),
        _cache_key(request=OCRRequest()),
        _cache_key(request=CaptionRequest(focus="receipt currency")),
        _cache_key(request=CaptionRequest(focus="receipt total", max_sentences=4)),
        _cache_key(backend=_backend(name="custom.other")),
        _cache_key(backend=_backend(version="2.0")),
        _cache_key(backend=_backend(model_id="other-model")),
        _cache_key(backend=_backend(model_revision="revision-b")),
        _cache_key(preprocessing_version="preprocessing-b"),
    )

    assert len(baseline) == 64
    assert all(character in "0123456789abcdef" for character in baseline)
    assert len(set((baseline, *variants))) == len(variants) + 1


def test_cache_key_misses_across_model_revisions_of_one_model() -> None:
    resolved = _backend(model_revision="a" * 40)
    other_snapshot = _backend(model_revision="b" * 40)
    unresolved = _backend(model_revision=None)

    keys = (
        _cache_key(backend=resolved),
        _cache_key(backend=other_snapshot),
        _cache_key(backend=unresolved),
    )

    assert len(set(keys)) == 3
    assert _cache_key(backend=resolved) == _cache_key(backend=_backend(model_revision="a" * 40))


@pytest.mark.parametrize(
    ("model_id", "model_revision", "eligible"),
    [
        (None, None, True),
        (None, "a" * 40, True),
        ("caption-model", "a" * 40, True),
        ("caption-model", None, False),
    ],
)
def test_durable_cache_eligibility_truth_table(
    model_id: str | None,
    model_revision: str | None,
    eligible: bool,
) -> None:
    descriptor = _backend(model_id=model_id, model_revision=model_revision)

    assert descriptor.durable_cache_eligible is eligible


def test_shipped_process_local_caches_declare_themselves_ephemeral() -> None:
    assert NullCache.durable is False
    assert MemoryLRUCache.durable is False
    assert is_durable_cache(NullCache()) is False
    assert is_durable_cache(MemoryLRUCache()) is False


def test_cache_without_durable_declaration_is_treated_as_durable() -> None:
    class UndeclaredCache(_ProtocolOnlyCache):
        """A caller cache written against the protocol without declaring durability."""

    assert hasattr(UndeclaredCache(), "durable") is False
    assert is_durable_cache(UndeclaredCache()) is True
    assert is_durable_cache(object()) is True


def test_cache_declaring_durable_false_is_ephemeral() -> None:
    class EphemeralCache(_ProtocolOnlyCache):
        durable = False

    assert is_durable_cache(EphemeralCache()) is False


def test_cache_declaring_durable_true_is_durable() -> None:
    class DurableCache(_ProtocolOnlyCache):
        durable = True

    assert is_durable_cache(DurableCache()) is True


@pytest.mark.parametrize(
    "declared",
    [0, 1, None, "", "no", "yes", (), []],
)
def test_only_the_exact_false_sentinel_opts_out_of_durability(declared: object) -> None:
    class ConfusedCache(_ProtocolOnlyCache):
        durable = declared

    assert is_durable_cache(ConfusedCache()) is True


def test_cache_key_is_deterministic_and_validates_components() -> None:
    assert _cache_key() == _cache_key()

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _cache_key(asset_digest_sha256="A" * 64)
    with pytest.raises(ValueError, match="schema_version"):
        _cache_key(schema_version=" ")
    with pytest.raises(ValueError, match="preprocessing_version"):
        _cache_key(preprocessing_version="bad\x00version")


@pytest.mark.asyncio
async def test_single_flight_shares_one_population_for_same_key() -> None:
    coordinator: SingleFlightCoordinator[str] = SingleFlightCoordinator()
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def populate() -> str:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return "shared-result"

    waiters = [asyncio.create_task(coordinator.run("same-key", populate)) for _ in range(3)]
    await started.wait()
    await asyncio.sleep(0)

    assert coordinator.active_keys == ("same-key",)

    release.set()

    assert await asyncio.gather(*waiters) == ["shared-result"] * 3
    assert calls == 1
    assert not coordinator.active_keys


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_cancel_shared_population() -> None:
    coordinator: SingleFlightCoordinator[str] = SingleFlightCoordinator()
    started = asyncio.Event()
    release = asyncio.Event()
    producer_cancelled = False
    calls = 0

    async def populate() -> str:
        nonlocal calls, producer_cancelled
        calls += 1
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            producer_cancelled = True
            raise
        return "survived"

    cancelled_waiter = asyncio.create_task(coordinator.run("shared-key", populate))
    await started.wait()
    remaining_waiter = asyncio.create_task(coordinator.run("shared-key", populate))
    await asyncio.sleep(0)

    cancelled_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_waiter

    assert not producer_cancelled

    release.set()

    assert await remaining_waiter == "survived"
    assert calls == 1
    assert not producer_cancelled
    assert coordinator.active_keys == ()


@pytest.mark.asyncio
async def test_single_flight_is_scoped_per_key_and_closes_idempotently() -> None:
    coordinator: SingleFlightCoordinator[str] = SingleFlightCoordinator()
    calls: list[str] = []

    def population(key: str) -> Callable[[], Awaitable[str]]:
        async def populate() -> str:
            calls.append(key)
            await asyncio.sleep(0)
            return key

        return populate

    results = await asyncio.gather(
        coordinator.run("first", population("first")),
        coordinator.run("second", population("second")),
    )
    assert list(results) == ["first", "second"]
    assert sorted(calls) == ["first", "second"]

    await coordinator.aclose()
    await coordinator.aclose()

    assert coordinator.closed
    with pytest.raises(RuntimeError, match="closed"):
        await coordinator.run("third", population("third"))
