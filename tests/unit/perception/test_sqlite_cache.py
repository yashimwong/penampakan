from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import stat
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

import pytest

from penampakan.errors import PenampakanError
from penampakan.perception.cache import CACHE_SCHEMA_VERSION, is_durable_cache
from penampakan.perception.sqlite_cache import (
    DATABASE_SCHEMA_VERSION,
    LOW_WATERMARK_RATIO,
    VALUE_ENCODING_VERSION,
    SQLiteCache,
    _CacheUnavailableError,
    _is_contention,
    _quarantine_rename,
    _run_within_deadline,
)
from penampakan.protocols import ManagedCache

_GATE_TIMEOUT_S = 10.0
_WAIT_TIMEOUT_S = 10.0
_POSIX = os.name == "posix"

posix_only = pytest.mark.skipif(not _POSIX, reason="requires POSIX file permissions")


def _cache_path(tmp_path: Path) -> Path:
    # The parent directory is deliberately absent so the cache creates it and
    # owns its permissions.
    return tmp_path / "cache" / "perception.db"


def _payload(size: int = 8) -> bytes:
    """Return strict UTF-8 JSON bytes of exactly ``size`` bytes."""

    return b'"' + b"x" * (size - 2) + b'"'


async def _store(cache: SQLiteCache, key: str, value: bytes) -> None:
    await cache.set(key, value, size=len(value))


async def _wait_until(predicate: Callable[[], bool], message: str) -> None:
    deadline = time.monotonic() + _WAIT_TIMEOUT_S
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(message)
        await asyncio.sleep(0.001)


class _Clock:
    """A deterministic injected wall clock in UTC epoch seconds."""

    def __init__(self, value: float = 1_700_000_000.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class _GatedClock(_Clock):
    """A clock whose first worker-side read blocks until the test opens the gate."""

    def __init__(self, value: float = 1_700_000_000.0) -> None:
        super().__init__(value)
        self.gate = threading.Event()
        self.entered = threading.Event()
        self.reads = 0

    def __call__(self) -> float:
        self.reads += 1
        if self.reads == 1:
            self.entered.set()
            if not self.gate.wait(_GATE_TIMEOUT_S):  # pragma: no cover - hung worker
                raise AssertionError("the gated worker was never released")
        return self.value


def _rows(path: Path) -> list[tuple[str, float, float, float | None]]:
    connection = sqlite3.connect(str(path))
    try:
        return [
            (str(row[0]), float(row[1]), float(row[2]), row[3])
            for row in connection.execute(
                "SELECT key, created_at, accessed_at, expires_at FROM entries ORDER BY key"
            ).fetchall()
        ]
    finally:
        connection.close()


def _write_database(path: Path, *, schema_version: str, key: str = "legacy") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path))
    try:
        connection.execute(
            "CREATE TABLE entries ("
            " key TEXT NOT NULL PRIMARY KEY, value BLOB NOT NULL, size INTEGER NOT NULL,"
            " created_at REAL NOT NULL, accessed_at REAL NOT NULL, expires_at REAL) STRICT"
        )
        connection.execute(
            "CREATE TABLE meta (name TEXT NOT NULL PRIMARY KEY, value TEXT NOT NULL) STRICT"
        )
        connection.executemany(
            "INSERT INTO meta (name, value) VALUES (?, ?)",
            (
                ("schema_version", schema_version),
                ("cache_schema_version", CACHE_SCHEMA_VERSION),
                ("value_encoding_version", str(VALUE_ENCODING_VERSION)),
            ),
        )
        connection.execute(
            "INSERT INTO entries VALUES (?, ?, ?, ?, ?, NULL)",
            (key, _payload(), len(_payload()), 1.0, 1.0),
        )
        connection.commit()
    finally:
        connection.close()


def _quarantined(path: Path, kind: str) -> list[Path]:
    return sorted(
        candidate
        for candidate in path.parent.iterdir()
        if candidate.name.startswith(f"{path.name}.{kind}-")
        and not candidate.name.endswith(("-wal", "-shm"))
    )


# --------------------------------------------------------------- D3 execution


@pytest.mark.asyncio
async def test_one_dedicated_thread_owns_every_connection_operation(tmp_path: Path) -> None:
    cache = SQLiteCache(_cache_path(tmp_path), now=_Clock())
    try:
        assert cache.available
        await _store(cache, "a", _payload())
        assert await cache.get("a") == _payload()
        await cache.stats()
        await cache.prune()
        await cache.clear()
    finally:
        await cache.aclose()
    idents = cache._connection_thread_idents
    assert len(idents) == 1
    assert threading.get_ident() not in idents
    assert idents == {cache._thread.ident}
    assert not cache._thread.is_alive()


@pytest.mark.asyncio
async def test_the_event_loop_keeps_running_while_the_worker_is_busy(tmp_path: Path) -> None:
    clock = _GatedClock()
    cache = SQLiteCache(_cache_path(tmp_path), now=clock)
    ticks = 0

    async def spin() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0)

    spinner = asyncio.create_task(spin())
    write = asyncio.create_task(_store(cache, "a", _payload()))
    try:
        await _wait_until(clock.entered.is_set, "the worker never started the write")
        await _wait_until(lambda: ticks > 100, "the event loop was blocked by the worker")
        assert not write.done()
    finally:
        clock.gate.set()
        spinner.cancel()
        with suppress(asyncio.CancelledError):
            await spinner
        await write
        await cache.aclose()
    assert await _reopened_value(cache.path, "a") == _payload()


async def _reopened_value(path: Path, key: str) -> bytes | None:
    reopened = SQLiteCache(path, now=_Clock())
    try:
        return await reopened.get(key)
    finally:
        await reopened.aclose()


@pytest.mark.asyncio
async def test_values_survive_close_and_reopen(tmp_path: Path) -> None:
    path = _cache_path(tmp_path)
    cache = SQLiteCache(path, now=_Clock())
    await _store(cache, "a", _payload(32))
    await cache.aclose()

    reopened = SQLiteCache(path, now=_Clock())
    try:
        assert reopened.available
        assert await reopened.get("a") == _payload(32)
        assert (await reopened.stats()).entry_count == 1
    finally:
        await reopened.aclose()


@pytest.mark.asyncio
async def test_close_drains_work_that_was_already_queued(tmp_path: Path) -> None:
    clock = _GatedClock()
    path = _cache_path(tmp_path)
    cache = SQLiteCache(path, now=clock)
    writes = [asyncio.create_task(_store(cache, f"k{index}", _payload())) for index in range(3)]
    await _wait_until(clock.entered.is_set, "the first write never reached the worker")
    await _wait_until(lambda: cache._queue.qsize() == 2, "the later writes were never queued")

    closing = asyncio.create_task(cache.aclose())
    await _wait_until(lambda: cache._queue.qsize() == 3, "close was not queued behind the writes")
    clock.gate.set()
    await asyncio.gather(*writes, closing)

    assert cache.closed
    assert {row[0] for row in _rows(path)} == {"k0", "k1", "k2"}


@pytest.mark.asyncio
async def test_a_cancelled_operation_is_skipped_and_the_worker_survives(tmp_path: Path) -> None:
    clock = _GatedClock()
    path = _cache_path(tmp_path)
    cache = SQLiteCache(path, now=clock)
    try:
        running = asyncio.create_task(_store(cache, "running", _payload()))
        await _wait_until(clock.entered.is_set, "the first write never reached the worker")
        abandoned = asyncio.create_task(_store(cache, "abandoned", _payload()))
        await _wait_until(lambda: cache._queue.qsize() == 1, "the second write was never queued")

        abandoned.cancel()
        with pytest.raises(asyncio.CancelledError):
            await abandoned
        clock.gate.set()
        await running

        assert await cache.get("running") == _payload()
        assert await cache.get("abandoned") is None
    finally:
        await cache.aclose()


@pytest.mark.asyncio
async def test_repeated_close_and_the_post_close_contract(tmp_path: Path) -> None:
    cache = SQLiteCache(_cache_path(tmp_path), now=_Clock())
    await _store(cache, "a", _payload())
    await cache.aclose()
    await cache.aclose()
    await cache.aclose()

    assert cache.closed
    assert await cache.get("a") is None
    await _store(cache, "b", _payload())
    for administration in (cache.stats(), cache.clear(), cache.prune()):
        with pytest.raises(PenampakanError) as raised:
            await administration
        assert raised.value.code == "cache_closed"


@pytest.mark.asyncio
async def test_contention_beyond_the_deadline_degrades_the_session_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = SQLiteCache(_cache_path(tmp_path), busy_timeout_s=0.01, now=_Clock())
    try:
        await _store(cache, "a", _payload())

        def _locked() -> float:
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(cache, "_clock", _locked)
        assert await cache.get("a") is None
        await _store(cache, "b", _payload())
        assert {warning.code for warning in cache.warnings} == {"cache_unavailable"}

        with pytest.raises(PenampakanError) as raised:
            await cache.stats()
        assert raised.value.code == "cache_unavailable"
        assert raised.value.retryable
    finally:
        await cache.aclose()


def test_the_retry_helper_is_bounded_by_the_operation_deadline() -> None:
    elapsed = [0.0]
    slept: list[float] = []

    def _sleep(seconds: float) -> None:
        slept.append(seconds)
        elapsed[0] += seconds

    def _locked() -> None:
        raise sqlite3.OperationalError("database is locked")

    with pytest.raises(_CacheUnavailableError):
        _run_within_deadline(
            _locked,
            timeout_s=0.5,
            sleep=_sleep,
            monotonic=lambda: elapsed[0],
            jitter=lambda: 0.5,
        )
    assert slept
    assert sum(slept) <= 0.5


def test_the_retry_helper_reraises_a_failure_that_is_not_contention() -> None:
    def _broken() -> None:
        raise sqlite3.OperationalError("no such table: entries")

    with pytest.raises(sqlite3.OperationalError):
        _run_within_deadline(_broken, timeout_s=0.5, sleep=lambda _: None)


@pytest.mark.asyncio
async def test_the_cache_declares_itself_durable_and_administrable(tmp_path: Path) -> None:
    cache = SQLiteCache(_cache_path(tmp_path), now=_Clock())
    try:
        assert SQLiteCache.durable is True
        assert is_durable_cache(cache)
        assert isinstance(cache, ManagedCache)
    finally:
        await cache.aclose()


# ------------------------------------------------------- D4 schema behaviour


@pytest.mark.asyncio
async def test_an_unsupported_sqlite_runtime_disables_the_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 36, 0))
    path = _cache_path(tmp_path)
    cache = SQLiteCache(path, now=_Clock())

    assert not cache.available
    assert cache.status is not None
    assert cache.status.code == "sqlite_version_unsupported"
    # Nothing was created, because a runtime without STRICT tables cannot
    # enforce the schema this cache promises.
    assert not path.exists()
    assert await cache.get("a") is None
    await _store(cache, "a", _payload())
    with pytest.raises(PenampakanError) as raised:
        await cache.stats()
    assert raised.value.code == "sqlite_version_unsupported"
    await cache.aclose()
    await cache.aclose()


@pytest.mark.asyncio
async def test_a_newer_schema_version_is_never_dropped(tmp_path: Path) -> None:
    path = _cache_path(tmp_path)
    _write_database(path, schema_version=str(DATABASE_SCHEMA_VERSION + 1), key="future")
    cache = SQLiteCache(path, now=_Clock())
    try:
        assert not cache.available
        assert cache.status is not None
        assert cache.status.code == "cache_schema_too_new"
        assert await cache.get("future") is None
    finally:
        await cache.aclose()

    assert [row[0] for row in _rows(path)] == ["future"]
    assert _quarantined(path, "superseded") == []
    assert _quarantined(path, "corrupt") == []


@pytest.mark.asyncio
async def test_an_older_schema_version_is_quarantined_and_replaced(tmp_path: Path) -> None:
    path = _cache_path(tmp_path)
    _write_database(path, schema_version="0", key="legacy")
    cache = SQLiteCache(path, now=_Clock())
    try:
        assert cache.available
        assert cache.status is None
        assert await cache.get("legacy") is None
        await _store(cache, "fresh", _payload())
        assert await cache.get("fresh") == _payload()
    finally:
        await cache.aclose()

    preserved = _quarantined(path, "superseded")
    assert len(preserved) == 1
    assert [row[0] for row in _rows(preserved[0])] == ["legacy"]
    assert [row[0] for row in _rows(path)] == ["fresh"]


@pytest.mark.asyncio
async def test_a_foreign_database_is_quarantined_and_replaced(tmp_path: Path) -> None:
    path = _cache_path(tmp_path)
    path.parent.mkdir(parents=True)
    connection = sqlite3.connect(str(path))
    connection.execute("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)")
    connection.commit()
    connection.close()

    cache = SQLiteCache(path, now=_Clock())
    try:
        assert cache.available
        await _store(cache, "fresh", _payload())
    finally:
        await cache.aclose()
    assert len(_quarantined(path, "superseded")) == 1


@pytest.mark.asyncio
async def test_a_corrupt_file_is_quarantined_and_replaced(tmp_path: Path) -> None:
    path = _cache_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"this is definitely not a SQLite database" * 32)
    original = path.read_bytes()

    cache = SQLiteCache(path, now=_Clock())
    try:
        assert cache.available
        assert cache.status is None
        await _store(cache, "fresh", _payload())
        assert await cache.get("fresh") == _payload()
    finally:
        await cache.aclose()

    preserved = _quarantined(path, "corrupt")
    assert len(preserved) == 1
    assert preserved[0].read_bytes() == original


@pytest.mark.asyncio
async def test_a_failed_quarantine_disables_without_overwriting_the_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _cache_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"this is definitely not a SQLite database" * 32)
    original = path.read_bytes()

    def _refuse(source: Path, target: Path) -> None:
        raise OSError("quarantine is not permitted")

    monkeypatch.setattr(
        "penampakan.perception.sqlite_cache._quarantine_rename",
        _refuse,
    )
    cache = SQLiteCache(path, now=_Clock())
    try:
        assert not cache.available
        assert cache.status is not None
        assert cache.status.code == "cache_quarantine_failed"
        assert await cache.get("fresh") is None
    finally:
        await cache.aclose()

    assert path.read_bytes() == original
    assert _quarantined(path, "corrupt") == []
    assert callable(_quarantine_rename)


# ---------------------------------------------------- D5 filesystem security


@posix_only
@pytest.mark.asyncio
async def test_created_directories_and_artifacts_are_private(tmp_path: Path) -> None:
    path = _cache_path(tmp_path)
    cache = SQLiteCache(path, now=_Clock())
    try:
        await _store(cache, "a", _payload())
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        for suffix in ("-wal", "-shm"):
            sidecar = path.with_name(path.name + suffix)
            if sidecar.exists():
                assert stat.S_IMODE(sidecar.stat().st_mode) == 0o600
        assert cache.warnings == ()
    finally:
        await cache.aclose()


@posix_only
@pytest.mark.asyncio
async def test_broader_existing_permissions_are_reported(tmp_path: Path) -> None:
    directory = tmp_path / "shared"
    directory.mkdir(mode=0o755)
    os.chmod(directory, 0o755)
    path = directory / "perception.db"
    path.touch(mode=0o644)
    os.chmod(path, 0o644)

    cache = SQLiteCache(path, now=_Clock())
    try:
        assert cache.available
        codes = {warning.code for warning in cache.warnings}
        assert "cache_directory_permissions" in codes
        assert "cache_file_permissions" in codes
    finally:
        await cache.aclose()


@pytest.mark.asyncio
@pytest.mark.skipif(not hasattr(os, "symlink"), reason="requires symbolic links")
async def test_a_symlinked_path_is_refused_unless_explicitly_allowed(tmp_path: Path) -> None:
    target = tmp_path / "real" / "perception.db"
    target.parent.mkdir(parents=True)
    link = tmp_path / "real" / "link.db"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):  # pragma: no cover - unsupported platform
        pytest.skip("symbolic links are unavailable")

    refused = SQLiteCache(link, now=_Clock())
    try:
        assert not refused.available
        assert refused.status is not None
        assert refused.status.code == "cache_symlink_rejected"
        assert await refused.get("a") is None
    finally:
        await refused.aclose()

    allowed = SQLiteCache(link, allow_symlink=True, now=_Clock())
    try:
        assert allowed.available
        await _store(allowed, "a", _payload())
        assert await allowed.get("a") == _payload()
    finally:
        await allowed.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "candidate",
    [
        "relative/perception.db",
        "~/perception.db",
        "/tmp/$CACHE_HOME/perception.db",
        "/tmp/penampakan-*/perception.db",
    ],
)
async def test_an_underived_or_unresolved_path_is_refused(candidate: str) -> None:
    cache = SQLiteCache(candidate, now=_Clock())
    try:
        assert not cache.available
        assert cache.status is not None
        assert cache.status.code == "cache_path_unusable"
        assert await cache.get("a") is None
        await _store(cache, "a", _payload())
    finally:
        await cache.aclose()
    assert not Path(candidate).exists()


@posix_only
@pytest.mark.asyncio
async def test_an_unwritable_directory_disables_rather_than_breaking_construction(
    tmp_path: Path,
) -> None:
    if os.geteuid() == 0:  # pragma: no cover - permission checks are meaningless as root
        pytest.skip("the superuser bypasses directory permissions")
    directory = tmp_path / "readonly"
    directory.mkdir(mode=0o500)
    os.chmod(directory, 0o500)
    try:
        cache = SQLiteCache(directory / "perception.db", now=_Clock())
        try:
            assert not cache.available
            assert cache.status is not None
            assert cache.status.code == "cache_path_unusable"
        finally:
            await cache.aclose()
    finally:
        os.chmod(directory, 0o700)


@pytest.mark.asyncio
async def test_clear_removes_every_entry_without_promising_erasure(tmp_path: Path) -> None:
    path = _cache_path(tmp_path)
    cache = SQLiteCache(path, now=_Clock())
    try:
        for index in range(4):
            await _store(cache, f"k{index}", _payload())
        await cache.clear()
        assert (await cache.stats()).entry_count == 0
        assert await cache.get("k0") is None
    finally:
        await cache.aclose()
    assert _rows(path) == []


# ------------------------------------------------- D6 TTL, LRU, and accounting


@pytest.mark.asyncio
async def test_ttl_expires_exactly_at_created_at_plus_ttl(tmp_path: Path) -> None:
    clock = _Clock()
    path = _cache_path(tmp_path)
    cache = SQLiteCache(path, ttl_s=10.0, now=clock)
    try:
        created_at = clock.value
        await _store(cache, "a", _payload())
        assert _rows(path) == [("a", created_at, created_at, created_at + 10.0)]

        clock.advance(9.999)
        assert await cache.get("a") == _payload()
        clock.advance(0.001)
        assert await cache.get("a") is None
        # The expired read is also the collection point, so the row is gone
        # rather than merely hidden.
        assert _rows(path) == []
    finally:
        await cache.aclose()


@pytest.mark.asyncio
async def test_an_expired_read_collects_every_other_expired_entry(tmp_path: Path) -> None:
    clock = _Clock()
    path = _cache_path(tmp_path)
    cache = SQLiteCache(path, ttl_s=5.0, now=clock)
    try:
        await _store(cache, "a", _payload())
        await _store(cache, "b", _payload())
        clock.advance(5.0)
        assert await cache.get("a") is None
        assert _rows(path) == []
    finally:
        await cache.aclose()


@pytest.mark.asyncio
async def test_reads_touch_recency_at_most_once_per_interval(tmp_path: Path) -> None:
    clock = _Clock()
    path = _cache_path(tmp_path)
    cache = SQLiteCache(path, touch_interval_s=60.0, now=clock)
    try:
        created_at = clock.value
        await _store(cache, "a", _payload())

        clock.advance(59.0)
        assert await cache.get("a") == _payload()
        # A reader inside the interval must not become a writer.
        assert _rows(path)[0][2] == created_at

        clock.advance(1.0)
        assert await cache.get("a") == _payload()
        assert _rows(path)[0][2] == created_at + 60.0
    finally:
        await cache.aclose()


@pytest.mark.asyncio
async def test_eviction_follows_approximate_recency(tmp_path: Path) -> None:
    clock = _Clock()
    cache = SQLiteCache(_cache_path(tmp_path), max_entries=4, touch_interval_s=0.0, now=clock)
    try:
        for key in ("a", "b", "c", "d"):
            await _store(cache, key, _payload())
            clock.advance(1.0)
        clock.advance(10.0)
        assert await cache.get("a") == _payload()

        clock.advance(1.0)
        await _store(cache, "e", _payload())

        assert await cache.get("a") == _payload()
        assert await cache.get("d") == _payload()
        assert await cache.get("e") == _payload()
        assert await cache.get("b") is None
        assert await cache.get("c") is None
    finally:
        await cache.aclose()


@pytest.mark.asyncio
async def test_a_write_crosses_the_high_watermark_by_one_entry_then_evicts_to_low(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    cache = SQLiteCache(_cache_path(tmp_path), max_entries=4, now=clock)
    try:
        assert cache.low_watermark_entries == max(1, int(4 * LOW_WATERMARK_RATIO))
        for index in range(4):
            await _store(cache, f"k{index}", _payload())
            clock.advance(1.0)
        # The configured capacity is usable in full: nothing is evicted until a
        # write actually crosses the high watermark.
        assert (await cache.stats()).entry_count == 4

        await _store(cache, "k4", _payload())
        assert (await cache.stats()).entry_count == cache.low_watermark_entries
        assert await cache.get("k4") == _payload()
    finally:
        await cache.aclose()


@pytest.mark.asyncio
async def test_byte_watermarks_evict_until_the_low_byte_total_is_restored(tmp_path: Path) -> None:
    clock = _Clock()
    cache = SQLiteCache(_cache_path(tmp_path), max_bytes=100, now=clock)
    try:
        assert cache.low_watermark_bytes == 90
        for key in ("a", "b"):
            await _store(cache, key, _payload(40))
            clock.advance(1.0)
        assert (await cache.stats()).total_bytes == 80

        await _store(cache, "c", _payload(40))
        stored = await cache.stats()
        assert stored.total_bytes == 80
        assert stored.entry_count == 2
        assert await cache.get("a") is None
        assert await cache.get("c") == _payload(40)
    finally:
        await cache.aclose()


@pytest.mark.asyncio
async def test_an_oversized_value_is_rejected_as_a_no_op_with_a_warning(tmp_path: Path) -> None:
    cache = SQLiteCache(_cache_path(tmp_path), max_bytes=100, now=_Clock())
    try:
        await _store(cache, "a", _payload(40))
        await _store(cache, "big", _payload(200))

        assert await cache.get("big") is None
        assert await cache.get("a") == _payload(40)
        stored = await cache.stats()
        assert stored.entry_count == 1
        assert stored.total_bytes == 40
        assert [warning.code for warning in cache.warnings] == ["cache_value_too_large"]
    finally:
        await cache.aclose()


@pytest.mark.asyncio
async def test_caller_supplied_accounting_must_match_the_value(tmp_path: Path) -> None:
    cache = SQLiteCache(_cache_path(tmp_path), now=_Clock())
    try:
        value = _payload(16)
        with pytest.raises(ValueError):
            await cache.set("a", value, size=len(value) + 1)
        with pytest.raises(ValueError):
            await cache.set("a", value, size=0)
        with pytest.raises(TypeError):
            await cache.set("a", "not bytes", size=9)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            await cache.set("a", b"not json", size=8)
        with pytest.raises(ValueError):
            await cache.set("", value, size=len(value))
        with pytest.raises(ValueError):
            await cache.get("bad\x00key")
        assert (await cache.stats()).entry_count == 0
    finally:
        await cache.aclose()


@pytest.mark.asyncio
async def test_statistics_derive_from_stored_sizes_rather_than_caller_accounting(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    cache = SQLiteCache(_cache_path(tmp_path), max_entries=8, max_bytes=4096, now=clock)
    try:
        for index, size in enumerate((16, 32, 64)):
            await _store(cache, f"k{index}", _payload(size))
            clock.advance(1.0)
        stored = await cache.stats()
        assert stored.entry_count == 3
        assert stored.total_bytes == 16 + 32 + 64
        assert stored.max_entries == 8
        assert stored.max_bytes == 4096
        assert stored.removed_entries == 0
        assert stored.removed_bytes == 0
    finally:
        await cache.aclose()


@pytest.mark.asyncio
async def test_prune_reports_expired_and_over_watermark_removals(tmp_path: Path) -> None:
    clock = _Clock()
    cache = SQLiteCache(_cache_path(tmp_path), max_entries=8, ttl_s=10.0, now=clock)
    try:
        await _store(cache, "a", _payload(16))
        await _store(cache, "b", _payload(32))
        clock.advance(5.0)
        await _store(cache, "c", _payload(64))

        assert (await cache.prune()).removed_entries == 0
        clock.advance(5.0)
        pruned = await cache.prune()
        assert pruned.removed_entries == 2
        assert pruned.removed_bytes == 16 + 32
        assert pruned.entry_count == 1
        assert pruned.total_bytes == 64
    finally:
        await cache.aclose()


@pytest.mark.asyncio
async def test_a_replaced_value_restarts_its_absolute_time_to_live(tmp_path: Path) -> None:
    clock = _Clock()
    path = _cache_path(tmp_path)
    cache = SQLiteCache(path, ttl_s=10.0, now=clock)
    try:
        await _store(cache, "a", _payload(16))
        clock.advance(9.0)
        await _store(cache, "a", _payload(32))
        assert _rows(path) == [("a", clock.value, clock.value, clock.value + 10.0)]
        assert (await cache.stats()).total_bytes == 32

        clock.advance(9.0)
        assert await cache.get("a") == _payload(32)
    finally:
        await cache.aclose()


def test_construction_rejects_programmer_errors_in_settings(tmp_path: Path) -> None:
    path = _cache_path(tmp_path)
    with pytest.raises(ValueError):
        SQLiteCache(path, max_entries=0)
    with pytest.raises(ValueError):
        SQLiteCache(path, max_bytes=-1)
    with pytest.raises(ValueError):
        SQLiteCache(path, ttl_s=0.0)
    with pytest.raises(ValueError):
        SQLiteCache(path, busy_timeout_s=0.0)
    with pytest.raises(ValueError):
        SQLiteCache(path, touch_interval_s=-1.0)
    with pytest.raises(TypeError):
        SQLiteCache(path, allow_symlink="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        SQLiteCache(path, now="now")  # type: ignore[arg-type]
    assert not path.exists()


@pytest.mark.asyncio
async def test_stored_values_are_the_exact_bytes_the_caller_supplied(tmp_path: Path) -> None:
    cache = SQLiteCache(_cache_path(tmp_path), now=_Clock())
    try:
        value = json.dumps(
            {"text": "sirkuit ünïcode", "boxes": [[1, 2, 3, 4]]},
            ensure_ascii=False,
        ).encode("utf-8")
        await _store(cache, "a", value)
        assert await cache.get("a") == value
    finally:
        await cache.aclose()


@pytest.mark.asyncio
async def test_a_value_between_the_watermarks_survives_its_own_eviction_pass(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    cache = SQLiteCache(_cache_path(tmp_path), max_bytes=100, now=clock)
    try:
        await _store(cache, "a", _payload(95))
        clock.advance(1.0)
        # The second write crosses the high watermark, and evicting everything
        # else still leaves the byte total above the low watermark. The entry the
        # write just accepted is never the one that pays for it.
        await _store(cache, "b", _payload(95))
        stored = await cache.stats()
        assert stored.entry_count == 1
        assert stored.total_bytes == 95
        assert await cache.get("b") == _payload(95)
        assert await cache.get("a") is None
    finally:
        await cache.aclose()


@pytest.mark.asyncio
async def test_a_failed_write_rolls_its_transaction_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = SQLiteCache(_cache_path(tmp_path), now=_Clock())
    try:
        await _store(cache, "kept", _payload(16))

        def _fail(
            self: SQLiteCache,
            connection: sqlite3.Connection,
            *,
            protected_key: str | None,
        ) -> tuple[int, int]:
            raise sqlite3.IntegrityError("eviction failed")

        monkeypatch.setattr(SQLiteCache, "_evict", _fail)
        await _store(cache, "rolled-back", _payload(16))
        monkeypatch.undo()

        assert await cache.get("rolled-back") is None
        assert await cache.get("kept") == _payload(16)
        assert (await cache.stats()).entry_count == 1
        assert "cache_operation_failed" in {warning.code for warning in cache.warnings}
    finally:
        await cache.aclose()


@pytest.mark.asyncio
async def test_a_directory_at_the_cache_path_disables_the_cache(tmp_path: Path) -> None:
    directory = tmp_path / "perception.db"
    directory.mkdir()
    cache = SQLiteCache(directory, now=_Clock())
    try:
        assert not cache.available
        assert cache.status is not None
        assert cache.status.code == "cache_path_unusable"
    finally:
        await cache.aclose()
    assert directory.is_dir()


@posix_only
@pytest.mark.asyncio
async def test_an_unreadable_database_file_disables_the_cache(tmp_path: Path) -> None:
    if os.geteuid() == 0:  # pragma: no cover - permission checks are meaningless as root
        pytest.skip("the superuser bypasses file permissions")
    path = _cache_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.touch(mode=0o600)
    os.chmod(path, 0o000)
    try:
        cache = SQLiteCache(path, now=_Clock())
        try:
            assert not cache.available
            assert cache.status is not None
            assert cache.status.code == "cache_path_unusable"
        finally:
            await cache.aclose()
    finally:
        os.chmod(path, 0o600)


def test_real_lock_contention_is_recognized(tmp_path: Path) -> None:
    path = tmp_path / "contended.db"
    holder = sqlite3.connect(str(path), isolation_level=None)
    blocked = sqlite3.connect(str(path), isolation_level=None, timeout=0)
    try:
        holder.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        holder.execute("BEGIN EXCLUSIVE")
        with pytest.raises(sqlite3.OperationalError) as raised:
            blocked.execute("BEGIN IMMEDIATE")
        assert _is_contention(raised.value)
    finally:
        holder.close()
        blocked.close()


@pytest.mark.asyncio
async def test_configured_limits_are_reported(tmp_path: Path) -> None:
    path = _cache_path(tmp_path)
    cache = SQLiteCache(path, max_entries=12, max_bytes=2048, ttl_s=30.0, now=_Clock())
    try:
        assert cache.path == path
        assert cache.max_entries == 12
        assert cache.max_bytes == 2048
        assert cache.ttl_s == 30.0
        assert cache.status is None
        assert not cache.closed
    finally:
        await cache.aclose()


@pytest.mark.asyncio
async def test_an_injected_clock_must_return_a_real_number(tmp_path: Path) -> None:
    cache = SQLiteCache(_cache_path(tmp_path), now=lambda: "now")  # type: ignore[arg-type,return-value]
    try:
        with pytest.raises(TypeError):
            await cache.get("a")
    finally:
        await cache.aclose()


@pytest.mark.asyncio
async def test_an_empty_or_home_relative_path_is_refused() -> None:
    for candidate in ("", "/tmp/~penampakan/perception.db"):
        cache = SQLiteCache(candidate, now=_Clock())
        try:
            assert not cache.available
            assert cache.status is not None
            assert cache.status.code == "cache_path_unusable"
        finally:
            await cache.aclose()


def test_construction_rejects_non_numeric_settings(tmp_path: Path) -> None:
    path = _cache_path(tmp_path)
    with pytest.raises(TypeError):
        SQLiteCache(path, ttl_s="10")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        SQLiteCache(path, busy_timeout_s="5")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        SQLiteCache(path, touch_interval_s="60")  # type: ignore[arg-type]
