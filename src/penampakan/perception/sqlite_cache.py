"""Durable perception cache owned by one dedicated SQLite worker thread.

SQLite connections are bound to the thread that created them, so this module
never touches its connection from the event loop and never hands connection work
to ``asyncio.to_thread``: an executor is free to run successive calls on
different threads, which would violate that contract. One worker thread creates,
owns, and closes the connection and drains an ordered queue of operations
submitted by the loop, and results travel back through
:meth:`asyncio.loop.call_soon_threadsafe`.

Content retention is a deliberate feature of this cache: it stores derived
descriptions of user images on disk, unencrypted, until they expire or are
cleared. Nothing here promises secure erasure.
"""

from __future__ import annotations

import asyncio
import os
import queue
import random
import secrets
import sqlite3
import stat
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final, Generic, TypeVar

from ..errors import PenampakanError
from ..models import CacheStats, WarningInfo
from .cache import (
    CACHE_SCHEMA_VERSION,
    validate_accounted_size,
    validate_json_bytes,
    validate_key,
    validate_positive_limit,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

# ``STRICT`` tables are the only way SQLite enforces the column types this
# schema depends on, and they were introduced in 3.37. Dropping the keyword on
# an older runtime would silently accept values this cache promises to reject.
MINIMUM_SQLITE_VERSION: Final = (3, 37, 0)

# The on-disk layout version. It is compared numerically on open: a newer
# database is never touched, an older one is quarantined and replaced.
DATABASE_SCHEMA_VERSION: Final = 1

# How stored values are encoded. Version 1 is the exact strict UTF-8 JSON bytes
# the caller supplied, stored as a BLOB without transformation.
VALUE_ENCODING_VERSION: Final = 1

# Eviction runs down to a low watermark rather than to the limit itself, so a
# steady write stream does not evict on every single accepted entry.
LOW_WATERMARK_RATIO: Final = 0.9

_EVICTION_BATCH: Final = 64
_RETRY_BASE_DELAY_S: Final = 0.005
_RETRY_MAX_BACKOFF_STEPS: Final = 6
_MAX_RETAINED_WARNINGS: Final = 16
_DIRECTORY_MODE: Final = 0o700
_FILE_MODE: Final = 0o600
_SQLITE_BUSY: Final = 5
_SQLITE_LOCKED: Final = 6
_SIDECAR_SUFFIXES: Final = ("-wal", "-shm")
_POSIX_PERMISSIONS: Final = os.name == "posix"

_ENTRIES_TABLE: Final = """
CREATE TABLE IF NOT EXISTS entries (
    key TEXT NOT NULL PRIMARY KEY,
    value BLOB NOT NULL,
    size INTEGER NOT NULL,
    created_at REAL NOT NULL,
    accessed_at REAL NOT NULL,
    expires_at REAL
) STRICT
"""
_META_TABLE: Final = """
CREATE TABLE IF NOT EXISTS meta (
    name TEXT NOT NULL PRIMARY KEY,
    value TEXT NOT NULL
) STRICT
"""
_ACCESSED_INDEX: Final = "CREATE INDEX IF NOT EXISTS entries_accessed_at ON entries (accessed_at)"
_EXPIRES_INDEX: Final = "CREATE INDEX IF NOT EXISTS entries_expires_at ON entries (expires_at)"
_UPSERT: Final = """
INSERT INTO entries (key, value, size, created_at, accessed_at, expires_at)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT (key) DO UPDATE SET
    value = excluded.value,
    size = excluded.size,
    created_at = excluded.created_at,
    accessed_at = excluded.accessed_at,
    expires_at = excluded.expires_at
"""
_LIVE_PREDICATE: Final = "(expires_at IS NULL OR expires_at > ?)"
_EXPIRED_PREDICATE: Final = "(expires_at IS NOT NULL AND expires_at <= ?)"

_SCHEMA_VERSION_KEY: Final = "schema_version"
_CACHE_SCHEMA_KEY: Final = "cache_schema_version"
_VALUE_ENCODING_KEY: Final = "value_encoding_version"

_ResultT = TypeVar("_ResultT")


class _CacheUnavailableError(Exception):
    """Internal outcome for a cache operation that could not reach its data.

    It never escapes this module. The session-facing surface converts it into a
    miss or a no-op, and the administrative surface converts it into a typed
    operator-facing error.
    """


@dataclass(slots=True)
class _Job(Generic[_ResultT]):
    """One queued connection operation and the loop future awaiting its result."""

    work: Callable[[sqlite3.Connection], _ResultT]
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future[_ResultT]
    abandoned: bool = False


@dataclass(slots=True)
class _CloseJob:
    """The ordered end of the queue, resolved once the connection is closed."""

    loop: asyncio.AbstractEventLoop
    future: asyncio.Future[None]


@dataclass(frozen=True, slots=True)
class _OpenOutcome:
    """What the worker learned while opening the database."""

    status: WarningInfo | None
    warnings: tuple[WarningInfo, ...]


@dataclass(frozen=True, slots=True)
class _Recreate:
    """The existing file cannot be used and must be quarantined first."""

    kind: str
    reason: str


@dataclass(frozen=True, slots=True)
class _Disabled:
    """The cache cannot open and must report why instead of failing construction."""

    status: WarningInfo


class SQLiteCache:
    """A durable perception cache serialized through one dedicated worker thread.

    Every connection operation runs on that worker inside a short explicit
    transaction. Contention is retried with bounded jitter inside the operation
    deadline and then degrades: the session-facing surface reports a miss or a
    no-op, while :meth:`stats`, :meth:`clear`, and :meth:`prune` raise, because
    an operator who asked a direct question is misled by silence.

    A runtime path or data problem disables the instance instead of breaking
    construction. A disabled instance is still a valid cache: it misses, accepts
    no values, and closes cleanly, and :attr:`status` says why.
    """

    __slots__ = (
        "_allow_symlink",
        "_available",
        "_busy_timeout_s",
        "_clock",
        "_closed",
        "_connection_thread_idents",
        "_lock",
        "_low_bytes",
        "_low_entries",
        "_max_bytes",
        "_max_entries",
        "_opened",
        "_path",
        "_pending_close",
        "_queue",
        "_status",
        "_thread",
        "_touch_interval_s",
        "_ttl_s",
        "_warnings",
    )

    durable: ClassVar[bool] = True

    def __init__(
        self,
        path: Path | str,
        *,
        max_entries: int = 256,
        max_bytes: int = 128 * 1024 * 1024,
        ttl_s: float | None = None,
        busy_timeout_s: float = 5.0,
        allow_symlink: bool = False,
        touch_interval_s: float = 60.0,
        now: Callable[[], float] | None = None,
    ) -> None:
        self._max_entries = validate_positive_limit("max_entries", max_entries)
        self._max_bytes = validate_positive_limit("max_bytes", max_bytes)
        self._ttl_s = None if ttl_s is None else _validate_positive_seconds("ttl_s", ttl_s)
        self._busy_timeout_s = _validate_positive_seconds("busy_timeout_s", busy_timeout_s)
        self._touch_interval_s = _validate_interval("touch_interval_s", touch_interval_s)
        if not isinstance(allow_symlink, bool):
            raise TypeError("allow_symlink must be a bool")
        if now is not None and not callable(now):
            raise TypeError("now must be a callable returning epoch seconds")
        self._allow_symlink = allow_symlink
        self._clock: Callable[[], float] = time.time if now is None else now
        self._path = Path(path)
        self._low_entries = max(1, int(self._max_entries * LOW_WATERMARK_RATIO))
        self._low_bytes = max(1, int(self._max_bytes * LOW_WATERMARK_RATIO))
        self._queue: queue.Queue[_Job[Any] | _CloseJob] = queue.Queue()
        self._lock = threading.Lock()
        self._opened = threading.Event()
        self._warnings: list[WarningInfo] = []
        self._connection_thread_idents: set[int] = set()
        self._pending_close: _CloseJob | None = None
        self._status: WarningInfo | None = None
        self._available = False
        self._closed = False
        self._thread = threading.Thread(
            target=self._worker_main,
            name=f"penampakan-sqlite-cache-{id(self):x}",
            daemon=True,
        )
        self._thread.start()
        # Opening is the one operation whose result the constructor must publish
        # synchronously, because ``available`` and ``status`` are read straight
        # after construction. It touches one small file and does not wait on any
        # queued work.
        self._opened.wait()

    async def get(self, key: str) -> bytes | None:
        """Return the cached bytes for a key, or a miss when it is absent or expired."""

        validate_key(key)
        if not self._usable:
            return None
        try:
            return await self._submit(lambda connection: self._get_operation(connection, key))
        except (_CacheUnavailableError, sqlite3.Error) as error:
            self._note_degradation(error)
            return None

    async def set(self, key: str, value: bytes, *, size: int) -> None:
        """Store validated JSON bytes under a key, evicting down to the low watermark."""

        validate_key(key)
        stored = validate_json_bytes(value)
        accounted_size = validate_accounted_size(size, stored)
        if accounted_size > self._max_bytes:
            # Accepting it would evict the entire cache to make room for a value
            # that still would not fit under the byte limit.
            self._add_warning(
                "cache_value_too_large",
                "A cache value larger than the configured byte limit was not stored.",
                {"size": accounted_size, "max_bytes": self._max_bytes},
            )
            return
        if not self._usable:
            return
        try:
            await self._submit(
                lambda connection: self._set_operation(connection, key, stored, accounted_size)
            )
        except (_CacheUnavailableError, sqlite3.Error) as error:
            self._note_degradation(error)

    async def aclose(self) -> None:
        """Drain queued work, close the connection, and stop the worker idempotently."""

        loop = asyncio.get_running_loop()
        close_future: asyncio.Future[None] = loop.create_future()
        close_job = _CloseJob(loop=loop, future=close_future)
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if not self._available:
                return
            # Work already queued sits ahead of this sentinel and is drained in
            # order, so an accepted write is never dropped by a later close.
            self._queue.put(close_job)
        with suppress(_CacheUnavailableError, sqlite3.Error):
            await asyncio.shield(close_job.future)
        # The worker resolves that future as its last action, so this join only
        # reaps an already finished thread.
        self._thread.join(timeout=self._busy_timeout_s)

    async def stats(self) -> CacheStats:
        """Return a transactional snapshot derived from verified stored sizes."""

        self._require_administrable()
        try:
            return await self._submit(self._stats_operation)
        except (_CacheUnavailableError, sqlite3.Error) as error:
            raise _unavailable_error(error) from error

    async def clear(self) -> None:
        """Remove every logical entry in one transaction.

        This is reclamation, not secure erasure: it promises nothing about
        database pages, the write-ahead log, backups, or filesystem snapshots.
        """

        self._require_administrable()
        try:
            await self._submit(self._clear_operation)
        except (_CacheUnavailableError, sqlite3.Error) as error:
            raise _unavailable_error(error) from error

    async def prune(self) -> CacheStats:
        """Drop expired and over-watermark entries and report what was discarded."""

        self._require_administrable()
        try:
            return await self._submit(self._prune_operation)
        except (_CacheUnavailableError, sqlite3.Error) as error:
            raise _unavailable_error(error) from error

    @property
    def path(self) -> Path:
        """Return the configured database path."""

        return self._path

    @property
    def available(self) -> bool:
        """Return whether this instance retains data rather than having disabled itself."""

        return self._available

    @property
    def status(self) -> WarningInfo | None:
        """Return why the cache disabled itself, or ``None`` when it is available."""

        return self._status

    @property
    def closed(self) -> bool:
        """Return whether close has been requested."""

        return self._closed

    @property
    def warnings(self) -> tuple[WarningInfo, ...]:
        """Return the deduplicated non-fatal warnings observed so far."""

        with self._lock:
            return tuple(self._warnings)

    @property
    def max_entries(self) -> int:
        """Return the configured maximum entry count."""

        return self._max_entries

    @property
    def max_bytes(self) -> int:
        """Return the configured maximum stored byte size."""

        return self._max_bytes

    @property
    def ttl_s(self) -> float | None:
        """Return the configured absolute time to live in seconds."""

        return self._ttl_s

    @property
    def low_watermark_entries(self) -> int:
        """Return the entry count eviction restores after an accepted write."""

        return self._low_entries

    @property
    def low_watermark_bytes(self) -> int:
        """Return the byte total eviction restores after an accepted write."""

        return self._low_bytes

    @property
    def _usable(self) -> bool:
        return self._available and not self._closed

    async def _submit(self, work: Callable[[sqlite3.Connection], _ResultT]) -> _ResultT:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[_ResultT] = loop.create_future()
        job = _Job(work=work, loop=loop, future=future)
        with self._lock:
            if self._closed or not self._available:
                raise _CacheUnavailableError
            self._queue.put(job)
        try:
            return await job.future
        except asyncio.CancelledError:
            # The worker skips a job whose caller stopped waiting, so a cancelled
            # await does not leave a write that nobody expects to have happened.
            job.abandoned = True
            raise

    def _require_administrable(self) -> None:
        if self._closed:
            raise PenampakanError(code="cache_closed")
        if not self._available:
            raise PenampakanError(
                code="cache_disabled" if self._status is None else self._status.code
            )

    def _note_degradation(self, error: Exception) -> None:
        if isinstance(error, _CacheUnavailableError):
            self._add_warning(
                "cache_unavailable",
                "The cache could not be reached within its deadline and was bypassed.",
                {},
            )
            return
        self._add_warning(
            "cache_operation_failed",
            "A cache operation failed and was bypassed.",
            {"error": type(error).__name__},
        )

    def _add_warning(self, code: str, message: str, details: dict[str, Any]) -> None:
        warning = WarningInfo(code=code, message=message, details=details)
        with self._lock:
            if any(existing.code == code for existing in self._warnings):
                return
            if len(self._warnings) >= _MAX_RETAINED_WARNINGS:
                return
            self._warnings.append(warning)

    # ---------------------------------------------------------------- worker

    def _worker_main(self) -> None:
        connection: sqlite3.Connection | None = None
        try:
            connection, outcome = self._open()
            self._status = outcome.status
            self._warnings.extend(outcome.warnings)
            self._available = connection is not None
        except BaseException as error:  # pragma: no cover - defensive
            self._status = _disable_status("cache_path_unusable", type(error).__name__)
            self._available = False
            connection = None
        finally:
            # Construction blocks on this event, so it must be set even when the
            # open path failed in a way this module did not anticipate.
            self._opened.set()
        if connection is None:
            return
        try:
            self._serve(connection)
        finally:
            with suppress(sqlite3.Error):
                connection.close()
            self._finish_close()

    def _serve(self, connection: sqlite3.Connection) -> None:
        while True:
            job = self._queue.get()
            if isinstance(job, _CloseJob):
                self._pending_close = job
                return
            if job.abandoned:
                continue
            try:
                result = self._run_operation(connection, job.work)
            except BaseException as error:
                _post_exception(job.loop, job.future, error)
            else:
                _post_result(job.loop, job.future, result)

    def _finish_close(self) -> None:
        # Anything still queued cannot run any more, so its caller is told the
        # cache became unreachable rather than being left awaiting forever.
        while True:
            try:
                job = self._queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(job, _CloseJob):
                _post_result(job.loop, job.future, None)
            else:
                _post_exception(job.loop, job.future, _CacheUnavailableError())
        close_job = self._pending_close
        self._pending_close = None
        if close_job is not None:
            _post_result(close_job.loop, close_job.future, None)

    def _run_operation(
        self,
        connection: sqlite3.Connection,
        work: Callable[[sqlite3.Connection], _ResultT],
    ) -> _ResultT:
        self._connection_thread_idents.add(threading.get_ident())
        return _run_within_deadline(
            lambda: work(connection),
            timeout_s=self._busy_timeout_s,
            on_retry=lambda: _safe_rollback(connection),
        )

    # ------------------------------------------------------------ operations

    def _get_operation(self, connection: sqlite3.Connection, key: str) -> bytes | None:
        now = self._read_clock()
        row = connection.execute(
            "SELECT value, accessed_at, expires_at FROM entries WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        value, accessed_at, expires_at = bytes(row[0]), float(row[1]), row[2]
        if expires_at is not None and float(expires_at) <= now:
            # A read that found expired data is the cheapest moment to collect
            # every other entry that expired with it.
            with self._transaction(connection):
                connection.execute(f"DELETE FROM entries WHERE {_EXPIRED_PREDICATE}", (now,))
            return None
        if now - accessed_at >= self._touch_interval_s:
            # Approximate LRU: recency is refreshed at most once per interval so
            # a read-mostly workload does not turn every reader into a writer.
            with self._transaction(connection):
                connection.execute(
                    "UPDATE entries SET accessed_at = ? WHERE key = ?",
                    (now, key),
                )
        return value

    def _set_operation(
        self,
        connection: sqlite3.Connection,
        key: str,
        value: bytes,
        size: int,
    ) -> None:
        now = self._read_clock()
        expires_at = None if self._ttl_s is None else now + self._ttl_s
        with self._transaction(connection):
            connection.execute(f"DELETE FROM entries WHERE {_EXPIRED_PREDICATE}", (now,))
            connection.execute(_UPSERT, (key, value, size, now, now, expires_at))
            self._evict(connection, protected_key=key)

    def _stats_operation(self, connection: sqlite3.Connection) -> CacheStats:
        now = self._read_clock()
        # One statement is one transaction, so the count and the byte total
        # cannot be read from two different versions of the table.
        entry_count, total_bytes = self._totals(connection, now)
        return CacheStats(
            entry_count=entry_count,
            total_bytes=total_bytes,
            max_entries=self._max_entries,
            max_bytes=self._max_bytes,
        )

    def _clear_operation(self, connection: sqlite3.Connection) -> None:
        with self._transaction(connection):
            connection.execute("DELETE FROM entries")

    def _prune_operation(self, connection: sqlite3.Connection) -> CacheStats:
        now = self._read_clock()
        with self._transaction(connection):
            expired_entries, expired_bytes = connection.execute(
                f"SELECT COUNT(*), COALESCE(SUM(size), 0) FROM entries WHERE {_EXPIRED_PREDICATE}",
                (now,),
            ).fetchone()
            connection.execute(f"DELETE FROM entries WHERE {_EXPIRED_PREDICATE}", (now,))
            evicted_entries, evicted_bytes = self._evict(connection, protected_key=None)
            entry_count, total_bytes = self._totals(connection, now)
        return CacheStats(
            entry_count=entry_count,
            total_bytes=total_bytes,
            max_entries=self._max_entries,
            max_bytes=self._max_bytes,
            removed_entries=int(expired_entries) + evicted_entries,
            removed_bytes=int(expired_bytes) + evicted_bytes,
        )

    def _totals(self, connection: sqlite3.Connection, now: float) -> tuple[int, int]:
        row = connection.execute(
            f"SELECT COUNT(*), COALESCE(SUM(size), 0) FROM entries WHERE {_LIVE_PREDICATE}",
            (now,),
        ).fetchone()
        return int(row[0]), int(row[1])

    def _evict(
        self, connection: sqlite3.Connection, *, protected_key: str | None
    ) -> tuple[int, int]:
        row = connection.execute("SELECT COUNT(*), COALESCE(SUM(size), 0) FROM entries").fetchone()
        entry_count, total_bytes = int(row[0]), int(row[1])
        removed_entries = 0
        removed_bytes = 0
        if entry_count <= self._max_entries and total_bytes <= self._max_bytes:
            # Nothing crossed the high watermark, so the configured capacity is
            # usable in full rather than being permanently held at the low one.
            return removed_entries, removed_bytes
        while entry_count > self._low_entries or total_bytes > self._low_bytes:
            # ``key IS NOT ?`` protects the entry this write just accepted while
            # still selecting every row when nothing is protected.
            candidates = connection.execute(
                "SELECT key, size FROM entries WHERE key IS NOT ?"
                " ORDER BY accessed_at ASC, created_at ASC, key ASC LIMIT ?",
                (protected_key, _EVICTION_BATCH),
            ).fetchall()
            if not candidates:
                return removed_entries, removed_bytes
            for candidate_key, candidate_size in candidates:
                connection.execute("DELETE FROM entries WHERE key = ?", (candidate_key,))
                entry_count -= 1
                total_bytes -= int(candidate_size)
                removed_entries += 1
                removed_bytes += int(candidate_size)
                if entry_count <= self._low_entries and total_bytes <= self._low_bytes:
                    break
        return removed_entries, removed_bytes

    def _read_clock(self) -> float:
        value = self._clock()
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError("now must return epoch seconds as a real number")
        return float(value)

    @contextmanager
    def _transaction(self, connection: sqlite3.Connection) -> Iterator[None]:
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            _safe_rollback(connection)
            raise
        connection.execute("COMMIT")

    # ------------------------------------------------------------------ open

    def _open(self) -> tuple[sqlite3.Connection | None, _OpenOutcome]:
        warnings: list[WarningInfo] = []
        if sqlite3.sqlite_version_info < MINIMUM_SQLITE_VERSION:
            return None, _OpenOutcome(
                status=_disable_status(
                    "sqlite_version_unsupported",
                    "strict tables require SQLite 3.37 or later",
                ),
                warnings=(),
            )
        path_status = self._prepare_path(warnings)
        if path_status is not None:
            return None, _OpenOutcome(status=path_status, warnings=tuple(warnings))
        # One recreate attempt is enough: the second pass runs against a file
        # this cache just created, and looping on a path that keeps failing
        # would quarantine the directory one file at a time.
        for attempt in range(2):
            outcome = self._try_open()
            if isinstance(outcome, _Disabled):
                return None, _OpenOutcome(status=outcome.status, warnings=tuple(warnings))
            if isinstance(outcome, sqlite3.Connection):
                self._apply_artifact_permissions(warnings)
                return outcome, _OpenOutcome(status=None, warnings=tuple(warnings))
            if attempt == 1:  # pragma: no cover - a freshly created file re-failing
                return None, _OpenOutcome(
                    status=_disable_status("cache_path_unusable", outcome.reason),
                    warnings=tuple(warnings),
                )
            try:
                self._quarantine(outcome.kind)
            except OSError as error:
                # The original file is left exactly as it was: overwriting data
                # this cache could not identify would destroy it.
                return None, _OpenOutcome(
                    status=_disable_status("cache_quarantine_failed", type(error).__name__),
                    warnings=tuple(warnings),
                )
            recreate_status = self._create_private_file()
            if recreate_status is not None:  # pragma: no cover - the directory just worked
                return None, _OpenOutcome(status=recreate_status, warnings=tuple(warnings))
        raise AssertionError("unreachable")  # pragma: no cover

    def _try_open(self) -> sqlite3.Connection | _Recreate | _Disabled:
        try:
            connection = sqlite3.connect(
                str(self._path),
                timeout=self._busy_timeout_s,
                isolation_level=None,
                check_same_thread=True,
            )
        except sqlite3.Error as error:
            return _Disabled(_disable_status("cache_path_unusable", type(error).__name__))
        keep_open = False
        try:
            self._connection_thread_idents.add(threading.get_ident())
            outcome = _run_within_deadline(
                lambda: self._configure(connection),
                timeout_s=self._busy_timeout_s,
                on_retry=lambda: _safe_rollback(connection),
            )
            keep_open = outcome is None
            return connection if outcome is None else outcome
        except _CacheUnavailableError:
            return _Disabled(_disable_status("cache_path_unusable", "database is locked"))
        except sqlite3.OperationalError as error:
            # Read-only files, missing directories, and I/O errors are path
            # problems; they are never a reason to quarantine user data.
            return _Disabled(_disable_status("cache_path_unusable", type(error).__name__))
        except sqlite3.DatabaseError:
            return _Recreate(kind="corrupt", reason="file is not a usable database")
        finally:
            if not keep_open:
                with suppress(sqlite3.Error):
                    connection.close()

    def _configure(self, connection: sqlite3.Connection) -> _Recreate | _Disabled | None:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute(f"PRAGMA busy_timeout={int(self._busy_timeout_s * 1000)}")
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        if "meta" not in tables:
            if tables:
                return _Recreate(kind="superseded", reason="unrecognized schema")
            self._create_schema(connection)
            return None
        stored = {
            str(row[0]): str(row[1])
            for row in connection.execute("SELECT name, value FROM meta").fetchall()
        }
        version = _parse_int(stored.get(_SCHEMA_VERSION_KEY))
        if version is not None and version > DATABASE_SCHEMA_VERSION:
            # An older binary must never drop data a newer one wrote, so the
            # file is left untouched and this instance stops here.
            return _Disabled(
                _disable_status("cache_schema_too_new", f"database schema version {version}")
            )
        if (
            version != DATABASE_SCHEMA_VERSION
            or stored.get(_CACHE_SCHEMA_KEY) != CACHE_SCHEMA_VERSION
            or _parse_int(stored.get(_VALUE_ENCODING_KEY)) != VALUE_ENCODING_VERSION
            or "entries" not in tables
        ):
            # No migration is defined for any released layout yet, so an older
            # or differently encoded database is preserved beside a fresh one.
            return _Recreate(kind="superseded", reason="unsupported schema version")
        return None

    def _create_schema(self, connection: sqlite3.Connection) -> None:
        with self._transaction(connection):
            connection.execute(_ENTRIES_TABLE)
            connection.execute(_META_TABLE)
            connection.execute(_ACCESSED_INDEX)
            connection.execute(_EXPIRES_INDEX)
            connection.executemany(
                "INSERT INTO meta (name, value) VALUES (?, ?)"
                " ON CONFLICT (name) DO UPDATE SET value = excluded.value",
                (
                    (_SCHEMA_VERSION_KEY, str(DATABASE_SCHEMA_VERSION)),
                    (_CACHE_SCHEMA_KEY, CACHE_SCHEMA_VERSION),
                    (_VALUE_ENCODING_KEY, str(VALUE_ENCODING_VERSION)),
                ),
            )

    # ------------------------------------------------------------ filesystem

    def _prepare_path(self, warnings: list[WarningInfo]) -> WarningInfo | None:
        unusable = _validate_path_shape(self._path)
        if unusable is not None:
            return _disable_status("cache_path_unusable", unusable)
        if self._path.is_symlink() and not self._allow_symlink:
            return _disable_status("cache_symlink_rejected", "the cache path is a symbolic link")
        if self._path.is_dir():
            return _disable_status("cache_path_unusable", "the cache path is a directory")
        missing = [parent for parent in self._path.parents if not parent.exists()]
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            return _disable_status("cache_path_unusable", type(error).__name__)
        if _POSIX_PERMISSIONS:
            for created in missing:
                # ``mkdir(mode=...)`` is masked by the umask, so the private mode
                # is applied explicitly to every directory this cache created.
                with suppress(OSError):
                    os.chmod(created, _DIRECTORY_MODE)
        if not missing:
            _check_permissions(
                self._path.parent,
                _DIRECTORY_MODE,
                "cache_directory_permissions",
                warnings,
            )
        if self._path.exists():
            _check_permissions(self._path, _FILE_MODE, "cache_file_permissions", warnings)
            return None
        return self._create_private_file()

    def _create_private_file(self) -> WarningInfo | None:
        try:
            # Creating the file before SQLite does is the only way to know it
            # never briefly existed with umask-wide permissions.
            descriptor = os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_RDWR, _FILE_MODE)
        except FileExistsError:
            return None
        except OSError as error:
            return _disable_status("cache_path_unusable", type(error).__name__)
        os.close(descriptor)
        if _POSIX_PERMISSIONS:
            # The umask can only clear bits, so it may leave the owner unable to
            # read the file it just created.
            with suppress(OSError):
                os.chmod(self._path, _FILE_MODE)
        return None

    def _apply_artifact_permissions(self, warnings: list[WarningInfo]) -> None:
        if not _POSIX_PERMISSIONS:
            return
        for suffix in _SIDECAR_SUFFIXES:
            sidecar = self._path.with_name(self._path.name + suffix)
            if not sidecar.exists():
                continue
            # SQLite creates the write-ahead log and shared-memory files itself,
            # under the process umask, so they are tightened here.
            with suppress(OSError):
                os.chmod(sidecar, _FILE_MODE)
            _check_permissions(sidecar, _FILE_MODE, "cache_file_permissions", warnings)

    def _quarantine(self, kind: str) -> Path:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        target = self._path.with_name(f"{self._path.name}.{kind}-{stamp}-{secrets.token_hex(4)}")
        _quarantine_rename(self._path, target)
        for suffix in _SIDECAR_SUFFIXES:
            sidecar = self._path.with_name(self._path.name + suffix)
            if not sidecar.exists():
                continue
            # A stale write-ahead log left beside a fresh database would be
            # replayed into it, so the sidecars follow their database.
            with suppress(OSError):
                _quarantine_rename(sidecar, target.with_name(target.name + suffix))
        return target


def _quarantine_rename(source: Path, target: Path) -> None:
    """Atomically move one cache artifact aside, preserving its contents."""

    os.replace(source, target)


def _run_within_deadline(
    operation: Callable[[], _ResultT],
    *,
    timeout_s: float,
    on_retry: Callable[[], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    jitter: Callable[[], float] = random.random,
) -> _ResultT:
    """Run one connection operation, retrying lock contention inside its deadline.

    Write-ahead logging and a busy timeout reduce contention; they do not remove
    it. Retries are bounded by the deadline and exhaustion becomes an internal
    unavailable outcome rather than an unbounded wait.
    """

    deadline = monotonic() + timeout_s
    attempt = 0
    while True:
        try:
            return operation()
        except sqlite3.OperationalError as error:
            if not _is_contention(error):
                raise
            if on_retry is not None:
                on_retry()
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise _CacheUnavailableError from error
            backoff = _RETRY_BASE_DELAY_S * float(2 ** min(attempt, _RETRY_MAX_BACKOFF_STEPS))
            attempt += 1
            sleep(max(0.0, min(remaining, backoff * (0.5 + jitter()))))


def _is_contention(error: sqlite3.OperationalError) -> bool:
    code = getattr(error, "sqlite_errorcode", None)
    if isinstance(code, int):
        return (code & 0xFF) in (_SQLITE_BUSY, _SQLITE_LOCKED)
    text = str(error).lower()
    return "is locked" in text


def _safe_rollback(connection: sqlite3.Connection) -> None:
    with suppress(sqlite3.Error):
        connection.rollback()


def _post_result(
    loop: asyncio.AbstractEventLoop,
    future: asyncio.Future[_ResultT],
    result: _ResultT,
) -> None:
    def _apply() -> None:
        if not future.done():
            future.set_result(result)

    with suppress(RuntimeError):
        loop.call_soon_threadsafe(_apply)


def _post_exception(
    loop: asyncio.AbstractEventLoop,
    future: asyncio.Future[Any],
    error: BaseException,
) -> None:
    def _apply() -> None:
        if not future.done():
            future.set_exception(error)

    with suppress(RuntimeError):
        loop.call_soon_threadsafe(_apply)


def _disable_status(code: str, reason: str) -> WarningInfo:
    return WarningInfo(
        code=code,
        message="The durable perception cache disabled itself and retains nothing.",
        details={"reason": reason},
    )


def _unavailable_error(error: Exception) -> PenampakanError:
    # Administration is operator-facing, so an unreachable cache says so instead
    # of returning a snapshot that a healthy empty cache would also produce.
    return PenampakanError(code="cache_unavailable", retryable=True, cause=error)


def _check_permissions(
    target: Path,
    expected_mode: int,
    code: str,
    warnings: list[WarningInfo],
) -> None:
    if not _POSIX_PERMISSIONS:
        return
    try:
        mode = stat.S_IMODE(target.stat().st_mode)
    except OSError:  # pragma: no cover - the artifact was removed under us
        return
    if mode & ~expected_mode:
        warnings.append(
            WarningInfo(
                code=code,
                message="An existing cache artifact is readable beyond its owner.",
                details={
                    "mode": format(mode, "04o"),
                    "expected_mode": format(expected_mode, "04o"),
                },
            )
        )


def _validate_path_shape(path: Path) -> str | None:
    text = str(path)
    if not text or path.name == "":
        return "the cache path is empty"
    if "\x00" in text:
        return "the cache path contains a NUL byte"
    if not path.is_absolute():
        # A relative path resolves against whatever directory the process
        # happens to be in, which is exactly the kind of target this cache must
        # never derive for itself.
        return "the cache path is not absolute"
    for part in path.parts:
        if part.startswith("~"):
            return "the cache path contains an unexpanded home reference"
        if any(character in part for character in "*?["):
            return "the cache path contains glob metacharacters"
        if "$" in part or (part.startswith("%") and part.endswith("%") and len(part) > 1):
            return "the cache path contains an unexpanded variable reference"
    return None


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _validate_positive_seconds(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number of seconds")
    number = float(value)
    if not number > 0 or number != number or number == float("inf"):
        raise ValueError(f"{name} must be a positive finite number of seconds")
    return number


def _validate_interval(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number of seconds")
    number = float(value)
    if number < 0 or number != number or number == float("inf"):
        raise ValueError(f"{name} must be a non-negative finite number of seconds")
    return number


__all__ = [
    "DATABASE_SCHEMA_VERSION",
    "LOW_WATERMARK_RATIO",
    "MINIMUM_SQLITE_VERSION",
    "VALUE_ENCODING_VERSION",
    "SQLiteCache",
]
