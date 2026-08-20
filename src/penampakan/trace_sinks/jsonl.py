"""A bounded, process-local JSON Lines trace destination.

Each sink is bound to the event loop that first receives an event. One writer
task on that loop exclusively owns the file descriptor, serialized ordering,
flushes, and rotation. This makes concurrent sessions in one process safe, but
does not make sharing the configured path between processes safe. Applications
that need that must supply and test a cross-process locking scheme externally.

Trace events are already redacted before they reach this transport. The file is
still a retention destination and is not encrypted here. Newly created parents
and files use private POSIX modes where available; broader existing permissions
are reported, not silently changed. Symbolic-link paths are refused unless the
caller explicitly accepts their inherent time-of-check/time-of-use limitations.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast

from penampakan.models import JsonValue, TraceEvent, WarningInfo

Overflow = Literal["drop_new", "block"]

_DIRECTORY_MODE: Final = 0o700
_FILE_MODE: Final = 0o600
_POSIX_PERMISSIONS: Final = os.name == "posix"
_STOP: Final = object()


@dataclass(frozen=True, slots=True)
class JsonlTraceSinkStats:
    """Immutable accounting for accepted, persisted, and lost trace events."""

    accepted_events: int
    written_events: int
    dropped_events: int
    post_close_emits: int
    write_failures: int
    rotations: int
    queued_events: int
    warning_count: int

    @property
    def post_close_events(self) -> int:
        """Alias for events offered after admission stopped."""

        return self.post_close_emits


class JsonlTraceSink:
    """Write canonical UTF-8 JSON lines through one bounded writer task.

    ``drop_new`` is the non-blocking default: a full queue drops the offered
    event, increments :attr:`JsonlTraceSinkStats.dropped_events`, and records one
    safe warning. ``block`` explicitly opts the caller into async backpressure.

    Rotation is checked before each line is written. The current file becomes
    ``<path>.1`` and older generations move upward through ``rotation_count``.
    A line larger than ``rotation_bytes`` is kept intact in a file of its own.
    Close stops new admission, drains every accepted event, flushes according to
    ``fsync``, closes exactly once, and is safe to call repeatedly. An event
    offered after close is ignored and counted.

    Future event schema versions are serialized opaquely, including their
    original ``schema_version`` value. This transport never interprets or
    downgrades event schemas.
    """

    __slots__ = (
        "_accepted_events",
        "_allow_symlink",
        "_close_task",
        "_closed",
        "_dropped_events",
        "_fsync",
        "_loop",
        "_overflow",
        "_path",
        "_pending_events",
        "_post_close_emits",
        "_queue",
        "_rotation_bytes",
        "_rotation_count",
        "_rotations",
        "_warning_codes",
        "_warnings",
        "_write_failures",
        "_writer_task",
        "_written_events",
    )

    def __init__(
        self,
        path: Path | str,
        *,
        rotation_bytes: int = 10 * 1024 * 1024,
        rotation_count: int = 5,
        fsync: bool = False,
        queue_size: int = 1024,
        overflow: Overflow = "drop_new",
        allow_symlink: bool = False,
    ) -> None:
        self._path = Path(path)
        unusable = _validate_path_shape(self._path)
        if unusable is not None:
            raise ValueError(f"invalid trace path: {unusable}")
        self._rotation_bytes = _positive_int("rotation_bytes", rotation_bytes)
        self._rotation_count = _positive_int("rotation_count", rotation_count)
        self._queue: asyncio.Queue[bytes | object] = asyncio.Queue(
            maxsize=_positive_int("queue_size", queue_size)
        )
        if not isinstance(fsync, bool):
            raise TypeError("fsync must be a bool")
        if overflow not in ("drop_new", "block"):
            raise ValueError("overflow must be 'drop_new' or 'block'")
        if not isinstance(allow_symlink, bool):
            raise TypeError("allow_symlink must be a bool")
        self._fsync = fsync
        self._overflow: Overflow = overflow
        self._allow_symlink = allow_symlink
        self._loop: asyncio.AbstractEventLoop | None = None
        self._writer_task: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._closed = False
        self._accepted_events = 0
        self._written_events = 0
        self._dropped_events = 0
        self._post_close_emits = 0
        self._write_failures = 0
        self._rotations = 0
        self._pending_events = 0
        self._warnings: list[WarningInfo] = []
        self._warning_codes: set[str] = set()

    async def emit(self, event: TraceEvent) -> None:
        """Serialize and admit one already-redacted event."""

        if not isinstance(event, TraceEvent):
            raise TypeError("event must be a TraceEvent")
        if self._closed:
            self._post_close_emits += 1
            return
        self._bind_loop()
        line = _canonical_line(event)
        self._ensure_writer()
        if self._overflow == "drop_new":
            try:
                self._queue.put_nowait(line)
            except asyncio.QueueFull:
                self._dropped_events += 1
                self._record_warning(
                    "trace_sink_queue_overflow",
                    "The JSONL trace sink queue is full; new events are being dropped.",
                )
                return
            self._accepted_events += 1
            self._pending_events += 1
            return

        # Count the waiter as pending before yielding. That prevents a writer
        # awakened by this put from decrementing the value before emit resumes.
        self._accepted_events += 1
        self._pending_events += 1
        try:
            await self._queue.put(line)
        except BaseException:
            self._accepted_events -= 1
            self._pending_events -= 1
            raise

    async def aclose(self) -> None:
        """Stop admission and drain accepted lines idempotently."""

        close_task = self._close_task
        if self._closed and (close_task is None or close_task.done()):
            return
        self._bind_loop()
        if close_task is None:
            self._closed = True
            if self._writer_task is None:
                return
            close_task = asyncio.create_task(
                self._finish_close(),
                name=f"penampakan-jsonl-close-{id(self):x}",
            )
            self._close_task = close_task
        await asyncio.shield(close_task)

    def stats(self) -> JsonlTraceSinkStats:
        """Return an immutable, non-blocking accounting snapshot."""

        return JsonlTraceSinkStats(
            accepted_events=self._accepted_events,
            written_events=self._written_events,
            dropped_events=self._dropped_events,
            post_close_emits=self._post_close_emits,
            write_failures=self._write_failures,
            rotations=self._rotations,
            queued_events=self._pending_events,
            warning_count=len(self._warnings),
        )

    @property
    def path(self) -> Path:
        """Return the configured destination path."""

        return self._path

    @property
    def closed(self) -> bool:
        """Return whether this sink has stopped accepting events."""

        return self._closed

    @property
    def warnings(self) -> tuple[WarningInfo, ...]:
        """Return deduplicated safe warnings observed by the writer."""

        return tuple(self._warnings)

    def _bind_loop(self) -> asyncio.AbstractEventLoop:
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError("a JSONL trace sink may only be used from one event loop")
        return loop

    def _ensure_writer(self) -> None:
        if self._writer_task is None:
            self._writer_task = asyncio.create_task(
                self._writer_main(),
                name=f"penampakan-jsonl-writer-{id(self):x}",
            )

    async def _finish_close(self) -> None:
        writer = self._writer_task
        if writer is None:  # pragma: no cover - guarded by aclose
            return
        await self._queue.put(_STOP)
        await writer

    async def _writer_main(self) -> None:
        descriptor: int | None = None
        current_size = 0
        try:
            while True:
                item = await self._queue.get()
                if item is _STOP:
                    return
                line = cast(bytes, item)
                try:
                    if descriptor is None:
                        descriptor, current_size = self._open_file()
                    if current_size and current_size + len(line) > self._rotation_bytes:
                        os.close(descriptor)
                        descriptor = None
                        self._rotate_files()
                        descriptor, current_size = self._open_file()
                        self._rotations += 1
                    _write_all(descriptor, line)
                    if self._fsync:
                        os.fsync(descriptor)
                    current_size += len(line)
                    self._written_events += 1
                except Exception as error:
                    self._write_failures += 1
                    self._record_warning(
                        "trace_sink_write_failed",
                        "The JSONL trace sink could not persist one or more events.",
                        error_type=type(error).__name__,
                    )
                    if descriptor is not None:
                        with suppress(OSError):
                            os.close(descriptor)
                        descriptor = None
                        current_size = 0
                finally:
                    self._pending_events -= 1
        finally:
            if descriptor is not None:
                if self._fsync:
                    with suppress(OSError):
                        os.fsync(descriptor)
                with suppress(OSError):
                    os.close(descriptor)

    def _open_file(self) -> tuple[int, int]:
        self._reject_symlink_components()
        self._prepare_parent()
        self._reject_symlink_components()
        if self._path.is_dir():
            raise OSError("the trace path is a directory")

        existed = self._path.exists()
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        if not self._allow_symlink and hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self._path, flags, _FILE_MODE)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise OSError("the trace path is not a regular file")
            if _POSIX_PERMISSIONS:
                if existed:
                    self._warn_broad_permissions(
                        self._path,
                        _FILE_MODE,
                        "trace_sink_file_permissions",
                    )
                else:
                    # The umask can clear owner bits, so restore the intended
                    # private mode after creating the file.
                    os.fchmod(descriptor, _FILE_MODE)
            return descriptor, info.st_size
        except BaseException:
            os.close(descriptor)
            raise

    def _prepare_parent(self) -> None:
        missing = [parent for parent in self._path.parents if not parent.exists()]
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if _POSIX_PERMISSIONS:
            for created in missing:
                with suppress(OSError):
                    os.chmod(created, _DIRECTORY_MODE)
            if not missing:
                self._warn_broad_permissions(
                    self._path.parent,
                    _DIRECTORY_MODE,
                    "trace_sink_directory_permissions",
                )

    def _rotate_files(self) -> None:
        self._reject_symlink_components()
        oldest = _rotation_path(self._path, self._rotation_count)
        with suppress(FileNotFoundError):
            oldest.unlink()
        for generation in range(self._rotation_count - 1, 0, -1):
            source = _rotation_path(self._path, generation)
            if source.exists() or source.is_symlink():
                os.replace(source, _rotation_path(self._path, generation + 1))
        os.replace(self._path, _rotation_path(self._path, 1))

    def _reject_symlink_components(self) -> None:
        if self._allow_symlink:
            return
        for component in (*reversed(self._path.parents), self._path):
            if component.is_symlink():
                raise OSError("symbolic links in trace paths are disabled")

    def _warn_broad_permissions(self, target: Path, expected: int, code: str) -> None:
        try:
            mode = stat.S_IMODE(target.stat().st_mode)
        except OSError:  # pragma: no cover - removed concurrently
            return
        if mode & ~expected:
            self._record_warning(
                code,
                "An existing trace artifact is readable beyond its owner.",
                mode=format(mode, "04o"),
                expected_mode=format(expected, "04o"),
            )

    def _record_warning(self, code: str, message: str, **details: str) -> None:
        if code in self._warning_codes:
            return
        self._warning_codes.add(code)
        safe_details: dict[str, JsonValue] = {}
        for key, value in details.items():
            safe_details[key] = value
        self._warnings.append(WarningInfo(code=code, message=message, details=safe_details))


def _canonical_line(event: TraceEvent) -> bytes:
    # model_dump retains unknown model_construct fields such as a future
    # schema_version. The sink intentionally performs no version-dependent
    # interpretation; readers must decide whether to reject or preserve it.
    value = event.model_dump(mode="json")
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _write_all(descriptor: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:  # pragma: no cover - defensive OS contract check
            raise OSError("trace write made no progress")
        view = view[written:]


def _rotation_path(path: Path, generation: int) -> Path:
    return path.with_name(f"{path.name}.{generation}")


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_path_shape(path: Path) -> str | None:
    text = str(path)
    if not text or path.name == "":
        return "the path is empty"
    if "\x00" in text:
        return "the path contains a NUL byte"
    if not path.is_absolute():
        return "the path is not absolute"
    for part in path.parts:
        if part.startswith("~"):
            return "the path contains an unexpanded home reference"
        if any(character in part for character in "*?["):
            return "the path contains glob metacharacters"
        if "$" in part or (part.startswith("%") and part.endswith("%") and len(part) > 1):
            return "the path contains an unexpanded variable reference"
    return None


__all__ = ["JsonlTraceSink", "JsonlTraceSinkStats", "Overflow"]
