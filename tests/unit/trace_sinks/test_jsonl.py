import asyncio
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import pytest

from penampakan.models import TraceEvent
from penampakan.trace_sinks.jsonl import JsonlTraceSink


def event(sequence: int, *, label: str = "café") -> TraceEvent:
    return TraceEvent(
        schema_version=2,
        trace_id=UUID("00000000-0000-4000-8000-000000000001"),
        sequence=sequence,
        event_type="trace_point",
        occurred_at=datetime(2026, 8, 18, tzinfo=timezone.utc),
        parent_invocation_id="run-1",
        data={"z_label": label, "a_value": sequence},
    )


def read_lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


async def test_writes_canonical_utf8_lines_in_admission_order(tmp_path: Path) -> None:
    path = tmp_path / "private" / "trace.jsonl"
    sink = JsonlTraceSink(path)

    for sequence in range(3):
        await sink.emit(event(sequence))
    await sink.aclose()

    raw_lines = path.read_bytes().splitlines(keepends=True)
    assert all(line.endswith(b"\n") for line in raw_lines)
    assert b"caf\xc3\xa9" in raw_lines[0]
    assert raw_lines[0] == (
        json.dumps(
            event(0).model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    assert [line["sequence"] for line in read_lines(path)] == [0, 1, 2]
    stats = sink.stats()
    assert stats.accepted_events == 3
    assert stats.written_events == 3
    assert stats.queued_events == 0


async def test_drop_new_overflow_never_yields_and_records_one_warning(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    sink = JsonlTraceSink(path, queue_size=1)

    # drop_new emit contains no scheduling point, so the first line remains in
    # the bounded queue until this coroutine yields at close.
    await sink.emit(event(0))
    await sink.emit(event(1))
    await sink.emit(event(2))
    before_close = sink.stats()

    assert before_close.accepted_events == 1
    assert before_close.dropped_events == 2
    assert before_close.queued_events == 1
    assert [warning.code for warning in sink.warnings] == ["trace_sink_queue_overflow"]

    await sink.aclose()
    assert [line["sequence"] for line in read_lines(path)] == [0]
    assert sink.stats().warning_count == 1


async def test_block_overflow_applies_backpressure_without_loss(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    sink = JsonlTraceSink(path, queue_size=1, overflow="block")

    await asyncio.gather(*(sink.emit(event(sequence)) for sequence in range(20)))
    await sink.aclose()

    stats = sink.stats()
    assert stats.accepted_events == 20
    assert stats.written_events == 20
    assert stats.dropped_events == 0
    assert sorted(line["sequence"] for line in read_lines(path)) == list(range(20))


async def test_rotation_keeps_only_the_bounded_numbered_set(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    line_size = len(
        json.dumps(
            event(0).model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    sink = JsonlTraceSink(path, rotation_bytes=line_size + 1, rotation_count=2)

    for sequence in range(4):
        await sink.emit(event(sequence))
    await sink.aclose()

    assert [line["sequence"] for line in read_lines(path)] == [3]
    assert [line["sequence"] for line in read_lines(path.with_name("trace.jsonl.1"))] == [2]
    assert [line["sequence"] for line in read_lines(path.with_name("trace.jsonl.2"))] == [1]
    assert not path.with_name("trace.jsonl.3").exists()
    assert sink.stats().rotations == 3


async def test_close_is_idempotent_drains_and_counts_post_close(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    sink = JsonlTraceSink(path, queue_size=2, overflow="block")
    emissions = [asyncio.create_task(sink.emit(event(sequence))) for sequence in range(8)]
    await asyncio.gather(*emissions)

    await asyncio.gather(sink.aclose(), sink.aclose())
    await sink.aclose()
    await sink.emit(event(99))

    assert sink.closed
    assert [line["sequence"] for line in read_lines(path)] == list(range(8))
    stats = sink.stats()
    assert stats.written_events == 8
    assert stats.post_close_emits == 1
    assert stats.post_close_events == 1


async def test_close_without_events_does_not_create_a_file(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    sink = JsonlTraceSink(path)

    await sink.aclose()
    await sink.aclose()

    assert not path.exists()
    assert sink.stats().written_events == 0


async def test_future_schema_is_preserved_opaquely(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    sink = JsonlTraceSink(path)
    future = TraceEvent.model_construct(
        schema_version=3,
        trace_id=UUID("00000000-0000-4000-8000-000000000001"),
        sequence=0,
        event_type="future_event",
        occurred_at=datetime(2026, 8, 18, tzinfo=timezone.utc),
        duration_ms=None,
        invocation_id=None,
        parent_invocation_id=None,
        data={"future_field": "opaque"},
    )

    await sink.emit(future)
    await sink.aclose()

    assert read_lines(path)[0]["schema_version"] == 3


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX permission bits")
async def test_created_parent_and_file_have_private_permissions(tmp_path: Path) -> None:
    path = tmp_path / "private" / "traces.jsonl"
    sink = JsonlTraceSink(path)

    await sink.emit(event(0))
    await sink.aclose()

    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert sink.warnings == ()


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX permission bits")
async def test_broad_existing_permissions_are_reported_without_path_disclosure(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "shared"
    directory.mkdir(mode=0o755)
    os.chmod(directory, 0o755)
    path = directory / "traces.jsonl"
    path.touch(mode=0o644)
    os.chmod(path, 0o644)
    sink = JsonlTraceSink(path)

    await sink.emit(event(0))
    await sink.aclose()

    serialized = "".join(warning.model_dump_json() for warning in sink.warnings)
    assert {warning.code for warning in sink.warnings} == {
        "trace_sink_directory_permissions",
        "trace_sink_file_permissions",
    }
    assert str(path) not in serialized


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="requires symbolic links")
async def test_symlink_is_refused_unless_explicitly_allowed(tmp_path: Path) -> None:
    target = tmp_path / "target.jsonl"
    target.write_text("existing\n", encoding="utf-8")
    link = tmp_path / "link.jsonl"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):  # pragma: no cover - unsupported platform
        pytest.skip("symbolic links are unavailable")

    refused = JsonlTraceSink(link)
    await refused.emit(event(0))
    await refused.aclose()

    assert target.read_text(encoding="utf-8") == "existing\n"
    assert refused.stats().write_failures == 1
    assert [warning.code for warning in refused.warnings] == ["trace_sink_write_failed"]

    allowed = JsonlTraceSink(link, allow_symlink=True)
    await allowed.emit(event(1))
    await allowed.aclose()
    persisted = target.read_text(encoding="utf-8").splitlines()
    assert json.loads(persisted[-1])["sequence"] == 1


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="requires symbolic links")
async def test_symlinked_parent_directory_is_refused_unless_allowed(tmp_path: Path) -> None:
    target_directory = tmp_path / "target"
    target_directory.mkdir()
    linked_directory = tmp_path / "linked"
    try:
        linked_directory.symlink_to(target_directory, target_is_directory=True)
    except (OSError, NotImplementedError):  # pragma: no cover - unsupported platform
        pytest.skip("symbolic links are unavailable")
    path = linked_directory / "trace.jsonl"

    refused = JsonlTraceSink(path)
    await refused.emit(event(0))
    await refused.aclose()

    target = target_directory / "trace.jsonl"
    assert not target.exists()
    assert refused.stats().write_failures == 1

    allowed = JsonlTraceSink(path, allow_symlink=True)
    await allowed.emit(event(1))
    await allowed.aclose()
    assert read_lines(target)[0]["sequence"] == 1


@pytest.mark.parametrize(
    ("argument", "value", "error"),
    [
        ("rotation_bytes", 0, ValueError),
        ("rotation_count", True, TypeError),
        ("queue_size", -1, ValueError),
        ("fsync", 1, TypeError),
        ("overflow", "discard", ValueError),
        ("allow_symlink", "yes", TypeError),
    ],
)
def test_configuration_is_validated(
    tmp_path: Path,
    argument: str,
    value: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        JsonlTraceSink(tmp_path / "trace.jsonl", **{argument: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "candidate",
    [
        "relative/trace.jsonl",
        "~/trace.jsonl",
        "/tmp/$TRACE_HOME/trace.jsonl",
        "/tmp/penampakan-*/trace.jsonl",
    ],
)
def test_unsafe_path_shapes_are_rejected(candidate: str) -> None:
    with pytest.raises(ValueError, match="invalid trace path"):
        JsonlTraceSink(candidate)
