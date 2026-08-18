"""Cross-process behavior of the durable SQLite perception cache.

Specification 07 D8 asks for a real subprocess writer/reader hit test and for
two-process contention with bounded lock handling, and acceptance criterion 7
requires that a second process obtains an attributed cache hit while any key
dimension or unresolved weight identity produces a miss or a bypass. None of
that can be shown with threads: SQLite's cross-process guarantees are exactly
what is under test, so every process here is a separate interpreter started
from :data:`sys.executable`, reports its own ``os.getpid()`` so the test can
prove it, and is given a hard timeout so a hang fails instead of stalling CI.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import textwrap
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from penampakan.perception.cache import CACHE_SCHEMA_VERSION
from tests.fixtures.images import encode_image, quadrants_image

pytestmark = [
    pytest.mark.integration,
    pytest.mark.integration_category("cache"),
]

_ROOT = Path(__file__).parents[2]

# Every child does a handful of small SQLite operations, so anything close to
# this bound is a hang rather than a slow machine.
_CHILD_TIMEOUT_S = 60.0

# The deadline a contending child holds itself to. It is far above the work it
# performs and far below the hard subprocess timeout, so an unbounded retry
# loop is reported by the child before the parent has to kill it.
_CONTENTION_DEADLINE_S = 20.0
_CONTENTION_ROUNDS = 30

_VALUE = b'{"observations":[{"caption":"a red square"}]}'
_REVISION = "b" * 40

_BASELINE: dict[str, Any] = {
    "asset_digest_sha256": "a" * 64,
    "request": {"focus": "receipt total"},
    "backend": {
        "name": "tests.cache_probe",
        "version": "1.0",
        "model_id": "org/caption",
        "model_revision": _REVISION,
    },
    "preprocessing_version": "preprocessing-1",
    "schema_version": CACHE_SCHEMA_VERSION,
}

# One differing value per key dimension. Each variation changes exactly one
# dimension, so a miss it produces cannot be blamed on anything else.
_VARIATIONS: dict[str, dict[str, Any]] = {
    "asset_digest": {"asset_digest_sha256": "c" * 64},
    "request": {"request": {"focus": "signature block"}},
    "backend_name": {"backend": {"name": "tests.other_probe"}},
    "backend_version": {"backend": {"version": "2.0"}},
    "model_id": {"backend": {"model_id": "org/other-caption"}},
    "model_revision": {"backend": {"model_revision": "d" * 40}},
    "preprocessing_version": {"preprocessing_version": "preprocessing-2"},
    "schema_version": {"schema_version": "perception-cache-v2"},
}

_KEY_HELPERS = """
import asyncio
import json
import os
import sys

from penampakan.models import BackendDescriptor, Capability, CapabilityDescriptor, CaptionRequest
from penampakan.perception.cache import build_perception_cache_key
from penampakan.perception.sqlite_cache import SQLiteCache


def cache_key(spec):
    backend = spec["backend"]
    return build_perception_cache_key(
        asset_digest_sha256=spec["asset_digest_sha256"],
        request=CaptionRequest(**spec["request"]),
        backend=BackendDescriptor(
            name=backend["name"],
            version=backend["version"],
            model_id=backend["model_id"],
            model_revision=backend["model_revision"],
            capabilities=(CapabilityDescriptor(capability=Capability.CAPTION),),
        ),
        preprocessing_version=spec["preprocessing_version"],
        schema_version=spec["schema_version"],
    )
"""

_WRITER = (
    _KEY_HELPERS
    + """
async def main():
    request = json.loads(sys.argv[1])
    key = cache_key(request["spec"])
    value = request["value"].encode("utf-8")
    cache = SQLiteCache(request["path"])
    try:
        await cache.set(key, value, size=len(value))
        available = cache.available
    finally:
        await cache.aclose()
    print(json.dumps({"pid": os.getpid(), "key": key, "available": available}))


asyncio.run(main())
"""
)

_READER = (
    _KEY_HELPERS
    + """
async def main():
    request = json.loads(sys.argv[1])
    keys = {}
    values = {}
    cache = SQLiteCache(request["path"])
    try:
        for name, spec in request["specs"].items():
            keys[name] = cache_key(spec)
            stored = await cache.get(keys[name])
            values[name] = None if stored is None else stored.decode("utf-8")
        available = cache.available
    finally:
        await cache.aclose()
    print(json.dumps({"pid": os.getpid(), "keys": keys, "values": values, "available": available}))


asyncio.run(main())
"""
)

_SESSION = """
import asyncio
import json
import os
import sys
from pathlib import Path

from penampakan.backends.callable import CallableVisionBackend
from penampakan.client import AsyncPenampakan
from penampakan.config import AgentSettings, CacheSettings, RunLimits, Settings
from penampakan.models import (
    BackendDescriptor,
    Capability,
    CapabilityDescriptor,
    CaptionPayload,
    CaptionRequest,
    InspectionOperation,
    InspectionPlan,
    ObservationDraft,
    VisionResult,
)


async def main():
    request = json.loads(sys.argv[1])
    calls = 0

    async def analyze(image, vision_request):
        nonlocal calls
        calls += 1
        return VisionResult(
            observations=(
                ObservationDraft(
                    payload=CaptionPayload(text=request["caption"], focus=vision_request.focus)
                ),
            )
        )

    backend = CallableVisionBackend(
        BackendDescriptor(
            name="tests.cache_probe",
            version="1.0",
            model_id="org/caption",
            model_revision=request["model_revision"],
            capabilities=(CapabilityDescriptor(capability=Capability.CAPTION),),
        ),
        analyze,
    )
    settings = Settings(
        run=RunLimits(),
        cache=CacheSettings(mode="sqlite", path=Path(request["path"])),
        agent=AgentSettings(initial_capabilities=()),
    )
    plan = InspectionPlan(
        operations=(InspectionOperation(request=CaptionRequest()),),
        include_available_overview=False,
    )
    async with AsyncPenampakan(backends=(backend,), settings=settings) as client:
        result = await client.inspect(Path(request["image"]).read_bytes(), plan)
    observation = result.observations[0]
    print(
        json.dumps(
            {
                "pid": os.getpid(),
                "backend_calls": calls,
                "text": observation.payload.text,
                "cache_hit": observation.provenance.cache_hit,
                "backend_name": observation.provenance.backend_name,
                "model_revision": observation.provenance.model_revision,
            }
        )
    )


asyncio.run(main())
"""

_CONTENDER = """
import asyncio
import json
import os
import sys
import time

from penampakan.perception.sqlite_cache import SQLiteCache


async def main():
    request = json.loads(sys.argv[1])
    label = request["label"]
    started = time.monotonic()
    cache = SQLiteCache(request["path"], busy_timeout_s=request["busy_timeout_s"])
    unreadable = []
    try:
        for index in range(request["rounds"]):
            key = label + "-" + str(index)
            value = json.dumps({"label": label, "index": index}, separators=(",", ":"))
            encoded = value.encode("utf-8")
            await cache.set(key, encoded, size=len(encoded))
            if await cache.get(key) != encoded:
                unreadable.append(key)
        available = cache.available
        warnings = sorted({warning.code for warning in cache.warnings})
    finally:
        await cache.aclose()
    print(
        json.dumps(
            {
                "pid": os.getpid(),
                "elapsed_s": time.monotonic() - started,
                "available": available,
                "unreadable": unreadable,
                "warnings": warnings,
            }
        )
    )


asyncio.run(main())
"""


def _environment() -> dict[str, str]:
    """Return a child environment that can import this working tree's package."""

    environment = dict(os.environ)
    source = str(_ROOT / "src")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source if not existing else source + os.pathsep + existing
    return environment


def _start(script: str, request: dict[str, Any]) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(script), json.dumps(request)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=_environment(),
        cwd=str(_ROOT),
    )


def _finish(process: subprocess.Popen[str]) -> dict[str, Any]:
    try:
        stdout, stderr = process.communicate(timeout=_CHILD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        message = f"a cache subprocess did not finish within {_CHILD_TIMEOUT_S}s"
        raise AssertionError(message) from None
    assert process.returncode == 0, stderr
    reported: dict[str, Any] = json.loads(stdout)
    # A child that reported the parent's identity would not be proving anything
    # about cross-process behavior.
    assert reported["pid"] != os.getpid()
    return reported


def _run(script: str, request: dict[str, Any]) -> dict[str, Any]:
    return _finish(_start(script, request))


def _varied(dimension: str) -> dict[str, Any]:
    """Return the baseline specification with exactly one dimension changed."""

    spec = deepcopy(_BASELINE)
    for field, value in _VARIATIONS[dimension].items():
        if field == "backend":
            spec["backend"].update(value)
        else:
            spec[field] = value
    return spec


def _entry_count(path: Path) -> int:
    """Return the number of rows a cache database retains, read from outside it."""

    connection = sqlite3.connect(str(path))
    try:
        return int(connection.execute("SELECT COUNT(*) FROM entries").fetchone()[0])
    finally:
        connection.close()


@pytest.fixture(scope="module")
def cross_process_probe(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Write one entry in one process and read every key variation in another."""

    path = tmp_path_factory.mktemp("durable-cache") / "cache" / "perception.db"
    writer = _run(
        _WRITER,
        {"path": str(path), "value": _VALUE.decode("utf-8"), "spec": _BASELINE},
    )
    specs = {"baseline": _BASELINE, **{name: _varied(name) for name in _VARIATIONS}}
    reader = _run(_READER, {"path": str(path), "specs": specs})
    return {"path": path, "writer": writer, "reader": reader}


def test_a_second_process_reads_back_the_entry_the_writer_stored(
    cross_process_probe: dict[str, Any],
) -> None:
    writer = cross_process_probe["writer"]
    reader = cross_process_probe["reader"]

    assert writer["available"] and reader["available"]
    assert writer["pid"] != reader["pid"]
    assert _entry_count(cross_process_probe["path"]) == 1
    assert reader["values"]["baseline"] == _VALUE.decode("utf-8")


def test_a_second_process_reproduces_the_key_from_the_same_dimensions(
    cross_process_probe: dict[str, Any],
) -> None:
    # The hit above is attributed only because the reader rebuilt the same key
    # from the same asset digest, request, descriptor, and preprocessing
    # version rather than being handed the writer's digest.
    assert cross_process_probe["reader"]["keys"]["baseline"] == cross_process_probe["writer"]["key"]


@pytest.mark.parametrize("dimension", sorted(_VARIATIONS))
def test_a_varied_key_dimension_misses_in_the_second_process(
    cross_process_probe: dict[str, Any],
    dimension: str,
) -> None:
    reader = cross_process_probe["reader"]

    assert reader["keys"][dimension] != cross_process_probe["writer"]["key"]
    assert reader["values"][dimension] is None
    # The same reader process, the same database, and the same run produced the
    # baseline hit, so the miss is caused by this dimension and nothing else.
    assert reader["values"]["baseline"] == _VALUE.decode("utf-8")


def test_a_second_process_serves_an_attributed_session_hit(tmp_path: Path) -> None:
    path = tmp_path / "cache" / "perception.db"
    image = tmp_path / "image.png"
    image.write_bytes(_image_bytes())
    request = {"path": str(path), "image": str(image), "model_revision": _REVISION}

    first = _run(_SESSION, {**request, "caption": "the first process caption"})
    second = _run(_SESSION, {**request, "caption": "the second process caption"})

    assert first["pid"] != second["pid"]
    assert (first["cache_hit"], first["backend_calls"]) == (False, 1)
    # A miss in the second process would have returned its own caption, so the
    # retained text is what proves the hit crossed the process boundary.
    assert (second["cache_hit"], second["backend_calls"]) == (True, 0)
    assert second["text"] == "the first process caption"
    assert second["backend_name"] == "tests.cache_probe"
    assert second["model_revision"] == _REVISION
    assert _entry_count(path) == 1


def test_an_unresolved_weight_identity_bypasses_the_durable_cache(tmp_path: Path) -> None:
    path = tmp_path / "cache" / "perception.db"
    image = tmp_path / "image.png"
    image.write_bytes(_image_bytes())

    bypassed = _run(
        _SESSION,
        {
            "path": str(path),
            "image": str(image),
            "model_revision": None,
            "caption": "an unattributable caption",
        },
    )

    assert (bypassed["cache_hit"], bypassed["backend_calls"]) == (False, 1)
    assert bypassed["model_revision"] is None
    # Counting rows only works because the database was opened and its schema
    # created, so an empty table is a bypass rather than a disabled cache. The
    # resolved-revision run above retains one row from the identical setup.
    assert _entry_count(path) == 0


def test_two_processes_read_and_write_within_their_deadline(tmp_path: Path) -> None:
    path = tmp_path / "cache" / "perception.db"
    request = {
        "path": str(path),
        "rounds": _CONTENTION_ROUNDS,
        "busy_timeout_s": 5.0,
    }
    processes = [_start(_CONTENDER, {**request, "label": label}) for label in ("first", "second")]

    reports = [_finish(process) for process in processes]

    assert reports[0]["pid"] != reports[1]["pid"]
    for report in reports:
        assert report["available"]
        assert report["elapsed_s"] < _CONTENTION_DEADLINE_S
        # Contention is bounded and observable: it degrades to a recorded
        # warning, never to an exception that reaches the caller.
        assert set(report["warnings"]) <= {"cache_unavailable", "cache_operation_failed"}
        if not report["warnings"]:
            assert report["unreadable"] == []
    assert _entry_count(path) == 2 * _CONTENTION_ROUNDS


def _image_bytes() -> bytes:
    image = quadrants_image(16)
    try:
        return encode_image(image)
    finally:
        image.close()
