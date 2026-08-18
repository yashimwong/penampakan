"""Import-performance regression test for specification 05 D4.

D4 forbids a fixed wall-clock assertion such as "under 1.5 seconds": that number
is a property of the machine, not of the public contract. This module therefore
uses a *self-calibrating* envelope.

Calibration
-----------
Every measurement is the median of several fresh subprocesses that time
``import penampakan`` with ``time.perf_counter`` inside the child (a warmup run
is discarded, so the operating-system page cache is warm for every counted
sample). Interpreter startup is outside the timed region for both measurements.

In the same job, and interleaved with nothing else, the test measures a control
statement that imports only stdlib modules of comparable weight
(``import json, asyncio``). The assertion is on the *ratio*

    median(import penampakan) / median(import json, asyncio)

against the ratio recorded in ``import_baseline.json``, multiplied by a generous
``relative_tolerance``. A slower or noisier CI runner inflates both numbers, so
the ratio stays roughly constant while an actual regression — a provider SDK, a
Torch import, eager weight loading — moves it by an order of magnitude. Torch
alone costs seconds and would push the ratio past 50.

Because CI runners are noisy, a single unlucky median must not fail the build:
the ratio check is retried a few times and only a run where *every* attempt
exceeds the envelope fails. The absolute ``emergency_ceiling_s`` is an
independent backstop for a catastrophic regression, deliberately set several
times larger than any plausible healthy measurement on slow hardware.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

BASELINE_PATH = Path(__file__).with_name("import_baseline.json")

TIMER = (
    "import time; _start = time.perf_counter(); {statement}; print(time.perf_counter() - _start)"
)


def _median_seconds(statement: str, *, samples: int, warmups: int) -> float:
    timings: list[float] = []
    for index in range(samples + warmups):
        completed = subprocess.run(
            [sys.executable, "-c", TIMER.format(statement=statement)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        if index >= warmups:
            timings.append(float(completed.stdout.strip()))
    return statistics.median(timings)


def _measure(baseline: dict[str, Any]) -> tuple[float, float]:
    samples = int(baseline["samples"])
    warmups = int(baseline["warmup_samples"])
    package = _median_seconds("import penampakan", samples=samples, warmups=warmups)
    control = _median_seconds(str(baseline["control_statement"]), samples=samples, warmups=warmups)
    return package, control


def test_the_baseline_envelope_is_documented() -> None:
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    for key in (
        "description",
        "measurement",
        "control_statement",
        "samples",
        "warmup_samples",
        "retry_attempts",
        "baseline_ratio",
        "relative_tolerance",
        "emergency_ceiling_s",
        "recorded_on",
    ):
        assert key in baseline, key
    assert baseline["samples"] >= 5
    assert baseline["relative_tolerance"] >= 1.0
    assert baseline["emergency_ceiling_s"] >= 1.0
    for key in ("python_version", "platform", "penampakan_median_s", "control_median_s"):
        assert key in baseline["recorded_on"], key
    # The recorded numbers are machine-relative documentation, never the
    # assertion: only the ratio and the emergency ceiling are enforced.
    assert "machine-relative" in baseline["description"]


def test_import_time_stays_inside_the_calibrated_envelope() -> None:
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    allowed_ratio = float(baseline["baseline_ratio"]) * float(baseline["relative_tolerance"])
    ceiling = float(baseline["emergency_ceiling_s"])
    attempts = int(baseline["retry_attempts"])

    observations: list[tuple[float, float]] = []
    for _ in range(attempts):
        package, control = _measure(baseline)
        observations.append((package, control))
        if package / control <= allowed_ratio:
            break

    best_package = min(package for package, _ in observations)
    best_ratio = min(package / control for package, control in observations)
    report = ", ".join(
        f"penampakan={package * 1000:.1f}ms control={control * 1000:.1f}ms "
        f"ratio={package / control:.2f}"
        for package, control in observations
    )

    assert best_package <= ceiling, (
        f"import penampakan took {best_package:.3f}s, above the emergency ceiling "
        f"of {ceiling:.3f}s ({report})"
    )
    assert best_ratio <= allowed_ratio, (
        f"import penampakan is {best_ratio:.2f}x the control import, above the calibrated "
        f"envelope of {allowed_ratio:.2f}x "
        f"(baseline {baseline['baseline_ratio']} x tolerance {baseline['relative_tolerance']}); "
        f"observations: {report}"
    )
