#!/usr/bin/env python3
"""Compare end-to-end metadata inspection latency across image libraries."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from penampakan import (
    InspectionOperation,
    InspectionPlan,
    MetadataPayload,
    MetadataRequest,
    Penampakan,
    __version__,
)

Metadata = tuple[int, int, bool]
Run = Callable[[], Metadata]


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One installed library and its equivalent metadata operation."""

    name: str
    version: str
    run: Run


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Aggregate per-call timing for one benchmark case."""

    name: str
    version: str
    median_ms: float
    min_ms: float
    max_ms: float
    calls_per_second: float
    relative_to_fastest: float


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=_positive_int, default=640)
    parser.add_argument("--height", type=_positive_int, default=480)
    parser.add_argument("--warmups", type=_non_negative_int, default=3)
    parser.add_argument("--iterations", type=_positive_int, default=20)
    parser.add_argument("--rounds", type=_positive_int, default=5)
    parser.add_argument("--format", choices=("table", "json"), default="table")
    parser.add_argument(
        "--plot",
        type=Path,
        help="write a Matplotlib latency chart to this path (for example, benchmark.png)",
    )
    return parser.parse_args(argv)


def _fixture(width: int, height: int) -> bytes:
    """Build one deterministic, non-uniform RGBA PNG outside timed regions."""

    pixels = bytearray(width * height * 4)
    offset = 0
    for y in range(height):
        for x in range(width):
            pixels[offset] = (x * 17 + y * 3) % 256
            pixels[offset + 1] = (x * 5 + y * 11) % 256
            pixels[offset + 2] = (x * 13 + y * 7) % 256
            pixels[offset + 3] = 160 if (x + y) % 17 == 0 else 255
            offset += 4

    image = Image.frombytes("RGBA", (width, height), bytes(pixels))
    output = BytesIO()
    try:
        image.save(output, format="PNG", compress_level=6)
        return output.getvalue()
    finally:
        output.close()
        image.close()


def _distribution_version(name: str, fallback: str = "unknown") -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return fallback


def _pillow_case(payload: bytes) -> BenchmarkCase:
    def inspect() -> Metadata:
        with Image.open(BytesIO(payload)) as image:
            image.load()
            has_alpha = image.mode in {"RGBA", "LA"} or "transparency" in image.info
            return image.width, image.height, has_alpha

    return BenchmarkCase("Pillow (direct)", _distribution_version("Pillow"), inspect)


def _opencv_case(payload: bytes) -> BenchmarkCase:
    import cv2
    import numpy as np

    def inspect() -> Metadata:
        encoded = np.frombuffer(payload, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError("OpenCV could not decode the fixture")
        channels = 1 if image.ndim == 2 else image.shape[2]
        return int(image.shape[1]), int(image.shape[0]), channels == 4

    return BenchmarkCase("OpenCV", str(cv2.__version__), inspect)


def _imageio_case(payload: bytes) -> BenchmarkCase:
    import imageio.v3 as imageio

    def inspect() -> Metadata:
        image = imageio.imread(BytesIO(payload), extension=".png")
        channels = 1 if image.ndim == 2 else image.shape[2]
        return int(image.shape[1]), int(image.shape[0]), channels == 4

    return BenchmarkCase("ImageIO", _distribution_version("imageio"), inspect)


def _optional_case(
    name: str,
    factory: Callable[[], BenchmarkCase],
) -> tuple[BenchmarkCase | None, str | None]:
    try:
        return factory(), None
    except (ImportError, ModuleNotFoundError) as error:
        dependency = error.name or name
        return None, f"{name}: missing dependency {dependency!r}"


def _penampakan_case(payload: bytes, vision: Penampakan) -> BenchmarkCase:
    plan = InspectionPlan(
        operations=(InspectionOperation(request=MetadataRequest()),),
        include_available_overview=False,
    )

    def inspect() -> Metadata:
        result = vision.inspect(payload, plan)
        if len(result.observations) != 1:
            raise RuntimeError("Penampakan returned an unexpected observation count")
        metadata = result.observations[0].payload
        if not isinstance(metadata, MetadataPayload):
            raise RuntimeError("Penampakan did not return metadata")
        return metadata.width, metadata.height, metadata.has_alpha

    return BenchmarkCase("Penampakan", __version__, inspect)


def _cases(
    payload: bytes,
    vision: Penampakan,
) -> tuple[list[BenchmarkCase], list[str]]:
    cases = [_penampakan_case(payload, vision), _pillow_case(payload)]
    skipped: list[str] = []
    for name, factory in (
        ("OpenCV", lambda: _opencv_case(payload)),
        ("ImageIO", lambda: _imageio_case(payload)),
    ):
        case, reason = _optional_case(name, factory)
        if case is not None:
            cases.append(case)
        if reason is not None:
            skipped.append(reason)
    return cases, skipped


def _measure(case: BenchmarkCase, iterations: int) -> float:
    gc_was_enabled = gc.isenabled()
    gc.collect()
    if gc_was_enabled:
        gc.disable()
    try:
        started = time.perf_counter_ns()
        for _ in range(iterations):
            case.run()
        elapsed_ns = time.perf_counter_ns() - started
    finally:
        if gc_was_enabled:
            gc.enable()
    return elapsed_ns / iterations / 1_000_000


def _run(
    cases: Sequence[BenchmarkCase],
    *,
    expected: Metadata,
    warmups: int,
    iterations: int,
    rounds: int,
) -> list[BenchmarkResult]:
    samples: dict[str, list[float]] = {case.name: [] for case in cases}

    for case in cases:
        actual = case.run()
        if actual != expected:
            raise RuntimeError(f"{case.name} returned {actual!r}; expected {expected!r}")
        for _ in range(warmups):
            case.run()

    for round_index in range(rounds):
        rotation = round_index % len(cases)
        ordered = (*cases[rotation:], *cases[:rotation])
        for case in ordered:
            samples[case.name].append(_measure(case, iterations))

    medians = {name: statistics.median(values) for name, values in samples.items()}
    fastest = min(medians.values())
    return [
        BenchmarkResult(
            name=case.name,
            version=case.version,
            median_ms=medians[case.name],
            min_ms=min(samples[case.name]),
            max_ms=max(samples[case.name]),
            calls_per_second=1_000 / medians[case.name],
            relative_to_fastest=medians[case.name] / fastest,
        )
        for case in cases
    ]


def _payload(
    args: argparse.Namespace,
    fixture_bytes: int,
    results: Sequence[BenchmarkResult],
    skipped: Sequence[str],
) -> dict[str, Any]:
    return {
        "benchmark": "decode PNG bytes and return width, height, and alpha metadata",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "fixture": {
            "width": args.width,
            "height": args.height,
            "mode": "RGBA",
            "encoded_bytes": fixture_bytes,
        },
        "settings": {
            "warmups": args.warmups,
            "iterations": args.iterations,
            "rounds": args.rounds,
        },
        "results": [asdict(result) for result in results],
        "skipped": list(skipped),
    }


def _table(data: dict[str, Any]) -> str:
    fixture = data["fixture"]
    settings = data["settings"]
    results = data["results"]
    lines = [
        "Metadata inspection benchmark",
        (f"Python {data['environment']['python']} on {data['environment']['platform']}"),
        (
            f"Fixture: {fixture['width']}x{fixture['height']} {fixture['mode']} PNG "
            f"({fixture['encoded_bytes']:,} bytes)"
        ),
        (
            f"Timing: {settings['warmups']} warmups, {settings['iterations']} iterations "
            f"x {settings['rounds']} rounds"
        ),
        "",
    ]
    headers = ("Library", "Version", "Median ms", "Min ms", "Max ms", "Calls/s", "vs fastest")
    rows = [
        (
            result["name"],
            result["version"],
            f"{result['median_ms']:.3f}",
            f"{result['min_ms']:.3f}",
            f"{result['max_ms']:.3f}",
            f"{result['calls_per_second']:.1f}",
            f"{result['relative_to_fastest']:.2f}x",
        )
        for result in results
    ]
    widths = [
        max(len(str(row[index])) for row in (headers, *rows)) for index in range(len(headers))
    ]
    lines.append("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    lines.append("  ".join("-" * width for width in widths))
    lines.extend(
        "  ".join(str(value).ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    )
    if data["skipped"]:
        lines.extend(("", "Skipped:"))
        lines.extend(f"- {reason}" for reason in data["skipped"])
    lines.extend(
        (
            "",
            "Penampakan includes normalization, canonical encoding, hashing, validation,",
            "routing, tracing, and cleanup; direct alternatives only decode metadata.",
        )
    )
    return "\n".join(lines)


def _plot(data: dict[str, Any], output: Path) -> None:
    """Render median latency and the observed min-max range on a log scale."""

    try:
        import matplotlib
    except ImportError as error:
        raise RuntimeError(
            "plotting requires the benchmark extra: pip install -e '.[benchmark]'"
        ) from error

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    results = data["results"]
    fixture = data["fixture"]
    settings = data["settings"]
    names = [result["name"] for result in results]
    medians = [result["median_ms"] for result in results]
    minimums = [result["min_ms"] for result in results]
    maximums = [result["max_ms"] for result in results]
    positions = list(range(len(results)))
    colors = ["#d1495b" if name == "Penampakan" else "#3977a8" for name in names]

    with plt.rc_context(
        {
            "axes.edgecolor": "#d0d7de",
            "axes.labelcolor": "#24292f",
            "font.size": 11,
            "text.color": "#24292f",
            "xtick.color": "#57606a",
            "ytick.color": "#24292f",
        }
    ):
        figure, axis = plt.subplots(figsize=(9.2, 5.2))
        figure.patch.set_facecolor("white")
        axis.set_facecolor("white")

        for position, median, minimum, maximum, color in zip(
            positions, medians, minimums, maximums, colors, strict=True
        ):
            axis.hlines(position, minimum, maximum, color=color, linewidth=4, alpha=0.55)
            axis.scatter(median, position, color=color, edgecolor="white", s=110, zorder=3)
            axis.annotate(
                f"{median:.3f} ms",
                (median, position),
                xytext=(9, 0),
                textcoords="offset points",
                va="center",
                fontweight="bold",
            )

        axis.set_xscale("log")
        axis.set_xlim(min(minimums) * 0.65, max(maximums) * 1.8)
        axis.set_yticks(positions, names)
        axis.invert_yaxis()
        axis.set_xlabel("Median latency per inspection (milliseconds, log scale)")
        axis.set_title("Metadata inspection latency — lower is better", loc="left", pad=20)
        axis.grid(axis="x", which="both", color="#d8dee4", linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0, pad=10)

        figure.text(
            0.125,
            0.015,
            (
                f"{fixture['width']}x{fixture['height']} RGBA PNG; "
                f"{settings['iterations']} iterations x {settings['rounds']} rounds. "
                "Lines show the observed min-max range."
            ),
            color="#57606a",
            fontsize=9,
        )
        figure.tight_layout(rect=(0, 0.06, 1, 1))
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    encoded = _fixture(args.width, args.height)
    expected = (args.width, args.height, True)

    with Penampakan() as vision:
        cases, skipped = _cases(encoded, vision)
        results = _run(
            cases,
            expected=expected,
            warmups=args.warmups,
            iterations=args.iterations,
            rounds=args.rounds,
        )

    data = _payload(args, len(encoded), results, skipped)
    if args.plot is not None:
        _plot(data, args.plot)
    if args.format == "json":
        json.dump(data, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(_table(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
