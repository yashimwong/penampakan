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
    ImageLimitExceededError,
    ImageLimits,
    InspectionOperation,
    InspectionPlan,
    InspectionResult,
    InvalidImageError,
    MetadataPayload,
    MetadataRequest,
    Penampakan,
    RemoteSourceDisabledError,
    Settings,
    UnsupportedImageError,
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


@dataclass(frozen=True, slots=True)
class ReusableSessionResult:
    """Latency breakdown for repeated inspections on one normalized image."""

    calls_per_session: int
    median_amortized_ms: float
    min_amortized_ms: float
    max_amortized_ms: float
    median_open_ms: float
    median_warm_inspection_ms: float
    median_close_ms: float
    speedup_vs_penampakan_one_shot: float


@dataclass(frozen=True, slots=True)
class ContractCheck:
    """One executed behavior guaranteed by the Penampakan input contract."""

    name: str
    passed: bool
    detail: str


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
    parser.add_argument(
        "--reuse-count",
        type=_positive_int,
        default=20,
        help="metadata inspections to amortize over each reusable image session",
    )
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


def _encode_image(image: Image.Image, format_name: str, **options: Any) -> bytes:
    output = BytesIO()
    try:
        image.save(output, format=format_name, **options)
        return output.getvalue()
    finally:
        output.close()


def _oriented_jpeg() -> bytes:
    image = Image.new("RGB", (3, 2), "black")
    try:
        exif = Image.Exif()
        exif[274] = 6
        return _encode_image(image, "JPEG", exif=exif, quality=100, subsampling=0)
    finally:
        image.close()


def _animated_webp() -> bytes:
    first = Image.new("RGB", (2, 2), "red")
    second = Image.new("RGB", (2, 2), "blue")
    try:
        return _encode_image(
            first,
            "WEBP",
            save_all=True,
            append_images=[second],
            duration=100,
            loop=0,
            lossless=True,
        )
    finally:
        first.close()
        second.close()


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
    plan = _metadata_plan()

    def inspect() -> Metadata:
        result = vision.inspect(payload, plan)
        return _metadata_from_result(result)

    return BenchmarkCase("Penampakan", __version__, inspect)


def _metadata_plan() -> InspectionPlan:
    return InspectionPlan(
        operations=(InspectionOperation(request=MetadataRequest()),),
        include_available_overview=False,
    )


def _metadata_from_result(result: InspectionResult) -> Metadata:
    if len(result.observations) != 1:
        raise RuntimeError("Penampakan returned an unexpected observation count")
    metadata = result.observations[0].payload
    if not isinstance(metadata, MetadataPayload):
        raise RuntimeError("Penampakan did not return metadata")
    return metadata.width, metadata.height, metadata.has_alpha


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


def _run_reusable_session(
    payload: bytes,
    vision: Penampakan,
    *,
    expected: Metadata,
    warmups: int,
    calls_per_session: int,
    rounds: int,
    penampakan_one_shot_ms: float,
) -> ReusableSessionResult:
    """Measure a complete open-many-inspect-close application workflow."""

    plan = _metadata_plan()
    for _ in range(warmups):
        with vision.open_image(payload) as session:
            actual = _metadata_from_result(session.inspect(plan))
            if actual != expected:
                raise RuntimeError(
                    f"Penampakan reusable session returned {actual!r}; expected {expected!r}"
                )

    amortized_samples: list[float] = []
    open_samples: list[float] = []
    inspection_samples: list[float] = []
    close_samples: list[float] = []
    for _ in range(rounds):
        gc_was_enabled = gc.isenabled()
        gc.collect()
        if gc_was_enabled:
            gc.disable()
        try:
            workflow_started = time.perf_counter_ns()
            open_started = time.perf_counter_ns()
            session = vision.open_image(payload)
            open_samples.append((time.perf_counter_ns() - open_started) / 1_000_000)
            try:
                for _ in range(calls_per_session):
                    inspection_started = time.perf_counter_ns()
                    actual = _metadata_from_result(session.inspect(plan))
                    inspection_samples.append(
                        (time.perf_counter_ns() - inspection_started) / 1_000_000
                    )
                    if actual != expected:
                        raise RuntimeError(
                            f"Penampakan reusable session returned {actual!r}; "
                            f"expected {expected!r}"
                        )
            finally:
                close_started = time.perf_counter_ns()
                session.close()
                close_samples.append((time.perf_counter_ns() - close_started) / 1_000_000)
            workflow_ms = (time.perf_counter_ns() - workflow_started) / 1_000_000
            amortized_samples.append(workflow_ms / calls_per_session)
        finally:
            if gc_was_enabled:
                gc.enable()

    median_amortized = statistics.median(amortized_samples)
    return ReusableSessionResult(
        calls_per_session=calls_per_session,
        median_amortized_ms=median_amortized,
        min_amortized_ms=min(amortized_samples),
        max_amortized_ms=max(amortized_samples),
        median_open_ms=statistics.median(open_samples),
        median_warm_inspection_ms=statistics.median(inspection_samples),
        median_close_ms=statistics.median(close_samples),
        speedup_vs_penampakan_one_shot=penampakan_one_shot_ms / median_amortized,
    )


def _contract_checks(payload: bytes, vision: Penampakan) -> list[ContractCheck]:
    """Execute representative normalization, safety, and attribution guarantees."""

    plan = _metadata_plan()

    def inspect(source: bytes | str) -> tuple[Metadata, InspectionResult]:
        result = vision.inspect(source, plan)
        return _metadata_from_result(result), result

    def format_normalization() -> None:
        image = Image.new("RGB", (7, 5), "purple")
        try:
            fixtures = (
                _encode_image(image, "PNG"),
                _encode_image(image, "JPEG", quality=100, subsampling=0),
                _encode_image(image, "WEBP", lossless=True),
            )
        finally:
            image.close()
        for encoded in fixtures:
            actual, _ = inspect(encoded)
            if actual != (7, 5, False):
                raise RuntimeError(f"format normalization returned {actual!r}")

    def orientation_normalization() -> None:
        actual, _ = inspect(_oriented_jpeg())
        if actual != (2, 3, False):
            raise RuntimeError(f"orientation normalization returned {actual!r}")

    def alpha_normalization() -> None:
        opaque = Image.new("RGBA", (4, 3), (255, 0, 0, 255))
        transparent = opaque.copy()
        transparent.putpixel((0, 0), (255, 0, 0, 0))
        try:
            opaque_actual, _ = inspect(_encode_image(opaque, "PNG"))
            transparent_actual, _ = inspect(_encode_image(transparent, "PNG"))
        finally:
            opaque.close()
            transparent.close()
        if opaque_actual != (4, 3, False) or transparent_actual != (4, 3, True):
            raise RuntimeError(
                "alpha normalization did not distinguish opaque and transparent inputs"
            )

    def safe_rejection() -> None:
        for source, expected_error in (
            (b"not-an-image", InvalidImageError),
            (_animated_webp(), UnsupportedImageError),
        ):
            try:
                inspect(source)
            except expected_error:
                continue
            raise RuntimeError(f"{expected_error.__name__} was not raised")

    def source_policy_and_limits() -> None:
        try:
            inspect("https://example.test/image.png")
        except RemoteSourceDisabledError:
            pass
        else:
            raise RuntimeError("RemoteSourceDisabledError was not raised")
        settings = Settings(image=ImageLimits(max_input_bytes=len(payload) - 1))
        with Penampakan(settings=settings) as bounded:
            try:
                bounded.inspect(payload, plan)
            except ImageLimitExceededError:
                return
        raise RuntimeError("ImageLimitExceededError was not raised")

    def typed_attribution() -> None:
        _, result = inspect(payload)
        observation = result.observations[0]
        if observation.provenance.backend_name != "penampakan.pillow":
            raise RuntimeError("metadata provenance did not identify the authoritative backend")
        if result.trace.summary.backend_calls != 1 or not result.trace.events:
            raise RuntimeError("the completed inspection trace did not record the backend call")

    definitions = (
        (
            "format_normalization",
            "PNG, JPEG, and WebP normalize to equivalent typed metadata",
            format_normalization,
        ),
        (
            "exif_orientation",
            "EXIF orientation is applied before dimensions are reported",
            orientation_normalization,
        ),
        (
            "alpha_semantics",
            "opaque alpha is removed while real transparency is preserved",
            alpha_normalization,
        ),
        (
            "unsafe_input_rejection",
            "malformed and animated inputs fail through documented errors",
            safe_rejection,
        ),
        (
            "source_policy_and_limits",
            "remote sources and oversized inputs are rejected before perception",
            source_policy_and_limits,
        ),
        (
            "typed_attribution",
            "metadata includes authoritative provenance and a completed redacted trace",
            typed_attribution,
        ),
    )
    checks: list[ContractCheck] = []
    for name, detail, check in definitions:
        try:
            check()
        except Exception as error:
            checks.append(
                ContractCheck(
                    name=name,
                    passed=False,
                    detail=f"{detail}; failed with {type(error).__name__}",
                )
            )
        else:
            checks.append(ContractCheck(name=name, passed=True, detail=detail))
    return checks


def _payload(
    args: argparse.Namespace,
    fixture_bytes: int,
    results: Sequence[BenchmarkResult],
    reusable_session: ReusableSessionResult,
    contract_checks: Sequence[ContractCheck],
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
            "reuse_count": args.reuse_count,
        },
        "results": [asdict(result) for result in results],
        "reusable_session": asdict(reusable_session),
        "contract_checks": [asdict(check) for check in contract_checks],
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
        "One-shot end-to-end comparison",
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
    reusable = data["reusable_session"]
    lines.extend(
        (
            "",
            "Reusable Penampakan session",
            (f"Workflow: open once, inspect {reusable['calls_per_session']} times, close once"),
            (
                f"Median open {reusable['median_open_ms']:.3f} ms; "
                f"warm inspection {reusable['median_warm_inspection_ms']:.3f} ms; "
                f"close {reusable['median_close_ms']:.3f} ms"
            ),
            (
                f"Amortized {reusable['median_amortized_ms']:.3f} ms/inspection "
                f"({reusable['speedup_vs_penampakan_one_shot']:.2f}x faster than "
                "Penampakan one-shot)"
            ),
            "",
            "Executed Penampakan contract checks (not latency rankings)",
        )
    )
    lines.extend(
        f"- {'PASS' if check['passed'] else 'FAIL'} {check['name']}: {check['detail']}"
        for check in data["contract_checks"]
    )
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
        axis.set_title("One-shot metadata latency — lower is better", loc="left", pad=20)
        axis.grid(axis="x", which="both", color="#d8dee4", linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0, pad=10)

        figure.text(
            0.125,
            0.035,
            (
                f"{fixture['width']}x{fixture['height']} RGBA PNG; "
                f"{settings['iterations']} iterations x {settings['rounds']} rounds. "
                "Lines show the observed min-max range."
            ),
            color="#57606a",
            fontsize=9,
        )
        reusable = data["reusable_session"]
        passed = sum(check["passed"] for check in data["contract_checks"])
        total = len(data["contract_checks"])
        figure.text(
            0.125,
            0.012,
            (
                f"Reusable Penampakan session: {reusable['median_amortized_ms']:.3f} ms "
                f"amortized over {reusable['calls_per_session']} inspections; "
                f"normalization and safety checks: {passed}/{total} passed."
            ),
            color="#287a4b",
            fontsize=9,
            fontweight="bold",
        )
        figure.tight_layout(rect=(0, 0.09, 1, 1))
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
        penampakan_result = next(result for result in results if result.name == "Penampakan")
        reusable_session = _run_reusable_session(
            encoded,
            vision,
            expected=expected,
            warmups=args.warmups,
            calls_per_session=args.reuse_count,
            rounds=args.rounds,
            penampakan_one_shot_ms=penampakan_result.median_ms,
        )
        contract_checks = _contract_checks(encoded, vision)

    data = _payload(
        args,
        len(encoded),
        results,
        reusable_session,
        contract_checks,
        skipped,
    )
    if args.plot is not None:
        _plot(data, args.plot)
    if args.format == "json":
        json.dump(data, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(_table(data))
    return 0 if all(check.passed for check in contract_checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
