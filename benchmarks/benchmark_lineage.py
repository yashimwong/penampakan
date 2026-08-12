#!/usr/bin/env python3
"""Measure asset-lineage and context cost inside a saturated reasoning session.

This benchmark supplies the evidence required by requirement F4 of
`specs/06-correctness-fixes.md`. It drives a real `AsyncVisionSession` with a
deterministic scripted policy until the session holds the configured maximum
number of derived assets at the configured maximum derivation depth, then
attributes wall time to the lineage and context scans named by F4:

    AsyncVisionSession._lineage_for_asset   (src/penampakan/session.py)
    AsyncVisionSession._asset_root_ids      (src/penampakan/session.py)
    AsyncVisionSession._maximum_depth       (src/penampakan/session.py)
    AssetStore.snapshots                    (src/penampakan/image/assets.py)
    _eligible_assets                        (src/penampakan/reasoning/context.py)
    ContextCompiler.compile                 (src/penampakan/reasoning/context.py)

Three independent measurements are combined:

1. Unprofiled session wall time (`time.perf_counter`), repeated over rounds.
2. A `cProfile`/`pstats` run of one session, for exact per-function call counts
   and cumulative time.
3. A direct timing loop over each lineage function at the maximum asset count,
   which yields per-call cost without profiler overhead.

The reported per-session lineage cost is the profiled call count multiplied by
the unprofiled per-call median, divided by the median session wall time. The
denominator therefore includes transform work executed in `asyncio.to_thread`,
which `cProfile` cannot see.

Inputs are deterministic, nothing touches the network, and only the standard
library, Pillow, and Penampakan itself are imported, so a base install is
enough. Run it from the repository root:

    .venv/bin/python benchmarks/benchmark_lineage.py

Exit status is 0 when the F4 gate does not fire and 1 when it does.
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import gc
import json
import platform
import pstats
import statistics
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from io import BytesIO
from typing import Any

from PIL import Image

from penampakan import __version__
from penampakan.client import AsyncPenampakan
from penampakan.config import AgentSettings, RunLimits, Settings
from penampakan.image.assets import AssetStore
from penampakan.models import (
    AnswerAction,
    AnswerStatus,
    Capability,
    EvidenceRef,
    PolicyAction,
    PolicyInput,
    ToolAction,
)
from penampakan.reasoning.context import ContextCompiler
from penampakan.reasoning.context import _eligible_assets as eligible_assets
from penampakan.reasoning.context import _rank_observations as rank_observations
from penampakan.session import AsyncVisionSession

GATE_SHARE_PERCENT = 5.0
"""Share of session wall time above which F4 requires the AssetStore refactor."""

BUDGET_MS_PER_SESSION = 1.0
"""Absolute lineage-scan budget for one session at the maximum asset count."""

BUDGET_MS_PER_STEP = 0.15
"""Absolute lineage-scan budget for one reasoning step."""

ADVISORY_COMPILE_MS_PER_SESSION = 8.0
"""Advisory ceiling for whole-context compilation; not part of the F4 gate."""

_TILE_PLAN = ((2, 3), (2, 3), (2, 2))
"""Row-major tile fan-outs that saturate 16 derived assets across 3 depths."""

_FIXTURE_BLOCKS = 8
"""Distinct base-color blocks per axis, which keeps every derived tile unique."""


@dataclass(frozen=True, slots=True)
class SessionShape:
    """The saturated session state the measurement was taken against."""

    total_assets: int
    derived_assets: int
    max_derived_assets: int
    maximum_depth: int
    max_derivation_depth: int
    observations: int
    policy_steps: int
    deepest_lineage: int


@dataclass(frozen=True, slots=True)
class WallTime:
    """Unprofiled end-to-end timing for the complete scripted session."""

    rounds: int
    median_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float


@dataclass(frozen=True, slots=True)
class FunctionCost:
    """One attributed function measured both by profile and by direct timing."""

    label: str
    location: str
    calls_per_session: int
    profile_cumulative_ms: float
    direct_median_us: float
    direct_min_us: float
    direct_max_us: float
    session_ms: float = field(init=False)

    def __post_init__(self) -> None:
        cost = self.calls_per_session * self.direct_median_us / 1_000
        object.__setattr__(self, "session_ms", cost)


@dataclass(frozen=True, slots=True)
class Attribution:
    """Aggregate lineage cost against the F4 decision gate.

    `lineage_*` covers exactly the scans the F4 remedy would replace with
    `AssetStore.root_of` and incremental metadata, and is what the gate tests.
    `ranking_*` and `compile_*` are progressively wider envelopes reported for
    disclosure; neither is removed by the proposed refactor.
    """

    lineage_ms: float
    lineage_percent: float
    lineage_ms_per_step: float
    ranking_ms: float
    ranking_percent: float
    compile_ms: float
    compile_percent: float
    budget_ms_per_session: float
    budget_ms_per_step: float
    advisory_compile_ms: float
    gate_share_percent: float
    over_share: bool
    over_budget: bool
    envelope_over_share: bool
    envelope_over_advisory: bool

    @property
    def gate_fired(self) -> bool:
        """Return whether F4 requires moving lineage into `AssetStore`."""

        return self.over_share or self.over_budget


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
    parser = argparse.ArgumentParser(description="Measure F4 lineage and context cost.")
    parser.add_argument("--width", type=_positive_int, default=384)
    parser.add_argument("--height", type=_positive_int, default=288)
    parser.add_argument("--warmups", type=_non_negative_int, default=2)
    parser.add_argument(
        "--rounds",
        type=_positive_int,
        default=15,
        help="complete scripted sessions timed without the profiler",
    )
    parser.add_argument(
        "--iterations",
        type=_positive_int,
        default=2_000,
        help="direct calls per timing sample for each attributed function",
    )
    parser.add_argument(
        "--samples",
        type=_positive_int,
        default=7,
        help="timing samples collected for each attributed function",
    )
    parser.add_argument("--format", choices=("table", "json"), default="table")
    return parser.parse_args(argv)


def _fixture(width: int, height: int) -> bytes:
    """Build one deterministic RGB PNG whose every tile is content-unique.

    A block-varying base color keeps each region distinguishable so no derived
    tile deduplicates against another, while the smooth in-block gradient keeps
    PNG encoding cost representative rather than incompressible noise.
    """

    pixels = bytearray(width * height * 3)
    offset = 0
    for y in range(height):
        block_y = y * _FIXTURE_BLOCKS // height
        for x in range(width):
            block_x = x * _FIXTURE_BLOCKS // width
            base = (block_x * 31 + block_y * 71) % 256
            pixels[offset] = (base + x * 3 + y) % 256
            pixels[offset + 1] = (base * 2 + x + y * 3) % 256
            pixels[offset + 2] = (base * 3 + x * 2 + y * 2) % 256
            offset += 3

    image = Image.frombytes("RGB", (width, height), bytes(pixels))
    output = BytesIO()
    try:
        image.save(output, format="PNG", compress_level=1)
        return output.getvalue()
    finally:
        output.close()
        image.close()


def _settings() -> Settings:
    """Use default run limits so the benchmark saturates the shipped maximum."""

    return Settings(
        run=RunLimits(),
        agent=AgentSettings(
            initial_capabilities=(Capability.METADATA, Capability.COLORS),
        ),
    )


def _context_records(context: str) -> Iterator[dict[str, Any]]:
    for line in context.splitlines():
        if not line.startswith("{"):
            continue
        value = json.loads(line)
        if isinstance(value, dict) and "id" in value:
            yield value


class ScriptedPolicy:
    """Deterministically tile to the asset ceiling, then perceive and answer."""

    def __init__(self) -> None:
        self.tool_calls = 0
        self.policy_calls = 0
        self._tiled: list[str] = []

    async def next_action(self, input: PolicyInput) -> PolicyAction:
        """Return the next scripted action for the compiled visual context."""

        self.policy_calls += 1
        records = list(_context_records(input.context))
        if input.answer_only:
            return self._answer(records)
        step = self.tool_calls
        self.tool_calls += 1
        if step < len(_TILE_PLAN):
            return self._tile(records, step)
        return self._perceive(records, step)

    @staticmethod
    def _answer(records: Sequence[dict[str, Any]]) -> AnswerAction:
        metadata = next(record for record in records if record.get("type") == "metadata")
        return AnswerAction(
            status="answered",
            answer="The image is a deterministic synthetic gradient.",
            evidence=(
                EvidenceRef(
                    observation_id=str(metadata["id"]),
                    supports="Authoritative metadata reports the normalized dimensions.",
                ),
            ),
        )

    def _tile(self, records: Sequence[dict[str, Any]], step: int) -> ToolAction:
        if step == 0:
            target = str(records[0]["asset_id"])
        else:
            children = sorted(
                str(record["derived_asset_id"])
                for record in records
                if record.get("type") == "transform"
                and record.get("parent_asset_id") == self._tiled[-1]
            )
            target = children[0]
        self._tiled.append(target)
        rows, columns = _TILE_PLAN[step]
        return ToolAction(
            tool="tile",
            arguments={"asset_id": target, "rows": rows, "columns": columns},
            purpose="Divide the asset into row-major tiles for closer inspection.",
        )

    @staticmethod
    def _perceive(records: Sequence[dict[str, Any]], step: int) -> ToolAction:
        assets = sorted(
            {str(record["asset_id"]) for record in records}
            | {
                str(record["derived_asset_id"])
                for record in records
                if record.get("type") == "transform"
            }
        )
        target = assets[(step * 7) % len(assets)]
        if step % 2:
            return ToolAction(
                tool="get_metadata",
                arguments={"asset_id": target},
                purpose="Confirm the normalized dimensions of a derived asset.",
            )
        return ToolAction(
            tool="get_colors",
            arguments={"asset_id": target, "count": 3},
            purpose="Estimate the dominant colors of a derived asset.",
        )


async def _drive(payload: bytes) -> tuple[AsyncVisionSession, AsyncPenampakan, ScriptedPolicy]:
    """Run one complete scripted session and return it still open."""

    policy = ScriptedPolicy()
    vision = AsyncPenampakan(policy=policy, settings=_settings())
    session = await vision.open_image(payload)
    try:
        answer = await session.ask("Which region of this image carries the strongest gradient?")
    except BaseException:
        await session.aclose()
        await vision.aclose()
        raise
    if answer.status is not AnswerStatus.ANSWERED:
        await session.aclose()
        await vision.aclose()
        raise RuntimeError(f"the scripted session returned {answer.status.value!r}")
    return session, vision, policy


async def _run_once(payload: bytes) -> None:
    session, vision, _ = await _drive(payload)
    await session.aclose()
    await vision.aclose()


def _session_shape(
    session: AsyncVisionSession,
    policy: ScriptedPolicy,
    limits: RunLimits,
) -> SessionShape:
    assets = session.assets
    deepest = max(assets, key=lambda asset: asset.derivation_depth)
    lineage = session._lineage_for_asset(deepest.id)
    return SessionShape(
        total_assets=len(assets),
        derived_assets=len(assets) - 1,
        max_derived_assets=limits.max_derived_assets,
        maximum_depth=deepest.derivation_depth,
        max_derivation_depth=limits.max_derivation_depth,
        observations=len(session.observations),
        policy_steps=policy.policy_calls,
        deepest_lineage=len(lineage),
    )


def _require_saturated(shape: SessionShape) -> None:
    if shape.derived_assets != shape.max_derived_assets:
        raise RuntimeError(
            f"the scripted session created {shape.derived_assets} derived assets; "
            f"the configured maximum is {shape.max_derived_assets}"
        )
    if shape.maximum_depth != shape.max_derivation_depth:
        raise RuntimeError(
            f"the scripted session reached depth {shape.maximum_depth}; "
            f"the configured maximum is {shape.max_derivation_depth}"
        )


def _wall_time(payload: bytes, *, warmups: int, rounds: int) -> WallTime:
    for _ in range(warmups):
        asyncio.run(_run_once(payload))

    samples: list[float] = []
    for _ in range(rounds):
        gc_was_enabled = gc.isenabled()
        gc.collect()
        if gc_was_enabled:
            gc.disable()
        try:
            started = time.perf_counter_ns()
            asyncio.run(_run_once(payload))
            samples.append((time.perf_counter_ns() - started) / 1_000_000)
        finally:
            if gc_was_enabled:
                gc.enable()
    return WallTime(
        rounds=rounds,
        median_ms=statistics.median(samples),
        min_ms=min(samples),
        max_ms=max(samples),
        stdev_ms=statistics.stdev(samples) if len(samples) > 1 else 0.0,
    )


def _profile(payload: bytes) -> dict[tuple[str, str], tuple[int, float]]:
    """Return primitive call count and cumulative seconds keyed by module and name."""

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        asyncio.run(_run_once(payload))
    finally:
        profiler.disable()
    stats = pstats.Stats(profiler)
    attributed: dict[tuple[str, str], tuple[int, float]] = {}
    for (filename, _, function), entry in stats.stats.items():  # type: ignore[attr-defined]
        primitive_calls, _, _, cumulative, _ = entry
        module = filename.replace("\\", "/").rsplit("/", 1)[-1]
        key = (module, function)
        previous = attributed.get(key)
        if previous is None:
            attributed[key] = (primitive_calls, cumulative)
        else:
            attributed[key] = (previous[0] + primitive_calls, previous[1] + cumulative)
    return attributed


def _time_call(call: Any, *, iterations: int, samples: int) -> tuple[float, float, float]:
    """Return median, minimum, and maximum microseconds for one direct call."""

    call()
    measured: list[float] = []
    for _ in range(samples):
        gc_was_enabled = gc.isenabled()
        gc.collect()
        if gc_was_enabled:
            gc.disable()
        try:
            started = time.perf_counter_ns()
            for _ in range(iterations):
                call()
            elapsed = time.perf_counter_ns() - started
        finally:
            if gc_was_enabled:
                gc.enable()
        measured.append(elapsed / iterations / 1_000)
    return statistics.median(measured), min(measured), max(measured)


def _costs(
    session: AsyncVisionSession,
    profiled: dict[tuple[str, str], tuple[int, float]],
    limits: RunLimits,
    *,
    iterations: int,
    samples: int,
) -> list[FunctionCost]:
    assets = session.assets
    deepest = max(assets, key=lambda asset: asset.derivation_depth)
    observations = session.observations
    root_id = session.root_asset.id
    lineage = session._lineage_for_asset(deepest.id)
    compiler = ContextCompiler(limits.max_context_chars)
    question = "Which region of this image carries the strongest gradient?"
    store: AssetStore = session._assets
    observation_by_id = {observation.id: observation for observation in observations}

    definitions = (
        (
            "AsyncVisionSession._lineage_for_asset",
            "session.py",
            "_lineage_for_asset",
            lambda: session._lineage_for_asset(deepest.id),
        ),
        (
            "AsyncVisionSession._asset_root_ids",
            "session.py",
            "_asset_root_ids",
            session._asset_root_ids,
        ),
        (
            "AsyncVisionSession._maximum_depth",
            "session.py",
            "_maximum_depth",
            session._maximum_depth,
        ),
        (
            "AssetStore.snapshots",
            "assets.py",
            "snapshots",
            store.snapshots,
        ),
        (
            "context._eligible_assets",
            "context.py",
            "_eligible_assets",
            lambda: eligible_assets(
                observations,
                root_id,
                tuple(asset.id for asset in assets),
                lineage,
                (),
                observation_by_id,
            ),
        ),
        (
            "context._rank_observations",
            "context.py",
            "_rank_observations",
            lambda: rank_observations(
                question,
                observations,
                observations,
                root_id,
                frozenset(lineage),
                frozenset(),
                frozenset(),
                frozenset(),
            ),
        ),
        (
            "ContextCompiler.compile",
            "context.py",
            "compile",
            lambda: compiler.compile(
                question,
                observations,
                root_asset_id=root_id,
                relevant_asset_ids=tuple(asset.id for asset in assets),
                most_recent_asset_lineage=lineage,
            ),
        ),
    )

    costs: list[FunctionCost] = []
    for label, module, function, call in definitions:
        calls, cumulative = profiled.get((module, function), (0, 0.0))
        median_us, min_us, max_us = _time_call(call, iterations=iterations, samples=samples)
        costs.append(
            FunctionCost(
                label=label,
                location=f"{module}:{function}",
                calls_per_session=calls,
                profile_cumulative_ms=cumulative * 1_000,
                direct_median_us=median_us,
                direct_min_us=min_us,
                direct_max_us=max_us,
            )
        )
    return costs


def _attribution(costs: Sequence[FunctionCost], wall: WallTime, steps: int) -> Attribution:
    by_label = {cost.label: cost for cost in costs}
    lineage_labels = (
        "AsyncVisionSession._lineage_for_asset",
        "AsyncVisionSession._asset_root_ids",
        "AsyncVisionSession._maximum_depth",
        "context._eligible_assets",
    )
    lineage_ms = sum(by_label[label].session_ms for label in lineage_labels)
    ranking_ms = by_label["context._rank_observations"].session_ms
    compile_ms = by_label["ContextCompiler.compile"].session_ms
    lineage_percent = 100 * lineage_ms / wall.median_ms
    compile_percent = 100 * compile_ms / wall.median_ms
    lineage_per_step = lineage_ms / max(1, steps)
    return Attribution(
        lineage_ms=lineage_ms,
        lineage_percent=lineage_percent,
        lineage_ms_per_step=lineage_per_step,
        ranking_ms=ranking_ms,
        ranking_percent=100 * ranking_ms / wall.median_ms,
        compile_ms=compile_ms,
        compile_percent=compile_percent,
        budget_ms_per_session=BUDGET_MS_PER_SESSION,
        budget_ms_per_step=BUDGET_MS_PER_STEP,
        advisory_compile_ms=ADVISORY_COMPILE_MS_PER_SESSION,
        gate_share_percent=GATE_SHARE_PERCENT,
        over_share=lineage_percent >= GATE_SHARE_PERCENT,
        over_budget=(lineage_ms > BUDGET_MS_PER_SESSION or lineage_per_step > BUDGET_MS_PER_STEP),
        envelope_over_share=compile_percent >= GATE_SHARE_PERCENT,
        envelope_over_advisory=compile_ms > ADVISORY_COMPILE_MS_PER_SESSION,
    )


def _payload(
    args: argparse.Namespace,
    fixture_bytes: int,
    shape: SessionShape,
    wall: WallTime,
    costs: Sequence[FunctionCost],
    attribution: Attribution,
) -> dict[str, Any]:
    return {
        "benchmark": (
            "attribute session wall time to asset-lineage and context scans at the "
            "configured maximum derived-asset count"
        ),
        "requirement": "specs/06-correctness-fixes.md F4",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "penampakan": __version__,
        },
        "fixture": {
            "width": args.width,
            "height": args.height,
            "mode": "RGB",
            "encoded_bytes": fixture_bytes,
        },
        "settings": {
            "warmups": args.warmups,
            "rounds": args.rounds,
            "iterations": args.iterations,
            "samples": args.samples,
        },
        "session": asdict(shape),
        "wall_time": asdict(wall),
        "functions": [asdict(cost) for cost in costs],
        "attribution": {**asdict(attribution), "gate_fired": attribution.gate_fired},
    }


def _table(data: dict[str, Any]) -> str:
    fixture = data["fixture"]
    settings = data["settings"]
    session = data["session"]
    wall = data["wall_time"]
    attribution = data["attribution"]
    lines = [
        "Asset lineage and context cost (specs/06-correctness-fixes.md F4)",
        f"Python {data['environment']['python']} on {data['environment']['platform']}",
        f"Penampakan {data['environment']['penampakan']}",
        (
            f"Fixture: {fixture['width']}x{fixture['height']} {fixture['mode']} PNG "
            f"({fixture['encoded_bytes']:,} bytes)"
        ),
        (
            f"Sessions: {settings['rounds']} timed rounds after {settings['warmups']} warmups; "
            f"direct timing {settings['iterations']} calls x {settings['samples']} samples"
        ),
        "",
        "Saturated session state",
        (
            f"  assets {session['total_assets']} "
            f"({session['derived_assets']}/{session['max_derived_assets']} derived), "
            f"depth {session['maximum_depth']}/{session['max_derivation_depth']}, "
            f"longest lineage {session['deepest_lineage']}"
        ),
        (f"  observations {session['observations']}, policy steps {session['policy_steps']}"),
        "",
        (
            f"Session wall time: median {wall['median_ms']:.3f} ms "
            f"(min {wall['min_ms']:.3f}, max {wall['max_ms']:.3f}, "
            f"stdev {wall['stdev_ms']:.3f}) over {wall['rounds']} rounds"
        ),
        "",
        "Attributed functions",
        "",
    ]
    headers = (
        "Function",
        "Calls",
        "Profile cum ms",
        "Direct median us",
        "Direct min us",
        "Direct max us",
        "Session ms",
    )
    rows = [
        (
            cost["label"],
            str(cost["calls_per_session"]),
            f"{cost['profile_cumulative_ms']:.3f}",
            f"{cost['direct_median_us']:.3f}",
            f"{cost['direct_min_us']:.3f}",
            f"{cost['direct_max_us']:.3f}",
            f"{cost['session_ms']:.4f}",
        )
        for cost in data["functions"]
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
    lines.extend(
        (
            "",
            "Session ms multiplies the profiled call count by the unprofiled per-call median",
            "measured at the saturated asset count, so it over-states the earlier steps.",
            "AssetStore.snapshots is already inside the three session rows, and",
            "_rank_observations and ContextCompiler.compile are wider envelopes; only the three",
            "session rows plus _eligible_assets are summed into the gated total.",
            "",
            (
                f"Gated lineage scans: {attribution['lineage_ms']:.4f} ms "
                f"({attribution['lineage_percent']:.3f}% of session wall time), "
                f"{attribution['lineage_ms_per_step']:.4f} ms per reasoning step"
            ),
            (
                f"Lineage-aware ranking envelope: {attribution['ranking_ms']:.4f} ms "
                f"({attribution['ranking_percent']:.3f}%)"
            ),
            (
                f"Whole context compilation envelope: {attribution['compile_ms']:.4f} ms "
                f"({attribution['compile_percent']:.3f}%)"
            ),
            "",
            (
                f"F4 gate: share {attribution['lineage_percent']:.3f}% vs "
                f"{attribution['gate_share_percent']:.1f}%; budget "
                f"{attribution['lineage_ms']:.4f} ms vs "
                f"{attribution['budget_ms_per_session']:.2f} ms per session and "
                f"{attribution['lineage_ms_per_step']:.4f} ms vs "
                f"{attribution['budget_ms_per_step']:.2f} ms per step"
            ),
            (
                "Decision: MOVE lineage into AssetStore (gate fired)"
                if attribution["gate_fired"]
                else "Decision: DEFER the AssetStore lineage refactor (gate did not fire)"
            ),
        )
    )
    if attribution["envelope_over_share"]:
        lines.append("Advisory: whole context compilation reached the 5% share; it is dominated by")
        lines.append("observation serialization and tokenization, which F4's remedy would not fix.")
    if attribution["envelope_over_advisory"]:
        lines.append(
            f"Advisory: context compilation exceeded {attribution['advisory_compile_ms']:.1f} ms "
            "per session."
        )
    lines.extend(
        (
            "",
            "The policy and backends run in process, so the denominator excludes network",
            "language-model latency; a deployed session is slower and every share is lower.",
        )
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    payload = _fixture(args.width, args.height)
    limits = _settings().run

    profiled = _profile(payload)

    async def collect() -> tuple[SessionShape, list[FunctionCost]]:
        session, vision, policy = await _drive(payload)
        try:
            shape = _session_shape(session, policy, limits)
            _require_saturated(shape)
            costs = _costs(
                session,
                profiled,
                limits,
                iterations=args.iterations,
                samples=args.samples,
            )
            return shape, costs
        finally:
            await session.aclose()
            await vision.aclose()

    shape, costs = asyncio.run(collect())
    wall = _wall_time(payload, warmups=args.warmups, rounds=args.rounds)
    attribution = _attribution(costs, wall, shape.policy_steps)
    data = _payload(args, len(payload), shape, wall, costs, attribution)
    if args.format == "json":
        json.dump(data, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(_table(data))
    return 1 if attribution.gate_fired else 0


if __name__ == "__main__":
    raise SystemExit(main())
