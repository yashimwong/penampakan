"""Experimental pure metrics for evidence localization and tool traces."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from penampakan.models import Box, JsonValue, RunTrace, VisionAnswer


@dataclass(frozen=True, slots=True)
class CoverageMatch:
    """The best localized evidence overlap for one target region."""

    target_index: int
    evidence_observation_id: str | None
    intersection_area: float
    iou: float


@dataclass(frozen=True, slots=True)
class CoverageMetrics:
    """Aggregate localization metrics for cited answer evidence."""

    target_count: int
    cited_evidence_count: int
    localized_evidence_count: int
    global_evidence_count: int
    covered_target_count: int
    target_coverage: float
    mean_best_iou: float
    max_iou: float
    matches: tuple[CoverageMatch, ...]


@dataclass(frozen=True, slots=True)
class EfficiencyMetrics:
    """Aggregate action reuse and evidence-use metrics for one run trace."""

    tool_calls: int
    backend_calls: int
    repeated_tool_calls: int
    repeated_backend_calls: int
    observations_produced: int
    observations_cited: int
    unused_observations: int
    cited_evidence_from_tool_chain: bool


def evidence_region_coverage(
    answer: VisionAnswer,
    target_boxes: Sequence[Box],
) -> CoverageMetrics:
    """Compare cited localized evidence with normalized target regions."""
    targets = tuple(target_boxes)
    localized = tuple(item for item in answer.evidence if item.observation.region is not None)
    global_count = sum(item.observation.region is None for item in answer.evidence)
    matches: list[CoverageMatch] = []
    for target_index, target in enumerate(targets):
        best_id: str | None = None
        best_intersection = 0.0
        best_iou = 0.0
        for item in localized:
            region = item.observation.region
            if region is None:
                continue
            overlap = region.intersection(target)
            intersection = 0.0 if overlap is None else overlap.area
            iou = region.iou(target)
            candidate = (iou, intersection)
            current = (best_iou, best_intersection)
            if candidate > current:
                best_id = item.observation.id
                best_intersection = intersection
                best_iou = iou
        matches.append(
            CoverageMatch(
                target_index=target_index,
                evidence_observation_id=best_id,
                intersection_area=best_intersection,
                iou=best_iou,
            )
        )
    covered = sum(match.intersection_area > 0.0 for match in matches)
    mean_iou = sum(match.iou for match in matches) / len(matches) if matches else 0.0
    return CoverageMetrics(
        target_count=len(targets),
        cited_evidence_count=len(answer.evidence),
        localized_evidence_count=len(localized),
        global_evidence_count=global_count,
        covered_target_count=covered,
        target_coverage=covered / len(targets) if targets else 0.0,
        mean_best_iou=mean_iou,
        max_iou=max((match.iou for match in matches), default=0.0),
        matches=tuple(matches),
    )


def _strings(value: JsonValue | None) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(item for item in value if isinstance(item, str))
    return ()


def _first(data: Mapping[str, JsonValue], names: Iterable[str]) -> str | None:
    for name in names:
        value = data.get(name)
        if isinstance(value, str):
            return value
    return None


def _call_signatures(
    trace: RunTrace,
    event_type: str,
    identity_names: tuple[str, ...],
) -> Counter[str]:
    signatures: Counter[str] = Counter()
    for event in trace.events:
        if event.event_type != event_type:
            continue
        identity = _first(event.data, identity_names)
        if identity is not None:
            signatures[identity] += 1
    return signatures


def _repeated(signatures: Counter[str]) -> int:
    return sum(count - 1 for count in signatures.values() if count > 1)


def tool_trace_efficiency(trace: RunTrace) -> EfficiencyMetrics:
    """Summarize repeated calls and whether produced observations were cited."""
    tool_signatures = _call_signatures(
        trace,
        "tool_call_started",
        ("canonical_call", "action_hash", "request_hash", "tool_call_id"),
    )
    backend_signatures = _call_signatures(
        trace,
        "backend_call_started",
        ("canonical_call", "request_hash", "backend_call_id"),
    )
    produced: set[str] = set()
    cited: set[str] = set()
    for event in trace.events:
        if event.event_type == "observations_committed":
            produced.update(_strings(event.data.get("observation_ids")))
        elif event.event_type == "answer_validated":
            cited.update(_strings(event.data.get("evidence_ids")))
            cited.update(_strings(event.data.get("observation_ids")))
    used = produced & cited
    return EfficiencyMetrics(
        tool_calls=trace.summary.tool_calls,
        backend_calls=trace.summary.backend_calls,
        repeated_tool_calls=_repeated(tool_signatures),
        repeated_backend_calls=_repeated(backend_signatures),
        observations_produced=len(produced),
        observations_cited=len(cited),
        unused_observations=len(produced - cited),
        cited_evidence_from_tool_chain=bool(cited) and used == cited,
    )


__all__ = [
    "CoverageMatch",
    "CoverageMetrics",
    "EfficiencyMetrics",
    "evidence_region_coverage",
    "tool_trace_efficiency",
]
