from __future__ import annotations

import pytest

from penampakan.evaluation import evidence_region_coverage, tool_trace_efficiency
from penampakan.models import (
    AnswerStatus,
    Box,
    Evidence,
    TextPayload,
    VisionAnswer,
)
from tests.unit.reasoning.helpers import (
    make_observation,
    make_trace,
    make_trace_event,
)

_ROOT = "img_aaaaaaaaaaaaaaaa"


def _vision_answer(evidence: tuple[Evidence, ...]) -> VisionAnswer:
    return VisionAnswer(
        status=AnswerStatus.ANSWERED,
        answer="Answer",
        evidence=evidence,
        uncertainties=(),
        warnings=(),
        trace=make_trace(),
    )


def test_evidence_region_coverage_distinguishes_localized_and_global_evidence() -> None:
    localized = make_observation(
        1,
        TextPayload(text="RM 42.50"),
        region=Box(x_min=0.0, y_min=0.0, x_max=0.5, y_max=0.5),
    )
    global_observation = make_observation(2, TextPayload(text="Receipt"))
    answer = _vision_answer(
        (
            Evidence(observation=localized, supports="Localized amount"),
            Evidence(observation=global_observation, supports="Global receipt"),
        )
    )
    targets = (
        Box(x_min=0.25, y_min=0.25, x_max=0.75, y_max=0.75),
        Box(x_min=0.75, y_min=0.75, x_max=1.0, y_max=1.0),
    )

    metrics = evidence_region_coverage(answer, targets)

    assert metrics.target_count == 2
    assert metrics.cited_evidence_count == 2
    assert metrics.localized_evidence_count == 1
    assert metrics.global_evidence_count == 1
    assert metrics.covered_target_count == 1
    assert metrics.target_coverage == 0.5
    assert metrics.matches[0].intersection_area == pytest.approx(0.0625)
    assert metrics.matches[0].iou == pytest.approx(1 / 7)
    assert metrics.matches[1].evidence_observation_id is None


def test_evidence_region_coverage_handles_no_targets() -> None:
    metrics = evidence_region_coverage(_vision_answer(()), ())

    assert metrics.target_count == 0
    assert metrics.target_coverage == 0.0
    assert metrics.mean_best_iou == 0.0
    assert metrics.max_iou == 0.0
    assert metrics.matches == ()


def test_tool_trace_efficiency_reports_repeats_unused_and_citation_chain() -> None:
    trace = make_trace(
        (
            make_trace_event("tool_call_started", {"canonical_call": "read_text:a"}),
            make_trace_event("tool_call_started", {"canonical_call": "read_text:a"}),
            make_trace_event("tool_call_started", {"canonical_call": "crop:b"}),
            make_trace_event("backend_call_started", {"request_hash": "backend-a"}),
            make_trace_event("backend_call_started", {"request_hash": "backend-a"}),
            make_trace_event(
                "observations_committed",
                {"observation_ids": ["obs_000001", "obs_000002"]},
            ),
            make_trace_event("answer_validated", {"evidence_ids": ["obs_000002"]}),
        ),
        tool_calls=3,
        backend_calls=2,
    )

    metrics = tool_trace_efficiency(trace)

    assert metrics.tool_calls == 3
    assert metrics.backend_calls == 2
    assert metrics.repeated_tool_calls == 1
    assert metrics.repeated_backend_calls == 1
    assert metrics.observations_produced == 2
    assert metrics.observations_cited == 1
    assert metrics.unused_observations == 1
    assert metrics.cited_evidence_from_tool_chain


def test_tool_trace_efficiency_detects_external_citation() -> None:
    trace = make_trace(
        (
            make_trace_event(
                "observations_committed",
                {"observation_ids": ["obs_000001"]},
            ),
            make_trace_event("answer_validated", {"evidence_ids": ["obs_999999"]}),
        )
    )

    metrics = tool_trace_efficiency(trace)

    assert not metrics.cited_evidence_from_tool_chain
    assert metrics.unused_observations == 1
