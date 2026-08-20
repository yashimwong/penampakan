"""Referential evidence validation and final answer materialization."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from penampakan.errors import EvidenceValidationError
from penampakan.models import (
    AnswerAction,
    AnswerStatus,
    Evidence,
    MarkDescriptionPayload,
    MarkPayload,
    Observation,
    RunTrace,
    VisionAnswer,
    WarningInfo,
    WarningPayload,
)

_NOT_VISIBLE_PHRASES = (
    "not visible",
    "not shown",
    "cannot see",
    "can't see",
    "isn't visible",
    "unable to see",
)


def _contains_action_json(value: str) -> bool:
    decoder = json.JSONDecoder()
    for index, character in enumerate(value):
        if character != "{":
            continue
        try:
            decoded, _ = decoder.raw_decode(value, index)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict) and decoded.get("type") in {"tool", "answer"}:
            return True
    return False


def _only_reports_not_visible(answer: str) -> bool:
    normalized = answer.casefold()
    return any(phrase in normalized for phrase in _NOT_VISIBLE_PHRASES)


def _observation_index(
    observations: Mapping[str, Observation] | Sequence[Observation],
) -> dict[str, Observation]:
    if isinstance(observations, Mapping):
        return dict(observations)
    return {observation.id: observation for observation in observations}


def validate_evidence(
    action: AnswerAction,
    observations: Mapping[str, Observation] | Sequence[Observation],
    *,
    visible_observation_ids: Sequence[str] | set[str] | frozenset[str],
    root_asset_id: str,
    asset_root_ids: Mapping[str, str],
) -> tuple[Evidence, ...]:
    """Validate and snapshot the action's referential evidence."""
    indexed = _observation_index(observations)
    visible = frozenset(visible_observation_ids)
    evidence: list[Evidence] = []
    seen: set[str] = set()
    for reference in action.evidence:
        if reference.observation_id in seen:
            continue
        seen.add(reference.observation_id)
        observation = indexed.get(reference.observation_id)
        if observation is None or reference.observation_id not in visible:
            raise EvidenceValidationError()
        if isinstance(observation.payload, (MarkPayload, WarningPayload)):
            raise EvidenceValidationError()
        if _contains_action_json(reference.supports):
            raise EvidenceValidationError()
        observation_root = asset_root_ids.get(observation.asset_id, observation.asset_id)
        if observation_root != root_asset_id:
            raise EvidenceValidationError()
        evidence.append(Evidence(observation=observation, supports=reference.supports))
    cited_ids = {item.observation.id for item in evidence}
    for item in evidence:
        if not isinstance(item.observation.payload, MarkDescriptionPayload):
            continue
        mappings = (
            indexed.get(parent_id)
            for parent_id in item.observation.provenance.parent_observation_ids
        )
        source_ids = {
            mark.observation_id
            for mapping in mappings
            if mapping is not None and isinstance(mapping.payload, MarkPayload)
            for mark in mapping.payload.marks
        }
        if not source_ids or cited_ids.isdisjoint(source_ids):
            raise EvidenceValidationError()
    if (
        action.status == "answered"
        and not evidence
        and not _only_reports_not_visible(action.answer)
    ):
        raise EvidenceValidationError()
    return tuple(evidence)


def materialize_answer(
    action: AnswerAction,
    observations: Mapping[str, Observation] | Sequence[Observation],
    *,
    visible_observation_ids: Sequence[str] | set[str] | frozenset[str],
    root_asset_id: str,
    asset_root_ids: Mapping[str, str],
    warnings: Sequence[WarningInfo],
    trace: RunTrace,
) -> VisionAnswer:
    """Validate evidence and construct the immutable public answer."""
    evidence = validate_evidence(
        action,
        observations,
        visible_observation_ids=visible_observation_ids,
        root_asset_id=root_asset_id,
        asset_root_ids=asset_root_ids,
    )
    status = (
        AnswerStatus.ANSWERED if action.status == "answered" else AnswerStatus.INSUFFICIENT_EVIDENCE
    )
    return VisionAnswer(
        status=status,
        answer=action.answer,
        evidence=evidence,
        uncertainties=action.uncertainties,
        warnings=tuple(warnings),
        trace=trace,
    )


__all__ = ["materialize_answer", "validate_evidence"]
