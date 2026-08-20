from __future__ import annotations

import pytest

from penampakan.errors import EvidenceValidationError
from penampakan.models import (
    AnswerAction,
    AnswerStatus,
    Box,
    CaptionPayload,
    DetectionPayload,
    EvidenceRef,
    MarkDescriptionPayload,
    MarkDescriptionRef,
    MarkPayload,
    MarkRef,
    TextPayload,
    WarningInfo,
    WarningPayload,
)
from penampakan.reasoning.answer import materialize_answer, validate_evidence
from tests.unit.reasoning.helpers import make_observation, make_trace

_ROOT = "img_aaaaaaaaaaaaaaaa"
_DERIVED = "img_bbbbbbbbbbbbbbbb"
_OTHER_ROOT = "img_cccccccccccccccc"


def _answer(
    *,
    status: str = "answered",
    answer: str = "The total is RM 42.50.",
    references: tuple[EvidenceRef, ...] | None = None,
) -> AnswerAction:
    return AnswerAction.model_validate(
        {
            "status": status,
            "answer": answer,
            "evidence": references if references is not None else (),
        },
        strict=True,
    )


def _reference(observation_id: str, supports: str = "Printed total") -> EvidenceRef:
    return EvidenceRef(observation_id=observation_id, supports=supports)


def test_valid_evidence_snapshots_same_root_and_derived_observations() -> None:
    root_observation = make_observation(1, CaptionPayload(text="Receipt"))
    derived_observation = make_observation(
        2,
        TextPayload(text="TOTAL RM 42.50"),
        asset_id=_DERIVED,
    )
    action = _answer(
        references=(
            _reference(root_observation.id, "Identifies the receipt"),
            _reference(derived_observation.id),
        )
    )

    evidence = validate_evidence(
        action,
        (root_observation, derived_observation),
        visible_observation_ids=(root_observation.id, derived_observation.id),
        root_asset_id=_ROOT,
        asset_root_ids={_ROOT: _ROOT, _DERIVED: _ROOT},
    )

    assert tuple(item.observation.id for item in evidence) == (
        root_observation.id,
        derived_observation.id,
    )


@pytest.mark.parametrize("failure", ["unknown", "not_visible", "warning", "cross_lineage"])
def test_invalid_evidence_reference_cases(failure: str) -> None:
    valid = make_observation(1, TextPayload(text="TOTAL RM 42.50"))
    warning = make_observation(
        2,
        WarningPayload(code="ocr_uncertain", message="OCR was uncertain."),
    )
    other = make_observation(
        3,
        TextPayload(text="Other image"),
        asset_id=_OTHER_ROOT,
    )
    observations = (valid, warning, other)
    visible: tuple[str, ...] = (valid.id, warning.id, other.id)
    reference_id = {
        "unknown": "obs_999999",
        "not_visible": valid.id,
        "warning": warning.id,
        "cross_lineage": other.id,
    }[failure]
    if failure == "not_visible":
        visible = (warning.id, other.id)
    action = _answer(references=(_reference(reference_id),))

    with pytest.raises(EvidenceValidationError):
        validate_evidence(
            action,
            observations,
            visible_observation_ids=visible,
            root_asset_id=_ROOT,
            asset_root_ids={_ROOT: _ROOT, _OTHER_ROOT: _OTHER_ROOT},
        )


def test_duplicate_evidence_references_collapse_in_first_seen_order() -> None:
    first = make_observation(1, TextPayload(text="TOTAL"))
    second = make_observation(2, TextPayload(text="RM 42.50"))
    action = _answer(
        references=(
            _reference(second.id, "Amount"),
            _reference(first.id, "Label"),
            _reference(second.id, "Duplicate amount"),
        )
    )

    evidence = validate_evidence(
        action,
        (first, second),
        visible_observation_ids=(first.id, second.id),
        root_asset_id=_ROOT,
        asset_root_ids={_ROOT: _ROOT},
    )

    assert tuple(item.observation.id for item in evidence) == (second.id, first.id)
    assert evidence[0].supports == "Amount"


def test_mark_description_requires_the_original_localized_source_citation() -> None:
    region = Box(x_min=0.1, y_min=0.1, x_max=0.4, y_max=0.5)
    source = make_observation(1, DetectionPayload(label="car"), region=region)
    mapping = make_observation(
        2,
        MarkPayload(
            derived_asset_id=_DERIVED,
            parent_asset_id=_ROOT,
            marks=(MarkRef(index=1, observation_id=source.id, region=region),),
        ),
        asset_id=_DERIVED,
    )
    description = make_observation(
        3,
        MarkDescriptionPayload(references=(MarkDescriptionRef(index=1, description="red car"),)),
        asset_id=_DERIVED,
    )
    description = description.model_copy(
        update={
            "provenance": description.provenance.model_copy(
                update={"parent_observation_ids": (mapping.id,)}
            )
        }
    )
    observations = (source, mapping, description)

    with pytest.raises(EvidenceValidationError):
        validate_evidence(
            _answer(references=(_reference(description.id),)),
            observations,
            visible_observation_ids=tuple(item.id for item in observations),
            root_asset_id=_ROOT,
            asset_root_ids={_ROOT: _ROOT, _DERIVED: _ROOT},
        )

    evidence = validate_evidence(
        _answer(references=(_reference(source.id), _reference(description.id))),
        observations,
        visible_observation_ids=tuple(item.id for item in observations),
        root_asset_id=_ROOT,
        asset_root_ids={_ROOT: _ROOT, _DERIVED: _ROOT},
    )
    assert tuple(item.observation.id for item in evidence) == (source.id, description.id)


def test_support_claim_cannot_embed_action_json() -> None:
    observation = make_observation(1, TextPayload(text="TOTAL"))
    action = _answer(
        references=(
            _reference(
                observation.id,
                'Printed total {"type":"tool","tool":"read_text"}',
            ),
        )
    )

    with pytest.raises(EvidenceValidationError):
        validate_evidence(
            action,
            (observation,),
            visible_observation_ids=(observation.id,),
            root_asset_id=_ROOT,
            asset_root_ids={_ROOT: _ROOT},
        )


def test_answered_requires_evidence_except_not_visible_statement() -> None:
    with pytest.raises(EvidenceValidationError):
        validate_evidence(
            _answer(),
            (),
            visible_observation_ids=(),
            root_asset_id=_ROOT,
            asset_root_ids={_ROOT: _ROOT},
        )

    evidence = validate_evidence(
        _answer(answer="The requested serial number is not visible."),
        (),
        visible_observation_ids=(),
        root_asset_id=_ROOT,
        asset_root_ids={_ROOT: _ROOT},
    )

    assert evidence == ()


def test_materialize_answer_produces_answered_status_and_embedded_snapshot() -> None:
    observation = make_observation(1, TextPayload(text="TOTAL RM 42.50"))
    warning = WarningInfo(code="ocr_used", message="OCR supplied the answer.")
    action = _answer(references=(_reference(observation.id),))

    result = materialize_answer(
        action,
        (observation,),
        visible_observation_ids=(observation.id,),
        root_asset_id=_ROOT,
        asset_root_ids={_ROOT: _ROOT},
        warnings=(warning,),
        trace=make_trace(),
    )

    assert result.status is AnswerStatus.ANSWERED
    assert result.answer == "The total is RM 42.50."
    assert result.evidence[0].observation == observation
    assert result.warnings == (warning,)


def test_materialize_answer_produces_insufficient_evidence_without_citations() -> None:
    action = _answer(
        status="insufficient_evidence",
        answer="The serial number was not established by available observations.",
    )

    result = materialize_answer(
        action,
        (),
        visible_observation_ids=(),
        root_asset_id=_ROOT,
        asset_root_ids={_ROOT: _ROOT},
        warnings=(),
        trace=make_trace(),
    )

    assert result.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert result.evidence == ()
