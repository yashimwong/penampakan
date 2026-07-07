from __future__ import annotations

import pytest

from penampakan.models import AnswerAction, ToolAction
from penampakan.reasoning.actions import ActionParseError, parse_action, parse_policy_action


def test_parse_valid_tool_and_answer_actions() -> None:
    tool = parse_policy_action(
        ' \n {"type":"tool","tool":"read_text","arguments":{"asset_id":"img_aaaaaaaaaaaaaaaa"},'
        '"purpose":"Find the printed total"} \t'
    )
    answer = parse_action(
        '{"type":"answer","status":"answered","answer":"RM 42.50",'
        '"evidence":[{"observation_id":"obs_000001","supports":"Printed total"}],'
        '"uncertainties":[]}'
    )

    assert isinstance(tool, ToolAction)
    assert tool.arguments == {"asset_id": "img_aaaaaaaaaaaaaaaa"}
    assert isinstance(answer, AnswerAction)
    assert answer.evidence[0].observation_id == "obs_000001"


@pytest.mark.parametrize(
    "model_output",
    [
        "",
        "   ",
        '```json\n{"type":"answer"}\n```',
        'Here is the action: {"type":"answer"}',
        '{"type":"answer"} trailing',
        '{"type":"answer"}{"type":"answer"}',
        '{"type":"tool","type":"answer"}',
        '{"type":"tool","tool":"read_text","arguments":{"value":NaN},"purpose":"x"}',
        '{"type":"tool","tool":"read_text","arguments":{"value":Infinity},"purpose":"x"}',
        "[]",
        "null",
        '{"type":"unknown"}',
        '{"type":"answer","status":"answered","answer":"x","unknown":true}',
        '{"type":"answer","status":"wrong","answer":"x"}',
        '{"type":"tool","tool":"read_text","arguments":[],"purpose":"x"}',
    ],
)
def test_strict_rejection_matrix(model_output: str) -> None:
    with pytest.raises(ActionParseError) as captured:
        parse_policy_action(model_output)

    assert captured.value.feedback
    assert captured.value.invalid_model_output == model_output


def test_duplicate_and_nonfinite_failures_have_stable_feedback_codes() -> None:
    with pytest.raises(ActionParseError) as duplicate:
        parse_policy_action('{"type":"answer","type":"tool"}')
    with pytest.raises(ActionParseError) as nonfinite:
        parse_policy_action(
            '{"type":"tool","tool":"read_text","arguments":{"x":NaN},"purpose":"x"}'
        )

    assert duplicate.value.feedback[0].code == "duplicate_json_key"
    assert nonfinite.value.feedback[0].code == "non_finite_json_number"


def test_validation_feedback_exposes_safe_locations_without_raw_values() -> None:
    secret = "SENTINEL_PRIVATE_MODEL_OUTPUT"
    output = (
        '{"type":"answer","status":"answered","answer":"ok","evidence":['
        f'{{"observation_id":"bad-{secret}","supports":"x"}}]}}'
    )

    with pytest.raises(ActionParseError) as captured:
        parse_policy_action(output)

    feedback_dump = repr(captured.value.feedback)
    assert "$.evidence[0].observation_id" in feedback_dump
    assert secret not in feedback_dump
    assert secret not in repr(captured.value)


def test_non_ascii_outer_whitespace_is_not_accepted() -> None:
    output = '\u00a0{"type":"answer","status":"insufficient_evidence","answer":"Unknown"}'

    with pytest.raises(ActionParseError):
        parse_policy_action(output)
