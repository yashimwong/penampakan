from __future__ import annotations

import pytest

from penampakan.errors import InvalidModelActionError, LLMError
from penampakan.models import AnswerAction, LLMResponse, PolicyInput, ToolAction
from penampakan.reasoning.actions import ActionParseError
from penampakan.reasoning.policy import JsonActionPolicy
from tests.unit.reasoning.helpers import ScriptedTextLLM, make_policy_input


def _repair_input(input: PolicyInput, error: ActionParseError) -> PolicyInput:
    return input.model_copy(
        update={
            "validation_feedback": error.feedback,
            "invalid_model_output": error.invalid_model_output,
        }
    )


@pytest.mark.asyncio
async def test_policy_returns_valid_declared_tool_and_records_one_request() -> None:
    llm = ScriptedTextLLM(
        [
            '{"type":"tool","tool":"read_text",'
            '"arguments":{"asset_id":"img_aaaaaaaaaaaaaaaa"},'
            '"purpose":"Find the receipt total"}'
        ]
    )
    policy = JsonActionPolicy(llm, timeout_s=9.0, max_output_tokens=222)

    action = await policy.next_action(make_policy_input())

    assert isinstance(action, ToolAction)
    assert action.tool == "read_text"
    assert len(llm.requests) == 1
    assert llm.requests[0].timeout_s == 9.0
    assert llm.requests[0].max_output_tokens == 222
    assert llm.remaining == 0


@pytest.mark.asyncio
async def test_one_invalid_action_can_be_repaired_successfully() -> None:
    invalid = "not JSON"
    llm = ScriptedTextLLM(
        [
            invalid,
            '{"type":"answer","status":"insufficient_evidence",'
            '"answer":"The total is not established."}',
        ]
    )
    policy = JsonActionPolicy(llm)
    input = make_policy_input()

    with pytest.raises(ActionParseError) as captured:
        await policy.next_action(input)

    action = await policy.next_action(_repair_input(input, captured.value))

    assert isinstance(action, AnswerAction)
    assert action.status == "insufficient_evidence"
    assert len(llm.requests) == 2
    assert llm.requests[1].metadata["repair"] == "true"
    assert invalid in llm.requests[1].messages[1].content


@pytest.mark.asyncio
async def test_two_invalid_actions_terminate_without_a_third_call() -> None:
    llm = ScriptedTextLLM(["first invalid", "second invalid", "unused"])
    policy = JsonActionPolicy(llm)
    input = make_policy_input()

    with pytest.raises(ActionParseError) as first:
        await policy.next_action(input)
    with pytest.raises(InvalidModelActionError):
        await policy.next_action(_repair_input(input, first.value))

    assert len(llm.requests) == 2
    assert llm.remaining == 1


@pytest.mark.asyncio
async def test_undeclared_tool_uses_the_same_single_repair_boundary() -> None:
    output = (
        '{"type":"tool","tool":"delete_files","arguments":{},"purpose":"Change unrelated state"}'
    )
    llm = ScriptedTextLLM([output, output])
    policy = JsonActionPolicy(llm)
    input = make_policy_input()

    with pytest.raises(ActionParseError) as first:
        await policy.next_action(input)
    assert first.value.feedback[0].code == "undeclared_tool"

    with pytest.raises(InvalidModelActionError):
        await policy.next_action(_repair_input(input, first.value))


@pytest.mark.asyncio
async def test_answer_only_rejects_tool_even_if_input_carries_tool_metadata() -> None:
    output = (
        '{"type":"tool","tool":"read_text",'
        '"arguments":{"asset_id":"img_aaaaaaaaaaaaaaaa"},"purpose":"Read total"}'
    )
    llm = ScriptedTextLLM([output])
    policy = JsonActionPolicy(llm)

    with pytest.raises(ActionParseError) as captured:
        await policy.next_action(make_policy_input(answer_only=True))

    assert captured.value.feedback[0].code == "tool_action_not_allowed"
    assert '"const":"tool"' not in str(llm.requests[0].response_json_schema)


@pytest.mark.asyncio
async def test_provider_errors_are_safely_wrapped_and_llm_errors_preserved() -> None:
    provider_failure = RuntimeError("provider secret")
    wrapped_policy = JsonActionPolicy(ScriptedTextLLM([provider_failure]))
    original = LLMError(code="provider_failure")
    preserved_policy = JsonActionPolicy(ScriptedTextLLM([original]))

    with pytest.raises(LLMError) as wrapped:
        await wrapped_policy.next_action(make_policy_input())
    with pytest.raises(LLMError) as preserved:
        await preserved_policy.next_action(make_policy_input())

    assert wrapped.value.__cause__ is provider_failure
    assert preserved.value is original
    assert "provider secret" not in str(wrapped.value)


@pytest.mark.asyncio
async def test_policy_accepts_explicit_llm_response() -> None:
    response = LLMResponse(
        text=(
            '{"type":"answer","status":"insufficient_evidence",'
            '"answer":"More evidence is required."}'
        ),
        model_id="text-model",
    )
    policy = JsonActionPolicy(ScriptedTextLLM([response]))

    action = await policy.next_action(make_policy_input())

    assert isinstance(action, AnswerAction)
    assert policy.prompt_version == "agent-v1"


def test_policy_constructor_rejects_unsupported_prompt_and_output_limit() -> None:
    llm = ScriptedTextLLM([])

    with pytest.raises(ValueError, match="unsupported prompt"):
        JsonActionPolicy(llm, prompt_version="agent-v2")
    with pytest.raises(ValueError, match="positive"):
        JsonActionPolicy(llm, max_output_tokens=0)
