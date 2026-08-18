from __future__ import annotations

from typing import cast

import pytest

from penampakan.errors import InvalidModelActionError, LLMError
from penampakan.models import (
    AnswerAction,
    LLMResponse,
    PolicyInput,
    SchemaEnforcement,
    ToolAction,
)
from penampakan.reasoning import supported_prompt_versions
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


def test_policy_accepts_every_supported_prompt_version() -> None:
    llm = ScriptedTextLLM([])

    for version in supported_prompt_versions():
        policy = JsonActionPolicy(llm, prompt_version=version)
        assert policy.prompt_version == version


class _ClosableTextLLM(ScriptedTextLLM):
    """A scripted language model that counts its close calls."""

    def __init__(self, responses: list[str | LLMResponse]) -> None:
        super().__init__(responses)
        self.close_calls = 0

    async def aclose(self) -> None:
        """Record one close request."""

        self.close_calls += 1


def _answer(text: str = "More evidence is required.") -> str:
    return f'{{"type":"answer","status":"insufficient_evidence","answer":"{text}"}}'


@pytest.mark.asyncio
async def test_json_only_enforcement_is_reported_once_as_typed_policy_state() -> None:
    degraded = LLMResponse(text=_answer(), schema_enforcement=SchemaEnforcement.JSON_ONLY)
    policy = JsonActionPolicy(ScriptedTextLLM([degraded, degraded]))

    assert policy.degradations == ()

    await policy.next_action(make_policy_input())
    await policy.next_action(make_policy_input())

    assert len(policy.degradations) == 1
    warning = policy.degradations[0]
    assert warning.code == "degraded_schema_enforcement"
    assert warning.details == {"schema_enforcement": "json_only"}


@pytest.mark.asyncio
async def test_strict_enforcement_reports_no_degradation() -> None:
    policy = JsonActionPolicy(ScriptedTextLLM([_answer()]))

    await policy.next_action(make_policy_input())

    assert policy.degradations == ()


@pytest.mark.asyncio
async def test_policy_closes_an_owned_language_model_exactly_once() -> None:
    llm = _ClosableTextLLM([])
    policy = JsonActionPolicy(llm, owns_llm=True)

    assert policy.owns_llm is True
    await policy.aclose()
    await policy.aclose()

    assert llm.close_calls == 1


@pytest.mark.asyncio
async def test_policy_leaves_a_caller_owned_language_model_open() -> None:
    llm = _ClosableTextLLM([])
    policy = JsonActionPolicy(llm)

    assert policy.owns_llm is False
    async with policy:
        pass

    assert llm.close_calls == 0


@pytest.mark.asyncio
async def test_policy_context_manager_closes_an_owned_language_model() -> None:
    llm = _ClosableTextLLM([_answer()])

    async with JsonActionPolicy(llm, owns_llm=True) as policy:
        await policy.next_action(make_policy_input())

    assert llm.close_calls == 1


@pytest.mark.asyncio
async def test_policy_forwards_a_configured_temperature() -> None:
    llm = ScriptedTextLLM([_answer()])
    policy = JsonActionPolicy(llm, temperature=1.0)

    await policy.next_action(make_policy_input())

    assert llm.requests[0].temperature == 1.0


@pytest.mark.asyncio
async def test_policy_defaults_to_a_deterministic_temperature() -> None:
    llm = ScriptedTextLLM([_answer()])

    await JsonActionPolicy(llm).next_action(make_policy_input())

    assert llm.requests[0].temperature == 0.0


def test_policy_constructor_rejects_invalid_ownership_and_temperature() -> None:
    llm = ScriptedTextLLM([])

    with pytest.raises(TypeError, match="owns_llm"):
        JsonActionPolicy(llm, owns_llm=cast(bool, "yes"))
    with pytest.raises(TypeError, match="temperature"):
        JsonActionPolicy(llm, temperature=cast(float, "hot"))
