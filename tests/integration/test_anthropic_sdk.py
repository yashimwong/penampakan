"""Real-SDK shape tests for the Anthropic adapter using a local mock transport.

These exercise request serialization and response parsing through the actual
``anthropic`` package, with no network and no credentials, so a fake that models
a desired API instead of the real SDK cannot pass.
"""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from penampakan.errors import LLMError
from penampakan.llms._anthropic_transport import TOOL_NAME
from penampakan.llms.anthropic import AnthropicTextLLM
from penampakan.llms.schema import SchemaTarget, compile_action_schema
from penampakan.models import (
    LLMRequest,
    PolicyInput,
    RemainingBudget,
    RetryPolicy,
    SchemaEnforcement,
    TokenUsage,
    ToolSpec,
)
from penampakan.reasoning.prompts import build_policy_request

_HAS_ANTHROPIC = importlib.util.find_spec("anthropic") is not None
_HAS_HTTPX = importlib.util.find_spec("httpx") is not None

pytestmark = pytest.mark.skipif(
    not (_HAS_ANTHROPIC and _HAS_HTTPX),
    reason="requires the anthropic package and its httpx transport",
)

MODEL = "claude-opus-5"
ANSWER_ACTION = {"type": "answer", "status": "answered", "answer": "The sign says STOP."}
CANONICAL_ANSWER = json.dumps(ANSWER_ACTION, separators=(",", ":"), sort_keys=True)


def tool_spec() -> ToolSpec:
    return ToolSpec(
        name="read_text",
        description="Read the text inside a region of an image.",
        arguments_json_schema={
            "type": "object",
            "properties": {"asset_id": {"type": "string"}},
            "required": ["asset_id"],
            "additionalProperties": False,
        },
    )


def policy_request() -> LLMRequest:
    policy_input = PolicyInput(
        question="What does the sign say?",
        context="observation: a red octagonal sign",
        tools=(tool_spec(),),
        prior_actions=(),
        remaining=RemainingBudget(
            steps=3,
            llm_calls=3,
            tool_calls=3,
            backend_calls=3,
            derived_assets=1,
            derivation_depth=1,
            context_chars=20_000,
            remaining_time_s=30.0,
        ),
    )
    return build_policy_request(policy_input, timeout_s=30.0)


def message_payload(
    content: list[dict[str, Any]],
    *,
    stop_reason: str = "end_turn",
    stop_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a response body from the real Messages API response schema."""
    return {
        "id": "msg_01EeWyXxfu5pfWkrYcMdjWG",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": content,
        "stop_reason": stop_reason,
        "stop_details": stop_details,
        "stop_sequence": None,
        "usage": {
            "input_tokens": 1_284,
            "output_tokens": 96,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


@contextmanager
def recording_adapter(
    respond: Callable[[], dict[str, Any]],
    *,
    structured_mode: str,
) -> Iterator[tuple[AnthropicTextLLM, list[Any]]]:
    """Yield an adapter wired to a real client over ``httpx.MockTransport``."""
    import anthropic
    import httpx

    recorded: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        return httpx.Response(200, json=respond())

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = anthropic.AsyncAnthropic(api_key="test", max_retries=0, http_client=http_client)
    llm = AnthropicTextLLM(
        model=MODEL,
        client=client,
        structured_mode=structured_mode,  # type: ignore[arg-type]
        retry_policy=RetryPolicy(max_attempts=1),
        owns_client=True,
    )
    # Closing the adapter closes the SDK client, which closes this transport.
    yield llm, recorded


def sent_body(recorded: list[Any]) -> dict[str, Any]:
    assert len(recorded) == 1
    request = recorded[0]
    assert request.method == "POST"
    assert request.url.path == "/v1/messages"
    body = json.loads(request.content)
    assert isinstance(body, dict)
    return body


async def test_json_output_request_and_response_survive_the_real_sdk() -> None:
    request = policy_request()
    compiled = compile_action_schema(
        request.response_json_schema, target=SchemaTarget.ANTHROPIC_STRICT
    )
    envelope = json.dumps({"action": ANSWER_ACTION})

    with recording_adapter(
        lambda: message_payload([{"type": "text", "text": envelope}]),
        structured_mode="json_output",
    ) as (llm, recorded):
        async with llm:
            response = await llm.complete(request)

    body = sent_body(recorded)
    assert body["model"] == MODEL
    assert body["max_tokens"] == request.max_output_tokens
    assert isinstance(body["system"], str) and body["system"]
    assert [entry["role"] for entry in body["messages"]] == ["user"]
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["output_config"]["format"]["schema"] == compiled.schema
    assert "tools" not in body

    assert response.text == CANONICAL_ANSWER
    assert response.model_id == MODEL
    assert response.request_id == "msg_01EeWyXxfu5pfWkrYcMdjWG"
    assert response.finish_reason == "end_turn"
    assert response.provider == "anthropic"
    assert response.usage == TokenUsage(input_tokens=1_284, output_tokens=96)
    assert response.attempts == 1
    assert response.schema_enforcement is SchemaEnforcement.STRICT
    assert (response.backend_fingerprint or "").startswith("anthropic/json_output/")


async def test_strict_tool_request_and_response_survive_the_real_sdk() -> None:
    request = policy_request()
    compiled = compile_action_schema(
        request.response_json_schema, target=SchemaTarget.ANTHROPIC_STRICT
    )

    def respond() -> dict[str, Any]:
        return message_payload(
            [
                {
                    "type": "tool_use",
                    "id": "toolu_01A09q90qw90lq917835lq9",
                    "name": TOOL_NAME,
                    "input": {"action": ANSWER_ACTION},
                }
            ],
            stop_reason="tool_use",
        )

    with recording_adapter(respond, structured_mode="strict_tool") as (llm, recorded):
        async with llm:
            response = await llm.complete(request)

    body = sent_body(recorded)
    assert isinstance(body["system"], str) and body["system"]
    assert body["max_tokens"] == request.max_output_tokens
    assert len(body["tools"]) == 1
    assert body["tools"][0]["name"] == TOOL_NAME
    assert body["tools"][0]["strict"] is True
    assert body["tools"][0]["input_schema"] == compiled.schema
    assert body["tool_choice"]["type"] == "tool"
    assert body["tool_choice"]["name"] == TOOL_NAME
    assert "output_config" not in body

    assert response.text == CANONICAL_ANSWER
    assert response.finish_reason == "tool_use"
    assert (response.backend_fingerprint or "").startswith("anthropic/strict_tool/")


@pytest.mark.parametrize("structured_mode", ["json_output", "strict_tool"])
async def test_real_sdk_does_not_transform_the_compiled_schema(structured_mode: str) -> None:
    """Pin what the installed SDK actually serializes for the compiled schema."""
    import anthropic

    request = policy_request()
    compiled = compile_action_schema(
        request.response_json_schema, target=SchemaTarget.ANTHROPIC_STRICT
    )
    envelope = json.dumps({"action": ANSWER_ACTION})

    def respond() -> dict[str, Any]:
        if structured_mode == "json_output":
            return message_payload([{"type": "text", "text": envelope}])
        return message_payload(
            [
                {
                    "type": "tool_use",
                    "id": "toolu_01A09q90qw90lq917835lq9",
                    "name": TOOL_NAME,
                    "input": {"action": ANSWER_ACTION},
                }
            ],
            stop_reason="tool_use",
        )

    with recording_adapter(respond, structured_mode=structured_mode) as (llm, recorded):
        async with llm:
            await llm.complete(request)

    body = sent_body(recorded)
    sent = (
        body["output_config"]["format"]["schema"]
        if structured_mode == "json_output"
        else body["tools"][0]["input_schema"]
    )
    version = anthropic.__version__
    assert sent == compiled.schema, f"anthropic {version} rewrote the compiled schema"
    assert json.dumps(sent, sort_keys=True) == json.dumps(compiled.schema, sort_keys=True), (
        f"anthropic {version} dropped part of the compiled schema"
    )
    # The compiler, not the SDK, is what changes the schema: the adapter keeps
    # the original provider-neutral schema for local post-validation.
    assert compiled.schema != request.response_json_schema


@pytest.mark.parametrize("structured_mode", ["json_output", "strict_tool"])
async def test_local_post_validation_uses_the_original_schema(structured_mode: str) -> None:
    request = policy_request()
    invalid = {"action": {"type": "answer", "status": "not-a-status", "answer": "x"}}

    def respond() -> dict[str, Any]:
        if structured_mode == "json_output":
            return message_payload([{"type": "text", "text": json.dumps(invalid)}])
        return message_payload(
            [
                {
                    "type": "tool_use",
                    "id": "toolu_01A09q90qw90lq917835lq9",
                    "name": TOOL_NAME,
                    "input": invalid,
                }
            ],
            stop_reason="tool_use",
        )

    with recording_adapter(respond, structured_mode=structured_mode) as (llm, recorded):
        async with llm:
            with pytest.raises(LLMError) as raised:
                await llm.complete(request)

    assert raised.value.code == "llm_schema_validation_failed"
    assert raised.value.provider == "anthropic"
    assert len(recorded) == 1


async def test_refusal_stop_reason_survives_real_sdk_deserialization() -> None:
    def respond() -> dict[str, Any]:
        return message_payload(
            [],
            stop_reason="refusal",
            stop_details={
                "type": "refusal",
                "category": "cyber",
                "explanation": "declined by policy",
            },
        )

    with recording_adapter(respond, structured_mode="json_output") as (llm, recorded):
        async with llm:
            with pytest.raises(LLMError) as raised:
                await llm.complete(policy_request())

    assert raised.value.code == "llm_refused"
    assert raised.value.retryable is False
    assert "declined by policy" not in f"{raised.value!r}{raised.value}"
    assert len(recorded) == 1


async def test_max_tokens_stop_reason_survives_real_sdk_deserialization() -> None:
    def respond() -> dict[str, Any]:
        return message_payload(
            [{"type": "text", "text": '{"action": {"type": "answer"'}],
            stop_reason="max_tokens",
        )

    with recording_adapter(respond, structured_mode="strict_tool") as (llm, recorded):
        async with llm:
            with pytest.raises(LLMError) as raised:
                await llm.complete(policy_request())

    assert raised.value.code == "llm_output_truncated"
    assert raised.value.retryable is False
    assert len(recorded) == 1
