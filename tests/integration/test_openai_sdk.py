"""Real-SDK shape tests for the OpenAI adapter over a local mock transport."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import pytest

from penampakan.errors import LLMError
from penampakan.llms.openai import OpenAITextLLM
from penampakan.llms.schema import SchemaTarget, compile_action_schema
from penampakan.models import (
    LLMRequest,
    Message,
    MessageRole,
    RetryPolicy,
    SchemaEnforcement,
    TokenUsage,
    ToolSpec,
)
from penampakan.reasoning.prompts import build_action_schema

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("openai") is None or importlib.util.find_spec("httpx") is None,
    reason="the openai extra is not installed",
)

if importlib.util.find_spec("httpx") is not None:  # pragma: no branch - guarded by pytestmark
    import httpx

SYSTEM_PROMPT = "TRUSTED SYSTEM PROMPT for the integration transport."
USER_PROMPT = "TRUSTED QUESTION for the integration transport."
ANSWER_PAYLOAD = (
    '{"action":{"type":"answer","status":"answered","answer":"A red sign.",'
    '"evidence":null,"uncertainties":null}}'
)

_TOOL = ToolSpec(
    name="read_text",
    description="Read visible text from the image.",
    arguments_json_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {"asset_id": {"type": "string"}},
        "required": ["asset_id"],
    },
)


def _schema() -> dict[str, Any]:
    return build_action_schema((_TOOL,))


def _request() -> LLMRequest:
    return LLMRequest(
        messages=(
            Message(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
            Message(role=MessageRole.USER, content=USER_PROMPT),
        ),
        response_json_schema=_schema(),
        max_output_tokens=256,
        timeout_s=10.0,
    )


def _responses_payload(
    *,
    status: str = "completed",
    output: list[dict[str, Any]] | None = None,
    incomplete_details: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a payload matching the real Responses JSON response schema."""
    if output is None:
        output = [
            {
                "id": "msg_68b1c0",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": ANSWER_PAYLOAD, "annotations": []},
                ],
            }
        ]
    return {
        "id": "resp_68b1c0f2a1",
        "object": "response",
        "created_at": 1_752_000_000,
        "model": "gpt-4.1-mini-2025-04-14",
        "status": status,
        "error": None,
        "incomplete_details": incomplete_details,
        "instructions": None,
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 341,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 27,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 368,
        },
    }


_Handler = Callable[["httpx.Request"], "httpx.Response"]


@asynccontextmanager
async def _adapter(handler: _Handler, **kwargs: Any) -> AsyncIterator[tuple[Any, Any]]:
    """Yield an adapter driven by a real SDK client over a mock transport."""
    import openai

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = openai.AsyncOpenAI(api_key="test", max_retries=0, http_client=http_client)
        llm = OpenAITextLLM(model="gpt-4.1-mini", client=client, **kwargs)
        # Deterministic full-jitter source: private test construction only.
        llm._random_source = lambda: 0.0
        try:
            yield llm, http_client
        finally:
            await llm.aclose()


async def test_serialized_request_and_parsed_response_match_the_real_sdk() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_responses_payload())

    async with _adapter(handler) as (llm, http_client):
        response = await llm.complete(_request())
        await llm.aclose()
        # The injected client is caller-owned, so its transport stays usable.
        assert http_client.is_closed is False

    compiled = compile_action_schema(_schema(), target=SchemaTarget.OPENAI_STRICT)
    assert seen["method"] == "POST"
    assert seen["path"] == "/v1/responses"
    body = seen["body"]
    assert body["model"] == "gpt-4.1-mini"
    assert body["max_output_tokens"] == 256
    assert body["temperature"] == 0.0
    assert [item["role"] for item in body["input"]] == ["system", "user"]
    assert [item["content"] for item in body["input"]] == [SYSTEM_PROMPT, USER_PROMPT]
    text_format = body["text"]["format"]
    assert text_format["type"] == "json_schema"
    assert text_format["name"] == "policy_action"
    assert text_format["strict"] is True
    assert text_format["schema"] == compiled.schema

    assert response.text == '{"answer":"A red sign.","status":"answered","type":"answer"}'
    assert response.model_id == "gpt-4.1-mini-2025-04-14"
    assert response.usage == TokenUsage(input_tokens=341, output_tokens=27)
    assert response.finish_reason == "stop"
    assert response.provider == "openai"
    assert response.request_id == "resp_68b1c0f2a1"
    assert response.attempts == 1
    assert response.schema_enforcement is SchemaEnforcement.STRICT
    assert response.backend_fingerprint is not None
    assert response.backend_fingerprint.endswith(compiled.fingerprint_sha256[:16])


async def test_real_refusal_payload_maps_to_a_typed_refusal() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = _responses_payload(
            output=[
                {
                    "id": "msg_68b1c1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "refusal", "refusal": "I cannot help with that."}],
                }
            ]
        )
        return httpx.Response(200, json=payload)

    async with _adapter(handler) as (llm, _):
        with pytest.raises(LLMError) as info:
            await llm.complete(_request())

    assert info.value.code == "llm_refused"
    assert info.value.provider == "openai"


async def test_real_incomplete_payload_maps_to_typed_truncation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = _responses_payload(
            status="incomplete",
            incomplete_details={"reason": "max_output_tokens"},
            output=[
                {
                    "id": "msg_68b1c2",
                    "type": "message",
                    "role": "assistant",
                    "status": "incomplete",
                    "content": [
                        {"type": "output_text", "text": '{"action":{"type":"an', "annotations": []},
                    ],
                }
            ],
        )
        return httpx.Response(200, json=payload)

    async with _adapter(handler) as (llm, _):
        with pytest.raises(LLMError) as info:
            await llm.complete(_request())

    assert info.value.code == "llm_output_truncated"


async def test_real_rate_limit_error_is_retried_once_and_then_succeeds() -> None:
    statuses: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if not statuses:
            statuses.append(429)
            return httpx.Response(
                429,
                json={
                    "error": {
                        "message": "Rate limit reached.",
                        "type": "requests",
                        "code": "rate_limit_exceeded",
                        "param": None,
                    }
                },
            )
        statuses.append(200)
        return httpx.Response(200, json=_responses_payload())

    policy = RetryPolicy(max_attempts=3, base_delay_s=0.001, max_delay_s=0.002)
    async with _adapter(handler, retry_policy=policy) as (llm, _):
        response = await llm.complete(_request())

    assert statuses == [429, 200]
    assert response.attempts == 2


async def test_real_bad_request_error_is_terminal() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "Invalid schema.",
                    "type": "invalid_request_error",
                    "code": "invalid_value",
                    "param": "text.format.schema",
                }
            },
        )

    policy = RetryPolicy(max_attempts=3, base_delay_s=0.001, max_delay_s=0.002)
    async with _adapter(handler, retry_policy=policy) as (llm, _):
        with pytest.raises(LLMError) as info:
            await llm.complete(_request())

    assert calls == ["/v1/responses"]
    assert info.value.code == "llm_request_failed"
    assert info.value.retryable is False
    assert info.value.provider_status == 400
    assert info.value.provider_code == "invalid_value"
