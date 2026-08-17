from __future__ import annotations

import asyncio
import builtins
import importlib
import importlib.util
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from penampakan.errors import ConfigurationError, LLMError
from penampakan.llms._anthropic_transport import (
    ADAPTER_BODY_KEYS,
    TOOL_NAME,
    classify_failure,
    load_sdk,
    resolve_capability,
    resolve_structured_mode,
)
from penampakan.llms.anthropic import AnthropicTextLLM
from penampakan.models import (
    LLMRequest,
    Message,
    MessageRole,
    PolicyInput,
    RemainingBudget,
    RetryPolicy,
    SchemaEnforcement,
    TokenUsage,
    ToolSpec,
)
from penampakan.reasoning.prompts import build_policy_request

_HAS_ANTHROPIC = importlib.util.find_spec("anthropic") is not None

pytestmark = pytest.mark.skipif(not _HAS_ANTHROPIC, reason="requires the anthropic package")

SUPPORTED_MODEL = "claude-opus-5"
API_KEY = "sk-ant-secret-key-value"
BASE_URL = "https://private-gateway.invalid/anthropic"

ANSWER_ACTION = {"type": "answer", "status": "answered", "answer": "The sign says STOP."}
ANSWER_ENVELOPE = json.dumps({"action": ANSWER_ACTION})
CANONICAL_ANSWER = json.dumps(ANSWER_ACTION, separators=(",", ":"), sort_keys=True)
INVALID_ENVELOPE = json.dumps({"action": {"type": "answer", "status": "maybe"}})


# --------------------------------------------------------------------------- #
# Fakes that mirror the real SDK field names.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FakeTextBlock:
    text: str
    type: str = "text"


@dataclass(frozen=True)
class FakeToolUseBlock:
    name: str
    input: dict[str, Any]
    id: str = "toolu_01"
    type: str = "tool_use"


@dataclass(frozen=True)
class FakeThinkingBlock:
    thinking: str = ""
    type: str = "thinking"


@dataclass(frozen=True)
class FakeUsage:
    input_tokens: int = 412
    output_tokens: int = 57


@dataclass(frozen=True)
class FakeMessage:
    content: Sequence[object]
    id: str = "msg_01XYZabc"
    model: str = SUPPORTED_MODEL
    role: str = "assistant"
    stop_reason: str | None = "end_turn"
    stop_sequence: None = None
    type: str = "message"
    usage: FakeUsage = field(default_factory=FakeUsage)


class FakeMessages:
    """Records every call and replays scripted results or exceptions."""

    def __init__(self, results: Sequence[object]) -> None:
        self._results = list(results)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        result = self._results[min(len(self.calls) - 1, len(self._results) - 1)]
        if isinstance(result, BaseException):
            raise result
        if callable(result):
            return result()
        return result


class FakeClient:
    """A minimal stand-in exposing only `.messages.create` and `.aclose`."""

    def __init__(self, *results: object) -> None:
        self.messages = FakeMessages(results)
        self.closes = 0

    async def aclose(self) -> None:
        self.closes += 1


def text_message(payload: str = ANSWER_ENVELOPE, **overrides: Any) -> FakeMessage:
    return FakeMessage(content=[FakeThinkingBlock(), FakeTextBlock(text=payload)], **overrides)


def tool_message(
    action: dict[str, Any] | None = None,
    *,
    name: str = TOOL_NAME,
    **overrides: Any,
) -> FakeMessage:
    payload = {"action": ANSWER_ACTION} if action is None else action
    return FakeMessage(content=[FakeToolUseBlock(name=name, input=payload)], **overrides)


def status_error(status: int) -> BaseException:
    import anthropic
    import httpx

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(
        status,
        request=request,
        json={"type": "error", "error": {"type": "api_error", "message": "provider prose"}},
    )
    classes: dict[int, Any] = {
        400: anthropic.BadRequestError,
        401: anthropic.AuthenticationError,
        403: anthropic.PermissionDeniedError,
        404: anthropic.NotFoundError,
        429: anthropic.RateLimitError,
        500: anthropic.InternalServerError,
    }
    error_class = classes.get(status, anthropic.APIStatusError)
    return error_class("provider prose", response=response, body=None)


def connection_error() -> BaseException:
    import anthropic
    import httpx

    return anthropic.APIConnectionError(
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    )


def timeout_error() -> BaseException:
    import anthropic
    import httpx

    return anthropic.APITimeoutError(
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    )


# --------------------------------------------------------------------------- #
# Request fixtures built from the library itself.
# --------------------------------------------------------------------------- #


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


def policy_request(*, timeout_s: float | None = None) -> LLMRequest:
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
    return build_policy_request(policy_input, timeout_s=timeout_s)


def adapter(
    client: FakeClient,
    *,
    structured_mode: Any = "auto",
    model: str = SUPPORTED_MODEL,
    retry_policy: RetryPolicy | None = None,
    owns_client: bool | None = None,
    extra_body: dict[str, object] | None = None,
    random_source: Callable[[], float] | None = None,
) -> AnthropicTextLLM:
    llm = AnthropicTextLLM(
        model=model,
        client=client,
        structured_mode=structured_mode,
        retry_policy=retry_policy,
        owns_client=owns_client,
        extra_body=extra_body,
    )
    if random_source is not None:
        # Deterministic jitter is supplied privately, exactly as the retry
        # contract documents; the public wire contract exposes no callables.
        llm._random_source = random_source
    return llm


# --------------------------------------------------------------------------- #
# Happy paths and metadata propagation.
# --------------------------------------------------------------------------- #


async def test_json_output_mode_populates_all_response_metadata() -> None:
    client = FakeClient(text_message())
    llm = adapter(client)
    request = policy_request()

    response = await llm.complete(request)

    assert llm.structured_mode == "json_output"
    assert response.text == CANONICAL_ANSWER
    assert response.model_id == SUPPORTED_MODEL
    assert response.usage == TokenUsage(input_tokens=412, output_tokens=57)
    assert response.finish_reason == "end_turn"
    assert response.provider == "anthropic"
    assert response.request_id == "msg_01XYZabc"
    assert response.attempts == 1
    assert response.schema_enforcement is SchemaEnforcement.STRICT
    assert response.backend_fingerprint is not None
    assert response.backend_fingerprint.startswith("anthropic/json_output/")

    call = client.messages.calls[0]
    assert call["model"] == SUPPORTED_MODEL
    assert call["max_tokens"] == request.max_output_tokens
    assert call["temperature"] == request.temperature
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["output_config"]["format"]["schema"]["required"] == ["action"]
    assert "tools" not in call
    assert "tool_choice" not in call


async def test_strict_tool_mode_forces_one_strict_tool() -> None:
    client = FakeClient(tool_message(stop_reason="tool_use"))
    llm = adapter(client, structured_mode="strict_tool")

    response = await llm.complete(policy_request())

    assert llm.structured_mode == "strict_tool"
    assert response.text == CANONICAL_ANSWER
    assert response.finish_reason == "tool_use"
    assert response.attempts == 1
    assert response.backend_fingerprint is not None
    assert response.backend_fingerprint.startswith("anthropic/strict_tool/")

    call = client.messages.calls[0]
    assert call["tools"][0]["name"] == TOOL_NAME
    assert call["tools"][0]["strict"] is True
    assert call["tools"][0]["input_schema"]["required"] == ["action"]
    assert call["tool_choice"] == {"type": "tool", "name": TOOL_NAME}
    assert "output_config" not in call


async def test_mode_is_observable_in_the_fingerprint_for_both_paths() -> None:
    json_client = FakeClient(text_message())
    tool_client = FakeClient(tool_message())
    json_llm = adapter(json_client, structured_mode="json_output")
    tool_llm = adapter(tool_client, structured_mode="strict_tool")

    json_response = await json_llm.complete(policy_request())
    tool_response = await tool_llm.complete(policy_request())

    json_fingerprint = json_response.backend_fingerprint or ""
    tool_fingerprint = tool_response.backend_fingerprint or ""
    assert json_fingerprint.split("/")[1] == "json_output"
    assert tool_fingerprint.split("/")[1] == "strict_tool"
    # Same compiled schema, different strict path.
    assert json_fingerprint.split("/")[3] == tool_fingerprint.split("/")[3]
    assert json_fingerprint.strip() == json_fingerprint


async def test_system_instruction_uses_the_top_level_field_only() -> None:
    client = FakeClient(text_message())
    llm = adapter(client)
    request = policy_request()
    system_message = next(m for m in request.messages if m.role is MessageRole.SYSTEM)

    await llm.complete(request)

    call = client.messages.calls[0]
    assert isinstance(call["system"], str)
    assert call["system"] == system_message.content
    assert [entry["role"] for entry in call["messages"]] == ["user"]
    for entry in call["messages"]:
        assert entry["content"] != system_message.content
        assert system_message.content not in entry["content"]


async def test_multiple_system_messages_are_joined_into_the_system_field() -> None:
    client = FakeClient(text_message())
    llm = adapter(client)
    request = LLMRequest(
        messages=(
            Message(role=MessageRole.SYSTEM, content="first rule"),
            Message(role=MessageRole.SYSTEM, content="second rule"),
            Message(role=MessageRole.USER, content="question"),
        ),
        response_json_schema=policy_request().response_json_schema,
    )

    await llm.complete(request)

    assert client.messages.calls[0]["system"] == "first rule\n\nsecond rule"


async def test_request_without_a_conversation_message_is_rejected() -> None:
    client = FakeClient(text_message())
    llm = adapter(client)
    request = LLMRequest(
        messages=(Message(role=MessageRole.SYSTEM, content="only a system rule"),),
        response_json_schema=policy_request().response_json_schema,
    )

    with pytest.raises(ConfigurationError) as raised:
        await llm.complete(request)

    assert raised.value.code == "unsupported_request_messages"
    assert client.messages.calls == []


# --------------------------------------------------------------------------- #
# Distinct typed provider outcomes.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", ["json_output", "strict_tool"])
async def test_refusal_is_a_distinct_terminal_error(mode: str) -> None:
    client = FakeClient(FakeMessage(content=[], stop_reason="refusal"))
    llm = adapter(client, structured_mode=mode)

    with pytest.raises(LLMError) as raised:
        await llm.complete(policy_request())

    assert raised.value.code == "llm_refused"
    assert raised.value.retryable is False
    assert raised.value.provider == "anthropic"
    assert raised.value.attempts == 1
    assert len(client.messages.calls) == 1


@pytest.mark.parametrize("stop_reason", ["max_tokens", "model_context_window_exceeded"])
async def test_truncation_is_a_distinct_terminal_error(stop_reason: str) -> None:
    client = FakeClient(text_message(stop_reason=stop_reason))
    llm = adapter(client)

    with pytest.raises(LLMError) as raised:
        await llm.complete(policy_request())

    assert raised.value.code == "llm_output_truncated"
    assert raised.value.retryable is False
    assert len(client.messages.calls) == 1


async def test_missing_text_block_in_json_output_mode_is_typed() -> None:
    client = FakeClient(FakeMessage(content=[FakeThinkingBlock()]))
    llm = adapter(client, structured_mode="json_output")

    with pytest.raises(LLMError) as raised:
        await llm.complete(policy_request())

    assert raised.value.code == "llm_missing_structured_output"
    assert raised.value.retryable is False


async def test_missing_tool_use_block_in_strict_tool_mode_is_typed() -> None:
    client = FakeClient(tool_message(name="some_other_tool"))
    llm = adapter(client, structured_mode="strict_tool")

    with pytest.raises(LLMError) as raised:
        await llm.complete(policy_request())

    assert raised.value.code == "llm_missing_structured_output"


async def test_text_only_response_in_strict_tool_mode_is_missing_output() -> None:
    client = FakeClient(text_message())
    llm = adapter(client, structured_mode="strict_tool")

    with pytest.raises(LLMError) as raised:
        await llm.complete(policy_request())

    assert raised.value.code == "llm_missing_structured_output"


async def test_non_object_tool_input_is_invalid_structured_output() -> None:
    client = FakeClient(
        FakeMessage(content=[FakeToolUseBlock(name=TOOL_NAME, input=None)])  # type: ignore[arg-type]
    )
    llm = adapter(client, structured_mode="strict_tool")

    with pytest.raises(LLMError) as raised:
        await llm.complete(policy_request())

    assert raised.value.code == "llm_invalid_structured_output"


@pytest.mark.parametrize("mode", ["json_output", "strict_tool"])
async def test_local_post_validation_rejects_schema_violating_output(mode: str) -> None:
    invalid = json.loads(INVALID_ENVELOPE)
    result = (
        text_message(INVALID_ENVELOPE)
        if mode == "json_output"
        else tool_message(invalid, stop_reason="tool_use")
    )
    client = FakeClient(result)
    llm = adapter(client, structured_mode=mode)

    with pytest.raises(LLMError) as raised:
        await llm.complete(policy_request())

    # Local validation runs even after provider strict enforcement.
    assert raised.value.code == "llm_schema_validation_failed"
    assert raised.value.provider == "anthropic"
    assert raised.value.attempts == 1


async def test_non_json_text_output_is_invalid_structured_output() -> None:
    client = FakeClient(text_message("not json at all"))
    llm = adapter(client)

    with pytest.raises(LLMError) as raised:
        await llm.complete(policy_request())

    assert raised.value.code == "llm_invalid_structured_output"


async def test_partial_usage_and_unsafe_identifiers_degrade_to_none() -> None:
    client = FakeClient(
        FakeMessage(
            content=[FakeTextBlock(text=ANSWER_ENVELOPE)],
            id="msg id with spaces",
            model="",
            stop_reason="END_TURN",
            usage=FakeUsage(input_tokens=-1, output_tokens=3),
        )
    )
    llm = adapter(client)

    response = await llm.complete(policy_request())

    assert response.request_id is None
    assert response.model_id is None
    assert response.finish_reason is None
    assert response.usage == TokenUsage(input_tokens=None, output_tokens=3)


# --------------------------------------------------------------------------- #
# Capability table.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-5",
        "claude-opus-5-20260101",
        "claude-opus-4-8",
        "claude-fable-5",
        "claude-mythos-5",
        "claude-sonnet-5",
        "claude-haiku-4-5",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-5",
        "claude-opus-4-5@20251101",
        "claude-opus-4-1",
        "anthropic.claude-opus-5",
        "us.anthropic.claude-opus-5-v1:0",
    ],
)
def test_supported_models_prefer_native_json_output(model: str) -> None:
    capability = resolve_capability(model)

    assert capability.json_output is True
    assert capability.strict_tools is True
    assert resolve_structured_mode(model, "auto") == "json_output"
    assert resolve_structured_mode(model, "strict_tool") == "strict_tool"


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-4-6",
        "claude-sonnet-4-5",
        "claude-3-opus-20240229",
        "claude-2.1",
        "claude-something-unknown",
        "not-a-claude-model",
    ],
)
def test_unsupported_models_declare_no_strict_path(model: str) -> None:
    capability = resolve_capability(model)

    assert capability.json_output is False
    assert capability.strict_tools is False


@pytest.mark.parametrize("model", ["claude-opus-4-6", "claude-3-opus-20240229", "gpt-4"])
@pytest.mark.parametrize("mode", ["auto", "json_output", "strict_tool"])
def test_unsupported_model_fails_construction(model: str, mode: str) -> None:
    with pytest.raises(ConfigurationError) as raised:
        AnthropicTextLLM(model=model, client=FakeClient(), structured_mode=mode)  # type: ignore[arg-type]

    assert raised.value.code == "strict_output_unsupported"


def test_explicit_mode_overrides_the_table_preference() -> None:
    client = FakeClient()

    assert adapter(client, structured_mode="strict_tool").structured_mode == "strict_tool"
    assert adapter(client, structured_mode="json_output").structured_mode == "json_output"
    assert adapter(client, structured_mode="auto").structured_mode == "json_output"


def test_invalid_structured_mode_is_rejected() -> None:
    with pytest.raises(ConfigurationError) as raised:
        AnthropicTextLLM(
            model=SUPPORTED_MODEL,
            client=FakeClient(),
            structured_mode="degraded",  # type: ignore[arg-type]
        )

    assert raised.value.code == "invalid_structured_mode"


def test_invalid_model_is_rejected() -> None:
    with pytest.raises(ConfigurationError) as raised:
        AnthropicTextLLM(model="   ", client=FakeClient())

    assert raised.value.code == "invalid_model"


def test_client_without_messages_create_is_rejected() -> None:
    with pytest.raises(ConfigurationError) as raised:
        AnthropicTextLLM(model=SUPPORTED_MODEL, client=object())

    assert raised.value.code == "invalid_client"


# --------------------------------------------------------------------------- #
# extra_body.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("key", sorted(ADAPTER_BODY_KEYS))
def test_extra_body_conflicting_with_adapter_keys_is_rejected(key: str) -> None:
    with pytest.raises(ConfigurationError) as raised:
        AnthropicTextLLM(
            model=SUPPORTED_MODEL,
            client=FakeClient(),
            extra_body={key: "hijacked"},
        )

    assert raised.value.code == "conflicting_extra_body"


async def test_extra_body_passes_caller_options_through() -> None:
    client = FakeClient(text_message())
    llm = adapter(client, extra_body={"service_tier": "standard_only"})

    await llm.complete(policy_request())

    assert client.messages.calls[0]["extra_body"] == {"service_tier": "standard_only"}


# --------------------------------------------------------------------------- #
# Retries, deadlines, and classification.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "failure_factory",
    [
        lambda: status_error(429),
        lambda: status_error(500),
        lambda: status_error(529),
        connection_error,
        timeout_error,
    ],
    ids=["rate_limited", "server_error", "overloaded", "connection", "timeout"],
)
async def test_retryable_failures_are_retried_and_counted(
    failure_factory: Callable[[], BaseException],
) -> None:
    failure = failure_factory()
    client = FakeClient(failure, text_message())
    llm = adapter(
        client,
        retry_policy=RetryPolicy(max_attempts=3, base_delay_s=0.01, max_delay_s=0.01),
        random_source=lambda: 0.0,
    )

    response = await llm.complete(policy_request())

    assert response.attempts == 2
    assert len(client.messages.calls) == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404])
async def test_terminal_status_codes_are_not_retried(status: int) -> None:
    client = FakeClient(status_error(status))
    llm = adapter(
        client,
        retry_policy=RetryPolicy(max_attempts=3, base_delay_s=0.01, max_delay_s=0.01),
        random_source=lambda: 0.0,
    )

    with pytest.raises(LLMError) as raised:
        await llm.complete(policy_request())

    assert raised.value.code == "llm_request_failed"
    assert raised.value.retryable is False
    assert raised.value.attempts == 1
    assert raised.value.provider_status == status
    assert len(client.messages.calls) == 1


async def test_retries_are_exhausted_with_a_safe_attempt_count() -> None:
    client = FakeClient(status_error(500))
    llm = adapter(
        client,
        retry_policy=RetryPolicy(max_attempts=3, base_delay_s=0.01, max_delay_s=0.01),
        random_source=lambda: 0.0,
    )

    with pytest.raises(LLMError) as raised:
        await llm.complete(policy_request())

    assert raised.value.code == "llm_retries_exhausted"
    assert raised.value.attempts == 3
    assert raised.value.provider_status == 500
    assert len(client.messages.calls) == 3


async def test_deadline_prevents_a_retry_that_cannot_fit() -> None:
    client = FakeClient(status_error(429))
    llm = adapter(
        client,
        retry_policy=RetryPolicy(max_attempts=4, base_delay_s=1.0, max_delay_s=1.0),
        random_source=lambda: 1.0,
    )

    with pytest.raises(LLMError) as raised:
        await llm.complete(policy_request(timeout_s=0.2))

    assert raised.value.code == "llm_timeout"
    assert raised.value.attempts == 1
    assert len(client.messages.calls) == 1


async def test_sdk_timeout_follows_the_remaining_total_budget() -> None:
    bounded = FakeClient(text_message())
    unbounded = FakeClient(text_message())

    await adapter(bounded).complete(policy_request(timeout_s=5.0))
    await adapter(unbounded).complete(policy_request())

    timeout = bounded.messages.calls[0]["timeout"]
    assert 0.0 < timeout <= 5.0
    assert "timeout" not in unbounded.messages.calls[0]


def test_classification_never_carries_provider_prose() -> None:
    import anthropic

    sdk = load_sdk(anthropic)

    rate_limited = classify_failure(status_error(429), sdk=sdk)
    assert rate_limited.retryable is True
    assert rate_limited.status == 429
    assert rate_limited.code == "rate_limited"

    denied = classify_failure(status_error(403), sdk=sdk)
    assert denied.retryable is False
    assert denied.code == "permission_denied"

    unknown = classify_failure(RuntimeError("provider prose"), sdk=sdk)
    assert unknown.retryable is False
    assert unknown.code == "provider_error"
    assert unknown.status is None

    assert classify_failure(TimeoutError(), sdk=sdk).code == "timeout"
    assert classify_failure(connection_error(), sdk=sdk).code == "connection_failed"


# --------------------------------------------------------------------------- #
# Ownership and lifecycle.
# --------------------------------------------------------------------------- #


async def test_injected_client_stays_caller_owned() -> None:
    client = FakeClient(text_message())
    llm = adapter(client)

    await llm.complete(policy_request())
    await llm.aclose()

    assert client.closes == 0


async def test_owns_client_override_closes_an_injected_client_once() -> None:
    client = FakeClient(text_message())
    llm = adapter(client, owns_client=True)

    await asyncio.gather(llm.aclose(), llm.aclose(), llm.aclose())

    assert client.closes == 1


async def test_async_context_manager_closes_only_owned_clients() -> None:
    caller_owned = FakeClient(text_message())
    adapter_owned = FakeClient(text_message())

    async with adapter(caller_owned) as llm:
        assert isinstance(llm, AnthropicTextLLM)
    async with adapter(adapter_owned, owns_client=True):
        pass

    assert caller_owned.closes == 0
    assert adapter_owned.closes == 1


async def test_completion_after_close_is_rejected() -> None:
    client = FakeClient(text_message())
    llm = adapter(client, owns_client=True)

    await llm.aclose()

    with pytest.raises(LLMError) as raised:
        await llm.complete(policy_request())

    assert raised.value.code == "llm_closed"
    assert client.messages.calls == []


def test_constructed_client_disables_native_retries() -> None:
    recorded: list[dict[str, Any]] = []

    class RecordingFactory:
        def __call__(self, **kwargs: Any) -> FakeClient:
            recorded.append(kwargs)
            return FakeClient()

    llm = AnthropicTextLLM(model=SUPPORTED_MODEL, client=FakeClient())
    llm._sdk = load_sdk_with_factory(RecordingFactory())
    resolved = llm._resolve_client(None, api_key=API_KEY, base_url=BASE_URL)

    assert isinstance(resolved, FakeClient)
    assert recorded == [{"max_retries": 0, "api_key": API_KEY, "base_url": BASE_URL}]


def load_sdk_with_factory(factory: Callable[..., object]) -> Any:
    import anthropic

    from penampakan.llms._anthropic_transport import AnthropicSDK

    sdk = load_sdk(anthropic)
    return AnthropicSDK(
        version=sdk.version,
        client_factory=factory,
        timeout_errors=sdk.timeout_errors,
        connection_errors=sdk.connection_errors,
    )


def test_injected_client_rejects_endpoint_options() -> None:
    with pytest.raises(ConfigurationError) as raised:
        AnthropicTextLLM(model=SUPPORTED_MODEL, client=FakeClient(), api_key=API_KEY)

    assert raised.value.code == "conflicting_client_options"


# --------------------------------------------------------------------------- #
# Redaction.
# --------------------------------------------------------------------------- #


def error_surface(error: Exception) -> str:
    parts = [str(error), repr(error), *(str(value) for value in vars(error).values())]
    parts.extend(str(argument) for argument in error.args)
    return "\n".join(parts)


async def test_raised_errors_carry_no_prompt_schema_or_credential_text() -> None:
    request = policy_request()
    secrets = (
        API_KEY,
        BASE_URL,
        "provider prose",
        request.messages[0].content,
        request.messages[1].content,
        "response_json_schema",
        "additionalProperties",
    )

    class LeakyMessages:
        async def create(self, **kwargs: Any) -> object:
            raise RuntimeError(f"{API_KEY} {BASE_URL} {kwargs['system']}")

    class LeakyClient:
        def __init__(self) -> None:
            self.messages = LeakyMessages()

    transport_llm = AnthropicTextLLM(model=SUPPORTED_MODEL, client=LeakyClient())
    with pytest.raises(LLMError) as transport_failure:
        await transport_llm.complete(request)

    validation_llm = adapter(FakeClient(text_message(INVALID_ENVELOPE)))
    with pytest.raises(LLMError) as validation_failure:
        await validation_llm.complete(request)

    with pytest.raises(ConfigurationError) as configuration_failure:
        AnthropicTextLLM(model="claude-3-opus-20240229", client=FakeClient())

    for error in (
        transport_failure.value,
        validation_failure.value,
        configuration_failure.value,
    ):
        surface = error_surface(error)
        for secret in secrets:
            assert secret not in surface
        assert error.cause_summary in {None, "RuntimeError", "cause details redacted"}


# --------------------------------------------------------------------------- #
# Optional dependency.
# --------------------------------------------------------------------------- #


def test_module_imports_and_construction_fails_without_the_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("penampakan.llms.anthropic")
    real_import = builtins.__import__

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "anthropic" or name.startswith("anthropic."):
            raise ImportError("No module named 'anthropic'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    monkeypatch.delitem(importlib.import_module("sys").modules, "anthropic", raising=False)
    reloaded = importlib.reload(module)

    with pytest.raises(ConfigurationError) as raised:
        reloaded.AnthropicTextLLM(model=SUPPORTED_MODEL)

    assert raised.value.code == "missing_optional_dependency"
    assert raised.value.cause_summary == "cause details redacted"

    monkeypatch.undo()
    importlib.reload(module)
