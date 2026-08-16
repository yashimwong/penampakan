"""Contract tests running the same action cases through every provider adapter.

Each adapter is driven through a provider-shaped fake so one table of valid and
invalid actions, plus LiteLLM's JSON-only degradation, is enforced identically
across providers.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Protocol, cast

import pytest

from penampakan.errors import ConfigurationError, LLMError
from penampakan.llms import AnthropicTextLLM, LiteLLMTextLLM, OpenAITextLLM
from penampakan.llms import litellm as litellm_module
from penampakan.models import (
    Capability,
    JsonValue,
    LLMRequest,
    LLMResponse,
    PolicyInput,
    RemainingBudget,
    RetryPolicy,
    SchemaEnforcement,
    ToolSpec,
)
from penampakan.perception.registry import ToolRegistry
from penampakan.reasoning.prompts import build_policy_request
from penampakan.tools.builtin import register_transform_tools
from penampakan.tools.vision import register_vision_tools

pytestmark = pytest.mark.providers

_ASSET_ID = "img_" + "a" * 16


class _Adapter(Protocol):
    """The provider-neutral surface every adapter contract test uses."""

    async def complete(self, request: LLMRequest) -> LLMResponse: ...

    async def aclose(self) -> None: ...


def _request() -> LLMRequest:
    registry = ToolRegistry()
    register_vision_tools(registry, set(Capability))
    register_transform_tools(registry)
    return build_policy_request(
        PolicyInput(
            question="What is the receipt total?",
            context='{"id":"obs_000001","type":"text","text":"RM 42.50"}',
            tools=cast(tuple[ToolSpec, ...], registry.specs),
            prior_actions=(),
            remaining=RemainingBudget(
                steps=3,
                llm_calls=3,
                tool_calls=3,
                backend_calls=3,
                derived_assets=3,
                derivation_depth=2,
                context_chars=4_000,
                remaining_time_s=30.0,
            ),
        )
    )


def _envelope(action: JsonValue) -> str:
    return json.dumps({"action": action})


_VALID_ACTIONS: tuple[tuple[str, JsonValue], ...] = (
    (
        "answer",
        {
            "type": "answer",
            "status": "answered",
            "answer": "The total is RM 42.50.",
            "evidence": [{"observation_id": "obs_000001", "supports": "The printed total."}],
            "uncertainties": [],
        },
    ),
    (
        "abstention",
        {"type": "answer", "status": "insufficient_evidence", "answer": "The total is unclear."},
    ),
    (
        "answer_with_nullable_optionals",
        {
            "type": "answer",
            "status": "answered",
            "answer": "The total is RM 42.50.",
            "evidence": None,
            "uncertainties": None,
        },
    ),
    (
        "tool",
        {
            "type": "tool",
            "tool": "read_text",
            "arguments": {"asset_id": _ASSET_ID, "region": None},
            "purpose": "Read the printed total.",
        },
    ),
)

_INVALID_ACTIONS: tuple[tuple[str, JsonValue], ...] = (
    ("unknown_discriminator", {"type": "guess", "answer": "x"}),
    ("undeclared_tool", {"type": "tool", "tool": "not_declared", "arguments": {}, "purpose": "x"}),
    (
        "surplus_property",
        {"type": "answer", "status": "answered", "answer": "x", "surplus": True},
    ),
    ("invalid_enum", {"type": "answer", "status": "maybe", "answer": "x"}),
    ("empty_string", {"type": "answer", "status": "answered", "answer": ""}),
    (
        "invalid_pattern",
        {
            "type": "answer",
            "status": "answered",
            "answer": "x",
            "evidence": [{"observation_id": "not-an-id", "supports": "x"}],
        },
    ),
    (
        "wrong_type",
        {"type": "answer", "status": "answered", "answer": "x", "uncertainties": "not-a-list"},
    ),
    ("missing_required", {"type": "answer", "status": "answered"}),
)


def _openai_response(payload: str) -> object:
    return SimpleNamespace(
        id="resp_contract",
        model="gpt-4.1",
        status="completed",
        incomplete_details=None,
        output=[
            SimpleNamespace(
                type="message",
                role="assistant",
                status="completed",
                content=[SimpleNamespace(type="output_text", text=payload)],
            )
        ],
        output_text=payload,
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
    )


def _anthropic_response(payload: str) -> object:
    return SimpleNamespace(
        id="msg_contract",
        model="claude-opus-5",
        role="assistant",
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text=payload)],
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
    )


def _litellm_response(payload: str) -> object:
    return SimpleNamespace(
        id="chatcmpl-contract",
        model="gpt-4o",
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=payload, refusal=None),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
    )


class _ProviderStatusError(Exception):
    """A provider exception carrying only an HTTP status code."""

    def __init__(self, status_code: int) -> None:
        super().__init__("provider status")
        self.status_code = status_code


_RETRY_POLICY = RetryPolicy(max_attempts=2, base_delay_s=0.001, max_delay_s=0.002)


def _openai_adapter(payload: str) -> _Adapter:
    async def create(**_kwargs: object) -> object:
        return _openai_response(payload)

    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    return OpenAITextLLM(model="gpt-4.1", client=client, retry_policy=RetryPolicy(max_attempts=1))


def _openai_failing_adapter(status: int) -> tuple[_Adapter, list[int]]:
    attempts: list[int] = []

    async def create(**_kwargs: object) -> object:
        attempts.append(status)
        raise _ProviderStatusError(status)

    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    adapter = OpenAITextLLM(model="gpt-4.1", client=client, retry_policy=_RETRY_POLICY)
    adapter._random_source = lambda: 0.0
    return adapter, attempts


def _anthropic_failing_adapter(status: int) -> tuple[_Adapter, list[int]]:
    attempts: list[int] = []

    async def create(**_kwargs: object) -> object:
        attempts.append(status)
        raise _ProviderStatusError(status)

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    adapter = AnthropicTextLLM(model="claude-opus-5", client=client, retry_policy=_RETRY_POLICY)
    adapter._random_source = lambda: 0.0
    return adapter, attempts


def _anthropic_adapter(payload: str) -> _Adapter:
    async def create(**_kwargs: object) -> object:
        return _anthropic_response(payload)

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    return AnthropicTextLLM(
        model="claude-opus-5",
        client=client,
        retry_policy=RetryPolicy(max_attempts=1),
    )


@dataclass(frozen=True, slots=True)
class _LiteLLMFactory:
    """Build LiteLLM adapters over a fake runtime with declared capabilities."""

    monkeypatch: pytest.MonkeyPatch
    strict: bool

    def __call__(self, payload: str, *, allow_json_only: bool = False) -> _Adapter:
        async def acompletion(**_kwargs: object) -> object:
            return _litellm_response(payload)

        runtime = litellm_module._Runtime(
            acompletion=acompletion,
            supported_params=lambda _model: ["response_format", "max_tokens"],
            supports_response_schema=lambda _model: self.strict,
            version="1.96.2",
            retryable_errors=(),
            terminal_errors=(),
        )
        self.monkeypatch.setattr(litellm_module, "_resolve_runtime", lambda: runtime)
        return LiteLLMTextLLM(
            model="gpt-4o",
            allow_json_only=allow_json_only,
            retry_policy=RetryPolicy(max_attempts=1),
        )


AdapterFactory = Callable[[str], "_Adapter"]


@pytest.fixture
def strict_adapters(monkeypatch: pytest.MonkeyPatch) -> dict[str, AdapterFactory]:
    """Return one factory per adapter that enforces the schema strictly."""
    litellm_factory = _LiteLLMFactory(monkeypatch, strict=True)
    return {
        "openai": _openai_adapter,
        "anthropic": _anthropic_adapter,
        "litellm": cast(AdapterFactory, litellm_factory),
    }


@pytest.fixture
def request_payload() -> LLMRequest:
    """Return one real policy request compiled from the library's tool set."""
    return _request()


def _adapter_names() -> Iterator[str]:
    yield from ("openai", "anthropic", "litellm")


@pytest.mark.parametrize("provider", list(_adapter_names()))
@pytest.mark.parametrize("case", _VALID_ACTIONS, ids=[name for name, _ in _VALID_ACTIONS])
async def test_valid_actions_are_accepted_by_every_strict_adapter(
    strict_adapters: dict[str, AdapterFactory],
    request_payload: LLMRequest,
    provider: str,
    case: tuple[str, JsonValue],
) -> None:
    _name, action = case
    adapter = strict_adapters[provider](_envelope(action))
    try:
        response = await adapter.complete(request_payload)
    finally:
        await adapter.aclose()

    parsed = json.loads(response.text)
    assert parsed["type"] == cast(dict[str, JsonValue], action)["type"]
    # Optional properties expressed as nullable unions are pruned back out.
    assert all(value is not None for value in parsed.values())
    assert response.provider == provider
    assert response.schema_enforcement is SchemaEnforcement.STRICT
    assert response.attempts == 1
    assert response.backend_fingerprint is not None
    assert response.request_id is not None
    assert response.usage is not None


@pytest.mark.parametrize("provider", list(_adapter_names()))
@pytest.mark.parametrize("case", _INVALID_ACTIONS, ids=[name for name, _ in _INVALID_ACTIONS])
async def test_invalid_actions_are_rejected_identically(
    strict_adapters: dict[str, AdapterFactory],
    request_payload: LLMRequest,
    provider: str,
    case: tuple[str, JsonValue],
) -> None:
    _name, action = case
    adapter = strict_adapters[provider](_envelope(action))
    try:
        with pytest.raises(LLMError) as failure:
            await adapter.complete(request_payload)
    finally:
        await adapter.aclose()

    assert failure.value.code == "llm_schema_validation_failed"
    assert failure.value.provider == provider


@pytest.mark.parametrize("provider", list(_adapter_names()))
@pytest.mark.parametrize(
    "payload",
    ("not json", "[]", '"text"', '{"unexpected": {}}', '{"action": []}', "{}"),
    ids=("syntax", "array", "string", "wrong_key", "wrong_action_type", "empty"),
)
async def test_malformed_envelopes_are_rejected_identically(
    strict_adapters: dict[str, AdapterFactory],
    request_payload: LLMRequest,
    provider: str,
    payload: str,
) -> None:
    adapter = strict_adapters[provider](payload)
    try:
        with pytest.raises(LLMError) as failure:
            await adapter.complete(request_payload)
    finally:
        await adapter.aclose()

    assert failure.value.code == "llm_invalid_structured_output"
    assert failure.value.provider == provider


@pytest.mark.parametrize("provider", list(_adapter_names()))
async def test_adapters_report_provider_neutral_metadata(
    strict_adapters: dict[str, AdapterFactory],
    request_payload: LLMRequest,
    provider: str,
) -> None:
    _name, action = _VALID_ACTIONS[0]
    adapter = strict_adapters[provider](_envelope(action))
    try:
        response = await adapter.complete(request_payload)
    finally:
        await adapter.aclose()

    assert response.model_id is not None
    assert response.finish_reason is not None
    assert response.usage is not None
    assert response.usage.input_tokens == 11
    assert response.usage.output_tokens == 7
    # The fingerprint names the provider, its SDK version, and the schema.
    fingerprint = response.backend_fingerprint or ""
    assert fingerprint.startswith(provider)
    assert len(fingerprint.split("/")) >= 3


@pytest.mark.parametrize("provider", list(_adapter_names()))
async def test_closed_adapters_reject_further_requests(
    strict_adapters: dict[str, AdapterFactory],
    request_payload: LLMRequest,
    provider: str,
) -> None:
    _name, action = _VALID_ACTIONS[0]
    adapter = strict_adapters[provider](_envelope(action))
    await adapter.aclose()
    await adapter.aclose()

    with pytest.raises(LLMError) as failure:
        await adapter.complete(request_payload)
    assert failure.value.code == "llm_closed"


async def test_litellm_requires_an_explicit_opt_in_before_degrading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _LiteLLMFactory(monkeypatch, strict=False)
    _name, action = _VALID_ACTIONS[0]
    adapter = factory(_envelope(action))
    try:
        with pytest.raises(ConfigurationError) as failure:
            await adapter.complete(_request())
    finally:
        await adapter.aclose()

    assert failure.value.code == "strict_schema_unavailable"


async def test_litellm_json_only_degradation_is_typed_and_post_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _LiteLLMFactory(monkeypatch, strict=False)
    _name, action = _VALID_ACTIONS[0]
    adapter = factory(_envelope(action), allow_json_only=True)
    try:
        response = await adapter.complete(_request())
    finally:
        await adapter.aclose()

    assert response.schema_enforcement is SchemaEnforcement.JSON_ONLY
    assert json.loads(response.text)["type"] == "answer"


@pytest.mark.parametrize("case", _INVALID_ACTIONS, ids=[name for name, _ in _INVALID_ACTIONS])
async def test_litellm_json_only_still_rejects_invalid_actions(
    monkeypatch: pytest.MonkeyPatch,
    case: tuple[str, JsonValue],
) -> None:
    _name, action = case
    factory = _LiteLLMFactory(monkeypatch, strict=False)
    adapter = factory(_envelope(action), allow_json_only=True)
    try:
        with pytest.raises(LLMError) as failure:
            await adapter.complete(_request())
    finally:
        await adapter.aclose()

    assert failure.value.code == "llm_schema_validation_failed"


@pytest.mark.parametrize("provider", list(_adapter_names()))
async def test_adapter_failures_never_carry_request_or_response_content(
    strict_adapters: dict[str, AdapterFactory],
    provider: str,
) -> None:
    secret = "SECRET-RECEIPT-TOTAL-99.99"
    request = _request().model_copy(update={"metadata": {"prompt_version": "agent-v1"}})
    payload = _envelope({"type": "answer", "status": "answered", "answer": secret, "surplus": 1})
    adapter = strict_adapters[provider](payload)
    try:
        with pytest.raises(LLMError) as failure:
            await adapter.complete(request)
    finally:
        await adapter.aclose()

    error = failure.value
    rendered = " ".join(
        (
            str(error),
            repr(error),
            str(error.args),
            str(error.cause_summary),
            str(error.provider_code),
        )
    )
    assert secret not in rendered
    assert "RM 42.50" not in rendered
    assert "obs_000001" not in rendered


def _litellm_failing_adapter(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
) -> tuple[_Adapter, list[int]]:
    attempts: list[int] = []

    async def acompletion(**_kwargs: object) -> object:
        attempts.append(status)
        raise _ProviderStatusError(status)

    runtime = litellm_module._Runtime(
        acompletion=acompletion,
        supported_params=lambda _model: ["response_format"],
        supports_response_schema=lambda _model: True,
        version="1.96.2",
        retryable_errors=(),
        terminal_errors=(),
    )
    monkeypatch.setattr(litellm_module, "_resolve_runtime", lambda: runtime)
    adapter = LiteLLMTextLLM(model="gpt-4o", retry_policy=_RETRY_POLICY)
    adapter._random_source = lambda: 0.0
    return adapter, attempts


def _failing_adapter(
    provider: str,
    status: int,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_Adapter, list[int]]:
    if provider == "openai":
        return _openai_failing_adapter(status)
    if provider == "anthropic":
        return _anthropic_failing_adapter(status)
    return _litellm_failing_adapter(monkeypatch, status)


@pytest.mark.parametrize("provider", list(_adapter_names()))
@pytest.mark.parametrize("status", (408, 429, 500, 502, 503, 529))
async def test_every_adapter_retries_the_same_transient_statuses(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    status: int,
) -> None:
    adapter, attempts = _failing_adapter(provider, status, monkeypatch)
    try:
        with pytest.raises(LLMError) as failure:
            await adapter.complete(_request())
    finally:
        await adapter.aclose()

    assert len(attempts) == 2
    assert failure.value.code == "llm_retries_exhausted"
    assert failure.value.attempts == 2
    assert failure.value.provider_status == status


@pytest.mark.parametrize("provider", list(_adapter_names()))
@pytest.mark.parametrize("status", (400, 401, 403, 404, 409, 413, 422, 425))
async def test_every_adapter_treats_other_client_errors_as_terminal(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    status: int,
) -> None:
    adapter, attempts = _failing_adapter(provider, status, monkeypatch)
    try:
        with pytest.raises(LLMError) as failure:
            await adapter.complete(_request())
    finally:
        await adapter.aclose()

    assert len(attempts) == 1
    assert failure.value.code == "llm_request_failed"
    assert failure.value.attempts == 1
    assert failure.value.retryable is False
    assert failure.value.provider_status == status
