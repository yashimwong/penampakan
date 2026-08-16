"""Real-SDK shape tests for the LiteLLM adapter, without network or credentials."""

from __future__ import annotations

import importlib.util
import json
import socket
from collections.abc import Iterator

import pytest

from penampakan.errors import ConfigurationError, LLMError
from penampakan.llms.litellm import LiteLLMTextLLM
from penampakan.llms.schema import SchemaTarget, canonical_json, compile_action_schema
from penampakan.models import LLMRequest, RetryPolicy, SchemaEnforcement
from penampakan.reasoning.prompts import build_policy_request
from tests.unit.reasoning.helpers import make_policy_input

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("litellm") is None,
    reason="the litellm extra is not installed",
)

# LiteLLM reports strict response-schema support for this documented model.
_STRICT_MODEL = "gpt-4o"
# LiteLLM reports JSON mode but no response-schema support for this model.
_JSON_ONLY_MODEL = "gpt-3.5-turbo-instruct"
# LiteLLM maps this to no provider at all, so capabilities stay unknown.
_UNKNOWN_MODEL = "definitely-not-a-real-model-xyz"

_ACTION = {
    "type": "answer",
    "status": "answered",
    "answer": "RM 42.50",
    "evidence": [{"observation_id": "obs_000001", "supports": "total line"}],
    "uncertainties": [],
}
_PAYLOAD = json.dumps({"action": _ACTION})

_CREDENTIAL_ENV = (
    "ANTHROPIC_API_KEY",
    "AZURE_API_BASE",
    "AZURE_API_KEY",
    "COHERE_API_KEY",
    "LITELLM_API_KEY",
    "LITELLM_PROXY_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_ORGANIZATION",
)


@pytest.fixture
def hermetic(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove provider credentials and fail the test on any outbound socket."""
    for name in _CREDENTIAL_ENV:
        monkeypatch.delenv(name, raising=False)

    def blocked(*args: object, **kwargs: object) -> object:
        raise AssertionError("the SDK-shape test must not open a network connection")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    yield


def _request() -> LLMRequest:
    return build_policy_request(make_policy_input(), timeout_s=10.0)


async def test_real_probes_select_the_strict_path_for_a_documented_model(
    hermetic: None,
) -> None:
    import litellm

    request = _request()
    compiled = compile_action_schema(
        request.response_json_schema,
        target=SchemaTarget.OPENAI_STRICT,
    )
    params = litellm.get_supported_openai_params(_STRICT_MODEL)
    assert params is not None and "response_format" in params
    assert litellm.supports_response_schema(_STRICT_MODEL) is True

    async with LiteLLMTextLLM(model=_STRICT_MODEL, mock_response=_PAYLOAD) as llm:
        response = await llm.complete(request)

    assert response.schema_enforcement is SchemaEnforcement.STRICT
    assert response.text == canonical_json(_ACTION)
    assert response.provider == "litellm"
    assert response.model_id == _STRICT_MODEL
    assert response.finish_reason == "stop"
    assert response.attempts == 1
    assert response.request_id is not None and response.request_id.startswith("chatcmpl-")
    assert response.usage is not None
    assert response.usage.input_tokens is not None and response.usage.input_tokens >= 0
    assert response.backend_fingerprint is not None
    assert response.backend_fingerprint.endswith(compiled.fingerprint_sha256[:16])
    assert response.backend_fingerprint.startswith("litellm/")


async def test_real_probes_select_json_only_when_the_model_has_no_response_schema(
    hermetic: None,
) -> None:
    import litellm

    params = litellm.get_supported_openai_params(_JSON_ONLY_MODEL)
    assert params is not None and "response_format" in params
    assert litellm.supports_response_schema(_JSON_ONLY_MODEL) is False

    with pytest.raises(ConfigurationError) as refused:
        async with LiteLLMTextLLM(model=_JSON_ONLY_MODEL, mock_response=_PAYLOAD) as strict:
            await strict.complete(_request())
    assert refused.value.code == "strict_schema_unavailable"

    async with LiteLLMTextLLM(
        model=_JSON_ONLY_MODEL,
        allow_json_only=True,
        mock_response=_PAYLOAD,
    ) as llm:
        response = await llm.complete(_request())

    assert response.schema_enforcement is SchemaEnforcement.JSON_ONLY
    assert response.text == canonical_json(_ACTION)
    assert response.model_id == _JSON_ONLY_MODEL


async def test_real_probes_treat_an_unmapped_model_as_capability_unknown(
    hermetic: None,
) -> None:
    import litellm

    assert litellm.get_supported_openai_params(_UNKNOWN_MODEL) is None
    assert litellm.supports_response_schema(_UNKNOWN_MODEL) is False

    async with LiteLLMTextLLM(model=_UNKNOWN_MODEL, mock_response=_PAYLOAD) as llm:
        with pytest.raises(ConfigurationError) as info:
            await llm.complete(_request())

    assert info.value.code == "strict_schema_unavailable"


async def test_real_sdk_receives_the_compiled_strict_request_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
    hermetic: None,
) -> None:
    import litellm

    real_acompletion = litellm.acompletion
    recorded: list[dict[str, object]] = []

    async def recording(**kwargs: object) -> object:
        recorded.append(dict(kwargs))
        return await real_acompletion(**kwargs)

    monkeypatch.setattr(litellm, "acompletion", recording)
    request = _request()
    compiled = compile_action_schema(
        request.response_json_schema,
        target=SchemaTarget.OPENAI_STRICT,
    )

    async with LiteLLMTextLLM(model=_STRICT_MODEL, mock_response=_PAYLOAD) as llm:
        response = await llm.complete(request)

    assert response.schema_enforcement is SchemaEnforcement.STRICT
    assert len(recorded) == 1
    call = recorded[0]
    assert call["model"] == _STRICT_MODEL
    assert call["num_retries"] == 0
    assert call["max_tokens"] == request.max_output_tokens
    assert call["temperature"] == request.temperature
    timeout = call["timeout"]
    assert isinstance(timeout, float) and 0.0 < timeout <= 10.0
    assert call["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "policy_action", "strict": True, "schema": compiled.schema},
    }
    assert call["messages"] == [
        {"role": "system", "content": request.messages[0].content},
        {"role": "user", "content": request.messages[1].content},
    ]


@pytest.mark.parametrize(
    ("mock_error", "provider_code", "status"),
    [
        pytest.param("litellm.RateLimitError", "rate_limited", 429, id="rate_limited"),
        pytest.param("litellm.InternalServerError", "server_error", 500, id="server_error"),
    ],
)
async def test_real_sdk_retryable_exceptions_are_retried_without_multiplying(
    monkeypatch: pytest.MonkeyPatch,
    hermetic: None,
    mock_error: str,
    provider_code: str,
    status: int,
) -> None:
    import litellm

    real_acompletion = litellm.acompletion
    calls: list[None] = []

    async def recording(**kwargs: object) -> object:
        calls.append(None)
        return await real_acompletion(**kwargs)

    monkeypatch.setattr(litellm, "acompletion", recording)
    llm = LiteLLMTextLLM(
        model=_STRICT_MODEL,
        retry_policy=RetryPolicy(max_attempts=2, base_delay_s=0.001, max_delay_s=0.002),
        mock_response=mock_error,
    )

    async with llm:
        with pytest.raises(LLMError) as info:
            await llm.complete(_request())

    assert info.value.code == "llm_retries_exhausted"
    assert info.value.attempts == 2
    assert info.value.provider_status == status
    assert info.value.provider_code == provider_code
    # One provider call per adapter attempt: LiteLLM's own retries stay disabled.
    assert len(calls) == 2


async def test_real_sdk_terminal_exception_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
    hermetic: None,
) -> None:
    import litellm

    real_acompletion = litellm.acompletion
    calls: list[None] = []

    async def recording(**kwargs: object) -> object:
        calls.append(None)
        return await real_acompletion(**kwargs)

    monkeypatch.setattr(litellm, "acompletion", recording)
    llm = LiteLLMTextLLM(
        model=_STRICT_MODEL,
        mock_response="litellm.ContextWindowExceededError",
    )

    async with llm:
        with pytest.raises(LLMError) as info:
            await llm.complete(_request())

    assert info.value.code == "llm_request_failed"
    assert info.value.retryable is False
    assert info.value.attempts == 1
    assert info.value.provider_code == "context_window_exceeded"
    assert len(calls) == 1


async def test_real_sdk_response_without_structured_output_is_typed(
    hermetic: None,
) -> None:
    async with LiteLLMTextLLM(model=_STRICT_MODEL, mock_response="   ") as llm:
        with pytest.raises(LLMError) as info:
            await llm.complete(_request())

    assert info.value.code == "llm_missing_structured_output"


async def test_real_sdk_non_json_output_fails_local_post_validation(
    hermetic: None,
) -> None:
    async with LiteLLMTextLLM(
        model=_JSON_ONLY_MODEL,
        allow_json_only=True,
        mock_response="I am not JSON.",
    ) as llm:
        with pytest.raises(LLMError) as info:
            await llm.complete(_request())

    assert info.value.code == "llm_invalid_structured_output"
