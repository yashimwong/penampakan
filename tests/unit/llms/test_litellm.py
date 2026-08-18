from __future__ import annotations

import asyncio
import builtins
import json
import subprocess
import sys
import textwrap
from collections.abc import Sequence
from typing import Any

import pytest

from penampakan.errors import ConfigurationError, LLMError
from penampakan.llms import litellm as litellm_module
from penampakan.llms.litellm import LiteLLMTextLLM
from penampakan.llms.schema import SchemaTarget, canonical_json, compile_action_schema
from penampakan.models import (
    LLMRequest,
    Message,
    MessageRole,
    RetryPolicy,
    SchemaEnforcement,
    TokenUsage,
)
from penampakan.reasoning.prompts import build_policy_request
from tests.unit.reasoning.helpers import make_policy_input

_MODEL = "gpt-4o"
_API_KEY = "sk-unit-test-secret-key"
_BASE_URL = "https://private.gateway.example/v1"

_VALID_ACTION = {
    "action": {
        "type": "answer",
        "status": "answered",
        "answer": "RM 42.50",
        "evidence": [{"observation_id": "obs_000001", "supports": "total line"}],
        "uncertainties": [],
    }
}
_INVALID_ACTION = {"action": {"type": "answer", "status": "not_a_status", "answer": "RM 42.50"}}


class _FakeUsage:
    """Usage counters shaped like the LiteLLM/OpenAI usage object."""

    def __init__(self, prompt_tokens: object = 11, completion_tokens: object = 22) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeMessage:
    """An assistant message shaped like the LiteLLM/OpenAI message object."""

    def __init__(self, content: object = None, refusal: str | None = None) -> None:
        self.content = content
        self.refusal = refusal


class _FakeChoice:
    """One choice shaped like the LiteLLM/OpenAI choice object."""

    def __init__(self, message: _FakeMessage, finish_reason: object = "stop") -> None:
        self.message = message
        self.finish_reason = finish_reason


class _FakeResponse:
    """A LiteLLM-shaped chat completion response."""

    def __init__(
        self,
        content: object = None,
        *,
        finish_reason: object = "stop",
        refusal: str | None = None,
        response_id: object = "chatcmpl-000000000001",
        model: object = "gpt-4o-2024-08-06",
        usage: _FakeUsage | None = None,
    ) -> None:
        self.id = response_id
        self.model = model
        self.choices = [_FakeChoice(_FakeMessage(content, refusal), finish_reason)]
        self.usage = _FakeUsage() if usage is None else usage


def _response(payload: object = _VALID_ACTION) -> _FakeResponse:
    return _FakeResponse(json.dumps(payload))


class _RateLimited(Exception):
    status_code = 429


class _ServerError(Exception):
    status_code = 503


class _ConnectionFailed(Exception):
    """A connection failure that carries no HTTP status, like the real SDK."""


class _AuthFailed(Exception):
    status_code = 401


class _PermissionDenied(Exception):
    status_code = 403


class _BadRequest(Exception):
    status_code = 400


class _FakeSDK:
    """A stand-in for the resolved LiteLLM surface, injected through the seam."""

    def __init__(
        self,
        results: Sequence[object],
        *,
        supported_params: object = ("response_format", "temperature"),
        schema_support: object = True,
        probe_error: bool = False,
        probes_present: bool = True,
        delay_s: float = 0.0,
    ) -> None:
        self._results = list(results)
        self._supported_params = supported_params
        self._schema_support = schema_support
        self._probe_error = probe_error
        self._probes_present = probes_present
        self._delay_s = delay_s
        self.calls: list[dict[str, object]] = []
        self.param_probes: list[str] = []
        self.schema_probes: list[str] = []

    async def acompletion(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        result = self._results[min(len(self.calls), len(self._results)) - 1]
        if isinstance(result, BaseException):
            raise result
        return result

    def get_supported_openai_params(self, model: str) -> object:
        self.param_probes.append(model)
        if self._probe_error:
            raise _BadRequest("unmapped provider")
        params = self._supported_params
        return list(params) if isinstance(params, tuple) else params

    def supports_response_schema(self, model: str) -> object:
        self.schema_probes.append(model)
        if self._probe_error:
            raise _BadRequest("unmapped provider")
        return self._schema_support

    def runtime(self) -> litellm_module._Runtime:
        """Build the private runtime record the adapter normally resolves."""
        return litellm_module._Runtime(
            acompletion=self.acompletion,
            supported_params=self.get_supported_openai_params if self._probes_present else None,
            supports_response_schema=(
                self.supports_response_schema if self._probes_present else None
            ),
            version="1.96.2",
            retryable_errors=(
                (_RateLimited, "rate_limited"),
                (_ConnectionFailed, "connection_failed"),
                (_ServerError, "server_error"),
            ),
            terminal_errors=(
                (_AuthFailed, "authentication_failed"),
                (_PermissionDenied, "permission_denied"),
                (_BadRequest, "invalid_request"),
            ),
        )


def _install(monkeypatch: pytest.MonkeyPatch, sdk: _FakeSDK) -> None:
    monkeypatch.setattr(litellm_module, "_resolve_runtime", sdk.runtime)


def _request(*, timeout_s: float | None = 5.0) -> LLMRequest:
    return build_policy_request(make_policy_input(), timeout_s=timeout_s)


def _adapter(
    monkeypatch: pytest.MonkeyPatch,
    sdk: _FakeSDK,
    **kwargs: Any,
) -> LiteLLMTextLLM:
    _install(monkeypatch, sdk)
    return LiteLLMTextLLM(model=_MODEL, **kwargs)


async def test_strict_path_sends_compiled_schema_and_reports_full_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSDK([_response()])
    llm = _adapter(monkeypatch, sdk)
    request = _request()
    compiled = compile_action_schema(
        request.response_json_schema,
        target=SchemaTarget.OPENAI_STRICT,
    )

    response = await llm.complete(request)

    assert response.schema_enforcement is SchemaEnforcement.STRICT
    assert response.text == canonical_json(_VALID_ACTION["action"])
    assert response.provider == "litellm"
    assert response.model_id == "gpt-4o-2024-08-06"
    assert response.request_id == "chatcmpl-000000000001"
    assert response.finish_reason == "stop"
    assert response.usage == TokenUsage(input_tokens=11, output_tokens=22)
    assert response.attempts == 1
    assert response.backend_fingerprint == (f"litellm/1.96.2/{compiled.fingerprint_sha256[:16]}")

    assert sdk.param_probes == [_MODEL]
    assert sdk.schema_probes == [_MODEL]
    call = sdk.calls[0]
    assert call["model"] == _MODEL
    assert call["max_tokens"] == request.max_output_tokens
    assert call["temperature"] == request.temperature
    assert call["num_retries"] == 0
    timeout = call["timeout"]
    assert isinstance(timeout, float) and 0.0 < timeout <= 5.0
    assert call["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "policy_action",
            "strict": True,
            "schema": compiled.schema,
        },
    }
    assert call["messages"] == [
        {"role": "system", "content": request.messages[0].content},
        {"role": "user", "content": request.messages[1].content},
    ]


async def test_system_messages_are_joined_and_other_roles_map_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSDK([_response()])
    llm = _adapter(monkeypatch, sdk)
    base = _request()
    request = base.model_copy(
        update={
            "messages": (
                Message(role=MessageRole.SYSTEM, content="rule one"),
                Message(role=MessageRole.USER, content="question"),
                Message(role=MessageRole.ASSISTANT, content="prior"),
                Message(role=MessageRole.SYSTEM, content="rule two"),
            )
        }
    )

    await llm.complete(request)

    assert sdk.calls[0]["messages"] == [
        {"role": "system", "content": "rule one\n\nrule two"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "prior"},
    ]


@pytest.mark.parametrize(
    "sdk_kwargs",
    [
        pytest.param({"supported_params": None}, id="params_none"),
        pytest.param({"supported_params": ("temperature",)}, id="response_format_absent"),
        pytest.param({"schema_support": False}, id="schema_unsupported"),
        pytest.param({"schema_support": None}, id="schema_unknown"),
        pytest.param({"probe_error": True}, id="probe_raises"),
        pytest.param({"probes_present": False}, id="probes_absent"),
    ],
)
async def test_unavailable_strict_schema_without_opt_in_is_a_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
    sdk_kwargs: dict[str, object],
) -> None:
    sdk = _FakeSDK([_response()], **sdk_kwargs)
    llm = _adapter(monkeypatch, sdk)

    with pytest.raises(ConfigurationError) as info:
        await llm.complete(_request())

    assert info.value.code == "strict_schema_unavailable"
    assert sdk.calls == []


async def test_json_only_opt_in_uses_json_mode_with_an_explicit_instruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSDK([_response()], schema_support=False)
    llm = _adapter(monkeypatch, sdk, allow_json_only=True)
    request = _request()
    compiled = compile_action_schema(request.response_json_schema, target=SchemaTarget.JSON_ONLY)

    response = await llm.complete(request)

    assert response.schema_enforcement is SchemaEnforcement.JSON_ONLY
    assert response.text == canonical_json(_VALID_ACTION["action"])
    assert response.backend_fingerprint == f"litellm/1.96.2/{compiled.fingerprint_sha256[:16]}"
    call = sdk.calls[0]
    assert call["response_format"] == {"type": "json_object"}
    messages = call["messages"]
    assert isinstance(messages, list)
    instruction = messages[0]["content"]
    assert messages[0]["role"] == "system"
    assert request.messages[0].content in instruction
    assert litellm_module._JSON_ONLY_INSTRUCTION in instruction
    assert '"action"' in instruction
    assert canonical_json(compiled.schema) in instruction


async def test_json_only_mode_still_post_validates_the_original_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSDK([_response(_INVALID_ACTION)], schema_support=False)
    llm = _adapter(monkeypatch, sdk, allow_json_only=True)

    with pytest.raises(LLMError) as info:
        await llm.complete(_request())

    assert info.value.code == "llm_schema_validation_failed"


async def test_json_only_mode_rejects_non_json_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSDK([_FakeResponse("not json at all")], schema_support=False)
    llm = _adapter(monkeypatch, sdk, allow_json_only=True)

    with pytest.raises(LLMError) as info:
        await llm.complete(_request())

    assert info.value.code == "llm_invalid_structured_output"


async def test_capability_probe_is_cached_across_completions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSDK([_response(), _response()])
    llm = _adapter(monkeypatch, sdk)

    await llm.complete(_request())
    await llm.complete(_request())

    assert sdk.param_probes == [_MODEL]
    assert sdk.schema_probes == [_MODEL]
    assert len(sdk.calls) == 2


@pytest.mark.parametrize(
    ("response", "code"),
    [
        pytest.param(
            _FakeResponse(json.dumps(_VALID_ACTION), finish_reason="length"),
            "llm_output_truncated",
            id="truncated",
        ),
        pytest.param(
            _FakeResponse(None, finish_reason="content_filter"),
            "llm_refused",
            id="content_filter",
        ),
        pytest.param(
            _FakeResponse(None, refusal="I cannot help with that."),
            "llm_refused",
            id="refusal_field",
        ),
        pytest.param(_FakeResponse(None), "llm_missing_structured_output", id="content_absent"),
        pytest.param(_FakeResponse("   "), "llm_missing_structured_output", id="content_blank"),
        pytest.param(_FakeResponse(42), "llm_missing_structured_output", id="content_not_text"),
    ],
)
async def test_typed_outcomes_are_raised_before_parsing(
    monkeypatch: pytest.MonkeyPatch,
    response: _FakeResponse,
    code: str,
) -> None:
    sdk = _FakeSDK([response])
    llm = _adapter(monkeypatch, sdk)

    with pytest.raises(LLMError) as info:
        await llm.complete(_request())

    assert info.value.code == code
    assert info.value.retryable is False
    assert info.value.attempts == 1
    assert info.value.provider == "litellm"
    assert len(sdk.calls) == 1


async def test_missing_choices_are_reported_as_missing_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty = _response()
    empty.choices = []
    sdk = _FakeSDK([empty])
    llm = _adapter(monkeypatch, sdk)

    with pytest.raises(LLMError) as info:
        await llm.complete(_request())

    assert info.value.code == "llm_missing_structured_output"


@pytest.mark.parametrize(
    "key",
    # "model" is an explicit parameter, so it can never reach completion_kwargs.
    ["messages", "response_format", "max_tokens", "temperature", "timeout", "num_retries"],
)
def test_adapter_controlled_completion_kwargs_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    _install(monkeypatch, _FakeSDK([_response()]))

    with pytest.raises(ConfigurationError) as info:
        LiteLLMTextLLM(model=_MODEL, **{key: "anything"})

    assert info.value.code == "conflicting_completion_kwargs"


async def test_extra_completion_kwargs_pass_through_with_native_retries_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSDK([_response()])
    llm = _adapter(monkeypatch, sdk, api_key=_API_KEY, base_url=_BASE_URL, seed=7)

    await llm.complete(_request())

    call = sdk.calls[0]
    assert call["api_key"] == _API_KEY
    assert call["base_url"] == _BASE_URL
    assert call["seed"] == 7
    assert call["num_retries"] == 0


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(_RateLimited("429"), id="rate_limited"),
        pytest.param(_ServerError("503"), id="server_error"),
        pytest.param(_ConnectionFailed("connection reset"), id="connection"),
    ],
)
async def test_retryable_failures_are_retried_until_one_attempt_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    jitter_draws: list[float] = []
    sdk = _FakeSDK([failure, _response()])
    llm = _adapter(
        monkeypatch,
        sdk,
        retry_policy=RetryPolicy(max_attempts=3, base_delay_s=0.001, max_delay_s=0.002),
    )
    llm._random_source = lambda: jitter_draws.append(0.0) or 0.0

    response = await llm.complete(_request(timeout_s=None))

    assert response.attempts == 2
    assert len(sdk.calls) == 2
    assert jitter_draws == [0.0]
    assert "timeout" not in sdk.calls[0]


async def test_retry_exhaustion_reports_a_safe_attempt_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSDK([_ServerError("503")])
    llm = _adapter(
        monkeypatch,
        sdk,
        retry_policy=RetryPolicy(max_attempts=3, base_delay_s=0.001, max_delay_s=0.002),
    )
    llm._random_source = lambda: 0.0

    with pytest.raises(LLMError) as info:
        await llm.complete(_request(timeout_s=None))

    assert info.value.code == "llm_retries_exhausted"
    assert info.value.attempts == 3
    assert info.value.provider_status == 503
    assert info.value.provider_code == "server_error"
    assert len(sdk.calls) == 3


@pytest.mark.parametrize(
    ("failure", "status", "code"),
    [
        pytest.param(_BadRequest("400"), 400, "invalid_request", id="bad_request"),
        pytest.param(_AuthFailed("401"), 401, "authentication_failed", id="authentication"),
        pytest.param(_PermissionDenied("403"), 403, "permission_denied", id="permission"),
    ],
)
async def test_terminal_failures_are_never_retried(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    status: int,
    code: str,
) -> None:
    sdk = _FakeSDK([failure])
    llm = _adapter(monkeypatch, sdk)

    with pytest.raises(LLMError) as info:
        await llm.complete(_request(timeout_s=None))

    assert info.value.code == "llm_request_failed"
    assert info.value.retryable is False
    assert info.value.attempts == 1
    assert info.value.provider_status == status
    assert info.value.provider_code == code
    assert len(sdk.calls) == 1


async def test_unknown_exceptions_are_classified_without_provider_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSDK([ValueError("boom with prompt text")])
    llm = _adapter(monkeypatch, sdk)

    with pytest.raises(LLMError) as info:
        await llm.complete(_request(timeout_s=None))

    assert info.value.code == "llm_request_failed"
    assert info.value.provider_code == "provider_error"
    assert info.value.provider_status is None
    assert "boom" not in str(info.value)


async def test_deadline_exhaustion_raises_a_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSDK([_response()], delay_s=5.0)
    llm = _adapter(monkeypatch, sdk)

    with pytest.raises(LLMError) as info:
        await llm.complete(_request(timeout_s=0.05))

    assert info.value.code == "llm_timeout"
    assert info.value.retryable is True


async def test_aclose_is_idempotent_and_a_closed_adapter_rejects_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSDK([_response()])
    llm = _adapter(monkeypatch, sdk)

    await asyncio.gather(llm.aclose(), llm.aclose(), llm.aclose())

    with pytest.raises(LLMError) as info:
        await llm.complete(_request())

    assert info.value.code == "llm_closed"
    assert sdk.calls == []


async def test_async_context_manager_closes_the_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSDK([_response()])
    _install(monkeypatch, sdk)

    async with LiteLLMTextLLM(model=_MODEL) as llm:
        assert (await llm.complete(_request())).attempts == 1

    with pytest.raises(LLMError):
        await llm.complete(_request())


async def test_raised_errors_never_carry_prompts_schemas_or_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    secrets = (
        _API_KEY,
        _BASE_URL,
        request.messages[0].content,
        request.messages[1].content,
        json.dumps(request.response_json_schema),
    )
    leak = _BadRequest(f"{_API_KEY} {_BASE_URL} {request.messages[1].content}")
    sdk = _FakeSDK([leak])
    llm = _adapter(monkeypatch, sdk, api_key=_API_KEY, base_url=_BASE_URL)

    with pytest.raises(LLMError) as info:
        await llm.complete(request)

    error = info.value
    exposed = " ".join([str(error), repr(error), *(str(value) for value in vars(error).values())])
    for secret in secrets:
        assert secret not in exposed
    assert error.cause_summary == "_BadRequest"


async def test_configuration_failures_never_carry_prompts_or_schemas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    sdk = _FakeSDK([_response()], schema_support=False)
    llm = _adapter(monkeypatch, sdk, api_key=_API_KEY)

    with pytest.raises(ConfigurationError) as info:
        await llm.complete(request)

    error = info.value
    exposed = " ".join([str(error), repr(error), *(str(value) for value in vars(error).values())])
    for secret in (_API_KEY, request.messages[1].content, json.dumps(request.response_json_schema)):
        assert secret not in exposed


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        pytest.param({"model": ""}, "invalid_model", id="empty_model"),
        pytest.param({"model": "   "}, "invalid_model", id="blank_model"),
        pytest.param(
            {"model": _MODEL, "retry_policy": object()}, "invalid_retry_policy", id="policy"
        ),
        pytest.param(
            {"model": _MODEL, "allow_json_only": "yes"},
            "invalid_json_only_opt_in",
            id="opt_in",
        ),
    ],
)
def test_constructor_rejects_invalid_configuration(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, object],
    code: str,
) -> None:
    _install(monkeypatch, _FakeSDK([_response()]))

    with pytest.raises(ConfigurationError) as info:
        LiteLLMTextLLM(**kwargs)  # type: ignore[arg-type]

    assert info.value.code == code


def test_construction_without_the_litellm_package_is_an_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "litellm" or name.startswith("litellm."):
            raise ImportError("No module named 'litellm'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(ConfigurationError) as info:
        LiteLLMTextLLM(model=_MODEL)

    assert info.value.code == "missing_optional_dependency"
    assert info.value.cause_summary == "cause details redacted"


def test_module_imports_and_types_without_the_litellm_package() -> None:
    script = textwrap.dedent(
        """
        import sys


        class _Blocker:
            def find_spec(self, name, path=None, target=None):
                if name == "litellm" or name.startswith("litellm."):
                    raise ImportError("blocked")
                return None


        sys.meta_path.insert(0, _Blocker())
        import penampakan.llms.litellm as module

        assert "litellm" not in sys.modules
        assert module.LiteLLMTextLLM.__name__ == "LiteLLMTextLLM"
        assert "LiteLLMTextLLM" in dir(module)
        try:
            module.LiteLLMTextLLM(model="gpt-4o")
        except Exception as error:
            assert getattr(error, "code", None) == "missing_optional_dependency", error
        else:
            raise AssertionError("construction must fail without litellm")
        print("ok")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")
