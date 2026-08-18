from __future__ import annotations

import asyncio
import builtins
import importlib.util
import re
import subprocess
import sys
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from penampakan.errors import ConfigurationError, LLMError
from penampakan.llms._openai_transport import (
    DEFAULT_CAPABILITIES,
    ModelCapabilities,
    classify_failure,
    finish_reason,
    resolve_capabilities,
    table_capabilities,
)
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

requires_sdk = pytest.mark.skipif(
    importlib.util.find_spec("openai") is None,
    reason="the openai extra is not installed",
)

SYSTEM_PROMPT = "TRUSTED SYSTEM PROMPT unicorn-system-marker"
USER_PROMPT = "TRUSTED QUESTION unicorn-user-marker"
SECRET_KEY = "sk-unicorn-secret-key"
SECRET_URL = "https://unicorn-secret-host.invalid/v1"

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
ANSWER_PAYLOAD = (
    '{"action": {"type": "answer", "status": "answered", "answer": "Hello.",'
    ' "evidence": null, "uncertainties": null}}'
)
CANONICAL_ANSWER = '{"answer":"Hello.","status":"answered","type":"answer"}'


def _schema() -> dict[str, Any]:
    return build_action_schema((_TOOL,))


def _request(
    *,
    temperature: float = 0.0,
    max_output_tokens: int = 512,
    timeout_s: float | None = None,
    with_assistant: bool = False,
) -> LLMRequest:
    messages = [
        Message(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        Message(role=MessageRole.USER, content=USER_PROMPT),
    ]
    if with_assistant:
        messages.append(Message(role=MessageRole.ASSISTANT, content="prior draft"))
    return LLMRequest(
        messages=tuple(messages),
        response_json_schema=_schema(),
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        timeout_s=timeout_s,
    )


@dataclass(frozen=True)
class _TextPart:
    """A stub ``output_text`` content part."""

    text: str
    type: str = "output_text"


@dataclass(frozen=True)
class _RefusalPart:
    """A stub ``refusal`` content part."""

    refusal: str
    type: str = "refusal"


@dataclass(frozen=True)
class _MessageItem:
    """A stub assistant message output item."""

    content: Sequence[object]
    status: str = "completed"
    role: str = "assistant"
    type: str = "message"


@dataclass(frozen=True)
class _ReasoningItem:
    """A stub non-message output item the parser must ignore."""

    type: str = "reasoning"


@dataclass(frozen=True)
class _Usage:
    """A stub Responses usage block."""

    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True)
class _IncompleteDetails:
    """A stub ``incomplete_details`` block."""

    reason: str


@dataclass(frozen=True)
class _Response:
    """A stub Responses object mirroring the fields the adapter reads."""

    output: Sequence[object] = field(default_factory=tuple)
    id: str = "resp_unicorn123"
    model: str = "gpt-4.1-mini-2025-04-14"
    status: str = "completed"
    usage: _Usage | None = None
    incomplete_details: _IncompleteDetails | None = None
    error: object | None = None

    @property
    def output_text(self) -> str:
        """Aggregate every text part exactly as the SDK helper does."""
        parts: list[str] = []
        for item in self.output:
            if isinstance(item, _MessageItem):
                parts.extend(part.text for part in item.content if isinstance(part, _TextPart))
        return "".join(parts)


def _completed(payload: str = ANSWER_PAYLOAD) -> _Response:
    return _Response(
        output=(_ReasoningItem(), _MessageItem(content=(_TextPart(text=payload),))),
        usage=_Usage(input_tokens=11, output_tokens=7),
    )


class _StatusError(Exception):
    """A stub provider status error shaped like ``openai.APIStatusError``."""

    def __init__(self, status_code: int, code: str | None = None, message: str = "failed") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class _FakeResponses:
    """A scripted ``client.responses`` namespace recording every call."""

    def __init__(self, results: Iterable[object]) -> None:
        self._results = deque(results)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> object:
        """Record the call and return or raise the next scripted item."""
        self.calls.append(kwargs)
        if not self._results:
            raise AssertionError("unexpected extra provider call")
        result = self._results.popleft()
        if isinstance(result, BaseException):
            raise result
        return result


class _FakeClient:
    """A fake async OpenAI client exposing only what the adapter uses."""

    def __init__(self, *results: object) -> None:
        self.responses = _FakeResponses(results)
        self.closes = 0

    async def close(self) -> None:
        """Count how often the adapter closed this client."""
        self.closes += 1


class _SlowClient:
    """A client whose single call outlives any test deadline."""

    def __init__(self) -> None:
        self.responses = _SlowResponses()
        self.closes = 0

    async def close(self) -> None:
        """Count how often the adapter closed this client."""
        self.closes += 1


class _SlowResponses:
    """A ``responses`` namespace that never completes in time."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> object:
        """Sleep past the caller deadline."""
        self.calls.append(kwargs)
        await asyncio.sleep(30)
        return _completed()


def _adapter(
    client: object,
    *,
    model: str = "gpt-4.1-mini",
    retry_policy: RetryPolicy | None = None,
    **kwargs: Any,
) -> OpenAITextLLM:
    llm = OpenAITextLLM(model=model, client=client, retry_policy=retry_policy, **kwargs)
    # Deterministic full-jitter source: private test construction only.
    llm._random_source = lambda: 0.0
    return llm


def _fast_policy(max_attempts: int = 3) -> RetryPolicy:
    return RetryPolicy(max_attempts=max_attempts, base_delay_s=0.001, max_delay_s=0.002)


@requires_sdk
async def test_happy_path_propagates_full_response_metadata() -> None:
    client = _FakeClient(_completed())
    llm = _adapter(client)

    response = await llm.complete(_request(timeout_s=5.0))

    compiled = compile_action_schema(_schema(), target=SchemaTarget.OPENAI_STRICT)
    assert response.text == CANONICAL_ANSWER
    assert response.model_id == "gpt-4.1-mini-2025-04-14"
    assert response.usage == TokenUsage(input_tokens=11, output_tokens=7)
    assert response.finish_reason == "stop"
    assert response.provider == "openai"
    assert response.request_id == "resp_unicorn123"
    assert response.attempts == 1
    assert response.schema_enforcement is SchemaEnforcement.STRICT
    fingerprint = response.backend_fingerprint
    assert fingerprint is not None
    assert re.fullmatch(r"openai/[0-9A-Za-z._-]+/[0-9a-f]{16}", fingerprint)
    assert fingerprint.endswith(compiled.fingerprint_sha256[:16])


@requires_sdk
async def test_request_carries_strict_schema_roles_and_bounded_parameters() -> None:
    client = _FakeClient(_completed())
    llm = _adapter(client)

    await llm.complete(_request(timeout_s=5.0, with_assistant=True))

    compiled = compile_action_schema(_schema(), target=SchemaTarget.OPENAI_STRICT)
    call = client.responses.calls[0]
    assert call["model"] == "gpt-4.1-mini"
    assert call["input"] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
        {"role": "assistant", "content": "prior draft"},
    ]
    assert call["text"] == {
        "format": {
            "type": "json_schema",
            "name": "policy_action",
            "strict": True,
            "schema": compiled.schema,
        }
    }
    assert call["max_output_tokens"] == 512
    assert call["temperature"] == 0.0
    assert 0.0 < call["timeout"] <= 5.0
    assert "extra_body" not in call


@requires_sdk
async def test_unbounded_request_sends_no_sdk_timeout() -> None:
    client = _FakeClient(_completed())
    llm = _adapter(client)

    await llm.complete(_request())

    assert "timeout" not in client.responses.calls[0]


@requires_sdk
async def test_refusal_part_is_surfaced_before_parsing() -> None:
    client = _FakeClient(
        _Response(output=(_MessageItem(content=(_RefusalPart(refusal="I cannot help."),)),))
    )
    llm = _adapter(client)

    with pytest.raises(LLMError) as info:
        await llm.complete(_request())

    assert info.value.code == "llm_refused"
    assert info.value.attempts == 1
    assert info.value.provider == "openai"


@requires_sdk
async def test_content_filter_incomplete_reason_is_a_refusal() -> None:
    client = _FakeClient(
        _Response(
            output=(_MessageItem(content=(_TextPart(text=ANSWER_PAYLOAD),)),),
            status="incomplete",
            incomplete_details=_IncompleteDetails(reason="content_filter"),
        )
    )
    llm = _adapter(client)

    with pytest.raises(LLMError) as info:
        await llm.complete(_request())

    assert info.value.code == "llm_refused"


@requires_sdk
async def test_truncation_from_incomplete_details_is_surfaced() -> None:
    client = _FakeClient(
        _Response(
            output=(_MessageItem(content=(_TextPart(text='{"action":'),)),),
            status="incomplete",
            incomplete_details=_IncompleteDetails(reason="max_output_tokens"),
        )
    )
    llm = _adapter(client)

    with pytest.raises(LLMError) as info:
        await llm.complete(_request())

    assert info.value.code == "llm_output_truncated"


@requires_sdk
async def test_truncation_from_message_item_status_is_surfaced() -> None:
    client = _FakeClient(
        _Response(
            output=(_MessageItem(content=(_TextPart(text='{"action":'),), status="incomplete"),)
        )
    )
    llm = _adapter(client)

    with pytest.raises(LLMError) as info:
        await llm.complete(_request())

    assert info.value.code == "llm_output_truncated"


@requires_sdk
async def test_failed_status_is_a_terminal_request_failure() -> None:
    client = _FakeClient(_Response(output=(), status="failed"))
    llm = _adapter(client)

    with pytest.raises(LLMError) as info:
        await llm.complete(_request())

    assert info.value.code == "llm_request_failed"


@requires_sdk
@pytest.mark.parametrize(
    "response",
    [
        _Response(output=()),
        _Response(output=(_ReasoningItem(),)),
        _Response(
            output=(
                _MessageItem(content=(_TextPart(text=ANSWER_PAYLOAD),)),
                _MessageItem(content=(_TextPart(text=ANSWER_PAYLOAD),)),
            )
        ),
        _Response(output=(_MessageItem(content=()),)),
    ],
    ids=["no_items", "no_message_item", "duplicate_messages", "no_text_part"],
)
async def test_missing_or_duplicate_structured_output_is_typed(response: _Response) -> None:
    llm = _adapter(_FakeClient(response))

    with pytest.raises(LLMError) as info:
        await llm.complete(_request())

    assert info.value.code == "llm_missing_structured_output"


@requires_sdk
async def test_non_json_output_fails_post_validation() -> None:
    llm = _adapter(_FakeClient(_completed(payload="not json at all")))

    with pytest.raises(LLMError) as info:
        await llm.complete(_request())

    assert info.value.code == "llm_invalid_structured_output"


@requires_sdk
async def test_json_violating_the_original_schema_fails_post_validation() -> None:
    payload = '{"action": {"type": "answer", "status": "unknown", "answer": "Hi."}}'
    llm = _adapter(_FakeClient(_completed(payload=payload)))

    with pytest.raises(LLMError) as info:
        await llm.complete(_request())

    assert info.value.code == "llm_schema_validation_failed"


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-5", ModelCapabilities(instruction_role="developer", supports_temperature=False)),
        ("gpt-5-mini", ModelCapabilities(instruction_role="developer", supports_temperature=False)),
        ("o1-preview", ModelCapabilities(instruction_role="developer", supports_temperature=False)),
        ("o3-mini", ModelCapabilities(instruction_role="developer", supports_temperature=False)),
        ("o4-mini", ModelCapabilities(instruction_role="developer", supports_temperature=False)),
        ("gpt-4.1-mini", ModelCapabilities(instruction_role="system", supports_temperature=True)),
        ("gpt-4o", ModelCapabilities(instruction_role="system", supports_temperature=True)),
        ("gpt-4-turbo", ModelCapabilities(instruction_role="system", supports_temperature=True)),
        ("GPT-4O-MINI", ModelCapabilities(instruction_role="system", supports_temperature=True)),
        ("some-future-model", DEFAULT_CAPABILITIES),
    ],
)
def test_capability_table_covers_every_documented_row(
    model: str,
    expected: ModelCapabilities,
) -> None:
    assert table_capabilities(model) == expected
    assert resolve_capabilities(model) == expected


def test_unknown_models_use_the_broadly_compatible_defaults() -> None:
    expected = ModelCapabilities(instruction_role="system", supports_temperature=True)

    assert table_capabilities("mystery-model-2099") == expected
    assert table_capabilities("gpt-3.5-turbo") == DEFAULT_CAPABILITIES
    assert DEFAULT_CAPABILITIES.instruction_role == "system"
    assert DEFAULT_CAPABILITIES.supports_temperature is True


@pytest.mark.parametrize("override", ["system", "developer"])
def test_explicit_instruction_role_overrides_the_table(override: str) -> None:
    capabilities = resolve_capabilities("gpt-5-mini", instruction_role=override)  # type: ignore[arg-type]

    assert capabilities.instruction_role == override
    assert capabilities.supports_temperature is False


def test_invalid_instruction_role_fails_configuration() -> None:
    with pytest.raises(ConfigurationError) as info:
        resolve_capabilities("gpt-4o", instruction_role="assistant")  # type: ignore[arg-type]

    assert info.value.code == "invalid_instruction_role"


@requires_sdk
async def test_developer_role_model_omits_temperature_and_uses_developer_input() -> None:
    client = _FakeClient(_completed())
    llm = _adapter(client, model="gpt-5-mini")

    await llm.complete(_request(temperature=1.0))

    call = client.responses.calls[0]
    assert "temperature" not in call
    assert call["input"][0]["role"] == "developer"


@requires_sdk
async def test_unsupported_temperature_fails_configuration() -> None:
    client = _FakeClient(_completed())
    llm = _adapter(client, model="gpt-5-mini")

    with pytest.raises(ConfigurationError) as info:
        await llm.complete(_request(temperature=0.2))

    assert info.value.code == "unsupported_request_parameter"
    assert client.responses.calls == []


@requires_sdk
def test_extra_body_conflict_fails_at_construction() -> None:
    with pytest.raises(ConfigurationError) as info:
        _adapter(_FakeClient(), extra_body={"temperature": 0.9})

    assert info.value.code == "conflicting_extra_body"


@requires_sdk
async def test_extra_body_passes_caller_options_through() -> None:
    client = _FakeClient(_completed())
    llm = _adapter(client, extra_body={"service_tier": "flex"})

    await llm.complete(_request())

    assert client.responses.calls[0]["extra_body"] == {"service_tier": "flex"}


@requires_sdk
@pytest.mark.parametrize(
    "failure",
    [_StatusError(429), _StatusError(503), ConnectionError("connection reset")],
)
async def test_retryable_failures_are_retried_and_reported(failure: BaseException) -> None:
    client = _FakeClient(failure, _completed())
    llm = _adapter(client, retry_policy=_fast_policy())

    response = await llm.complete(_request())

    assert response.attempts == 2
    assert len(client.responses.calls) == 2


@requires_sdk
@pytest.mark.parametrize("error_name", ["APIConnectionError", "APITimeoutError"])
async def test_real_sdk_transport_failures_are_retried(error_name: str) -> None:
    import httpx
    import openai

    error_type = getattr(openai, error_name)
    failure = error_type(request=httpx.Request("POST", "https://api.openai.com/v1/responses"))
    client = _FakeClient(failure, _completed())
    llm = _adapter(client, retry_policy=_fast_policy())

    assert (await llm.complete(_request())).attempts == 2


@requires_sdk
async def test_retry_exhaustion_reports_attempts_and_last_status() -> None:
    client = _FakeClient(
        _StatusError(500), _StatusError(500), _StatusError(500, code="server_error")
    )
    llm = _adapter(client, retry_policy=_fast_policy())

    with pytest.raises(LLMError) as info:
        await llm.complete(_request())

    assert info.value.code == "llm_retries_exhausted"
    assert info.value.attempts == 3
    assert info.value.provider_status == 500
    assert info.value.provider_code == "server_error"
    assert len(client.responses.calls) == 3


@requires_sdk
@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422])
async def test_terminal_status_errors_are_not_retried(status: int) -> None:
    client = _FakeClient(_StatusError(status))
    llm = _adapter(client, retry_policy=_fast_policy())

    with pytest.raises(LLMError) as info:
        await llm.complete(_request())

    assert info.value.code == "llm_request_failed"
    assert info.value.retryable is False
    assert info.value.attempts == 1
    assert info.value.provider_status == status
    assert len(client.responses.calls) == 1


@pytest.mark.parametrize(
    ("error", "retryable"),
    [
        (_StatusError(429), True),
        (_StatusError(500), True),
        (_StatusError(599), True),
        (_StatusError(400), False),
        (_StatusError(401), False),
        (_StatusError(403), False),
        (TimeoutError("slow"), True),
        (ConnectionError("down"), True),
        (ValueError("schema"), False),
    ],
)
def test_failure_classification_is_explicit(error: BaseException, retryable: bool) -> None:
    assert classify_failure(error).retryable is retryable


def test_unsafe_provider_codes_are_dropped() -> None:
    failure = classify_failure(_StatusError(400, code="Rejected: unicorn-user-marker"))

    assert failure.code is None


@pytest.mark.parametrize(
    ("status", "reason", "expected"),
    [
        ("completed", None, "stop"),
        ("incomplete", "max_output_tokens", "incomplete_max_output_tokens"),
        ("incomplete", "content_filter", "incomplete_content_filter"),
        ("failed", None, "failed"),
        (None, None, "stop"),
        ("Weird Status", None, "stop"),
    ],
)
def test_finish_reason_tokens_are_safe_and_stable(
    status: object,
    reason: object,
    expected: str,
) -> None:
    token = finish_reason(status, reason)

    assert token == expected
    assert re.fullmatch(r"^[a-z][a-z0-9_]*$", token)


@requires_sdk
async def test_deadline_expiry_during_a_call_reports_a_timeout() -> None:
    client = _SlowClient()
    llm = _adapter(client, retry_policy=_fast_policy())

    with pytest.raises(LLMError) as info:
        await llm.complete(_request(timeout_s=0.05))

    assert info.value.code == "llm_timeout"
    assert info.value.attempts == 1


@requires_sdk
async def test_retry_does_not_start_when_backoff_cannot_fit_the_deadline() -> None:
    client = _FakeClient(_StatusError(503), _completed())
    llm = OpenAITextLLM(
        model="gpt-4.1-mini",
        client=client,
        retry_policy=RetryPolicy(max_attempts=3, base_delay_s=1.0, max_delay_s=1.0),
    )
    llm._random_source = lambda: 1.0

    with pytest.raises(LLMError) as info:
        await llm.complete(_request(timeout_s=0.3))

    assert info.value.code == "llm_timeout"
    assert info.value.attempts == 1
    assert len(client.responses.calls) == 1


@requires_sdk
async def test_injected_client_is_never_closed() -> None:
    client = _FakeClient(_completed())
    llm = _adapter(client)

    await llm.complete(_request())
    await llm.aclose()
    await llm.aclose()

    assert client.closes == 0


@requires_sdk
async def test_injected_client_can_be_adopted_and_is_closed_once() -> None:
    client = _FakeClient()
    llm = _adapter(client, owns_client=True)

    await asyncio.gather(llm.aclose(), llm.aclose(), llm.aclose())

    assert client.closes == 1


@requires_sdk
async def test_constructed_client_disables_native_retries_and_closes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openai

    constructed: list[dict[str, Any]] = []

    class _RecordingClient(_FakeClient):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(_completed())
            constructed.append(kwargs)

    monkeypatch.setattr(openai, "AsyncOpenAI", _RecordingClient)
    llm = OpenAITextLLM(model="gpt-4.1-mini", api_key=SECRET_KEY, base_url=SECRET_URL)

    await llm.aclose()
    await llm.aclose()

    assert constructed == [{"max_retries": 0, "api_key": SECRET_KEY, "base_url": SECRET_URL}]
    client = llm._client
    assert isinstance(client, _FakeClient)
    assert client.closes == 1


@requires_sdk
async def test_owned_client_can_be_disowned(monkeypatch: pytest.MonkeyPatch) -> None:
    import openai

    class _RecordingClient(_FakeClient):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__()

    monkeypatch.setattr(openai, "AsyncOpenAI", _RecordingClient)
    llm = OpenAITextLLM(model="gpt-4.1-mini", owns_client=False)

    await llm.aclose()

    client = llm._client
    assert isinstance(client, _FakeClient)
    assert client.closes == 0


@requires_sdk
async def test_async_context_manager_closes_an_owned_client_and_blocks_reuse() -> None:
    client = _FakeClient(_completed())

    async with _adapter(client, owns_client=True) as llm:
        assert (await llm.complete(_request())).text == CANONICAL_ANSWER

    assert client.closes == 1
    with pytest.raises(LLMError) as info:
        await llm.complete(_request())
    assert info.value.code == "llm_closed"


@requires_sdk
def test_client_without_the_responses_api_fails_configuration() -> None:
    with pytest.raises(ConfigurationError) as info:
        OpenAITextLLM(model="gpt-4.1-mini", client=object())

    assert info.value.code == "invalid_client"


@requires_sdk
@pytest.mark.parametrize("model", ["", "   "])
def test_blank_model_fails_configuration(model: str) -> None:
    with pytest.raises(ConfigurationError) as info:
        OpenAITextLLM(model=model, client=_FakeClient())

    assert info.value.code == "invalid_model"


@requires_sdk
def test_invalid_retry_policy_fails_at_construction() -> None:
    with pytest.raises(ConfigurationError) as info:
        OpenAITextLLM(
            model="gpt-4.1-mini",
            client=_FakeClient(),
            retry_policy=object(),  # type: ignore[arg-type]
        )

    assert info.value.code == "invalid_retry_policy"


def _leaks(error: BaseException) -> list[str]:
    """Return every secret marker visible on a public error surface."""
    surfaces = [str(error), repr(error)]
    surfaces.extend(f"{name}={value!r}" for name, value in vars(error).items())
    haystack = "\n".join(surfaces)
    schema_marker = _TOOL.name
    markers = (SYSTEM_PROMPT, USER_PROMPT, SECRET_KEY, SECRET_URL, schema_marker, ANSWER_PAYLOAD)
    return [marker for marker in markers if marker in haystack]


@requires_sdk
async def test_provider_error_details_never_reach_the_public_error() -> None:
    leaking = _StatusError(
        500,
        code="internal",
        message=f"{SYSTEM_PROMPT} {USER_PROMPT} {SECRET_KEY} {SECRET_URL} {_TOOL.name}",
    )
    llm = _adapter(_FakeClient(leaking, leaking, leaking), retry_policy=_fast_policy())

    with pytest.raises(LLMError) as info:
        await llm.complete(_request())

    assert _leaks(info.value) == []


@requires_sdk
async def test_configuration_errors_never_carry_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openai

    class _RecordingClient(_FakeClient):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__()

    monkeypatch.setattr(openai, "AsyncOpenAI", _RecordingClient)
    llm = OpenAITextLLM(model="gpt-5-mini", api_key=SECRET_KEY, base_url=SECRET_URL)

    with pytest.raises(ConfigurationError) as info:
        await llm.complete(_request(temperature=0.2))

    assert _leaks(info.value) == []
    await llm.aclose()


@requires_sdk
async def test_schema_validation_failures_never_carry_prompt_content() -> None:
    payload = '{"action": {"type": "answer", "status": "unknown", "answer": "Hi."}}'
    llm = _adapter(_FakeClient(_completed(payload=payload)))

    with pytest.raises(LLMError) as info:
        await llm.complete(_request())

    assert _leaks(info.value) == []


def test_construction_without_the_sdk_raises_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: object = (),
        level: int = 0,
    ) -> Any:
        if name == "openai":
            raise ImportError("no module named openai")
        return real_import(name, globals, locals, fromlist, level)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ConfigurationError) as info:
        OpenAITextLLM(model="gpt-4.1-mini")

    assert info.value.code == "missing_optional_dependency"


_IMPORT_PROBE = """
import sys


class _BlockOpenAI:
    def find_spec(self, name, path=None, target=None):
        if name == "openai" or name.startswith("openai."):
            raise ImportError(name)
        return None


sys.meta_path.insert(0, _BlockOpenAI())
import penampakan.llms.openai as adapter
from penampakan.errors import ConfigurationError

try:
    adapter.OpenAITextLLM(model="gpt-4.1-mini")
except ConfigurationError as error:
    assert error.code == "missing_optional_dependency", error.code
    print("import-safe")
else:
    raise AssertionError("construction must fail without the SDK")
"""


def test_module_imports_with_the_sdk_absent() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "import-safe" in result.stdout
