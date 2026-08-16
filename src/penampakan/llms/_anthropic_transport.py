"""Adapter-private Anthropic Messages API request building, parsing, and classification.

Every Messages-API detail lives here so the public ``AnthropicTextLLM`` class
exposes only ``TextLLM`` contracts. One capability table decides which strict
structured-output path a model supports, so no prefix conditionals appear in the
request path. Nothing in this module imports the optional ``anthropic`` package
at import time; the adapter passes the imported module in during construction.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Final, Literal, cast

from penampakan.errors import ConfigurationError, LLMError
from penampakan.llms._retry import ProviderFailure
from penampakan.llms.schema import CompiledSchema
from penampakan.models import LLMRequest, MessageRole, TokenUsage

PROVIDER: Final = "anthropic"

# The forced tool used by the strict-tool path. The name and description are
# static library constants, never caller data.
TOOL_NAME: Final = "policy_action"
TOOL_DESCRIPTION: Final = "Emit the single policy action for this step as one strict JSON object."

# Request-body keys the adapter controls; caller extras may never replace them.
ADAPTER_BODY_KEYS: Final = frozenset(
    {"model", "messages", "system", "max_tokens", "tools", "tool_choice", "output_config"}
)

StructuredMode = Literal["json_output", "strict_tool"]
ConfiguredMode = Literal["auto", "json_output", "strict_tool"]

_REFUSAL_STOP: Final = "refusal"
_TRUNCATED_STOPS: Final = frozenset({"max_tokens", "model_context_window_exceeded"})
_FINISH_REASON: Final = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_IDENTIFIER: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,127}$")
_VERSION: Final = re.compile(r"^[0-9A-Za-z][0-9A-Za-z_.+-]{0,31}$")
# A capability prefix only matches at a model-name boundary, so `claude-opus-5`
# never claims `claude-opus-4-5` or a longer unrelated family.
_MODEL_BOUNDARIES: Final = frozenset({"-", "@", ":", "."})
_UNKNOWN_VERSION: Final = "unknown"


@dataclass(frozen=True, slots=True)
class ModelCapability:
    """Structured-output support declared for one Anthropic model family."""

    json_output: bool
    strict_tools: bool


_STRUCTURED: Final = ModelCapability(json_output=True, strict_tools=True)
_LEGACY: Final = ModelCapability(json_output=False, strict_tools=False)

# The one capability table for this adapter. Models documented as supporting
# structured outputs get native `output_config.format` JSON output and strict
# tool use; every other `claude-*` model gets neither.
CAPABILITY_TABLE: Final[tuple[tuple[str, ModelCapability], ...]] = (
    ("claude-opus-5", _STRUCTURED),
    ("claude-opus-4-8", _STRUCTURED),
    ("claude-fable-5", _STRUCTURED),
    ("claude-mythos-5", _STRUCTURED),
    ("claude-sonnet-5", _STRUCTURED),
    ("claude-haiku-4-5", _STRUCTURED),
    ("claude-opus-4-5", _STRUCTURED),
    ("claude-opus-4-1", _STRUCTURED),
)

UNKNOWN_CAPABILITY: Final = _LEGACY


@dataclass(frozen=True, slots=True)
class AnthropicSDK:
    """The optional-SDK entry points this adapter depends on."""

    version: str
    client_factory: Callable[..., object]
    timeout_errors: tuple[type[BaseException], ...]
    connection_errors: tuple[type[BaseException], ...]


@dataclass(frozen=True, slots=True)
class ProviderOutput:
    """One parsed Messages-API response, reduced to safe metadata."""

    payload: str
    request_id: str | None
    model_id: str | None
    finish_reason: str | None
    usage: TokenUsage | None


def _safe_version(value: object) -> str:
    if isinstance(value, str) and _VERSION.fullmatch(value):
        return value
    return _UNKNOWN_VERSION


def _error_classes(module: ModuleType, names: Sequence[str]) -> tuple[type[BaseException], ...]:
    found: list[type[BaseException]] = []
    for name in names:
        candidate = getattr(module, name, None)
        if isinstance(candidate, type) and issubclass(candidate, BaseException):
            found.append(candidate)
    return tuple(found)


def load_sdk(module: ModuleType) -> AnthropicSDK:
    """Capture the client factory, version, and transport error classes."""
    factory = getattr(module, "AsyncAnthropic", None)
    if not callable(factory):
        raise ConfigurationError(code="unsupported_anthropic_sdk")
    return AnthropicSDK(
        version=_safe_version(getattr(module, "__version__", None)),
        client_factory=cast(Callable[..., object], factory),
        timeout_errors=_error_classes(module, ("APITimeoutError",)),
        connection_errors=_error_classes(module, ("APIConnectionError",)),
    )


def normalize_model(model: str) -> str:
    """Return the comparable model name without a cloud-provider prefix."""
    lowered = model.strip().lower()
    marker = lowered.find("claude")
    return lowered if marker <= 0 else lowered[marker:]


def _matches(model: str, prefix: str) -> bool:
    if model == prefix:
        return True
    return model.startswith(prefix) and model[len(prefix)] in _MODEL_BOUNDARIES


def resolve_capability(model: str) -> ModelCapability:
    """Return the declared structured-output support for one model name."""
    normalized = normalize_model(model)
    for prefix, capability in CAPABILITY_TABLE:
        if _matches(normalized, prefix):
            return capability
    return UNKNOWN_CAPABILITY


def resolve_structured_mode(model: str, requested: ConfiguredMode) -> StructuredMode:
    """Resolve the strict path for one model, failing when none is supported."""
    capability = resolve_capability(model)
    if requested == "auto":
        if capability.json_output:
            return "json_output"
        if capability.strict_tools:
            return "strict_tool"
        raise ConfigurationError(code="strict_output_unsupported")
    if requested == "json_output":
        if capability.json_output:
            return "json_output"
        raise ConfigurationError(code="strict_output_unsupported")
    if requested == "strict_tool":
        if capability.strict_tools:
            return "strict_tool"
        raise ConfigurationError(code="strict_output_unsupported")
    raise ConfigurationError(code="invalid_structured_mode")


def validate_extra_body(extra_body: Mapping[str, object] | None) -> dict[str, object]:
    """Copy caller request extras, rejecting adapter-controlled keys."""
    if extra_body is None:
        return {}
    if not isinstance(extra_body, Mapping):
        raise ConfigurationError(code="invalid_extra_body")
    result: dict[str, object] = {}
    for key, value in extra_body.items():
        if not isinstance(key, str):
            raise ConfigurationError(code="invalid_extra_body")
        if key in ADAPTER_BODY_KEYS:
            raise ConfigurationError(code="conflicting_extra_body")
        result[key] = value
    return result


def backend_fingerprint(mode: StructuredMode, sdk_version: str, compiled: CompiledSchema) -> str:
    """Return the non-sensitive fingerprint naming the strict path that ran."""
    return f"{PROVIDER}/{mode}/{sdk_version}/{compiled.fingerprint_sha256[:16]}"


def build_request(
    *,
    model: str,
    request: LLMRequest,
    compiled: CompiledSchema,
    mode: StructuredMode,
    timeout_s: float | None,
    extra_body: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the Messages-API keyword arguments for one attempt."""
    instructions: list[str] = []
    conversation: list[dict[str, str]] = []
    for message in request.messages:
        if message.role is MessageRole.SYSTEM:
            # Anthropic carries the system instruction in a top-level field, so
            # it is never sent as a conversation message.
            instructions.append(message.content)
            continue
        conversation.append({"role": message.role.value, "content": message.content})
    if not conversation:
        raise ConfigurationError(code="unsupported_request_messages")

    payload: dict[str, object] = {
        "model": model,
        "messages": conversation,
        "max_tokens": request.max_output_tokens,
        "temperature": request.temperature,
    }
    if instructions:
        payload["system"] = "\n\n".join(instructions)
    if mode == "json_output":
        payload["output_config"] = {
            "format": {"type": "json_schema", "schema": compiled.schema},
        }
    else:
        payload["tools"] = [
            {
                "name": TOOL_NAME,
                "description": TOOL_DESCRIPTION,
                "input_schema": compiled.schema,
                "strict": True,
            }
        ]
        payload["tool_choice"] = {"type": "tool", "name": TOOL_NAME}
    if timeout_s is not None:
        # The SDK timeout follows the remaining total budget so a provider call
        # cannot outlive the adapter deadline.
        payload["timeout"] = timeout_s
    extras = validate_extra_body(extra_body)
    if extras:
        payload["extra_body"] = extras
    return payload


def _safe_identifier(value: object) -> str | None:
    if isinstance(value, str) and _IDENTIFIER.fullmatch(value):
        return value
    return None


def finish_reason(stop_reason: object) -> str | None:
    """Return the stable safe token derived from a provider stop reason."""
    if isinstance(stop_reason, str) and _FINISH_REASON.fullmatch(stop_reason):
        return stop_reason
    return None


def _token_count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _usage(message: object) -> TokenUsage | None:
    usage = getattr(message, "usage", None)
    if usage is None:
        return None
    input_tokens = _token_count(getattr(usage, "input_tokens", None))
    output_tokens = _token_count(getattr(usage, "output_tokens", None))
    if input_tokens is None and output_tokens is None:
        return None
    return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)


def _blocks(message: object) -> tuple[object, ...]:
    content = getattr(message, "content", None)
    if isinstance(content, Sequence) and not isinstance(content, str | bytes):
        return tuple(content)
    return ()


def _json_text(message: object) -> str | None:
    for block in _blocks(message):
        if getattr(block, "type", None) != "text":
            continue
        text = getattr(block, "text", None)
        if isinstance(text, str) and text.strip():
            return text
    return None


def _tool_payload(message: object, *, attempts: int) -> str | None:
    for block in _blocks(message):
        if getattr(block, "type", None) != "tool_use":
            continue
        if getattr(block, "name", None) != TOOL_NAME:
            continue
        arguments = getattr(block, "input", None)
        if not isinstance(arguments, Mapping):
            raise LLMError(
                code="llm_invalid_structured_output",
                attempts=attempts,
                provider=PROVIDER,
            )
        try:
            return json.dumps(dict(arguments), ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise LLMError(
                code="llm_invalid_structured_output",
                attempts=attempts,
                provider=PROVIDER,
                cause=error,
            ) from error
    return None


def parse_message(message: object, *, mode: StructuredMode, attempts: int) -> ProviderOutput:
    """Reject refusal and truncation, then extract the structured payload.

    Refusal, truncation, and absent structured output are distinct terminal
    outcomes checked before any parsing, and none of them is retried.
    """
    stop_reason = getattr(message, "stop_reason", None)
    if stop_reason == _REFUSAL_STOP:
        raise LLMError(code="llm_refused", attempts=attempts, provider=PROVIDER)
    if isinstance(stop_reason, str) and stop_reason in _TRUNCATED_STOPS:
        raise LLMError(code="llm_output_truncated", attempts=attempts, provider=PROVIDER)
    payload = (
        _json_text(message) if mode == "json_output" else _tool_payload(message, attempts=attempts)
    )
    if payload is None:
        raise LLMError(
            code="llm_missing_structured_output",
            attempts=attempts,
            provider=PROVIDER,
        )
    return ProviderOutput(
        payload=payload,
        request_id=_safe_identifier(getattr(message, "id", None)),
        model_id=_safe_identifier(getattr(message, "model", None)),
        finish_reason=finish_reason(stop_reason),
        usage=_usage(message),
    )


def _status_code(error: BaseException) -> int | None:
    status = getattr(error, "status_code", None)
    if isinstance(status, bool) or not isinstance(status, int):
        return None
    if not 100 <= status <= 599:
        return None
    return status


def _status_failure(status: int) -> ProviderFailure:
    if status == 408:
        # A timeout status is retried; no other 4xx is.
        return ProviderFailure(retryable=True, status=status, code="provider_timeout")
    if status == 429:
        return ProviderFailure(retryable=True, status=status, code="rate_limited")
    if status == 529:
        return ProviderFailure(retryable=True, status=status, code="overloaded")
    if status >= 500:
        return ProviderFailure(retryable=True, status=status, code="server_error")
    if status == 401:
        return ProviderFailure(retryable=False, status=status, code="authentication_failed")
    if status == 403:
        return ProviderFailure(retryable=False, status=status, code="permission_denied")
    if status == 404:
        return ProviderFailure(retryable=False, status=status, code="not_found")
    if status == 400:
        return ProviderFailure(retryable=False, status=status, code="invalid_request")
    return ProviderFailure(retryable=False, status=status, code="client_error")


def classify_failure(error: BaseException, *, sdk: AnthropicSDK) -> ProviderFailure:
    """Classify one provider exception without carrying provider prose.

    Only connection failures, timeouts, 429, and 5xx (including 529 overloaded)
    are retryable; authentication, permission, and ordinary 4xx failures are
    terminal.
    """
    status = _status_code(error)
    if status is not None:
        return _status_failure(status)
    if isinstance(error, (*sdk.timeout_errors, TimeoutError)):
        return ProviderFailure(retryable=True, code="timeout")
    if isinstance(error, sdk.connection_errors):
        return ProviderFailure(retryable=True, code="connection_failed")
    return ProviderFailure(retryable=False, code="provider_error")


__all__ = [
    "ADAPTER_BODY_KEYS",
    "CAPABILITY_TABLE",
    "PROVIDER",
    "TOOL_DESCRIPTION",
    "TOOL_NAME",
    "UNKNOWN_CAPABILITY",
    "AnthropicSDK",
    "ConfiguredMode",
    "ModelCapability",
    "ProviderOutput",
    "StructuredMode",
    "backend_fingerprint",
    "build_request",
    "classify_failure",
    "finish_reason",
    "load_sdk",
    "normalize_model",
    "parse_message",
    "resolve_capability",
    "resolve_structured_mode",
    "validate_extra_body",
]
