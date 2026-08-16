"""Adapter-private OpenAI Responses-API request shaping, parsing, and classification.

The public :class:`penampakan.llms.openai.OpenAITextLLM` exposes only the
provider-neutral ``TextLLM`` contracts; every Responses-specific detail lives
here. Nothing in this module imports the optional ``openai`` package at import
time, and nothing here ever carries prompt, response, schema, or credential
content into an exception.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Final, Literal

from penampakan.errors import ConfigurationError, LLMError
from penampakan.llms._retry import ProviderFailure
from penampakan.llms.schema import CompiledSchema
from penampakan.models import LLMRequest, MessageRole, TokenUsage

PROVIDER: Final = "openai"
SCHEMA_FORMAT_NAME: Final = "policy_action"
# The Responses API applies this temperature when the caller sends none, so a
# request that keeps it is compatible with models that reject the parameter.
PROVIDER_DEFAULT_TEMPERATURE: Final = 1.0

InstructionRole = Literal["system", "developer"]
InstructionRoleOption = Literal["auto", "system", "developer"]

_INSTRUCTION_ROLE_OPTIONS: Final = ("auto", "system", "developer")
# Keys the adapter controls; a caller cannot redefine them through extra_body.
_ADAPTER_CONTROLLED_KEYS: Final = frozenset(
    {"text", "input", "model", "max_output_tokens", "temperature"}
)
_MESSAGE_ROLES: Final = {MessageRole.USER: "user", MessageRole.ASSISTANT: "assistant"}
_FINISH_REASON = re.compile(r"^[a-z][a-z0-9_]*$")
_PROVIDER_CODE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_SDK_VERSION = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._-]{0,31}$")
_REFUSAL_PART_TYPES: Final = frozenset({"refusal", "output_refusal"})
# 408 is a timeout status; no other 4xx is retried.
_RETRYABLE_STATUS: Final = frozenset({408, 429})
_UNKNOWN_SDK_VERSION: Final = "unknown"


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """The request capabilities resolved for one OpenAI model."""

    instruction_role: InstructionRole
    supports_temperature: bool


@dataclass(frozen=True, slots=True)
class _CapabilityRow:
    """One capability-table row keyed by documented model-name prefixes."""

    prefixes: tuple[str, ...]
    capabilities: ModelCapabilities


# The single capability table. It holds model-family prefixes only: no prompt,
# schema, or credential content. Prefix conditionals are forbidden elsewhere.
_CAPABILITY_TABLE: Final = (
    _CapabilityRow(
        ("gpt-5", "o1", "o3", "o4"),
        ModelCapabilities(instruction_role="developer", supports_temperature=False),
    ),
    _CapabilityRow(
        ("gpt-4.1", "gpt-4o", "gpt-4-turbo"),
        ModelCapabilities(instruction_role="system", supports_temperature=True),
    ),
)
# Unknown models use the most broadly compatible documented role and assume the
# parameter is accepted; a provider rejection is normalized by the classifier.
DEFAULT_CAPABILITIES: Final = ModelCapabilities(
    instruction_role="system",
    supports_temperature=True,
)


def table_capabilities(model: str) -> ModelCapabilities:
    """Return the table capabilities for one model name."""
    name = model.strip().lower()
    for row in _CAPABILITY_TABLE:
        if name.startswith(row.prefixes):
            return row.capabilities
    return DEFAULT_CAPABILITIES


def resolve_capabilities(
    model: str,
    *,
    instruction_role: InstructionRoleOption = "auto",
) -> ModelCapabilities:
    """Resolve model capabilities, letting an explicit instruction role win."""
    if instruction_role not in _INSTRUCTION_ROLE_OPTIONS:
        raise ConfigurationError(code="invalid_instruction_role")
    capabilities = table_capabilities(model)
    if instruction_role == "auto":
        return capabilities
    return replace(capabilities, instruction_role=instruction_role)


def validate_extra_body(extra_body: Mapping[str, object] | None) -> dict[str, Any]:
    """Copy caller provider options, rejecting adapter-controlled keys."""
    if extra_body is None:
        return {}
    options: dict[str, Any] = {}
    for key, value in extra_body.items():
        if not isinstance(key, str):
            raise ConfigurationError(code="invalid_extra_body")
        if key in _ADAPTER_CONTROLLED_KEYS:
            raise ConfigurationError(code="conflicting_extra_body")
        options[key] = value
    return options


def safe_sdk_version(version: object) -> str:
    """Return a fingerprint-safe SDK version token."""
    if isinstance(version, str) and _SDK_VERSION.fullmatch(version):
        return version
    return _UNKNOWN_SDK_VERSION


def backend_fingerprint(compiled: CompiledSchema, *, sdk_version: str) -> str:
    """Return the provider/SDK/compiled-schema fingerprint for one response."""
    return f"{PROVIDER}/{sdk_version}/{compiled.fingerprint_sha256[:16]}"


def _input_items(request: LLMRequest, capabilities: ModelCapabilities) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for message in request.messages:
        role = (
            capabilities.instruction_role
            if message.role is MessageRole.SYSTEM
            else _MESSAGE_ROLES[message.role]
        )
        items.append({"role": role, "content": message.content})
    return items


def build_request_kwargs(
    request: LLMRequest,
    *,
    model: str,
    compiled: CompiledSchema,
    capabilities: ModelCapabilities,
    extra_body: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Build the Responses-API keyword arguments for one policy request."""
    unsupported_temperature = (
        not capabilities.supports_temperature
        and request.temperature != PROVIDER_DEFAULT_TEMPERATURE
    )
    if unsupported_temperature:
        # An unsupported requested parameter fails configuration rather than
        # silently disappearing from the provider call.
        raise ConfigurationError(code="unsupported_request_parameter")
    kwargs: dict[str, Any] = {
        "model": model,
        "input": _input_items(request, capabilities),
        "text": {
            "format": {
                "type": "json_schema",
                "name": SCHEMA_FORMAT_NAME,
                "strict": True,
                "schema": compiled.schema,
            }
        },
        "max_output_tokens": request.max_output_tokens,
    }
    if capabilities.supports_temperature:
        kwargs["temperature"] = request.temperature
    options = validate_extra_body(extra_body)
    if options:
        kwargs["extra_body"] = options
    return kwargs


@dataclass(frozen=True, slots=True)
class ParsedResponse:
    """One structured assistant output plus its safe response metadata."""

    payload: str
    model_id: str | None
    usage: TokenUsage | None
    finish_reason: str
    request_id: str | None


def finish_reason(status: object, reason: object) -> str:
    """Derive a safe, stable finish-reason token from the response status."""
    if isinstance(reason, str) and _FINISH_REASON.fullmatch(reason.strip()):
        return f"incomplete_{reason.strip()}"
    if isinstance(status, str):
        stripped = status.strip()
        if stripped == "completed":
            return "stop"
        if _FINISH_REASON.fullmatch(stripped):
            return stripped
    return "stop"


def _clean_text(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _token_count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _usage(response: object) -> TokenUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    input_tokens = _token_count(getattr(usage, "input_tokens", None))
    output_tokens = _token_count(getattr(usage, "output_tokens", None))
    if input_tokens is None and output_tokens is None:
        return None
    return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)


def _output_items(response: object) -> Sequence[object]:
    output = getattr(response, "output", None)
    if isinstance(output, Sequence) and not isinstance(output, (str, bytes)):
        return output
    return ()


def _content_parts(item: object) -> Sequence[object]:
    content = getattr(item, "content", None)
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        return content
    return ()


def _fail(code: str, attempts: int) -> LLMError:
    # Only a stable code, the provider name, and the attempt count are exposed.
    return LLMError(code=code, attempts=attempts, provider=PROVIDER)


def parse_response(response: object, *, attempts: int) -> ParsedResponse:
    """Extract exactly one structured output, mapping typed failures first.

    Typed refusal and truncation items are inspected before any text is parsed,
    so a refusal or truncated generation never reaches schema validation.
    """
    status = getattr(response, "status", None)
    details = getattr(response, "incomplete_details", None)
    reason = getattr(details, "reason", None)
    messages: list[object] = []
    for item in _output_items(response):
        if getattr(item, "type", None) != "message":
            continue
        messages.append(item)
        for part in _content_parts(item):
            if getattr(part, "type", None) in _REFUSAL_PART_TYPES:
                raise _fail("llm_refused", attempts)
    if reason == "content_filter":
        raise _fail("llm_refused", attempts)
    if status == "incomplete" or reason == "max_output_tokens":
        raise _fail("llm_output_truncated", attempts)
    if any(getattr(item, "status", None) == "incomplete" for item in messages):
        raise _fail("llm_output_truncated", attempts)
    if isinstance(status, str) and status != "completed":
        raise LLMError(code="llm_request_failed", attempts=attempts, provider=PROVIDER)
    if len(messages) != 1:
        raise _fail("llm_missing_structured_output", attempts)
    parts: list[str] = []
    for part in _content_parts(messages[0]):
        text = getattr(part, "text", None)
        if getattr(part, "type", None) == "output_text" and isinstance(text, str):
            parts.append(text)
    if not parts:
        raise _fail("llm_missing_structured_output", attempts)
    aggregate = getattr(response, "output_text", None)
    payload = aggregate if isinstance(aggregate, str) and aggregate.strip() else "".join(parts)
    return ParsedResponse(
        payload=payload,
        model_id=_clean_text(getattr(response, "model", None)),
        usage=_usage(response),
        finish_reason=finish_reason(status, reason),
        request_id=_clean_text(getattr(response, "id", None)),
    )


def _status_code(error: BaseException) -> int | None:
    status = getattr(error, "status_code", None)
    if isinstance(status, bool) or not isinstance(status, int):
        return None
    if not 100 <= status <= 599:
        return None
    return status


def _provider_code(error: BaseException) -> str | None:
    code = getattr(error, "code", None)
    if isinstance(code, str) and _PROVIDER_CODE.fullmatch(code):
        return code
    return None


def _transport_error_types() -> tuple[type[BaseException], ...]:
    try:
        import openai
    except ImportError:  # pragma: no cover - the SDK is present during a call
        return ()
    return (openai.APITimeoutError, openai.APIConnectionError)


def classify_failure(error: BaseException) -> ProviderFailure:
    """Classify one provider exception into a redacted retry decision."""
    status = _status_code(error)
    code = _provider_code(error)
    if status is not None:
        # Timeouts, 429, and 5xx only; every other 4xx is terminal.
        retryable = status in _RETRYABLE_STATUS or 500 <= status <= 599
        return ProviderFailure(retryable=retryable, status=status, code=code)
    # OSError covers TimeoutError and ConnectionError; the SDK wraps transport
    # failures in its own non-status error types.
    transport = isinstance(error, (OSError, *_transport_error_types()))
    return ProviderFailure(retryable=transport, code=code)


__all__ = [
    "DEFAULT_CAPABILITIES",
    "PROVIDER",
    "PROVIDER_DEFAULT_TEMPERATURE",
    "SCHEMA_FORMAT_NAME",
    "InstructionRole",
    "InstructionRoleOption",
    "ModelCapabilities",
    "ParsedResponse",
    "backend_fingerprint",
    "build_request_kwargs",
    "classify_failure",
    "finish_reason",
    "parse_response",
    "resolve_capabilities",
    "safe_sdk_version",
    "table_capabilities",
    "validate_extra_body",
]
