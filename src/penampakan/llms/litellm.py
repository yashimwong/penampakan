"""Optional LiteLLM text-LLM adapter with capability-checked schema enforcement.

The LiteLLM package is imported during construction only, so this module stays
importable on a base install. LiteLLM normalizes many providers onto the
OpenAI-compatible request surface, so the compiled OpenAI strict subset is the
strict path here.

Provider reference: <https://docs.litellm.ai/docs/completion/json_mode>
"""

from __future__ import annotations

import importlib.metadata
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Final, cast

from penampakan.errors import ConfigurationError, LLMError
from penampakan.llms._base import (
    ProviderLifecycle,
    SchemaCompilerCache,
    finalize_action_text,
    missing_dependency,
)
from penampakan.llms._retry import (
    Deadline,
    ProviderFailure,
    RandomSource,
    call_with_retries,
)
from penampakan.llms.schema import CompiledSchema, SchemaTarget, canonical_json
from penampakan.models import (
    JsonValue,
    LLMRequest,
    LLMResponse,
    MessageRole,
    RetryPolicy,
    SchemaEnforcement,
    TokenUsage,
)

_PROVIDER: Final = "litellm"
_SCHEMA_NAME: Final = "policy_action"
_PACKAGE: Final = "litellm"

# Adapter-controlled request keys. A caller keyword that would overwrite one of
# these is rejected instead of silently changing enforcement or budgets.
_CONTROLLED_KEYS: Final = frozenset(
    {
        "model",
        "messages",
        "response_format",
        "max_tokens",
        "temperature",
        "timeout",
        "num_retries",
    }
)

# LiteLLM re-exports the OpenAI exception hierarchy. Names are resolved from the
# installed package, so an absent name simply drops out of the table. The
# retryable table is checked first and lists subclasses before their bases.
_RETRYABLE_ERRORS: Final = (
    ("Timeout", "provider_timeout"),
    ("RateLimitError", "rate_limited"),
    ("APIConnectionError", "connection_failed"),
    ("InternalServerError", "server_error"),
    ("ServiceUnavailableError", "server_error"),
)
_TERMINAL_ERRORS: Final = (
    ("AuthenticationError", "authentication_failed"),
    ("PermissionDeniedError", "permission_denied"),
    ("ContentPolicyViolationError", "content_filtered"),
    ("ContextWindowExceededError", "context_window_exceeded"),
    ("NotFoundError", "not_found"),
    ("UnprocessableEntityError", "invalid_request"),
    ("BadRequestError", "invalid_request"),
)

# 408 is a timeout status; every other 4xx is terminal.
_RETRYABLE_STATUS: Final = frozenset({408, 429})

_MODEL_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,127}$")
_REQUEST_ID_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_FINISH_REASON_TOKEN = re.compile(r"^[a-z][a-z0-9_]*$")
_VERSION_TOKEN = re.compile(r"^[0-9][0-9A-Za-z._+-]{0,31}$")
_UNKNOWN_VERSION: Final = "unknown"

# The degraded prompt instruction. The envelope shape is named explicitly and the
# JSON-only compiled schema is appended, because nothing in JSON mode constrains
# the action beyond syntactic JSON.
_JSON_ONLY_INSTRUCTION: Final = (
    "Output requirements: reply with exactly one JSON object and nothing else. "
    "No prose, no explanation, and no code fences. The object must have exactly "
    'one property named "action", whose value is the single chosen action '
    "object. The complete reply must validate against this JSON Schema:"
)

ErrorTable = tuple[tuple[type[BaseException], str], ...]
CompletionCallable = Callable[..., Awaitable[object]]
ProbeCallable = Callable[[str], object]


@dataclass(frozen=True, slots=True)
class _Runtime:
    """The LiteLLM entry points, capability probes, and exception tables."""

    acompletion: CompletionCallable
    supported_params: ProbeCallable | None
    supports_response_schema: ProbeCallable | None
    version: str
    retryable_errors: ErrorTable
    terminal_errors: ErrorTable


def _error_table(module: ModuleType, names: Sequence[tuple[str, str]]) -> ErrorTable:
    """Resolve the exception classes this LiteLLM version actually exposes."""
    resolved: list[tuple[type[BaseException], str]] = []
    for name, code in names:
        candidate = getattr(module, name, None)
        if isinstance(candidate, type) and issubclass(candidate, BaseException):
            resolved.append((candidate, code))
    return tuple(resolved)


def _optional_probe(module: ModuleType, name: str) -> ProbeCallable | None:
    """Return one capability probe when this LiteLLM version provides it."""
    candidate = getattr(module, name, None)
    if not callable(candidate):
        return None
    return cast(ProbeCallable, candidate)


def _sdk_version() -> str:
    """Return a safe LiteLLM distribution version for the fingerprint."""
    # LiteLLM exposes no ``__version__`` attribute, so the distribution metadata
    # is the only version source.
    try:
        raw = importlib.metadata.version(_PACKAGE)
    except importlib.metadata.PackageNotFoundError:
        return _UNKNOWN_VERSION
    return raw if _VERSION_TOKEN.fullmatch(raw) else _UNKNOWN_VERSION


def _resolve_runtime() -> _Runtime:
    """Import LiteLLM and resolve everything this adapter needs from it."""
    try:
        import litellm
    except ImportError as error:
        raise missing_dependency(_PACKAGE) from error
    module: ModuleType = litellm
    entry = getattr(module, "acompletion", None)
    if not callable(entry):
        raise missing_dependency(_PACKAGE)
    return _Runtime(
        acompletion=cast(CompletionCallable, entry),
        supported_params=_optional_probe(module, "get_supported_openai_params"),
        supports_response_schema=_optional_probe(module, "supports_response_schema"),
        version=_sdk_version(),
        retryable_errors=_error_table(module, _RETRYABLE_ERRORS),
        terminal_errors=_error_table(module, _TERMINAL_ERRORS),
    )


def _safe_token(value: object, pattern: re.Pattern[str]) -> str | None:
    """Return provider metadata only when it matches a conservative shape."""
    if isinstance(value, str) and pattern.fullmatch(value):
        return value
    return None


def _token_count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _usage(response: object) -> TokenUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    input_tokens = _token_count(getattr(usage, "prompt_tokens", None))
    output_tokens = _token_count(getattr(usage, "completion_tokens", None))
    if input_tokens is None and output_tokens is None:
        return None
    return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)


def _status_code(error: BaseException) -> int | None:
    status = getattr(error, "status_code", None)
    if isinstance(status, bool) or not isinstance(status, int):
        return None
    return status if 100 <= status <= 599 else None


def _match_error(error: BaseException, table: ErrorTable) -> str | None:
    for error_type, code in table:
        if isinstance(error, error_type):
            return code
    return None


def _first_choice(response: object) -> object | None:
    choices = getattr(response, "choices", None)
    if not isinstance(choices, (list, tuple)) or not choices:
        return None
    return cast(object, choices[0])


class LiteLLMTextLLM:
    """LiteLLM text LLM whose schema enforcement is decided by capability probes.

    Before the first request the adapter asks LiteLLM which OpenAI parameters and
    response schemas the configured model supports. That probe, not provider
    rejection, selects the request path; a provider rejection is still normalized
    into a typed redacted error because capability metadata can be wrong.

    When strict structured output is unavailable the adapter fails unless
    ``allow_json_only=True``. JSON-only mode is a degraded mode and is never
    equivalent to strict structured output: the provider guarantees syntactic
    JSON only and enforces nothing about the action shape, so the reply is always
    post-validated locally and reported as ``SchemaEnforcement.JSON_ONLY``.

    The adapter constructs no SDK client, so it owns nothing; ``aclose`` is an
    idempotent no-op that still closes the adapter to further requests.
    """

    def __init__(
        self,
        *,
        model: str,
        allow_json_only: bool = False,
        retry_policy: RetryPolicy | None = None,
        **completion_kwargs: object,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ConfigurationError(code="invalid_model")
        if not isinstance(allow_json_only, bool):
            raise ConfigurationError(code="invalid_json_only_opt_in")
        if retry_policy is not None and not isinstance(retry_policy, RetryPolicy):
            raise ConfigurationError(code="invalid_retry_policy")
        if any(key in _CONTROLLED_KEYS for key in completion_kwargs):
            # The key name is caller data, so only the stable code is reported.
            raise ConfigurationError(code="conflicting_completion_kwargs")
        self._runtime = _resolve_runtime()
        self._model = model
        self._allow_json_only = allow_json_only
        self._retry_policy = RetryPolicy() if retry_policy is None else retry_policy
        self._completion_kwargs = dict(completion_kwargs)
        self._strict_cache = SchemaCompilerCache(SchemaTarget.OPENAI_STRICT)
        self._json_only_cache = SchemaCompilerCache(SchemaTarget.JSON_ONLY)
        self._strict_available: bool | None = None
        self._lifecycle = ProviderLifecycle(None)
        # Test-only jitter seam; the public retry contract carries no callables.
        self._random_source: RandomSource | None = None

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send one policy request and return its validated structured output."""
        self._lifecycle.require_open()
        strict = self._strict_schema_available()
        if not strict and not self._allow_json_only:
            raise ConfigurationError(code="strict_schema_unavailable")
        enforcement = SchemaEnforcement.STRICT if strict else SchemaEnforcement.JSON_ONLY
        compiled = (
            self._strict_cache.compile(request.response_json_schema)
            if strict
            else self._json_only_cache.compile(request.response_json_schema)
        )
        messages = self._messages(request, None if strict else compiled)
        response_format = self._response_format(compiled, strict=strict)
        deadline = Deadline(request.timeout_s)

        async def operation() -> object:
            return await self._runtime.acompletion(
                **self._call_kwargs(
                    request,
                    messages=messages,
                    response_format=response_format,
                    deadline=deadline,
                )
            )

        response, attempts = await call_with_retries(
            operation,
            policy=self._retry_policy,
            classify=self._classify,
            deadline=deadline,
            provider=_PROVIDER,
            random_source=self._random_source,
        )
        payload = self._payload(response, attempts=attempts)
        return LLMResponse(
            text=finalize_action_text(
                payload,
                request=request,
                provider=_PROVIDER,
                attempts=attempts,
            ),
            model_id=_safe_token(getattr(response, "model", None), _MODEL_TOKEN),
            usage=_usage(response),
            finish_reason=_safe_token(
                getattr(_first_choice(response), "finish_reason", None),
                _FINISH_REASON_TOKEN,
            ),
            provider=_PROVIDER,
            request_id=_safe_token(getattr(response, "id", None), _REQUEST_ID_TOKEN),
            backend_fingerprint=(
                f"{_PROVIDER}/{self._runtime.version}/{compiled.fingerprint_sha256[:16]}"
            ),
            attempts=attempts,
            schema_enforcement=enforcement,
        )

    async def aclose(self) -> None:
        """Close the adapter exactly once; it owns no SDK client."""
        await self._lifecycle.aclose()

    async def __aenter__(self) -> LiteLLMTextLLM:
        """Enter the adapter context."""
        self._lifecycle.require_open()
        return self

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Close the adapter on context exit."""
        await self.aclose()

    def _strict_schema_available(self) -> bool:
        """Probe LiteLLM once per instance for strict structured-output support."""
        cached = self._strict_available
        if cached is not None:
            return cached
        available = self._probe_strict_schema()
        self._strict_available = available
        return available

    def _probe_strict_schema(self) -> bool:
        params = self._probe(self._runtime.supported_params)
        if not isinstance(params, (list, tuple)) or "response_format" not in params:
            # An unknown model reports ``None`` or omits the parameter; that is
            # "capability unknown", which is never treated as support.
            return False
        return self._probe(self._runtime.supports_response_schema) is True

    def _probe(self, probe: ProbeCallable | None) -> object:
        if probe is None:
            return None
        try:
            return probe(self._model)
        except Exception:
            # A probe failure means the capability is unknown, so the strict path
            # is not offered. The provider error text is never surfaced.
            return None

    def _messages(
        self,
        request: LLMRequest,
        json_only_schema: CompiledSchema | None,
    ) -> list[dict[str, str]]:
        """Map provider-neutral text messages onto the OpenAI chat surface."""
        instructions = [
            message.content for message in request.messages if message.role is MessageRole.SYSTEM
        ]
        if json_only_schema is not None:
            envelope = canonical_json(cast(JsonValue, json_only_schema.schema))
            instructions.append(f"{_JSON_ONLY_INSTRUCTION}\n{envelope}")
        messages: list[dict[str, str]] = []
        if instructions:
            messages.append({"role": "system", "content": "\n\n".join(instructions)})
        messages.extend(
            {"role": message.role.value, "content": message.content}
            for message in request.messages
            if message.role is not MessageRole.SYSTEM
        )
        return messages

    def _response_format(self, compiled: CompiledSchema, *, strict: bool) -> dict[str, object]:
        if not strict:
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "json_schema": {
                "name": _SCHEMA_NAME,
                "strict": True,
                "schema": compiled.schema,
            },
        }

    def _call_kwargs(
        self,
        request: LLMRequest,
        *,
        messages: list[dict[str, str]],
        response_format: dict[str, object],
        deadline: Deadline,
    ) -> dict[str, object]:
        kwargs = dict(self._completion_kwargs)
        kwargs.update(
            model=self._model,
            messages=messages,
            response_format=response_format,
            max_tokens=request.max_output_tokens,
            temperature=request.temperature,
            # LiteLLM's own retries are disabled so provider attempts cannot
            # multiply with the shared adapter retry budget.
            num_retries=0,
        )
        remaining = deadline.remaining()
        if remaining is not None:
            # The per-call timeout keeps the SDK inside the total request budget.
            kwargs["timeout"] = max(remaining, 0.001)
        return kwargs

    def _classify(self, error: BaseException) -> ProviderFailure:
        """Classify one LiteLLM exception without retaining provider text."""
        status = _status_code(error)
        code = _match_error(error, self._runtime.retryable_errors)
        if code is not None:
            return ProviderFailure(retryable=True, status=status, code=code)
        code = _match_error(error, self._runtime.terminal_errors)
        if code is not None:
            return ProviderFailure(retryable=False, status=status, code=code)
        if status is not None:
            retryable = status in _RETRYABLE_STATUS or status >= 500
            return ProviderFailure(
                retryable=retryable,
                status=status,
                code="server_error" if retryable else "invalid_request",
            )
        if isinstance(error, (TimeoutError, ConnectionError)):
            return ProviderFailure(retryable=True, code="connection_failed")
        return ProviderFailure(retryable=False, code="provider_error")

    def _payload(self, response: object, *, attempts: int) -> str:
        """Return the structured payload, raising typed outcomes before parsing."""
        choice = _first_choice(response)
        if choice is None:
            raise self._failure("llm_missing_structured_output", attempts)
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "length":
            raise self._failure("llm_output_truncated", attempts)
        message = getattr(choice, "message", None)
        refusal = getattr(message, "refusal", None)
        if finish_reason == "content_filter" or (isinstance(refusal, str) and refusal.strip()):
            raise self._failure("llm_refused", attempts)
        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise self._failure("llm_missing_structured_output", attempts)
        return content

    @staticmethod
    def _failure(code: str, attempts: int) -> LLMError:
        return LLMError(code=code, attempts=attempts, provider=_PROVIDER)


__all__ = ["LiteLLMTextLLM"]
