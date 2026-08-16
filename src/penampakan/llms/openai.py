"""OpenAI Responses-API text language-model adapter."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal, cast

from penampakan.errors import ConfigurationError
from penampakan.llms._base import (
    ProviderLifecycle,
    SchemaCompilerCache,
    finalize_action_text,
    missing_dependency,
    resolve_ownership,
)
from penampakan.llms._openai_transport import (
    PROVIDER,
    ParsedResponse,
    backend_fingerprint,
    build_request_kwargs,
    classify_failure,
    parse_response,
    resolve_capabilities,
    safe_sdk_version,
    validate_extra_body,
)
from penampakan.llms._retry import Deadline, RandomSource, call_with_retries
from penampakan.llms.schema import SchemaTarget
from penampakan.models import LLMRequest, LLMResponse, RetryPolicy, SchemaEnforcement

_CreateCall = Callable[..., Awaitable[object]]


class OpenAITextLLM:
    """Strict structured-output text LLM backed by the OpenAI Responses API.

    The module imports without the ``openai`` package installed; construction
    performs the optional import and raises a configuration error when it is
    absent. A client constructed here disables native SDK retries so provider
    attempts cannot multiply. An injected client stays caller-owned and MUST
    also have its native retries disabled (``max_retries=0``) whenever adapter
    retries are configured, otherwise the two retry layers multiply.
    """

    def __init__(
        self,
        *,
        model: str,
        client: object | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        retry_policy: RetryPolicy | None = None,
        owns_client: bool | None = None,
        instruction_role: Literal["auto", "system", "developer"] = "auto",
        extra_body: Mapping[str, object] | None = None,
    ) -> None:
        try:
            import openai
        except ImportError as error:
            raise missing_dependency("openai") from error
        if not isinstance(model, str) or not model.strip():
            raise ConfigurationError(code="invalid_model")
        if retry_policy is not None and not isinstance(retry_policy, RetryPolicy):
            raise ConfigurationError(code="invalid_retry_policy")
        self._model = model.strip()
        self._capabilities = resolve_capabilities(self._model, instruction_role=instruction_role)
        self._extra_body = validate_extra_body(extra_body)
        self._retry_policy = RetryPolicy() if retry_policy is None else retry_policy
        self._sdk_version = safe_sdk_version(getattr(openai, "__version__", None))
        self._schemas = SchemaCompilerCache(SchemaTarget.OPENAI_STRICT)
        injected = client
        if client is None:
            options: dict[str, Any] = {"max_retries": 0}
            if api_key is not None:
                options["api_key"] = api_key
            if base_url is not None:
                options["base_url"] = base_url
            client = openai.AsyncOpenAI(**options)
        self._client = client
        self._create = self._resolve_create(client)
        owned = resolve_ownership(client=injected, owns_client=owns_client)
        self._lifecycle = ProviderLifecycle(client if owned else None)
        # Private deterministic backoff source for adapter tests only; the
        # public retry contract never carries callable state.
        self._random_source: RandomSource | None = None

    @staticmethod
    def _resolve_create(client: object) -> _CreateCall:
        responses = getattr(client, "responses", None)
        create = getattr(responses, "create", None)
        if not callable(create):
            raise ConfigurationError(code="invalid_client")
        return cast("_CreateCall", create)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Complete one policy request with a strict compiled action schema."""
        self._lifecycle.require_open()
        compiled = self._schemas.compile(request.response_json_schema)
        base_kwargs = build_request_kwargs(
            request,
            model=self._model,
            compiled=compiled,
            capabilities=self._capabilities,
            extra_body=self._extra_body,
        )
        deadline = Deadline(request.timeout_s)
        attempted = 0

        async def operation() -> ParsedResponse:
            nonlocal attempted
            attempted += 1
            kwargs = dict(base_kwargs)
            remaining = deadline.remaining()
            if remaining is not None:
                # Bound the SDK call so it cannot outlive the total budget.
                kwargs["timeout"] = max(remaining, 0.0)
            response = await self._create(**kwargs)
            return parse_response(response, attempts=attempted)

        parsed, attempts = await call_with_retries(
            operation,
            policy=self._retry_policy,
            classify=classify_failure,
            deadline=deadline,
            provider=PROVIDER,
            random_source=self._random_source,
        )
        text = finalize_action_text(
            parsed.payload,
            request=request,
            provider=PROVIDER,
            attempts=attempts,
        )
        return LLMResponse(
            text=text,
            model_id=parsed.model_id,
            usage=parsed.usage,
            finish_reason=parsed.finish_reason,
            provider=PROVIDER,
            request_id=parsed.request_id,
            backend_fingerprint=backend_fingerprint(compiled, sdk_version=self._sdk_version),
            attempts=attempts,
            schema_enforcement=SchemaEnforcement.STRICT,
        )

    async def aclose(self) -> None:
        """Close an owned SDK client exactly once."""
        await self._lifecycle.aclose()

    async def __aenter__(self) -> OpenAITextLLM:
        """Enter an idempotent async context around this adapter."""
        return self

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Close the adapter when leaving its async context."""
        await self.aclose()


__all__ = ["OpenAITextLLM"]
