"""Anthropic Messages API text language model adapter with strict structured output."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Literal, Protocol, cast

from penampakan.errors import ConfigurationError
from penampakan.llms._anthropic_transport import (
    PROVIDER,
    AnthropicSDK,
    StructuredMode,
    backend_fingerprint,
    build_request,
    classify_failure,
    load_sdk,
    parse_message,
    resolve_structured_mode,
    validate_extra_body,
)
from penampakan.llms._base import (
    ProviderLifecycle,
    SchemaCompilerCache,
    finalize_action_text,
    missing_dependency,
    resolve_ownership,
)
from penampakan.llms._retry import Deadline, ProviderFailure, RandomSource, call_with_retries
from penampakan.llms.schema import SchemaTarget
from penampakan.models import LLMRequest, LLMResponse, RetryPolicy, SchemaEnforcement

_DEFAULT_RETRY_POLICY: Final = RetryPolicy()


class _MessagesResource(Protocol):
    """The single Messages-API operation this adapter calls."""

    async def create(self, **kwargs: object) -> object: ...


class AnthropicTextLLM:
    """Adapt the Anthropic Messages API as a strict-schema text language model.

    The module imports without the ``anthropic`` package; construction performs
    the optional import and raises
    ``ConfigurationError(code="missing_optional_dependency")`` when it is absent.

    In ``structured_mode="auto"`` the adapter uses native
    ``output_config.format`` JSON output when the configured model supports it
    and otherwise forces one client tool with ``strict: true``. A model that
    supports neither fails construction with
    ``ConfigurationError(code="strict_output_unsupported")``; this adapter has no
    degraded JSON-only behaviour.

    A client this adapter constructs disables native SDK retries so provider
    attempts cannot multiply. An **injected client must also disable its native
    retries** (``max_retries=0``) whenever adapter retries are configured, and
    stays caller-owned unless ``owns_client`` says otherwise::

        async with anthropic.AsyncAnthropic(max_retries=0) as client:
            async with AnthropicTextLLM(model="claude-opus-5", client=client) as llm:
                ...
    """

    __slots__ = (
        "_extra_body",
        "_lifecycle",
        "_messages",
        "_mode",
        "_model",
        "_policy",
        "_random_source",
        "_schemas",
        "_sdk",
    )

    def __init__(
        self,
        *,
        model: str,
        client: object | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        retry_policy: RetryPolicy | None = None,
        owns_client: bool | None = None,
        structured_mode: Literal["auto", "json_output", "strict_tool"] = "auto",
        extra_body: Mapping[str, object] | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ConfigurationError(code="invalid_model")
        if retry_policy is not None and not isinstance(retry_policy, RetryPolicy):
            raise ConfigurationError(code="invalid_retry_policy")
        try:
            import anthropic
        except ImportError as error:
            raise missing_dependency("anthropic") from error

        self._model = model.strip()
        self._sdk: AnthropicSDK = load_sdk(anthropic)
        self._mode: StructuredMode = resolve_structured_mode(self._model, structured_mode)
        self._extra_body = validate_extra_body(extra_body)
        self._policy = _DEFAULT_RETRY_POLICY if retry_policy is None else retry_policy
        self._schemas = SchemaCompilerCache(SchemaTarget.ANTHROPIC_STRICT)
        # A deterministic random source is supplied privately by adapter tests;
        # the public wire contract never exposes callable state.
        self._random_source: RandomSource | None = None

        resolved = self._resolve_client(client, api_key=api_key, base_url=base_url)
        self._messages = _messages_resource(resolved)
        owned = resolve_ownership(client=client, owns_client=owns_client)
        self._lifecycle = ProviderLifecycle(resolved if owned else None)

    def _resolve_client(
        self,
        client: object | None,
        *,
        api_key: str | None,
        base_url: str | None,
    ) -> object:
        if client is not None:
            if api_key is not None or base_url is not None:
                raise ConfigurationError(code="conflicting_client_options")
            return client
        options: dict[str, object] = {"max_retries": 0}
        if api_key is not None:
            options["api_key"] = api_key
        if base_url is not None:
            options["base_url"] = base_url
        try:
            # Native SDK retries are disabled so adapter retries cannot multiply.
            return self._sdk.client_factory(**options)
        except Exception as error:
            # Endpoint credentials and base URLs never reach the exception.
            raise ConfigurationError(code="invalid_client_options", cause=error) from error

    @property
    def structured_mode(self) -> StructuredMode:
        """Return the strict path resolved for the configured model."""
        return self._mode

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Request one strict policy action and locally validate the result."""
        self._lifecycle.require_open()
        compiled = self._schemas.compile(request.response_json_schema)
        deadline = Deadline(request.timeout_s)
        # The request is built once, before the retry loop, so a configuration
        # failure is never reported as a provider transport failure.
        base = build_request(
            model=self._model,
            request=request,
            compiled=compiled,
            mode=self._mode,
            timeout_s=None,
            extra_body=self._extra_body,
        )

        async def operation() -> object:
            payload = dict(base)
            remaining = deadline.remaining()
            if remaining is not None:
                # The SDK timeout tracks the remaining total budget, so a
                # provider call cannot outlive the adapter deadline.
                payload["timeout"] = remaining
            return await self._messages.create(**payload)

        message, attempts = await call_with_retries(
            operation,
            policy=self._policy,
            classify=self._classify,
            deadline=deadline,
            provider=PROVIDER,
            random_source=self._random_source,
        )
        output = parse_message(message, mode=self._mode, attempts=attempts)
        # Local post-validation always runs, even after provider strict enforcement.
        text = finalize_action_text(
            output.payload,
            request=request,
            provider=PROVIDER,
            attempts=attempts,
        )
        return LLMResponse(
            text=text,
            model_id=output.model_id,
            usage=output.usage,
            finish_reason=output.finish_reason,
            provider=PROVIDER,
            request_id=output.request_id,
            backend_fingerprint=backend_fingerprint(self._mode, self._sdk.version, compiled),
            attempts=attempts,
            schema_enforcement=SchemaEnforcement.STRICT,
        )

    def _classify(self, error: BaseException) -> ProviderFailure:
        return classify_failure(error, sdk=self._sdk)

    async def aclose(self) -> None:
        """Close an owned SDK client exactly once."""
        await self._lifecycle.aclose()

    async def __aenter__(self) -> AnthropicTextLLM:
        """Enter the adapter context without changing ownership."""
        return self

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Close an owned SDK client when the context ends."""
        await self.aclose()


def _messages_resource(client: object) -> _MessagesResource:
    messages = getattr(client, "messages", None)
    if messages is None or not callable(getattr(messages, "create", None)):
        raise ConfigurationError(code="invalid_client")
    return cast(_MessagesResource, messages)


__all__ = ["AnthropicTextLLM"]
