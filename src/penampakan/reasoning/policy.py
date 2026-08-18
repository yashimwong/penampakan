"""Structured JSON action policy for provider-neutral text language models."""

from __future__ import annotations

import asyncio

from penampakan.errors import InvalidModelActionError, LLMError
from penampakan.models import (
    LLMResponse,
    PolicyAction,
    PolicyInput,
    SchemaEnforcement,
    ToolAction,
    WarningInfo,
)
from penampakan.protocols import TextLLM
from penampakan.reasoning._metrics import record_policy_response
from penampakan.reasoning.actions import ActionParseError, parse_policy_action
from penampakan.reasoning.prompts import (
    PROMPT_VERSION,
    SUPPORTED_PROMPT_VERSIONS,
    build_policy_request,
)

_DEGRADED_SCHEMA_ENFORCEMENT = WarningInfo(
    code="degraded_schema_enforcement",
    message="The language model provider could not enforce the action schema strictly.",
    details={"schema_enforcement": SchemaEnforcement.JSON_ONLY.value},
)


class JsonActionPolicy:
    """Use one text-LLM call per invocation to select a strict JSON action.

    The policy owns the configured language model only when ``owns_llm`` is
    explicitly set, so a caller-supplied adapter stays caller-owned and remains
    usable after this policy closes.
    """

    def __init__(
        self,
        llm: TextLLM,
        *,
        prompt_version: str = PROMPT_VERSION,
        timeout_s: float | None = None,
        max_output_tokens: int = 800,
        temperature: float = 0.0,
        owns_llm: bool = False,
    ) -> None:
        if prompt_version not in SUPPORTED_PROMPT_VERSIONS:
            raise ValueError("unsupported prompt version")
        if isinstance(max_output_tokens, bool) or max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise TypeError("temperature must be a number")
        if not isinstance(owns_llm, bool):
            raise TypeError("owns_llm must be a bool")
        self._llm = llm
        self._prompt_version = prompt_version
        self._timeout_s = timeout_s
        self._max_output_tokens = max_output_tokens
        self._temperature = float(temperature)
        self._owns_llm = owns_llm
        self._degradations: dict[str, WarningInfo] = {}
        self._close_task: asyncio.Task[None] | None = None

    @property
    def prompt_version(self) -> str:
        """Return the immutable prompt interface version selected at construction."""
        return self._prompt_version

    @property
    def owns_llm(self) -> bool:
        """Return whether closing this policy also closes its language model."""
        return self._owns_llm

    @property
    def degradations(self) -> tuple[WarningInfo, ...]:
        """Return typed provider degradations observed by this policy.

        The set only grows, so concurrent runs sharing one policy all observe a
        degradation instead of racing to consume it.
        """
        return tuple(self._degradations.values())

    async def next_action(self, input: PolicyInput) -> PolicyAction:
        """Request and parse one action, enforcing single-repair termination."""
        request = build_policy_request(
            input,
            timeout_s=self._timeout_s,
            max_output_tokens=self._max_output_tokens,
            temperature=self._temperature,
        )
        try:
            response = await self._llm.complete(request)
        except LLMError:
            raise
        except Exception as error:
            raise LLMError(cause=error) from error
        if not isinstance(response, LLMResponse):
            raise LLMError(code="invalid_llm_response")
        record_policy_response(response)
        self._record_enforcement(response)
        try:
            action = parse_policy_action(response.text)
            self._validate_declared_action(action, input, response.text)
        except ActionParseError as error:
            if input.validation_feedback or input.invalid_model_output is not None:
                raise InvalidModelActionError(cause=error) from error
            raise
        return action

    async def aclose(self) -> None:
        """Close an owned language model exactly once."""
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_owned())
        await asyncio.shield(self._close_task)

    async def __aenter__(self) -> JsonActionPolicy:
        """Enter this policy context."""
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        """Close this policy context."""
        await self.aclose()

    async def _close_owned(self) -> None:
        if not self._owns_llm:
            return
        closer = getattr(self._llm, "aclose", None)
        if closer is None or not callable(closer):
            return
        await closer()

    def _record_enforcement(self, response: LLMResponse) -> None:
        if response.schema_enforcement is not SchemaEnforcement.JSON_ONLY:
            return
        # Carried as typed policy state; never logged with request content.
        self._degradations.setdefault(
            _DEGRADED_SCHEMA_ENFORCEMENT.code,
            _DEGRADED_SCHEMA_ENFORCEMENT,
        )

    @staticmethod
    def _validate_declared_action(
        action: PolicyAction,
        input: PolicyInput,
        model_output: str,
    ) -> None:
        if not isinstance(action, ToolAction):
            return
        if input.answer_only:
            feedback = WarningInfo(
                code="tool_action_not_allowed",
                message="The response schema permits only an answer action.",
                details={"location": "$.type", "error_type": "answer_only"},
            )
            raise ActionParseError((feedback,), invalid_model_output=model_output)
        if action.tool not in {tool.name for tool in input.tools}:
            feedback = WarningInfo(
                code="undeclared_tool",
                message="The requested tool is not declared in the response schema.",
                details={"location": "$.tool", "error_type": "literal_error"},
            )
            raise ActionParseError((feedback,), invalid_model_output=model_output)


__all__ = ["JsonActionPolicy"]
