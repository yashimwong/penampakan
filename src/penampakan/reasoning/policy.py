"""Structured JSON action policy for provider-neutral text language models."""

from __future__ import annotations

from penampakan.errors import InvalidModelActionError, LLMError
from penampakan.models import LLMResponse, PolicyAction, PolicyInput, ToolAction, WarningInfo
from penampakan.protocols import TextLLM
from penampakan.reasoning.actions import ActionParseError, parse_policy_action
from penampakan.reasoning.prompts import PROMPT_VERSION, build_policy_request


class JsonActionPolicy:
    """Use one text-LLM call per invocation to select a strict JSON action."""

    def __init__(
        self,
        llm: TextLLM,
        *,
        prompt_version: str = PROMPT_VERSION,
        timeout_s: float | None = None,
        max_output_tokens: int = 800,
    ) -> None:
        if prompt_version != PROMPT_VERSION:
            raise ValueError("unsupported prompt version")
        if isinstance(max_output_tokens, bool) or max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        self._llm = llm
        self._timeout_s = timeout_s
        self._max_output_tokens = max_output_tokens

    @property
    def prompt_version(self) -> str:
        """Return the immutable prompt interface version."""
        return PROMPT_VERSION

    async def next_action(self, input: PolicyInput) -> PolicyAction:
        """Request and parse one action, enforcing single-repair termination."""
        request = build_policy_request(
            input,
            timeout_s=self._timeout_s,
            max_output_tokens=self._max_output_tokens,
        )
        try:
            response = await self._llm.complete(request)
        except LLMError:
            raise
        except Exception as error:
            raise LLMError(cause=error) from error
        if not isinstance(response, LLMResponse):
            raise LLMError(code="invalid_llm_response")
        try:
            action = parse_policy_action(response.text)
            self._validate_declared_action(action, input, response.text)
        except ActionParseError as error:
            if input.validation_feedback or input.invalid_model_output is not None:
                raise InvalidModelActionError(cause=error) from error
            raise
        return action

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
