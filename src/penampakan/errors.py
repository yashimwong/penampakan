from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar
from uuid import UUID

if TYPE_CHECKING:
    from .models import InspectionResult


_ERROR_CODE = re.compile(r"^[a-z][a-z0-9_]*$")
# Provider identifiers and error codes are only reported when they match this
# conservative shape, so no provider text can leak through an exception.
_SAFE_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")


class PenampakanError(Exception):
    """Base class for safe public Penampakan failures."""

    default_code: ClassVar[str] = "penampakan_error"
    default_message: ClassVar[str] = "Penampakan could not complete the operation."
    default_retryable: ClassVar[bool] = False

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        trace_id: UUID | None = None,
        backend_name: str | None = None,
        tool_name: str | None = None,
        retryable: bool | None = None,
        cause: BaseException | None = None,
        cause_summary: str | None = None,
    ) -> None:
        self.code = code if code is not None and _ERROR_CODE.fullmatch(code) else self.default_code
        self.trace_id = trace_id
        self.backend_name = backend_name
        self.tool_name = tool_name
        self.retryable = self.default_retryable if retryable is None else retryable
        self.cause_summary = self._summarize_cause(cause, cause_summary)
        self.safe_message = self.default_message
        super().__init__(self.safe_message)
        if cause is not None:
            self.__cause__ = cause

    @staticmethod
    def _summarize_cause(cause: BaseException | None, summary: str | None) -> str | None:
        if summary is not None:
            return "cause details redacted"
        if cause is None:
            return None
        return type(cause).__name__

    def __str__(self) -> str:
        return f"{self.safe_message} [{self.code}]"

    def __repr__(self) -> str:
        fields = [f"code={self.code!r}", f"retryable={self.retryable!r}"]
        if self.trace_id is not None:
            fields.append(f"trace_id={str(self.trace_id)!r}")
        return f"{type(self).__name__}({', '.join(fields)})"


class ConfigurationError(PenampakanError):
    """Raised when client configuration is inconsistent or invalid."""

    default_code = "configuration_error"
    default_message = "Penampakan configuration is invalid."

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        trace_id: UUID | None = None,
        backend_name: str | None = None,
        tool_name: str | None = None,
        retryable: bool | None = None,
        cause: BaseException | None = None,
        cause_summary: str | None = None,
        extra: str | None = None,
    ) -> None:
        # The installable extra is a static library constant, never caller data,
        # so naming it is the one detail a configuration failure must report for
        # the caller to be able to fix it.
        self.extra = extra if extra is not None and _SAFE_TOKEN.fullmatch(extra) else None
        super().__init__(
            message,
            code=code,
            trace_id=trace_id,
            backend_name=backend_name,
            tool_name=tool_name,
            retryable=retryable,
            cause=cause,
            cause_summary=cause_summary,
        )
        if self.extra is not None:
            self.safe_message = f"{self.safe_message} Install penampakan[{self.extra}]."
            self.args = (self.safe_message,)

    def __repr__(self) -> str:
        rendered = super().__repr__()
        if self.extra is None:
            return rendered
        return f"{rendered[:-1]}, extra={self.extra!r})"


class ImageError(PenampakanError):
    """Base class for image loading and normalization failures."""

    default_code = "image_error"
    default_message = "The image could not be processed."


class InvalidImageError(ImageError):
    """Raised when input bytes do not decode as a valid image."""

    default_code = "invalid_image"
    default_message = "The image is invalid or corrupted."


class UnsupportedImageError(ImageError):
    """Raised when an image format or image feature is unsupported."""

    default_code = "unsupported_image"
    default_message = "The image format or image feature is unsupported."


class ImageLimitExceededError(ImageError):
    """Raised when an input image exceeds a configured resource limit."""

    default_code = "image_limit_exceeded"
    default_message = "The image exceeds a configured resource limit."


class RemoteSourceDisabledError(ImageError):
    """Raised when a URL is supplied while remote image loading is disabled."""

    default_code = "remote_source_disabled"
    default_message = "Remote image sources are disabled."


class SessionError(PenampakanError):
    """Base class for session state and lookup failures."""

    default_code = "session_error"
    default_message = "The vision session could not complete the operation."


class SessionClosedError(SessionError):
    """Raised when an operation targets a closed session."""

    default_code = "session_closed"
    default_message = "The vision session is closed."


class AssetNotFoundError(SessionError):
    """Raised when an image asset ID is not present in the session."""

    default_code = "asset_not_found"
    default_message = "The requested image asset was not found."


class ObservationNotFoundError(SessionError):
    """Raised when an observation ID is not present in the session."""

    default_code = "observation_not_found"
    default_message = "The requested observation was not found."


class CapabilityError(PenampakanError):
    """Base class for perception capability failures."""

    default_code = "capability_error"
    default_message = "A perception capability could not complete the request."


class CapabilityUnavailableError(CapabilityError):
    """Raised when no registered backend supports a required capability."""

    default_code = "capability_unavailable"
    default_message = "The required perception capability is unavailable."


class BackendError(CapabilityError):
    """Base class for failures reported by a vision backend."""

    default_code = "backend_error"
    default_message = "A vision backend failed."


class BackendUnavailableError(BackendError):
    """Raised when a selected vision backend is unavailable."""

    default_code = "backend_unavailable"
    default_message = "A vision backend is unavailable."
    default_retryable = True


class BackendTimeoutError(BackendError):
    """Raised when a vision backend exceeds its deadline."""

    default_code = "backend_timeout"
    default_message = "A vision backend exceeded its deadline."
    default_retryable = True


class InvalidBackendOutputError(BackendError):
    """Raised when a backend returns data outside its strict contract."""

    default_code = "invalid_backend_output"
    default_message = "A vision backend returned invalid output."


class ToolExecutionError(CapabilityError):
    """Raised when a validated built-in tool cannot execute successfully."""

    default_code = "tool_execution_error"
    default_message = "A visual tool could not complete the request."


class ReasoningError(PenampakanError):
    """Base class for language-model, policy, and evidence failures."""

    default_code = "reasoning_error"
    default_message = "Visual reasoning could not complete the request."


class LLMNotConfiguredError(ReasoningError):
    """Raised when question answering is requested without a language model."""

    default_code = "llm_not_configured"
    default_message = "Question answering requires a configured language model."


class LLMError(ReasoningError):
    """Raised when the configured text language model fails.

    Provider adapters may attach a safe attempt count and the last provider
    status/error code. No prompt, response, header, or credential content is
    ever carried.
    """

    default_code = "llm_error"
    default_message = "The configured language model failed."

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        trace_id: UUID | None = None,
        backend_name: str | None = None,
        tool_name: str | None = None,
        retryable: bool | None = None,
        cause: BaseException | None = None,
        cause_summary: str | None = None,
        attempts: int | None = None,
        provider: str | None = None,
        provider_status: int | None = None,
        provider_code: str | None = None,
    ) -> None:
        self.attempts = attempts if isinstance(attempts, int) and attempts >= 1 else None
        self.provider = (
            provider if provider is not None and _SAFE_TOKEN.fullmatch(provider) else None
        )
        self.provider_status = (
            provider_status
            if isinstance(provider_status, int) and 100 <= provider_status <= 599
            else None
        )
        self.provider_code = (
            provider_code
            if provider_code is not None and _SAFE_TOKEN.fullmatch(provider_code)
            else None
        )
        super().__init__(
            message,
            code=code,
            trace_id=trace_id,
            backend_name=backend_name,
            tool_name=tool_name,
            retryable=retryable,
            cause=cause,
            cause_summary=cause_summary,
        )

    def __repr__(self) -> str:
        fields = [f"code={self.code!r}", f"retryable={self.retryable!r}"]
        if self.trace_id is not None:
            fields.append(f"trace_id={str(self.trace_id)!r}")
        if self.attempts is not None:
            fields.append(f"attempts={self.attempts!r}")
        if self.provider_status is not None:
            fields.append(f"provider_status={self.provider_status!r}")
        return f"{type(self).__name__}({', '.join(fields)})"


class InvalidModelActionError(ReasoningError):
    """Raised when a model action remains invalid after the allowed repair."""

    default_code = "invalid_model_action"
    default_message = "The language model returned an invalid action."


class PolicyDeniedError(ReasoningError):
    """Raised when policy authorization denies a requested action."""

    default_code = "policy_denied"
    default_message = "The requested action was denied by policy."


class EvidenceValidationError(ReasoningError):
    """Raised when final-answer evidence fails validation."""

    default_code = "evidence_validation"
    default_message = "The answer contains invalid evidence references."


class BudgetExceededError(PenampakanError):
    """Base class for bounded-run resource exhaustion."""

    default_code = "budget_exceeded"
    default_message = "The operation exhausted a configured run budget."


class StepLimitExceededError(BudgetExceededError):
    """Raised when an ask run exhausts its action-step budget."""

    default_code = "step_limit_exceeded"
    default_message = "The operation exhausted its action-step budget."


class LLMCallLimitExceededError(BudgetExceededError):
    """Raised when an ask run exhausts its policy-call budget."""

    default_code = "llm_call_limit_exceeded"
    default_message = "The operation exhausted its language-model call budget."


class ToolLimitExceededError(BudgetExceededError):
    """Raised when a run exhausts its tool-call budget."""

    default_code = "tool_limit_exceeded"
    default_message = "The operation exhausted its tool-call budget."


class BackendCallLimitExceededError(BudgetExceededError):
    """Raised when a run exhausts its backend-call budget."""

    default_code = "backend_call_limit_exceeded"
    default_message = "The operation exhausted its backend-call budget."


class AssetLimitExceededError(BudgetExceededError):
    """Raised when a run exhausts its derived-asset budget."""

    default_code = "asset_limit_exceeded"
    default_message = "The operation exhausted its derived-asset budget."


class DerivationDepthLimitExceededError(BudgetExceededError):
    """Raised when an image transform exceeds the derivation-depth limit."""

    default_code = "derivation_depth_limit_exceeded"
    default_message = "The image derivation-depth limit was exceeded."


class ContextLimitExceededError(BudgetExceededError):
    """Raised when no valid policy context fits within the context budget."""

    default_code = "context_limit_exceeded"
    default_message = "The operation exhausted its context budget."


class OperationTimeoutError(PenampakanError):
    """Raised when an overall operation exceeds its deadline."""

    default_code = "operation_timeout"
    default_message = "The operation exceeded its deadline."


class InspectionFailedError(PenampakanError):
    """Raised when inspection operations fail and exposes a safe partial result."""

    default_code = "inspection_failed"
    default_message = "Image inspection did not complete successfully."

    def __init__(
        self,
        message: str | None = None,
        *,
        partial_result: InspectionResult | None = None,
        code: str | None = None,
        trace_id: UUID | None = None,
        backend_name: str | None = None,
        tool_name: str | None = None,
        retryable: bool | None = None,
        cause: BaseException | None = None,
        cause_summary: str | None = None,
    ) -> None:
        self.partial_result = partial_result
        super().__init__(
            message,
            code=code,
            trace_id=trace_id,
            backend_name=backend_name,
            tool_name=tool_name,
            retryable=retryable,
            cause=cause,
            cause_summary=cause_summary,
        )


class SyncInAsyncContextError(PenampakanError):
    """Raised when the blocking facade is invoked from an asynchronous loop."""

    default_code = "sync_in_async_context"
    default_message = "The synchronous API cannot run inside an active event loop."


__all__ = [
    "AssetLimitExceededError",
    "AssetNotFoundError",
    "BackendCallLimitExceededError",
    "BackendError",
    "BackendTimeoutError",
    "BackendUnavailableError",
    "BudgetExceededError",
    "CapabilityError",
    "CapabilityUnavailableError",
    "ConfigurationError",
    "ContextLimitExceededError",
    "DerivationDepthLimitExceededError",
    "EvidenceValidationError",
    "ImageError",
    "ImageLimitExceededError",
    "InspectionFailedError",
    "InvalidBackendOutputError",
    "InvalidImageError",
    "InvalidModelActionError",
    "LLMCallLimitExceededError",
    "LLMError",
    "LLMNotConfiguredError",
    "ObservationNotFoundError",
    "OperationTimeoutError",
    "PenampakanError",
    "PolicyDeniedError",
    "ReasoningError",
    "RemoteSourceDisabledError",
    "SessionClosedError",
    "SessionError",
    "StepLimitExceededError",
    "SyncInAsyncContextError",
    "ToolExecutionError",
    "ToolLimitExceededError",
    "UnsupportedImageError",
]
