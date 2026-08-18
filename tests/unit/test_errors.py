from penampakan.errors import (
    BackendUnavailableError,
    ConfigurationError,
    InspectionFailedError,
    InvalidImageError,
    PenampakanError,
)


def test_errors_expose_safe_structured_details() -> None:
    error = BackendUnavailableError(
        backend_name="private-backend",
        cause=RuntimeError("credential-token-secret"),
    )

    assert isinstance(error, PenampakanError)
    assert error.code == "backend_unavailable"
    assert error.retryable is True
    assert error.backend_name == "private-backend"
    assert error.cause_summary == "RuntimeError"


def test_error_text_redacts_messages_and_causes() -> None:
    sentinel = "credential-token-secret"
    error = InvalidImageError(sentinel, cause=RuntimeError(sentinel), cause_summary=sentinel)

    assert sentinel not in str(error)
    assert sentinel not in repr(error)
    assert error.cause_summary == "cause details redacted"


def test_configuration_error_names_a_safe_extra_and_rejects_unsafe_text() -> None:
    named = ConfigurationError(code="missing_optional_dependency", extra="ocr")

    assert named.extra == "ocr"
    assert "penampakan[ocr]" in str(named)
    assert "extra='ocr'" in repr(named)

    # Only a conservative token shape is reportable, so no free-form text can
    # reach the public message through this field.
    unsafe = ConfigurationError(code="missing_optional_dependency", extra="Not A Token")

    assert unsafe.extra is None
    assert "Not A Token" not in str(unsafe)
    assert "Not A Token" not in repr(unsafe)

    plain = ConfigurationError(code="invalid_llm")

    assert plain.extra is None
    assert str(plain) == "Penampakan configuration is invalid. [invalid_llm]"


def test_invalid_custom_code_falls_back_to_stable_default() -> None:
    error = PenampakanError(code="Invalid Code")

    assert error.code == "penampakan_error"


def test_inspection_partial_result_is_not_rendered() -> None:
    partial = object()
    error = InspectionFailedError(partial_result=partial)

    assert error.partial_result is partial
    assert "partial" not in str(error)
    assert "object" not in repr(error)
