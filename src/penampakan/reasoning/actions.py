"""Strict parsing for untrusted language-model policy actions."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from typing import TypeAlias, cast

from pydantic import TypeAdapter, ValidationError

from penampakan.models import JsonValue, PolicyAction, WarningInfo

_ASCII_JSON_WHITESPACE = " \t\r\n"
_SAFE_LOCATION = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_KNOWN_FIELDS = frozenset(
    {
        "answer",
        "arguments",
        "evidence",
        "observation_id",
        "purpose",
        "status",
        "supports",
        "tool",
        "type",
        "uncertainties",
    }
)
_BRANCH_NAMES = frozenset({"answer", "tool"})
_MAX_FEEDBACK_ITEMS = 16
_POLICY_ACTION_ADAPTER: TypeAdapter[PolicyAction] = TypeAdapter(PolicyAction)

JsonObject: TypeAlias = dict[str, object]


class _DuplicateKeyError(ValueError):
    pass


class _NonFiniteNumberError(ValueError):
    pass


class ActionParseError(Exception):
    """A redacted internal parse failure carrying repair-safe feedback."""

    def __init__(
        self,
        feedback: tuple[WarningInfo, ...],
        *,
        invalid_model_output: str,
    ) -> None:
        self.feedback = feedback
        self.invalid_model_output = invalid_model_output
        super().__init__("The language model returned an invalid JSON action.")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(feedback_count={len(self.feedback)})"


def _reject_duplicate_keys(pairs: Sequence[tuple[str, object]]) -> JsonObject:
    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise _NonFiniteNumberError from None


def _strict_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise _NonFiniteNumberError
    return parsed


def _warning(
    code: str,
    message: str,
    *,
    location: str = "$",
    error_type: str | None = None,
    line: int | None = None,
    column: int | None = None,
) -> WarningInfo:
    details: dict[str, JsonValue] = {"location": location}
    if error_type is not None:
        details["error_type"] = error_type
    if line is not None:
        details["line"] = line
    if column is not None:
        details["column"] = column
    return WarningInfo(code=code, message=message, details=details)


def _syntax_feedback(error: BaseException) -> tuple[WarningInfo, ...]:
    if isinstance(error, _DuplicateKeyError):
        return (
            _warning(
                "duplicate_json_key",
                "The JSON action contains a duplicate object key.",
                error_type="duplicate_key",
            ),
        )
    if isinstance(error, _NonFiniteNumberError):
        return (
            _warning(
                "non_finite_json_number",
                "The JSON action contains a non-finite number.",
                error_type="non_finite_number",
            ),
        )
    if isinstance(error, json.JSONDecodeError):
        return (
            _warning(
                "invalid_action_json",
                "The response is not one complete JSON object.",
                error_type="json_syntax",
                line=error.lineno,
                column=error.colno,
            ),
        )
    return (
        _warning(
            "invalid_action_json",
            "The response is not one complete JSON object.",
            error_type="json_syntax",
        ),
    )


def _safe_location(location: tuple[object, ...]) -> str:
    segments = list(location)
    if segments and isinstance(segments[0], str) and segments[0] in _BRANCH_NAMES:
        segments.pop(0)
    result = "$"
    for segment in segments:
        if isinstance(segment, int) and not isinstance(segment, bool) and segment >= 0:
            result += f"[{segment}]"
        elif (
            isinstance(segment, str)
            and segment in _KNOWN_FIELDS
            and _SAFE_LOCATION.fullmatch(segment)
        ):
            result += f".{segment}"
        else:
            result += ".[unknown]"
    return result


def _safe_error_type(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value):
        return "validation_error"
    return value


def _validation_message(error_type: str) -> str:
    if error_type == "missing":
        return "A required action field is missing."
    if error_type == "extra_forbidden":
        return "An unknown action field is not allowed."
    if error_type in {"union_tag_invalid", "union_tag_not_found"}:
        return "The action type discriminator is missing or invalid."
    if error_type in {"literal_error", "string_pattern_mismatch"}:
        return "An action field has an unsupported value."
    if error_type in {"too_long", "string_too_long"}:
        return "An action field exceeds its maximum length."
    if error_type in {"too_short", "string_too_short"}:
        return "An action field is shorter than its minimum length."
    if error_type == "finite_number":
        return "An action field must contain a finite number."
    if error_type.endswith("_type"):
        return "An action field has the wrong JSON type."
    return "An action field does not satisfy the required schema."


def _validation_feedback(error: ValidationError) -> tuple[WarningInfo, ...]:
    feedback: list[WarningInfo] = []
    errors = error.errors(include_url=False, include_context=False, include_input=False)
    for item in errors[:_MAX_FEEDBACK_ITEMS]:
        error_type = _safe_error_type(item.get("type"))
        location = tuple(item.get("loc", ()))
        feedback.append(
            _warning(
                "invalid_action_schema",
                _validation_message(error_type),
                location=_safe_location(location),
                error_type=error_type,
            )
        )
    omitted = len(errors) - len(feedback)
    if omitted > 0:
        feedback.append(
            WarningInfo(
                code="invalid_action_errors_omitted",
                message="Additional action validation errors were omitted.",
                details={"count": omitted},
            )
        )
    if not feedback:
        feedback.append(
            _warning(
                "invalid_action_schema",
                "The JSON action does not satisfy the required schema.",
                error_type="validation_error",
            )
        )
    return tuple(feedback)


def _prepare_json_arrays(value: JsonObject) -> JsonObject:
    if value.get("type") != "answer":
        return value
    prepared = dict(value)
    evidence = prepared.get("evidence")
    uncertainties = prepared.get("uncertainties")
    if isinstance(evidence, list):
        prepared["evidence"] = tuple(evidence)
    if isinstance(uncertainties, list):
        prepared["uncertainties"] = tuple(uncertainties)
    return prepared


def parse_policy_action(model_output: str) -> PolicyAction:
    """Parse exactly one strict JSON policy action from untrusted model text."""
    if not isinstance(model_output, str):
        raise TypeError("model_output must be text")
    candidate = model_output.strip(_ASCII_JSON_WHITESPACE)
    try:
        if not candidate:
            raise json.JSONDecodeError("empty", candidate, 0)
        decoded = json.loads(
            candidate,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
            parse_float=_strict_float,
        )
    except (json.JSONDecodeError, _DuplicateKeyError, _NonFiniteNumberError) as error:
        raise ActionParseError(
            _syntax_feedback(error),
            invalid_model_output=model_output,
        ) from None
    if not isinstance(decoded, dict):
        raise ActionParseError(
            (
                _warning(
                    "action_must_be_object",
                    "The response must be one JSON object.",
                    error_type="object_required",
                ),
            ),
            invalid_model_output=model_output,
        ) from None
    prepared = _prepare_json_arrays(cast(JsonObject, decoded))
    try:
        return _POLICY_ACTION_ADAPTER.validate_python(prepared, strict=True)
    except ValidationError as error:
        raise ActionParseError(
            _validation_feedback(error),
            invalid_model_output=model_output,
        ) from None


def parse_action(model_output: str) -> PolicyAction:
    """Parse exactly one strict JSON policy action from untrusted model text."""
    return parse_policy_action(model_output)


__all__ = [
    "ActionParseError",
    "parse_action",
    "parse_policy_action",
]
