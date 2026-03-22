import math

import pytest
from pydantic import ValidationError

from penampakan.config import RunLimits, Settings, validate_timeout_s
from penampakan.models import Capability


def test_settings_defaults_are_isolated_and_immutable() -> None:
    first = Settings()
    second = Settings()

    assert first is not second
    assert first.image is not second.image
    assert first.backend_preferences == {}
    with pytest.raises(ValidationError):
        first.run = RunLimits()


def test_run_limits_validate_cross_field_relationships() -> None:
    with pytest.raises(ValidationError):
        RunLimits(max_parallel_tools=13)
    with pytest.raises(ValidationError):
        RunLimits(max_steps=10, max_llm_calls=10)
    with pytest.raises(ValidationError):
        RunLimits(default_timeout_s=30.0, backend_timeout_s=31.0)
    with pytest.raises(ValidationError):
        RunLimits(default_timeout_s=30.0, llm_timeout_s=31.0)


@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, -math.inf, math.nan])
def test_call_timeout_rejects_non_positive_or_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError):
        validate_timeout_s(value)


def test_call_timeout_accepts_positive_values_and_none() -> None:
    assert validate_timeout_s(None) is None
    assert validate_timeout_s(0.25) == 0.25


def test_backend_preferences_validate_names_and_duplicates() -> None:
    settings = Settings(
        backend_preferences={Capability.CAPTION: ("local.caption", "remote-caption")}
    )

    assert settings.backend_preferences[Capability.CAPTION] == (
        "local.caption",
        "remote-caption",
    )
    with pytest.raises(ValidationError):
        Settings(backend_preferences={Capability.CAPTION: ("duplicate", "duplicate")})
    with pytest.raises(ValidationError):
        Settings(backend_preferences={Capability.CAPTION: ("Invalid Name",)})


def test_settings_reject_unknown_and_coercive_values() -> None:
    with pytest.raises(ValidationError):
        RunLimits(max_steps="8")
    with pytest.raises(ValidationError):
        Settings.model_validate({"unknown": True})
