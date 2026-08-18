import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from penampakan.config import (
    AgentSettings,
    CacheSettings,
    RunLimits,
    Settings,
    TraceContentPolicy,
    validate_timeout_s,
)
from penampakan.models import Capability
from penampakan.reasoning import PROMPT_VERSION, supported_prompt_versions


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


def test_agent_settings_default_prompt_version_tracks_canonical_constant() -> None:
    default = AgentSettings().prompt_version

    assert default == PROMPT_VERSION
    assert default in supported_prompt_versions()
    assert Settings().agent.prompt_version == PROMPT_VERSION


def test_cache_retention_is_off_by_default() -> None:
    settings = CacheSettings()

    assert settings.mode == "off"
    assert settings.path is None
    assert settings.ttl_s is None
    assert settings.allow_symlink is False


def test_cache_path_is_required_only_for_durable_retention() -> None:
    durable = CacheSettings(mode="sqlite", path=Path("/tmp/penampakan/cache.sqlite3"))

    assert durable.path == Path("/tmp/penampakan/cache.sqlite3")

    with pytest.raises(ValidationError):
        CacheSettings(mode="sqlite")
    with pytest.raises(ValidationError):
        CacheSettings(mode="memory", path=Path("/tmp/penampakan/cache.sqlite3"))
    with pytest.raises(ValidationError):
        CacheSettings(mode="off", path=Path("/tmp/penampakan/cache.sqlite3"))


def test_cache_retention_mode_is_a_closed_set() -> None:
    with pytest.raises(ValidationError):
        CacheSettings(mode="redis")
    with pytest.raises(ValidationError):
        CacheSettings(mode="disk")


def test_cache_bounds_reject_non_positive_and_non_finite_values() -> None:
    for invalid in (0, -1):
        with pytest.raises(ValidationError):
            CacheSettings(max_entries=invalid)
        with pytest.raises(ValidationError):
            CacheSettings(max_bytes=invalid)
    with pytest.raises(ValidationError):
        CacheSettings(ttl_s=0.0)
    with pytest.raises(ValidationError):
        CacheSettings(ttl_s=math.inf)
    with pytest.raises(ValidationError):
        CacheSettings(busy_timeout_s=0.0)
    with pytest.raises(ValidationError):
        CacheSettings(busy_timeout_s=math.nan)


def test_cache_retention_is_independent_of_trace_content() -> None:
    settings = Settings(trace_content=TraceContentPolicy(include_observation_text=True))

    assert settings.cache.mode == "off"

    cached = Settings(cache=CacheSettings(mode="memory"))

    assert cached.trace_content == TraceContentPolicy()
    assert cached.trace_content.include_observation_text is False
