from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from .models import Capability
from .reasoning.prompts import PROMPT_VERSION

_BACKEND_NAME = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")


class _SettingsModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ImageLimits(_SettingsModel):
    """Limits applied while decoding and normalizing an input image."""

    max_input_bytes: int = Field(default=20 * 1024 * 1024, gt=0)
    max_pixels: int = Field(default=40_000_000, gt=0)
    max_width: int = Field(default=12_000, gt=0)
    max_height: int = Field(default=12_000, gt=0)


class RunLimits(_SettingsModel):
    """Resource and deadline limits for one inspection or question run."""

    max_steps: int = Field(default=8, gt=0)
    max_llm_calls: int = Field(default=10, gt=0)
    max_tool_calls: int = Field(default=12, gt=0)
    max_backend_calls: int = Field(default=16, gt=0)
    max_derived_assets: int = Field(default=16, gt=0)
    max_derivation_depth: int = Field(default=3, gt=0)
    max_parallel_tools: int = Field(default=4, gt=0)
    max_context_chars: int = Field(default=24_000, gt=0)
    max_ocr_chars_per_observation: int = Field(default=8_000, gt=0)
    default_timeout_s: float = Field(default=120.0, gt=0, allow_inf_nan=False)
    backend_timeout_s: float = Field(default=60.0, gt=0, allow_inf_nan=False)
    llm_timeout_s: float = Field(default=60.0, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_relationships(self) -> Self:
        """Validate relationships between run counters and deadlines."""
        if self.backend_timeout_s > self.default_timeout_s:
            raise ValueError("backend_timeout_s cannot exceed default_timeout_s")
        if self.llm_timeout_s > self.default_timeout_s:
            raise ValueError("llm_timeout_s cannot exceed default_timeout_s")
        if self.max_parallel_tools > self.max_tool_calls:
            raise ValueError("max_parallel_tools cannot exceed max_tool_calls")
        if self.max_llm_calls < self.max_steps + 1:
            raise ValueError("max_llm_calls must be at least max_steps + 1")
        return self


class TraceContentPolicy(_SettingsModel):
    """Explicit opt-ins for potentially sensitive trace content."""

    include_paths: bool = False
    include_questions: bool = False
    include_observation_text: bool = False
    include_model_output: bool = False
    include_answers: bool = False


class CacheSettings(_SettingsModel):
    """Retention configuration for the optional perception cache.

    A cache stores derived descriptions of user images, including OCR text and
    captions, so it is a content retention feature rather than a pure
    optimization. Retention is therefore off by default and is selected
    explicitly: ``memory`` retains within this process, ``sqlite`` retains on
    disk across processes. This setting is independent of
    :class:`TraceContentPolicy`; enabling trace content never enables a cache
    and enabling a cache never changes trace redaction.
    """

    mode: Literal["off", "memory", "sqlite"] = "off"
    path: Path | None = None
    max_entries: int = Field(default=256, gt=0)
    max_bytes: int = Field(default=128 * 1024 * 1024, gt=0)
    ttl_s: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    busy_timeout_s: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    allow_symlink: bool = False

    @model_validator(mode="after")
    def validate_path_matches_mode(self) -> Self:
        """Require a path for durable retention and reject one otherwise."""
        if self.mode == "sqlite":
            if self.path is None:
                raise ValueError("path is required when mode is 'sqlite'")
        elif self.path is not None:
            raise ValueError("path is only valid when mode is 'sqlite'")
        return self


class AgentSettings(_SettingsModel):
    """Configuration for policy orchestration and initial perception."""

    initial_capabilities: tuple[Capability, ...] = Field(
        default_factory=lambda: (
            Capability.METADATA,
            Capability.CAPTION,
            Capability.OCR,
        )
    )
    fallback_backends: bool = True
    strict_evidence: bool = True
    max_identical_actions: int = Field(default=2, gt=0)
    prompt_version: str = PROMPT_VERSION

    @field_validator("prompt_version")
    @classmethod
    def validate_prompt_version(cls, value: str) -> str:
        """Normalize and validate the public prompt version identifier."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("prompt_version cannot be empty")
        if "\x00" in normalized:
            raise ValueError("prompt_version cannot contain NUL")
        return normalized


class Settings(_SettingsModel):
    """Complete immutable configuration for a Penampakan client."""

    image: ImageLimits = Field(default_factory=ImageLimits)
    run: RunLimits = Field(default_factory=RunLimits)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    backend_preferences: dict[Capability, tuple[str, ...]] = Field(default_factory=dict)
    trace_content: TraceContentPolicy = Field(default_factory=TraceContentPolicy)

    @field_validator("backend_preferences")
    @classmethod
    def validate_backend_preferences(
        cls,
        value: dict[Capability, tuple[str, ...]],
    ) -> dict[Capability, tuple[str, ...]]:
        """Validate configured backend names and preference uniqueness."""
        for names in value.values():
            seen: set[str] = set()
            for name in names:
                if not _BACKEND_NAME.fullmatch(name):
                    raise ValueError("backend preference contains an invalid backend name")
                if name in seen:
                    raise ValueError("backend preference contains a duplicate backend name")
                seen.add(name)
        return value


def validate_timeout_s(timeout_s: float | None) -> float | None:
    """Validate a per-call timeout without changing configured sub-deadlines."""
    if timeout_s is None:
        return None
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
        raise TypeError("timeout_s must be a finite positive number")
    value = float(timeout_s)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("timeout_s must be a finite positive number")
    return value


__all__ = [
    "AgentSettings",
    "CacheSettings",
    "ImageLimits",
    "RunLimits",
    "Settings",
    "TraceContentPolicy",
]
