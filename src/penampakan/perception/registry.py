"""Typed declarations and executors for bounded visual tools."""

from __future__ import annotations

import inspect
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol

from PIL.Image import Image as PillowImage
from pydantic import BaseModel

from penampakan.errors import ConfigurationError, ToolExecutionError
from penampakan.image.assets import PendingAsset
from penampakan.models import (
    JsonValue,
    ObservationDraft,
    ToolSpec,
    VisionRequest,
    WarningInfo,
)

_TOOL_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Validated drafts, pending assets, and warnings from one tool call."""

    observations: tuple[ObservationDraft, ...] = field(default_factory=tuple)
    assets: tuple[PendingAsset, ...] = field(default_factory=tuple)
    warnings: tuple[WarningInfo, ...] = field(default_factory=tuple)


class ToolExecutionContext(Protocol):
    """Least-authority services available to a built-in tool executor."""

    def image(self, asset_id: str) -> PillowImage:
        """Return a caller-owned copy of a session asset."""
        ...

    def ensure_asset_capacity(self, parent_id: str, count: int) -> None:
        """Reserve worst-case asset capacity before transform rendering."""
        ...

    async def perceive(self, asset_id: str, request: VisionRequest) -> ToolResult:
        """Route and normalize one perception request."""
        ...


ToolExecutor = Callable[[ToolExecutionContext, BaseModel], Awaitable[ToolResult]]


@dataclass(frozen=True, slots=True)
class ToolRegistration:
    """One immutable tool declaration, validator, and private executor."""

    spec: ToolSpec
    arguments_model: type[BaseModel]
    executor: ToolExecutor = field(repr=False)


class ToolRegistry:
    """Ordered registry of uniquely named typed visual tools."""

    def __init__(self) -> None:
        self._registrations: dict[str, ToolRegistration] = {}

    @property
    def names(self) -> tuple[str, ...]:
        """Return registered tool names in deterministic insertion order."""

        return tuple(self._registrations)

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        """Return immutable LLM-visible declarations in registration order."""

        return tuple(item.spec for item in self._registrations.values())

    def register(
        self,
        *,
        name: str,
        description: str,
        arguments_model: type[BaseModel],
        executor: ToolExecutor,
        creates_assets: bool = False,
        cost_hint: int = 1,
    ) -> None:
        """Register one validated declaration and private executor."""

        if not isinstance(name, str) or _TOOL_NAME.fullmatch(name) is None:
            raise ConfigurationError(code="invalid_tool_name")
        if name in self._registrations:
            raise ConfigurationError(code="duplicate_tool_name")
        if not inspect.isclass(arguments_model) or not issubclass(arguments_model, BaseModel):
            raise ConfigurationError(code="invalid_tool_arguments_model")
        if not callable(executor):
            raise ConfigurationError(code="invalid_tool_executor")
        schema = arguments_model.model_json_schema(mode="validation")
        _require_closed_schemas(schema)
        spec = ToolSpec(
            name=name,
            description=description,
            arguments_json_schema=schema,
            creates_assets=creates_assets,
            cost_hint=cost_hint,
        )
        self._registrations[name] = ToolRegistration(spec, arguments_model, executor)

    def spec(self, name: str) -> ToolSpec:
        """Return one declaration or a safe unknown-tool failure."""

        return self._registration(name).spec

    def validate_arguments(self, name: str, arguments: dict[str, JsonValue]) -> BaseModel:
        """Strictly validate an untrusted tool argument object."""

        registration = self._registration(name)
        try:
            return registration.arguments_model.model_validate(arguments, strict=True)
        except Exception as error:
            raise ToolExecutionError(code="invalid_tool_arguments", tool_name=name) from error

    async def execute(
        self,
        context: ToolExecutionContext,
        name: str,
        arguments: dict[str, JsonValue],
    ) -> ToolResult:
        """Validate and execute one registered tool call."""

        registration = self._registration(name)
        validated = self.validate_arguments(name, arguments)
        try:
            result = await registration.executor(context, validated)
        except ToolExecutionError:
            raise
        except Exception as error:
            raise ToolExecutionError(code="tool_execution_failed", tool_name=name) from error
        if not isinstance(result, ToolResult):
            raise ToolExecutionError(code="invalid_tool_result", tool_name=name)
        return result

    def _registration(self, name: str) -> ToolRegistration:
        try:
            return self._registrations[name]
        except KeyError as error:
            raise ToolExecutionError(code="unknown_tool", tool_name=name) from error


def _require_closed_schemas(value: object) -> None:
    if isinstance(value, dict):
        if (
            value.get("type") == "object"
            and "properties" in value
            and value.get("additionalProperties") is not False
        ):
            raise ConfigurationError(code="open_tool_arguments_schema")
        for child in value.values():
            _require_closed_schemas(child)
    elif isinstance(value, list):
        for child in value:
            _require_closed_schemas(child)


__all__ = [
    "ToolExecutionContext",
    "ToolExecutor",
    "ToolRegistration",
    "ToolRegistry",
    "ToolResult",
]
