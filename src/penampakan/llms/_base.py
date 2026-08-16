"""Shared provider-adapter lifecycle, schema caching, and output finalization.

Every provider adapter reuses this module so ownership, idempotent closing,
schema compilation, and post-validation behave identically across providers.
Nothing here imports an optional provider SDK.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Final

from penampakan._callables import call_async_or_thread
from penampakan.errors import ConfigurationError, LLMError
from penampakan.llms.schema import (
    CompiledSchema,
    SchemaTarget,
    canonical_json,
    compile_action_schema,
    prune_optional_nulls,
    unwrap_action_envelope,
    validate_action_instance,
)
from penampakan.models import JsonValue, LLMRequest

# Compiled schemas are reused across calls; the action schema only changes when
# the declared tool set or answer-only mode changes.
_SCHEMA_CACHE_LIMIT: Final = 8


def missing_dependency(package: str) -> ConfigurationError:
    """Return the actionable error for an absent optional provider package."""
    # The package name is a static library constant, never caller data.
    return ConfigurationError(code="missing_optional_dependency", cause_summary=package)


class SchemaCompilerCache:
    """Compile the action schema once per distinct schema and target."""

    __slots__ = ("_entries", "_target")

    def __init__(self, target: SchemaTarget) -> None:
        self._target = target
        self._entries: dict[str, CompiledSchema] = {}

    @property
    def target(self) -> SchemaTarget:
        """Return the provider target this cache compiles for."""
        return self._target

    def compile(self, schema: Mapping[str, JsonValue]) -> CompiledSchema:
        """Return the compiled schema for one provider-neutral action schema."""
        key = canonical_json(dict(schema))
        cached = self._entries.get(key)
        if cached is not None:
            return cached
        compiled = compile_action_schema(schema, target=self._target)
        if len(self._entries) >= _SCHEMA_CACHE_LIMIT:
            self._entries.clear()
        self._entries[key] = compiled
        return compiled


def finalize_action_text(
    payload: str,
    *,
    request: LLMRequest,
    provider: str,
    attempts: int,
) -> str:
    """Parse, unwrap, and locally validate one provider structured output.

    Local validation runs even after provider strict enforcement, so schema
    weakening or provider drift is always detected here rather than downstream.
    """
    try:
        decoded = json.loads(payload)
    except ValueError as error:
        raise LLMError(
            code="llm_invalid_structured_output",
            attempts=attempts,
            provider=provider,
            cause=error,
        ) from error
    try:
        action = unwrap_action_envelope(decoded)
    except ValueError as error:
        raise LLMError(
            code="llm_invalid_structured_output",
            attempts=attempts,
            provider=provider,
            cause=error,
        ) from error
    pruned = prune_optional_nulls(action, request.response_json_schema)
    findings = validate_action_instance(pruned, request.response_json_schema)
    if findings:
        raise LLMError(
            code="llm_schema_validation_failed",
            attempts=attempts,
            provider=provider,
        )
    return canonical_json(pruned)


class ProviderLifecycle:
    """Idempotent ownership and closing for one provider adapter."""

    __slots__ = ("_close_task", "_closed", "_owned_client")

    def __init__(self, owned_client: object | None) -> None:
        self._owned_client = owned_client
        self._close_task: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def closed(self) -> bool:
        """Return whether this adapter has completed closing."""
        return self._closed

    def require_open(self) -> None:
        """Reject use of a closing or closed adapter."""
        if self._close_task is not None:
            raise LLMError(code="llm_closed")

    async def aclose(self) -> None:
        """Close an owned SDK client exactly once."""
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_owned())
        await asyncio.shield(self._close_task)

    async def _close_owned(self) -> None:
        try:
            client = self._owned_client
            if client is None:
                return
            closer = getattr(client, "aclose", None) or getattr(client, "close", None)
            if closer is None:
                return
            await call_async_or_thread(closer)
        finally:
            self._owned_client = None
            self._closed = True


def resolve_ownership(*, client: object | None, owns_client: bool | None) -> bool:
    """Resolve client ownership: constructed clients are owned by default."""
    if owns_client is not None:
        if not isinstance(owns_client, bool):
            raise ConfigurationError(code="invalid_ownership")
        return owns_client
    return client is None


__all__ = [
    "ProviderLifecycle",
    "SchemaCompilerCache",
    "finalize_action_text",
    "missing_dependency",
    "resolve_ownership",
]
