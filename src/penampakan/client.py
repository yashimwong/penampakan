"""Reusable asynchronous Penampakan client ownership and one-shot helpers."""

from __future__ import annotations

import asyncio
import threading
import time
import weakref
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import cast

from penampakan.backends.pillow import PillowBackend
from penampakan.config import Settings, validate_timeout_s
from penampakan.errors import (
    ConfigurationError,
    OperationTimeoutError,
    PenampakanError,
    SessionClosedError,
)
from penampakan.image.assets import AssetStore
from penampakan.image.loader import load_image
from penampakan.models import (
    Capability,
    ImageSource,
    InspectionPlan,
    InspectionResult,
    JsonValue,
    VisionAnswer,
    WarningInfo,
)
from penampakan.perception.cache import MemoryLRUCache, NullCache, SingleFlightCoordinator
from penampakan.perception.registry import ToolRegistry
from penampakan.perception.router import BackendRouter
from penampakan.protocols import ActionPolicy, Cache, TextLLM, TraceSink, VisionBackend
from penampakan.reasoning.policy import JsonActionPolicy
from penampakan.reasoning.prompts import PROMPT_VERSION
from penampakan.session import AsyncVisionSession
from penampakan.tools.builtin import register_transform_tools
from penampakan.tools.vision import register_vision_tools


@dataclass(slots=True)
class _OwnershipEntry:
    """One cross-client claim on a caller-supplied backend instance."""

    backend_ref: weakref.ReferenceType[object]
    owner_ref: weakref.ReferenceType[AsyncPenampakan]
    token: object


# A reentrant lock keeps a weakref callback that fires while the registry is
# being mutated on the same thread from deadlocking against itself.
_BACKEND_OWNERS_LOCK = threading.RLock()
_BACKEND_OWNERS: dict[int, _OwnershipEntry] = {}


def _live_backend_owner(key: int, backend: object) -> AsyncPenampakan | None:
    """Return the live owner of a backend, discarding stale registry entries."""
    entry = _BACKEND_OWNERS.get(key)
    if entry is None:
        return None
    if entry.backend_ref() is not backend:
        # A dead backend, or an unrelated object that reused the address.
        del _BACKEND_OWNERS[key]
        return None
    owner = entry.owner_ref()
    if owner is None:
        # The owning client was collected without closing.
        del _BACKEND_OWNERS[key]
        return None
    return owner


def _forget_backend(key: int, token: object, _reference: object) -> None:
    """Drop a claim for a collected backend without evicting a newer owner."""
    with _BACKEND_OWNERS_LOCK:
        entry = _BACKEND_OWNERS.get(key)
        if entry is not None and entry.token is token:
            del _BACKEND_OWNERS[key]


class AsyncPenampakan:
    """Own shared backends and open isolated reusable vision sessions."""

    def __init__(
        self,
        *,
        llm: TextLLM | None = None,
        backends: Sequence[VisionBackend] = (),
        policy: ActionPolicy | None = None,
        cache: Cache | None = None,
        settings: Settings | None = None,
        trace_sinks: Sequence[TraceSink] = (),
        owns_policy: bool = False,
        owns_llm: bool = False,
    ) -> None:
        self._settings = self._validated_settings(settings)
        if llm is not None and policy is not None:
            raise ConfigurationError(code="llm_and_policy_conflict")
        if llm is not None and not callable(getattr(llm, "complete", None)):
            raise ConfigurationError(code="invalid_llm")
        if policy is not None and not callable(getattr(policy, "next_action", None)):
            raise ConfigurationError(code="invalid_policy")
        if not isinstance(owns_policy, bool) or not isinstance(owns_llm, bool):
            raise ConfigurationError(code="invalid_ownership")
        if owns_llm and llm is None:
            raise ConfigurationError(code="invalid_ownership")
        if owns_policy and policy is None:
            raise ConfigurationError(code="invalid_ownership")
        if self._settings.agent.prompt_version != PROMPT_VERSION:
            raise ConfigurationError(code="unsupported_prompt_version")
        # The convenience path constructs the policy, so the client owns it and
        # cascades to the language model only when the caller handed ownership
        # over. Caller-supplied resources default to caller-owned.
        self._policy = (
            JsonActionPolicy(
                llm,
                prompt_version=self._settings.agent.prompt_version,
                timeout_s=self._settings.run.llm_timeout_s,
                owns_llm=owns_llm,
            )
            if llm is not None
            else policy
        )
        self._owns_policy = llm is not None or owns_policy
        self._trace_sinks = self._validate_trace_sinks(trace_sinks)
        self._cache = cache if cache is not None else self._default_cache(self._settings)
        self._validate_cache(self._cache)
        self._close_warnings: list[WarningInfo] = []
        caller_backends = tuple(backends)
        self._claim_backend_ownership(caller_backends)
        pillow = PillowBackend()
        metadata_preferences = self._settings.backend_preferences.get(Capability.METADATA)
        if metadata_preferences not in {None, (pillow.descriptor.name,)}:
            self._release_backend_ownership()
            raise ConfigurationError(code="authoritative_backend_preference")
        try:
            self._router = BackendRouter(
                (pillow, *caller_backends),
                preferences=self._settings.backend_preferences,
                fallback_backends=self._settings.agent.fallback_backends,
                backend_timeout_s=self._settings.run.backend_timeout_s,
                trusted_reserved_names=(pillow.descriptor.name,),
                authoritative_backends={Capability.METADATA: pillow.descriptor.name},
            )
            self._tools = self._build_tools(self._router)
        except BaseException:
            self._release_backend_ownership()
            raise
        self._singleflight: SingleFlightCoordinator[bytes] = SingleFlightCoordinator()
        self._sessions: set[AsyncVisionSession] = set()
        self._state_lock = asyncio.Lock()
        self._idle = asyncio.Event()
        self._idle.set()
        self._active_operations = 0
        self._closing = False
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._close_failure: BaseException | None = None

    @property
    def settings(self) -> Settings:
        """Return the immutable client settings."""
        return self._settings.model_copy(deep=True)

    @property
    def closed(self) -> bool:
        """Return whether all client-owned resources have closed."""
        return self._closed

    @property
    def close_warnings(self) -> tuple[WarningInfo, ...]:
        """Return redacted warnings for owned resources that failed to close."""
        return tuple(self._close_warnings)

    async def open_image(self, source: ImageSource) -> AsyncVisionSession:
        """Normalize an image and return a reusable isolated session."""
        await self._begin_operation()
        try:
            return await self._create_session(source)
        finally:
            self._end_operation()

    async def inspect(
        self,
        source: ImageSource,
        plan: InspectionPlan | None = None,
        *,
        timeout_s: float | None = None,
    ) -> InspectionResult:
        """Inspect one image and release its private session in all outcomes."""
        timeout = validate_timeout_s(timeout_s) or self._settings.run.default_timeout_s
        deadline = time.monotonic() + timeout
        await self._begin_operation()
        session: AsyncVisionSession | None = None
        failed = False
        try:
            session = await self._create_session_before(source, deadline)
            remaining = self._remaining_time(deadline)
            result = await session.inspect(plan, timeout_s=remaining)
            return result.model_copy(deep=True)
        except BaseException:
            failed = True
            raise
        finally:
            close_error: BaseException | None = None
            if session is not None:
                try:
                    await self._close_session(session)
                except BaseException as error:
                    close_error = error
            self._end_operation()
            if close_error is not None and not failed:
                raise close_error

    async def ask(
        self,
        source: ImageSource,
        question: str,
        *,
        timeout_s: float | None = None,
    ) -> VisionAnswer:
        """Answer one visual question and release its private session in all outcomes."""
        timeout = validate_timeout_s(timeout_s) or self._settings.run.default_timeout_s
        deadline = time.monotonic() + timeout
        await self._begin_operation()
        session: AsyncVisionSession | None = None
        failed = False
        try:
            session = await self._create_session_before(source, deadline)
            remaining = self._remaining_time(deadline)
            result = await session.ask(question, timeout_s=remaining)
            return result.model_copy(deep=True)
        except BaseException:
            failed = True
            raise
        finally:
            close_error = None
            if session is not None:
                try:
                    await self._close_session(session)
                except BaseException as error:
                    close_error = error
            self._end_operation()
            if close_error is not None and not failed:
                raise close_error

    async def aclose(self) -> None:
        """Drain operations and close all owned resources exactly once."""
        async with self._state_lock:
            if self._close_task is None:
                self._closing = True
                self._close_task = asyncio.create_task(self._close_owned())
            close_task = self._close_task
        try:
            await asyncio.shield(close_task)
        except asyncio.CancelledError:
            if not close_task.done():
                # The caller was cancelled: let the protected cleanup finish
                # before the caller's own cancellation resumes.
                await self._drain_close(close_task)
            raise
        if self._close_failure is not None:
            raise self._close_failure

    @staticmethod
    async def _drain_close(close_task: asyncio.Task[None]) -> None:
        while not close_task.done():
            try:
                await asyncio.shield(close_task)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break

    async def __aenter__(self) -> AsyncPenampakan:
        """Enter the reusable asynchronous client context."""
        async with self._state_lock:
            if self._closing or self._closed:
                raise SessionClosedError()
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        """Close the reusable asynchronous client context."""
        await self.aclose()

    async def _create_session(self, source: ImageSource) -> AsyncVisionSession:
        worker = asyncio.create_task(asyncio.to_thread(self._load_store, source))
        try:
            store, load_warnings = await asyncio.shield(worker)
        except asyncio.CancelledError:
            worker.add_done_callback(self._close_late_store)
            raise
        try:
            session = AsyncVisionSession(
                asset_store=store,
                router=self._router,
                tools=self._tools,
                policy=self._policy,
                cache=self._cache,
                singleflight=self._singleflight,
                settings=self._settings,
                trace_sinks=self._trace_sinks,
                load_warnings=load_warnings,
                on_close=self._session_closed,
            )
            async with self._state_lock:
                rejected = self._closing or self._closed
                if not rejected:
                    self._sessions.add(session)
            if rejected:
                await session.aclose()
                raise SessionClosedError()
            return session
        except BaseException:
            store.close()
            raise

    async def _create_session_before(
        self,
        source: ImageSource,
        deadline: float,
    ) -> AsyncVisionSession:
        try:
            return await asyncio.wait_for(
                self._create_session(source),
                timeout=self._remaining_time(deadline),
            )
        except asyncio.TimeoutError as error:
            raise OperationTimeoutError(cause=error) from error

    def _load_store(self, source: ImageSource) -> tuple[AssetStore, tuple[WarningInfo, ...]]:
        loaded = load_image(source, self._settings.image)
        load_warnings = loaded.warnings
        store = AssetStore.from_loaded(
            loaded,
            image_limits=self._settings.image,
            run_limits=self._settings.run,
        )
        return store, load_warnings

    @staticmethod
    def _close_late_store(
        worker: asyncio.Task[tuple[AssetStore, tuple[WarningInfo, ...]]],
    ) -> None:
        try:
            store, _ = worker.result()
        except (asyncio.CancelledError, Exception):
            return
        store.close()

    async def _begin_operation(self) -> None:
        async with self._state_lock:
            if self._closing or self._closed:
                raise SessionClosedError()
            self._active_operations += 1
            self._idle.clear()

    def _end_operation(self) -> None:
        self._active_operations -= 1
        if self._active_operations == 0:
            self._idle.set()

    def _session_closed(self, session: AsyncVisionSession) -> None:
        self._sessions.discard(session)

    async def _close_session(self, session: AsyncVisionSession) -> None:
        task = asyncio.create_task(session.aclose())
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise

    def _close_sequence(self) -> tuple[tuple[str, Callable[[], Awaitable[object]]], ...]:
        """Return owned close steps in dependency order.

        Active operations drain first so no session is closed underneath a
        caller. Sessions close before the resources they borrow. Single flight
        stops sharing populations before the router closes the backends those
        populations invoke, an owned policy closes after the sessions that call
        it and cascades to a language model it owns, the cache closes after the
        sessions and populations that write to it, and trace sinks close last so
        every earlier step can still be observed. Caller-owned resources,
        including a caller-supplied language model or action policy, are never
        part of this sequence, and no shared adapter or external client is
        closed implicitly.
        """
        return (
            ("active_operations", self._idle.wait),
            ("session", self._close_sessions),
            ("singleflight", self._singleflight.aclose),
            ("router", self._router.aclose),
            *self._owned_policy_step(),
            ("cache", self._cache.aclose),
            *((f"trace_sink_{index}", sink.aclose) for index, sink in enumerate(self._trace_sinks)),
        )

    def _owned_policy_step(self) -> tuple[tuple[str, Callable[[], Awaitable[object]]], ...]:
        if not self._owns_policy or self._policy is None:
            return ()
        closer = getattr(self._policy, "aclose", None)
        if not callable(closer):
            return ()
        return (("policy", cast("Callable[[], Awaitable[object]]", closer)),)

    async def _close_owned(self) -> None:
        """Attempt every owned resource once and retain the first base exception.

        The retained base exception is stored rather than raised out of this
        task. A task that raises ``KeyboardInterrupt`` or ``SystemExit`` hands
        it to the event loop instead of to the awaiting caller, which would
        replace the primary exception with an unrelated ``CancelledError``.
        ``aclose`` re-raises the stored failure so every caller observes it.
        See ``specs/adr/0002-close-failure-propagation.md``.
        """
        primary: BaseException | None = None
        try:
            for resource, close in self._close_sequence():
                try:
                    await close()
                except Exception as error:
                    # An ordinary failure never displaces a primary base
                    # exception and never stops the remaining cleanup.
                    self._record_close_warning(resource, error)
                except BaseException as error:
                    if primary is None:
                        primary = error
        finally:
            self._close_failure = primary
            self._release_backend_ownership()
            self._closed = True

    async def _close_sessions(self) -> None:
        sessions = tuple(self._sessions)
        if not sessions:
            return
        results = await asyncio.gather(*(self._close_one_session(session) for session in sessions))
        primary: BaseException | None = None
        for error in results:
            if error is None:
                continue
            if isinstance(error, Exception):
                self._record_close_warning("session", error)
            elif primary is None:
                primary = error
        if primary is not None:
            raise primary

    @staticmethod
    async def _close_one_session(session: AsyncVisionSession) -> BaseException | None:
        # Returning the failure keeps a base exception inside this task instead
        # of letting it abandon the remaining concurrent session closes.
        try:
            await session.aclose()
        except BaseException as error:
            return error
        return None

    def _record_close_warning(self, resource: str, error: Exception) -> None:
        details: dict[str, JsonValue] = {
            "resource": resource,
            "error_type": type(error).__name__,
        }
        # Only a library error code is known to be a safe stable identifier.
        if isinstance(error, PenampakanError):
            details["error_code"] = error.code
        self._close_warnings.append(
            WarningInfo(
                code="owned_resource_close_failed",
                message="An owned resource failed to close during shutdown.",
                details=details,
            )
        )

    @staticmethod
    def _build_tools(router: BackendRouter) -> ToolRegistry:
        capabilities = {
            capability.capability
            for descriptor in router.descriptors
            for capability in descriptor.capabilities
        }
        registry = ToolRegistry()
        register_vision_tools(registry, capabilities)
        register_transform_tools(registry)
        return registry

    @staticmethod
    def _default_cache(settings: Settings) -> Cache:
        if not settings.cache.enabled:
            return NullCache()
        return MemoryLRUCache(
            max_entries=settings.cache.max_entries,
            max_bytes=settings.cache.max_bytes,
        )

    @staticmethod
    def _validate_cache(cache: Cache) -> None:
        if not all(callable(getattr(cache, name, None)) for name in ("get", "set", "aclose")):
            raise ConfigurationError(code="invalid_cache")

    @staticmethod
    def _validate_trace_sinks(sinks: Sequence[TraceSink]) -> tuple[TraceSink, ...]:
        selected = tuple(sinks)
        for sink in selected:
            if not callable(getattr(sink, "emit", None)) or not callable(
                getattr(sink, "aclose", None)
            ):
                raise ConfigurationError(code="invalid_trace_sink")
        return selected

    @staticmethod
    def _validated_settings(settings: Settings | None) -> Settings:
        if settings is None:
            return Settings()
        if not isinstance(settings, Settings):
            raise ConfigurationError(code="invalid_settings")
        try:
            return Settings.model_validate(
                settings.model_dump(mode="python"),
                strict=True,
            )
        except Exception as error:
            raise ConfigurationError(code="invalid_settings", cause=error) from error

    @staticmethod
    def _remaining_time(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise OperationTimeoutError()
        return remaining

    def _claim_backend_ownership(self, backends: Sequence[VisionBackend]) -> None:
        selected: list[object] = []
        for backend in backends:
            # Identity comparison keeps duplicate detection independent of the
            # backend's hashing and equality behavior.
            if any(backend is existing for existing in selected):
                raise ConfigurationError(code="duplicate_backend_instance")
            selected.append(backend)
        claims: list[tuple[int, object]] = []
        with _BACKEND_OWNERS_LOCK:
            for candidate in selected:
                if _live_backend_owner(id(candidate), candidate) is not None:
                    raise ConfigurationError(code="backend_already_owned")
            for candidate in selected:
                key = id(candidate)
                token = object()
                try:
                    reference = weakref.ref(candidate, partial(_forget_backend, key, token))
                except TypeError:
                    # A backend without weak-reference support keeps local
                    # duplicate detection but skips the cross-client guard.
                    continue
                _BACKEND_OWNERS[key] = _OwnershipEntry(
                    backend_ref=reference,
                    owner_ref=weakref.ref(self),
                    token=token,
                )
                claims.append((key, token))
        self._ownership_claims: tuple[tuple[int, object], ...] = tuple(claims)

    def _release_backend_ownership(self) -> None:
        claims: tuple[tuple[int, object], ...] = getattr(self, "_ownership_claims", ())
        if not claims:
            return
        with _BACKEND_OWNERS_LOCK:
            for key, token in claims:
                entry = _BACKEND_OWNERS.get(key)
                if entry is not None and entry.token is token:
                    del _BACKEND_OWNERS[key]
        self._ownership_claims = ()


__all__ = ["AsyncPenampakan"]
