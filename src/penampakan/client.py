"""Reusable asynchronous Penampakan client ownership and one-shot helpers."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Sequence

from penampakan.backends.pillow import PillowBackend
from penampakan.config import Settings, validate_timeout_s
from penampakan.errors import ConfigurationError, OperationTimeoutError, SessionClosedError
from penampakan.image.assets import AssetStore
from penampakan.image.loader import load_image
from penampakan.models import (
    Capability,
    ImageSource,
    InspectionPlan,
    InspectionResult,
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

_BACKEND_OWNERS_LOCK = threading.Lock()
_OWNED_BACKEND_IDS: set[int] = set()


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
    ) -> None:
        self._settings = self._validated_settings(settings)
        if llm is not None and policy is not None:
            raise ConfigurationError(code="llm_and_policy_conflict")
        if llm is not None and not callable(getattr(llm, "complete", None)):
            raise ConfigurationError(code="invalid_llm")
        if policy is not None and not callable(getattr(policy, "next_action", None)):
            raise ConfigurationError(code="invalid_policy")
        if self._settings.agent.prompt_version != PROMPT_VERSION:
            raise ConfigurationError(code="unsupported_prompt_version")
        self._policy = (
            JsonActionPolicy(
                llm,
                prompt_version=self._settings.agent.prompt_version,
                timeout_s=self._settings.run.llm_timeout_s,
            )
            if llm is not None
            else policy
        )
        self._trace_sinks = self._validate_trace_sinks(trace_sinks)
        self._cache = cache if cache is not None else self._default_cache(self._settings)
        self._validate_cache(self._cache)
        caller_backends = tuple(backends)
        self._owned_backend_ids = self._claim_backend_ownership(caller_backends)
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

    @property
    def settings(self) -> Settings:
        """Return the immutable client settings."""
        return self._settings.model_copy(deep=True)

    @property
    def closed(self) -> bool:
        """Return whether all client-owned resources have closed."""
        return self._closed

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
        await asyncio.shield(close_task)

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

    async def _close_owned(self) -> None:
        try:
            await self._idle.wait()
            sessions = tuple(self._sessions)
            if sessions:
                await asyncio.gather(
                    *(session.aclose() for session in sessions),
                    return_exceptions=True,
                )
            for close in (
                self._singleflight.aclose,
                self._router.aclose,
                self._cache.aclose,
            ):
                try:
                    await close()
                except BaseException:
                    continue
            for sink in self._trace_sinks:
                try:
                    await sink.aclose()
                except BaseException:
                    continue
        finally:
            self._release_backend_ownership()
            self._closed = True

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

    @staticmethod
    def _claim_backend_ownership(backends: Sequence[VisionBackend]) -> set[int]:
        identifiers = [id(backend) for backend in backends]
        if len(identifiers) != len(set(identifiers)):
            raise ConfigurationError(code="duplicate_backend_instance")
        with _BACKEND_OWNERS_LOCK:
            if any(identifier in _OWNED_BACKEND_IDS for identifier in identifiers):
                raise ConfigurationError(code="backend_already_owned")
            _OWNED_BACKEND_IDS.update(identifiers)
        return set(identifiers)

    def _release_backend_ownership(self) -> None:
        identifiers: set[int] = getattr(self, "_owned_backend_ids", set())
        with _BACKEND_OWNERS_LOCK:
            _OWNED_BACKEND_IDS.difference_update(identifiers)
        identifiers.clear()


__all__ = ["AsyncPenampakan"]
