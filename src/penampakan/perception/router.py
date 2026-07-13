"""Deterministic, bounded, and caller-controlled perception routing."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from penampakan.errors import (
    BackendError,
    BackendTimeoutError,
    BackendUnavailableError,
    CapabilityUnavailableError,
    ConfigurationError,
    InvalidBackendOutputError,
    PenampakanError,
    SessionClosedError,
)
from penampakan.models import (
    BackendDescriptor,
    BackendImage,
    Capability,
    VisionRequest,
    VisionResult,
    WarningInfo,
)
from penampakan.protocols import VisionBackend

AttemptOutcome = Literal["success", "error", "retryable_error", "timeout", "unavailable"]


@dataclass(frozen=True, slots=True)
class RouteAttempt:
    """Safe metadata for one bounded backend invocation."""

    descriptor: BackendDescriptor
    outcome: AttemptOutcome
    duration_ms: int
    error_code: str | None = None
    retryable: bool = False

    @property
    def backend_name(self) -> str:
        """Return the attempted backend name."""
        return self.descriptor.name


@dataclass(frozen=True, slots=True)
class RouteResult:
    """A successful backend result with complete safe routing metadata."""

    result: VisionResult
    descriptor: BackendDescriptor
    attempts: tuple[RouteAttempt, ...]
    warnings: tuple[WarningInfo, ...]

    @property
    def backend_calls(self) -> int:
        """Return the number of actual backend attempts."""
        return len(self.attempts)


@dataclass(slots=True)
class _BackendEntry:
    backend: VisionBackend
    descriptor: BackendDescriptor
    semaphore: asyncio.Semaphore
    registration_index: int


def route_failure_attempts(error: BaseException) -> tuple[RouteAttempt, ...]:
    """Return safe attempts attached to a failed routed invocation."""
    value = getattr(error, "route_attempts", ())
    if not isinstance(value, tuple):
        return ()
    return cast(tuple[RouteAttempt, ...], value)


def route_failure_warnings(error: BaseException) -> tuple[WarningInfo, ...]:
    """Return safe fallback warnings attached to a failed routed invocation."""
    value = getattr(error, "route_warnings", ())
    if not isinstance(value, tuple):
        return ()
    return cast(tuple[WarningInfo, ...], value)


class BackendRouter:
    """Own and route validated vision backends under deterministic policy."""

    def __init__(
        self,
        backends: Sequence[VisionBackend],
        *,
        preferences: Mapping[Capability, Sequence[str]] | None = None,
        fallback_backends: bool = True,
        backend_timeout_s: float = 60.0,
        trusted_reserved_names: Collection[str] = (),
        authoritative_backends: Mapping[Capability, str] | None = None,
    ) -> None:
        self._fallback_backends = fallback_backends
        if not isinstance(fallback_backends, bool):
            raise ConfigurationError(code="invalid_fallback_setting")
        self._backend_timeout_s = self._validate_timeout(backend_timeout_s)
        self._trusted_reserved_names = frozenset(trusted_reserved_names)
        self._entries = self._build_entries(backends)
        self._by_name = {entry.descriptor.name: entry for entry in self._entries}
        self._authoritative_backends = self._validate_authoritative_backends(
            authoritative_backends or {}
        )
        self._preferences = self._validate_preferences(preferences or {})
        self._state_lock = asyncio.Lock()
        self._idle = asyncio.Event()
        self._idle.set()
        self._active_calls = 0
        self._closing = False
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    @property
    def descriptors(self) -> tuple[BackendDescriptor, ...]:
        """Return stable backend descriptors in registration order."""
        return tuple(entry.descriptor for entry in self._entries)

    @property
    def closed(self) -> bool:
        """Return whether all owned backends have been closed."""
        return self._closed

    def descriptor(self, backend_name: str) -> BackendDescriptor:
        """Return a registered backend descriptor by caller-controlled name."""
        try:
            return self._by_name[backend_name].descriptor
        except KeyError as error:
            raise CapabilityUnavailableError(
                code="backend_override_unavailable",
                backend_name=backend_name,
            ) from error

    def supports(self, request: VisionRequest, *, backend_name: str | None = None) -> bool:
        """Return whether caller-controlled routing has an exact compatible backend."""
        try:
            return bool(self.route(request, backend_name=backend_name))
        except CapabilityUnavailableError:
            return False

    def route(
        self,
        request: VisionRequest,
        *,
        backend_name: str | None = None,
    ) -> tuple[BackendDescriptor, ...]:
        """Resolve exact compatible candidates without invoking a backend."""
        if self._closing or self._closed:
            raise SessionClosedError()
        return self._route(request, backend_name)

    def _route(
        self,
        request: VisionRequest,
        backend_name: str | None,
    ) -> tuple[BackendDescriptor, ...]:
        capability = request.capability
        authoritative_name = self._authoritative_backends.get(capability)
        if authoritative_name is not None:
            if backend_name is not None and backend_name != authoritative_name:
                raise CapabilityUnavailableError(code="backend_override_incompatible")
            authoritative = self._by_name[authoritative_name]
            if not self._supports(authoritative, request):
                raise CapabilityUnavailableError(code="capability_option_unavailable")
            return (authoritative.descriptor,)
        ordered: list[_BackendEntry] = []
        selected: set[str] = set()
        if backend_name is not None:
            override = self._override_entry(backend_name, request)
            ordered.append(override)
            selected.add(override.descriptor.name)
        for preferred_name in self._preferences.get(capability, ()):
            entry = self._by_name[preferred_name]
            if entry.descriptor.name not in selected and self._supports(entry, request):
                ordered.append(entry)
                selected.add(entry.descriptor.name)
        for entry in self._entries:
            if entry.descriptor.name in selected:
                continue
            if not self._declares(entry.descriptor, capability):
                continue
            if self._supports(entry, request):
                ordered.append(entry)
                selected.add(entry.descriptor.name)
        if not ordered:
            declared = any(self._declares(entry.descriptor, capability) for entry in self._entries)
            code = "capability_option_unavailable" if declared else "capability_unavailable"
            raise CapabilityUnavailableError(code=code)
        return tuple(entry.descriptor for entry in ordered)

    async def analyze(
        self,
        image: BackendImage,
        request: VisionRequest,
        *,
        backend_name: str | None = None,
        timeout_s: float | None = None,
        before_attempt: Callable[[BackendDescriptor], Awaitable[None]] | None = None,
    ) -> RouteResult:
        """Analyze through deterministic fallback while preserving safe attempt data."""
        await self._begin_call()
        try:
            descriptors = self._route(request, backend_name)
            effective_timeout = self._effective_timeout(timeout_s)
            attempts: list[RouteAttempt] = []
            warnings: list[WarningInfo] = []
            for descriptor in descriptors:
                entry = self._by_name[descriptor.name]
                if before_attempt is not None:
                    await before_attempt(descriptor)
                started = time.monotonic()
                try:
                    result = await asyncio.wait_for(
                        self._invoke(entry, image, request),
                        timeout=effective_timeout,
                    )
                    if not isinstance(result, VisionResult):
                        raise InvalidBackendOutputError(
                            code="invalid_backend_output",
                            backend_name=descriptor.name,
                        )
                except asyncio.CancelledError:
                    raise
                except asyncio.TimeoutError as error:
                    timeout_error = BackendTimeoutError(
                        code="backend_timeout",
                        backend_name=descriptor.name,
                        cause=error,
                    )
                    attempt = self._failed_attempt(
                        descriptor,
                        timeout_error,
                        started,
                        outcome="timeout",
                    )
                    attempts.append(attempt)
                    warnings.append(self._fallback_warning(attempt))
                    if self._can_fallback(timeout_error, len(attempts), len(descriptors)):
                        continue
                    self._attach_failure_metadata(timeout_error, attempts, warnings)
                    raise timeout_error from error
                except BackendError as error:
                    self._set_backend_name(error, descriptor.name)
                    if isinstance(error, BackendUnavailableError):
                        outcome: AttemptOutcome = "unavailable"
                    elif error.retryable:
                        outcome = "retryable_error"
                    else:
                        outcome = "error"
                    attempt = self._failed_attempt(descriptor, error, started, outcome=outcome)
                    attempts.append(attempt)
                    warnings.append(self._fallback_warning(attempt))
                    if self._can_fallback(error, len(attempts), len(descriptors)):
                        continue
                    self._attach_failure_metadata(error, attempts, warnings)
                    raise
                except PenampakanError as error:
                    attempt = self._failed_attempt(
                        descriptor,
                        error,
                        started,
                        outcome="error",
                    )
                    attempts.append(attempt)
                    warnings.append(self._fallback_warning(attempt))
                    self._attach_failure_metadata(error, attempts, warnings)
                    raise
                except Exception as error:
                    unexpected_error = BackendError(
                        code="backend_error",
                        backend_name=descriptor.name,
                        cause=error,
                    )
                    attempt = self._failed_attempt(
                        descriptor,
                        unexpected_error,
                        started,
                        outcome="error",
                    )
                    attempts.append(attempt)
                    warnings.append(self._fallback_warning(attempt))
                    self._attach_failure_metadata(unexpected_error, attempts, warnings)
                    raise unexpected_error from error
                attempts.append(
                    RouteAttempt(
                        descriptor=descriptor,
                        outcome="success",
                        duration_ms=self._duration_ms(started),
                    )
                )
                return RouteResult(
                    result=result,
                    descriptor=descriptor,
                    attempts=tuple(attempts),
                    warnings=tuple(warnings),
                )
            raise RuntimeError("backend routing ended without a result")
        finally:
            await self._end_call()

    async def aclose(self) -> None:
        """Wait for active routes and idempotently close every owned backend."""
        async with self._state_lock:
            if self._close_task is None:
                self._closing = True
                self._close_task = asyncio.create_task(self._close_owned_backends())
            task = self._close_task
        await asyncio.shield(task)

    def _build_entries(self, backends: Sequence[VisionBackend]) -> tuple[_BackendEntry, ...]:
        entries: list[_BackendEntry] = []
        names: set[str] = set()
        instances: set[int] = set()
        for index, backend in enumerate(backends):
            if id(backend) in instances:
                raise ConfigurationError(code="duplicate_backend_instance")
            instances.add(id(backend))
            descriptor = self._stable_descriptor(backend)
            if descriptor.name in names:
                raise ConfigurationError(code="duplicate_backend_name")
            names.add(descriptor.name)
            if (
                self._is_reserved(descriptor.name)
                and descriptor.name not in self._trusted_reserved_names
            ):
                raise ConfigurationError(code="reserved_backend_name")
            entries.append(
                _BackendEntry(
                    backend=backend,
                    descriptor=descriptor,
                    semaphore=asyncio.Semaphore(descriptor.max_concurrency),
                    registration_index=index,
                )
            )
        unknown_trusted = self._trusted_reserved_names.difference(names)
        if unknown_trusted:
            raise ConfigurationError(code="unknown_trusted_backend")
        return tuple(entries)

    def _validate_preferences(
        self,
        preferences: Mapping[Capability, Sequence[str]],
    ) -> dict[Capability, tuple[str, ...]]:
        validated: dict[Capability, tuple[str, ...]] = {}
        for capability, names in preferences.items():
            if not isinstance(capability, Capability):
                raise ConfigurationError(code="invalid_backend_preference")
            ordered = tuple(names)
            if len(ordered) != len(set(ordered)):
                raise ConfigurationError(code="duplicate_backend_preference")
            for name in ordered:
                entry = self._by_name.get(name)
                if entry is None:
                    raise ConfigurationError(code="unknown_backend_preference")
                if not self._declares(entry.descriptor, capability):
                    raise ConfigurationError(code="incompatible_backend_preference")
            validated[capability] = ordered
        return validated

    def _validate_authoritative_backends(
        self,
        configured: Mapping[Capability, str],
    ) -> dict[Capability, str]:
        result: dict[Capability, str] = {}
        for capability, name in configured.items():
            entry = self._by_name.get(name)
            if not isinstance(capability, Capability) or entry is None:
                raise ConfigurationError(code="invalid_authoritative_backend")
            if not self._declares(entry.descriptor, capability):
                raise ConfigurationError(code="invalid_authoritative_backend")
            result[capability] = name
        return result

    @staticmethod
    def _stable_descriptor(backend: VisionBackend) -> BackendDescriptor:
        try:
            first = backend.descriptor
            second = backend.descriptor
        except Exception as error:
            raise ConfigurationError(code="invalid_backend_descriptor", cause=error) from error
        if not isinstance(first, BackendDescriptor) or first != second:
            raise ConfigurationError(code="unstable_backend_descriptor")
        if not callable(getattr(backend, "supports", None)):
            raise ConfigurationError(code="invalid_backend_contract")
        if not callable(getattr(backend, "analyze", None)):
            raise ConfigurationError(code="invalid_backend_contract")
        if not callable(getattr(backend, "aclose", None)):
            raise ConfigurationError(code="invalid_backend_contract")
        return first

    def _override_entry(self, backend_name: str, request: VisionRequest) -> _BackendEntry:
        try:
            entry = self._by_name[backend_name]
        except KeyError as error:
            raise CapabilityUnavailableError(
                code="backend_override_unavailable",
                backend_name=backend_name,
            ) from error
        if not self._declares(entry.descriptor, request.capability):
            raise CapabilityUnavailableError(
                code="backend_override_incompatible",
                backend_name=backend_name,
            )
        if not self._supports(entry, request):
            raise CapabilityUnavailableError(
                code="capability_option_unavailable",
                backend_name=backend_name,
            )
        return entry

    @staticmethod
    def _declares(descriptor: BackendDescriptor, capability: Capability) -> bool:
        return any(item.capability is capability for item in descriptor.capabilities)

    @staticmethod
    def _supports(entry: _BackendEntry, request: VisionRequest) -> bool:
        try:
            supported = entry.backend.supports(request)
        except Exception as error:
            raise InvalidBackendOutputError(
                code="backend_supports_failed",
                backend_name=entry.descriptor.name,
                cause=error,
            ) from error
        if not isinstance(supported, bool):
            raise InvalidBackendOutputError(
                code="invalid_backend_supports",
                backend_name=entry.descriptor.name,
            )
        return supported

    async def _invoke(
        self,
        entry: _BackendEntry,
        image: BackendImage,
        request: VisionRequest,
    ) -> VisionResult:
        async with entry.semaphore:
            return await entry.backend.analyze(image, request)

    def _effective_timeout(self, timeout_s: float | None) -> float:
        if timeout_s is None:
            return self._backend_timeout_s
        return min(self._backend_timeout_s, self._validate_timeout(timeout_s))

    @staticmethod
    def _validate_timeout(timeout_s: float) -> float:
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
            raise ConfigurationError(code="invalid_backend_timeout")
        result = float(timeout_s)
        if not math.isfinite(result) or result <= 0.0:
            raise ConfigurationError(code="invalid_backend_timeout")
        return result

    def _can_fallback(self, error: BackendError, attempts: int, candidates: int) -> bool:
        if not self._fallback_backends or attempts >= candidates:
            return False
        return isinstance(error, BackendUnavailableError) or error.retryable

    @staticmethod
    def _duration_ms(started: float) -> int:
        return max(0, int((time.monotonic() - started) * 1000))

    def _failed_attempt(
        self,
        descriptor: BackendDescriptor,
        error: PenampakanError,
        started: float,
        *,
        outcome: AttemptOutcome,
    ) -> RouteAttempt:
        return RouteAttempt(
            descriptor=descriptor,
            outcome=outcome,
            duration_ms=self._duration_ms(started),
            error_code=error.code,
            retryable=error.retryable,
        )

    @staticmethod
    def _fallback_warning(attempt: RouteAttempt) -> WarningInfo:
        return WarningInfo(
            code="backend_fallback",
            message="A compatible backend attempt failed during deterministic routing.",
            details={
                "backend_name": attempt.backend_name,
                "error_code": attempt.error_code,
                "retryable": attempt.retryable,
            },
        )

    @staticmethod
    def _set_backend_name(error: BackendError, backend_name: str) -> None:
        if error.backend_name is None:
            error.backend_name = backend_name

    @staticmethod
    def _attach_failure_metadata(
        error: PenampakanError,
        attempts: Sequence[RouteAttempt],
        warnings: Sequence[WarningInfo],
    ) -> None:
        vars(error)["route_attempts"] = tuple(attempts)
        vars(error)["route_warnings"] = tuple(warnings)

    async def _begin_call(self) -> None:
        async with self._state_lock:
            if self._closing or self._closed:
                raise SessionClosedError()
            self._active_calls += 1
            self._idle.clear()

    async def _end_call(self) -> None:
        async with self._state_lock:
            self._active_calls -= 1
            if self._active_calls == 0:
                self._idle.set()

    async def _close_owned_backends(self) -> None:
        await self._idle.wait()
        first_error: BaseException | None = None
        for entry in self._entries:
            try:
                await entry.backend.aclose()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if first_error is None:
                    first_error = error
        async with self._state_lock:
            self._closed = True
        if first_error is not None:
            raise BackendError(code="backend_close_failed", cause=first_error) from first_error

    @staticmethod
    def _is_reserved(name: str) -> bool:
        return name == "penampakan" or name.startswith("penampakan.")


__all__ = [
    "BackendRouter",
    "RouteAttempt",
    "RouteResult",
    "route_failure_attempts",
    "route_failure_warnings",
]
