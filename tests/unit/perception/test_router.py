import asyncio
from collections import deque
from collections.abc import Callable

import pytest

from penampakan.errors import (
    BackendError,
    BackendTimeoutError,
    BackendUnavailableError,
    CapabilityUnavailableError,
    ConfigurationError,
    PolicyDeniedError,
    SessionClosedError,
)
from penampakan.models import (
    BackendDescriptor,
    BackendImage,
    Capability,
    CapabilityDescriptor,
    CaptionRequest,
    ImageAsset,
    VisionRequest,
    VisionResult,
)
from penampakan.perception.router import (
    BackendRouter,
    route_failure_attempts,
    route_failure_warnings,
)

EMPTY_RESULT = VisionResult(observations=())


def make_backend_image() -> BackendImage:
    asset = ImageAsset(
        id="img_0123456789abcdef",
        width=1,
        height=1,
        mode="RGB",
        mime_type="image/png",
        original_format="PNG",
        digest_sha256="a" * 64,
        parent_id=None,
        derivation_depth=0,
        transform=None,
    )
    return BackendImage(asset=asset, content=b"canonical-png")


class ScriptedBackend:
    def __init__(
        self,
        name: str,
        *,
        capability: Capability = Capability.CAPTION,
        outcomes: tuple[VisionResult | BaseException, ...] = (EMPTY_RESULT,),
        supports: Callable[[VisionRequest], bool] | None = None,
        is_remote: bool = False,
        max_concurrency: int = 1,
    ) -> None:
        self._descriptor = BackendDescriptor(
            name=name,
            version="1.0.0",
            capabilities=(CapabilityDescriptor(capability=capability),),
            is_remote=is_remote,
            max_concurrency=max_concurrency,
        )
        self._outcomes = deque(outcomes)
        self._supports = supports
        self.requests: list[VisionRequest] = []
        self.active = 0
        self.peak_active = 0
        self.cancellations = 0
        self.close_count = 0
        self.gate: asyncio.Event | None = None

    @property
    def descriptor(self) -> BackendDescriptor:
        return self._descriptor

    def supports(self, request: VisionRequest) -> bool:
        if self._supports is not None:
            return self._supports(request)
        return request.capability is self._descriptor.capabilities[0].capability

    async def analyze(self, image: BackendImage, request: VisionRequest) -> VisionResult:
        self.requests.append(request)
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        try:
            if self.gate is not None:
                await self.gate.wait()
            outcome = self._outcomes[0] if len(self._outcomes) == 1 else self._outcomes.popleft()
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        except asyncio.CancelledError:
            self.cancellations += 1
            raise
        finally:
            self.active -= 1

    async def aclose(self) -> None:
        self.close_count += 1


async def wait_until(predicate: Callable[[], bool]) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout=1.0)


def route_names(router: BackendRouter, request: VisionRequest) -> tuple[str, ...]:
    return tuple(descriptor.name for descriptor in router.route(request))


async def test_override_preferences_and_registration_order_are_deterministic() -> None:
    first = ScriptedBackend("example.first")
    second = ScriptedBackend("example.second")
    remote = ScriptedBackend("example.remote", is_remote=True)
    router = BackendRouter(
        (first, second, remote),
        preferences={Capability.CAPTION: ("example.remote", "example.second")},
    )
    request = CaptionRequest()

    assert route_names(router, request) == (
        "example.remote",
        "example.second",
        "example.first",
    )
    assert tuple(
        descriptor.name for descriptor in router.route(request, backend_name="example.first")
    ) == ("example.first", "example.remote", "example.second")

    result = await router.analyze(make_backend_image(), request)

    assert result.descriptor.name == "example.remote"
    assert result.descriptor.is_remote is True
    assert remote.requests == [request]
    assert first.requests == []
    assert second.requests == []

    await router.aclose()


async def test_registration_order_breaks_ties_without_preferences() -> None:
    first = ScriptedBackend("example.first")
    second = ScriptedBackend("example.second")
    router = BackendRouter((first, second))

    assert route_names(router, CaptionRequest()) == ("example.first", "example.second")

    await router.aclose()


@pytest.mark.parametrize(
    "backends, code",
    (
        ((ScriptedBackend("penampakan.claimed"),), "reserved_backend_name"),
        (
            (ScriptedBackend("example.same"), ScriptedBackend("example.same")),
            "duplicate_backend_name",
        ),
    ),
)
def test_reserved_and_duplicate_backend_names_are_rejected(
    backends: tuple[ScriptedBackend, ...],
    code: str,
) -> None:
    with pytest.raises(ConfigurationError) as captured:
        BackendRouter(backends)

    assert captured.value.code == code


def test_duplicate_backend_instance_is_rejected() -> None:
    backend = ScriptedBackend("example.single")

    with pytest.raises(ConfigurationError) as captured:
        BackendRouter((backend, backend))

    assert captured.value.code == "duplicate_backend_instance"


@pytest.mark.parametrize(
    "preferences, code",
    (
        ({Capability.CAPTION: ("example.unknown",)}, "unknown_backend_preference"),
        (
            {Capability.CAPTION: ("example.caption", "example.caption")},
            "duplicate_backend_preference",
        ),
        ({Capability.METADATA: ("example.caption",)}, "incompatible_backend_preference"),
    ),
)
def test_invalid_preferences_are_rejected(
    preferences: dict[Capability, tuple[str, ...]],
    code: str,
) -> None:
    backend = ScriptedBackend("example.caption")

    with pytest.raises(ConfigurationError) as captured:
        BackendRouter((backend,), preferences=preferences)

    assert captured.value.code == code


async def test_option_specific_support_filters_routes_and_honors_override() -> None:
    global_only = ScriptedBackend(
        "example.global",
        supports=lambda request: isinstance(request, CaptionRequest) and request.focus is None,
    )
    focused = ScriptedBackend(
        "example.focused",
        supports=lambda request: isinstance(request, CaptionRequest) and request.focus is not None,
    )
    router = BackendRouter((global_only, focused))
    request = CaptionRequest(focus="Read the display")

    assert route_names(router, request) == ("example.focused",)
    assert router.supports(request)
    assert not router.supports(request, backend_name="example.global")

    with pytest.raises(CapabilityUnavailableError) as captured:
        router.route(request, backend_name="example.global")

    assert captured.value.code == "capability_option_unavailable"
    assert global_only.requests == []

    await router.aclose()


async def test_unavailable_and_retryable_errors_fall_back_with_safe_metadata() -> None:
    secret = "backend-secret-sentinel"
    unavailable = ScriptedBackend(
        "example.unavailable",
        outcomes=(BackendUnavailableError(cause=RuntimeError(secret)),),
    )
    retryable = ScriptedBackend(
        "example.retryable",
        outcomes=(BackendError(code="temporary_backend_error", retryable=True),),
    )
    successful = ScriptedBackend("example.successful")
    router = BackendRouter((unavailable, retryable, successful))

    result = await router.analyze(make_backend_image(), CaptionRequest())

    assert result.descriptor.name == "example.successful"
    assert result.backend_calls == 3
    assert tuple(attempt.outcome for attempt in result.attempts) == (
        "unavailable",
        "retryable_error",
        "success",
    )
    assert tuple(attempt.backend_name for attempt in result.attempts) == (
        "example.unavailable",
        "example.retryable",
        "example.successful",
    )
    assert tuple(warning.code for warning in result.warnings) == (
        "backend_fallback",
        "backend_fallback",
    )
    assert secret not in repr(result.attempts)
    assert secret not in repr(result.warnings)

    await router.aclose()


@pytest.mark.parametrize(
    "failure",
    (
        BackendError(code="non_retryable_backend_error"),
        PolicyDeniedError(),
    ),
)
async def test_prohibited_failures_propagate_without_fallback(
    failure: BackendError | PolicyDeniedError,
) -> None:
    failing = ScriptedBackend("example.failing", outcomes=(failure,))
    untouched = ScriptedBackend("example.untouched")
    router = BackendRouter((failing, untouched))

    with pytest.raises(type(failure)) as captured:
        await router.analyze(make_backend_image(), CaptionRequest())

    assert captured.value is failure
    assert untouched.requests == []
    assert len(route_failure_attempts(captured.value)) == 1
    assert len(route_failure_warnings(captured.value)) == 1

    await router.aclose()


async def test_disabled_fallback_propagates_retryable_failure() -> None:
    failure = BackendUnavailableError()
    failing = ScriptedBackend("example.failing", outcomes=(failure,))
    untouched = ScriptedBackend("example.untouched")
    router = BackendRouter((failing, untouched), fallback_backends=False)

    with pytest.raises(BackendUnavailableError) as captured:
        await router.analyze(make_backend_image(), CaptionRequest())

    assert captured.value is failure
    assert untouched.requests == []

    await router.aclose()


async def test_backend_semaphore_enforces_per_instance_concurrency() -> None:
    backend = ScriptedBackend("example.serialized", max_concurrency=2)
    backend.gate = asyncio.Event()
    router = BackendRouter((backend,))
    image = make_backend_image()
    request = CaptionRequest()
    tasks = tuple(asyncio.create_task(router.analyze(image, request)) for _ in range(5))

    await wait_until(lambda: backend.active == 2)

    assert backend.peak_active == 2
    assert len(backend.requests) == 2

    backend.gate.set()
    await asyncio.gather(*tasks)

    assert backend.peak_active == 2
    assert len(backend.requests) == 5

    await router.aclose()


async def test_component_timeout_falls_back_and_records_attempt() -> None:
    slow = ScriptedBackend("example.slow")
    slow.gate = asyncio.Event()
    successful = ScriptedBackend("example.successful")
    router = BackendRouter((slow, successful), backend_timeout_s=0.01)

    result = await router.analyze(make_backend_image(), CaptionRequest())

    assert result.descriptor.name == "example.successful"
    assert tuple(attempt.outcome for attempt in result.attempts) == ("timeout", "success")
    assert result.attempts[0].error_code == "backend_timeout"
    assert slow.cancellations == 1

    await router.aclose()


async def test_component_timeout_without_fallback_raises_safe_error() -> None:
    slow = ScriptedBackend("example.slow")
    slow.gate = asyncio.Event()
    router = BackendRouter((slow,), backend_timeout_s=0.01)

    with pytest.raises(BackendTimeoutError) as captured:
        await router.analyze(make_backend_image(), CaptionRequest())

    attempts = route_failure_attempts(captured.value)
    assert len(attempts) == 1
    assert attempts[0].outcome == "timeout"
    assert attempts[0].backend_name == "example.slow"

    await router.aclose()


async def test_cancellation_propagates_without_fallback_and_router_recovers() -> None:
    cancellable = ScriptedBackend("example.cancellable")
    cancellable.gate = asyncio.Event()
    untouched = ScriptedBackend("example.untouched")
    router = BackendRouter((cancellable, untouched))
    task = asyncio.create_task(router.analyze(make_backend_image(), CaptionRequest()))

    await wait_until(lambda: cancellable.active == 1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert cancellable.cancellations == 1
    assert untouched.requests == []

    cancellable.gate.set()
    result = await router.analyze(make_backend_image(), CaptionRequest())

    assert result.descriptor.name == "example.cancellable"

    await router.aclose()


async def test_close_waits_for_active_route_and_is_concurrently_idempotent() -> None:
    backend = ScriptedBackend("example.owned")
    backend.gate = asyncio.Event()
    router = BackendRouter((backend,))
    analysis = asyncio.create_task(router.analyze(make_backend_image(), CaptionRequest()))

    await wait_until(lambda: backend.active == 1)

    first_close = asyncio.create_task(router.aclose())
    second_close = asyncio.create_task(router.aclose())
    await asyncio.sleep(0)

    assert not first_close.done()
    assert not second_close.done()
    assert backend.close_count == 0

    with pytest.raises(SessionClosedError):
        await router.analyze(make_backend_image(), CaptionRequest())

    backend.gate.set()
    await analysis
    await asyncio.gather(first_close, second_close)

    assert router.closed
    assert backend.close_count == 1

    await router.aclose()

    assert backend.close_count == 1
