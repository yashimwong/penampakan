"""Reusable asynchronous image session orchestration."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from PIL.Image import Image as PillowImage

from penampakan.config import Settings, validate_timeout_s
from penampakan.errors import (
    CapabilityUnavailableError,
    InspectionFailedError,
    LLMNotConfiguredError,
    OperationTimeoutError,
    PenampakanError,
    SessionClosedError,
)
from penampakan.image.assets import AssetStore
from penampakan.models import (
    BackendDescriptor,
    Capability,
    CaptionRequest,
    ColorsRequest,
    ImageAsset,
    InspectionOperation,
    InspectionPlan,
    InspectionResult,
    MetadataRequest,
    Observation,
    OCRRequest,
    VisionAnswer,
    VisionRequest,
    VisionResult,
    WarningInfo,
)
from penampakan.perception.cache import (
    SingleFlightCoordinator,
    build_perception_cache_key,
    canonical_request_json,
)
from penampakan.perception.normalize import NormalizationLimits, normalize_backend_result
from penampakan.perception.registry import ToolRegistry, ToolResult
from penampakan.perception.router import BackendRouter, RouteResult
from penampakan.perception.store import ObservationStore, ProvenanceSpec
from penampakan.protocols import ActionPolicy, Cache, TraceSink
from penampakan.reasoning.budget import RunBudget
from penampakan.tracing import TraceBuilder

_PREPROCESSING_VERSION = "normalize-v1"


@dataclass(frozen=True, slots=True)
class _PerceptionOutcome:
    asset_id: str
    result: VisionResult
    provenance: ProvenanceSpec
    warnings: tuple[WarningInfo, ...]


@dataclass(frozen=True, slots=True)
class _PlannedOperation:
    operation: InspectionOperation
    tool_name: str
    default: bool


@dataclass(frozen=True, slots=True)
class _OperationOutcome:
    planned: _PlannedOperation
    perception: _PerceptionOutcome | None = None
    warning: WarningInfo | None = None
    error: BaseException | None = None


class AsyncVisionSession:
    """Own one normalized image lineage and its reusable visual observations."""

    def __init__(
        self,
        *,
        asset_store: AssetStore,
        router: BackendRouter,
        tools: ToolRegistry,
        policy: ActionPolicy | None,
        cache: Cache,
        singleflight: SingleFlightCoordinator[bytes],
        settings: Settings,
        trace_sinks: Sequence[TraceSink] = (),
        load_warnings: Sequence[WarningInfo] = (),
        on_close: Callable[[AsyncVisionSession], None] | None = None,
    ) -> None:
        self._assets = asset_store
        self._router = router
        self._tools = tools
        self._policy = policy
        self._cache = cache
        self._singleflight = singleflight
        self._settings = settings
        self._trace_sinks = tuple(trace_sinks)
        self._load_warnings = tuple(load_warnings)
        self._on_close = on_close
        self._observations = ObservationStore(asset_store)
        self._operation_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._closing = False
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._active_budget: RunBudget | None = None
        self._active_trace: TraceBuilder | None = None
        self._active_tool_name: str | None = None
        self._last_perception: _PerceptionOutcome | None = None

    @property
    def root_asset(self) -> ImageAsset:
        """Return the immutable root asset snapshot."""
        self._require_open()
        return self._assets.root

    @property
    def assets(self) -> tuple[ImageAsset, ...]:
        """Return immutable asset snapshots in creation order."""
        self._require_open()
        return self._assets.snapshots()

    @property
    def observations(self) -> tuple[Observation, ...]:
        """Return immutable observation snapshots in commit order."""
        self._require_open()
        return self._observations.snapshots()

    @property
    def closed(self) -> bool:
        """Return whether all session-owned image state has closed."""
        return self._closed

    def get_asset(self, asset_id: str) -> ImageAsset:
        """Return one owned asset snapshot by stable ID."""
        self._require_open()
        return self._assets.snapshot(asset_id)

    def get_observation(self, observation_id: str) -> Observation:
        """Return one committed observation snapshot by session-local ID."""
        self._require_open()
        return self._observations.get(observation_id)

    async def inspect(
        self,
        plan: InspectionPlan | None = None,
        *,
        timeout_s: float | None = None,
    ) -> InspectionResult:
        """Execute a deterministic bounded inspection plan."""
        selected_plan = plan or InspectionPlan()
        if not isinstance(selected_plan, InspectionPlan):
            raise TypeError("plan must be an InspectionPlan")
        timeout = validate_timeout_s(timeout_s)
        self._require_open()
        async with self._operation_lock:
            self._require_open()
            return await self._run_inspection(selected_plan, timeout)

    async def ask(
        self,
        question: str,
        *,
        timeout_s: float | None = None,
    ) -> VisionAnswer:
        """Answer a question through the configured bounded action policy."""
        raise LLMNotConfiguredError()

    def image(self, asset_id: str) -> PillowImage:
        """Return a caller-owned normalized image copy for a built-in tool."""
        self._require_open()
        return self._assets.image(asset_id)

    def ensure_asset_capacity(self, parent_id: str, count: int) -> None:
        """Validate session asset capacity before rendering a transform."""
        self._require_open()
        self._assets.ensure_capacity(parent_id, count)

    async def perceive(self, asset_id: str, request: VisionRequest) -> ToolResult:
        """Route one active ask-tool perception request."""
        budget = self._active_budget
        trace = self._active_trace
        tool_name = self._active_tool_name
        if budget is None or trace is None or tool_name is None:
            raise RuntimeError("perception requires an active tool call")
        outcome = await self._perceive(
            asset_id,
            request,
            tool_name=tool_name,
            budget=budget,
            trace=trace,
        )
        self._last_perception = outcome
        return ToolResult(
            observations=outcome.result.observations,
            warnings=outcome.warnings,
        )

    async def aclose(self) -> None:
        """Wait for active work and release private session state exactly once."""
        async with self._state_lock:
            if self._close_task is None:
                self._closing = True
                self._close_task = asyncio.create_task(self._close_owned())
            close_task = self._close_task
        await asyncio.shield(close_task)

    async def __aenter__(self) -> AsyncVisionSession:
        """Enter this open reusable session."""
        self._require_open()
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        """Close this reusable session context."""
        await self.aclose()

    async def _run_inspection(
        self,
        plan: InspectionPlan,
        timeout_s: float | None,
    ) -> InspectionResult:
        budget = RunBudget(self._settings.run, timeout_s=timeout_s)
        trace = TraceBuilder(
            content_policy=self._settings.trace_content,
            sinks=self._trace_sinks,
        )
        await trace.start({"operation": "inspect", "asset_id": self._assets.root_id})
        await trace.emit("image_loaded", {"asset_id": self._assets.root_id})
        try:
            result = await asyncio.wait_for(
                self._inspect_body(plan, budget, trace),
                timeout=budget.remaining_time_s(),
            )
        except asyncio.CancelledError:
            await trace.cancel()
            raise
        except asyncio.TimeoutError as error:
            timeout_error = OperationTimeoutError(trace_id=trace.trace_id, cause=error)
            await trace.fail(timeout_error)
            raise timeout_error from error
        except InspectionFailedError:
            raise
        except Exception as error:
            if isinstance(error, PenampakanError):
                error.trace_id = trace.trace_id
            await trace.fail(error)
            raise
        return result

    async def _inspect_body(
        self,
        plan: InspectionPlan,
        budget: RunBudget,
        trace: TraceBuilder,
    ) -> InspectionResult:
        planned = self._inspection_operations(plan)
        await trace.emit("initial_plan_started", {"operation_count": len(planned)})
        if plan.fail_fast:
            outcomes = await self._inspect_sequential(planned, budget, trace)
        else:
            outcomes = await self._inspect_parallel(planned, budget, trace)
        committed: list[Observation] = []
        warnings = list(self._load_warnings)
        failures: list[BaseException] = []
        required_failure = False
        executable = 0
        for outcome in outcomes:
            if outcome.warning is not None:
                warnings.append(outcome.warning)
                continue
            executable += 1
            if outcome.error is not None:
                failures.append(outcome.error)
                required_failure = required_failure or outcome.planned.operation.required
                continue
            perception = outcome.perception
            if perception is None:
                continue
            observations = self._observations.commit_result(
                perception.asset_id,
                perception.result,
                perception.provenance,
            )
            committed.extend(observations)
            warnings.extend(perception.warnings)
            await trace.emit(
                "observations_committed",
                {
                    "asset_id": perception.asset_id,
                    "observation_ids": [item.id for item in observations],
                },
            )
        failed_all = executable > 0 and not committed and bool(failures)
        if required_failure or failed_all:
            failed_trace = await trace.fail(failures[0] if failures else None)
            partial = InspectionResult(
                root_asset=self._assets.root,
                observations=tuple(committed),
                warnings=tuple(warnings),
                trace=failed_trace,
            )
            raise InspectionFailedError(
                partial_result=partial,
                trace_id=trace.trace_id,
                cause=failures[0] if failures else None,
            )
        completed_trace = await trace.finish()
        return InspectionResult(
            root_asset=self._assets.root,
            observations=tuple(committed),
            warnings=tuple(warnings),
            trace=completed_trace,
        )

    def _inspection_operations(self, plan: InspectionPlan) -> tuple[_PlannedOperation, ...]:
        explicit = tuple(plan.operations)
        for operation in explicit:
            self._assets.snapshot(operation.asset_id or self._assets.root_id)
        result: list[_PlannedOperation] = []
        if plan.include_available_overview:
            defaults: tuple[tuple[VisionRequest, str], ...] = (
                (MetadataRequest(), "get_metadata"),
                (ColorsRequest(), "get_colors"),
                (CaptionRequest(), "describe_image"),
                (OCRRequest(), "read_text"),
            )
            explicit_root_capabilities = {
                operation.request.capability
                for operation in explicit
                if operation.asset_id in {None, self._assets.root_id}
            }
            for request, tool_name in defaults:
                if request.capability in explicit_root_capabilities:
                    continue
                if request.capability is Capability.METADATA and self._has_root_metadata():
                    continue
                operation = InspectionOperation(request=request)
                if self._router.supports(request):
                    result.append(_PlannedOperation(operation, tool_name, True))
        result.extend(
            _PlannedOperation(
                operation,
                self._tool_name(operation.request.capability),
                False,
            )
            for operation in explicit
        )
        return tuple(result)

    async def _inspect_sequential(
        self,
        planned: tuple[_PlannedOperation, ...],
        budget: RunBudget,
        trace: TraceBuilder,
    ) -> tuple[_OperationOutcome, ...]:
        outcomes: list[_OperationOutcome] = []
        for item in planned:
            outcome = await self._inspect_one(item, budget, trace)
            outcomes.append(outcome)
            if outcome.error is not None:
                break
        return tuple(outcomes)

    async def _inspect_parallel(
        self,
        planned: tuple[_PlannedOperation, ...],
        budget: RunBudget,
        trace: TraceBuilder,
    ) -> tuple[_OperationOutcome, ...]:
        semaphore = asyncio.Semaphore(self._settings.run.max_parallel_tools)

        async def execute(item: _PlannedOperation) -> _OperationOutcome:
            async with semaphore:
                return await self._inspect_one(item, budget, trace)

        return tuple(await asyncio.gather(*(execute(item) for item in planned)))

    async def _inspect_one(
        self,
        planned: _PlannedOperation,
        budget: RunBudget,
        trace: TraceBuilder,
    ) -> _OperationOutcome:
        operation = planned.operation
        asset_id = operation.asset_id or self._assets.root_id
        try:
            if not self._router.supports(
                operation.request,
                backend_name=operation.backend,
            ):
                if operation.required:
                    return _OperationOutcome(
                        planned=planned,
                        error=CapabilityUnavailableError(code="capability_unavailable"),
                    )
                return _OperationOutcome(
                    planned=planned,
                    warning=WarningInfo(
                        code="capability_unavailable",
                        message="The requested optional capability is unavailable.",
                        details={"capability": operation.request.capability.value},
                    ),
                )
            await budget.reserve_tool_call()
            await trace.emit(
                "tool_call_started",
                {
                    "tool_name": planned.tool_name,
                    "asset_id": asset_id,
                    "request_hash": self._request_hash(operation.request),
                },
            )
            perception = await self._perceive(
                asset_id,
                operation.request,
                tool_name=planned.tool_name,
                budget=budget,
                trace=trace,
                backend_name=operation.backend,
            )
            return _OperationOutcome(planned=planned, perception=perception)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            return _OperationOutcome(planned=planned, error=error)

    async def _perceive(
        self,
        asset_id: str,
        request: VisionRequest,
        *,
        tool_name: str,
        budget: RunBudget,
        trace: TraceBuilder,
        backend_name: str | None = None,
    ) -> _PerceptionOutcome:
        image = self._assets.backend_image(asset_id)
        candidates = self._router.route(request, backend_name=backend_name)
        first = candidates[0]
        cache_key = build_perception_cache_key(
            asset_digest_sha256=image.asset.digest_sha256,
            request=request,
            backend=first,
            preprocessing_version=_PREPROCESSING_VERSION,
        )
        cache_warning: WarningInfo | None = None
        cached = await self._safe_cache_get(cache_key)
        if cached is not None:
            try:
                result = normalize_backend_result(
                    VisionResult.model_validate_json(cached, strict=True),
                    request,
                    limits=self._normalization_limits(),
                )
            except Exception:
                cache_warning = WarningInfo(
                    code="invalid_cache_entry",
                    message="An invalid cached perception result was ignored.",
                )
            else:
                await trace.emit(
                    "cache_hit",
                    {"asset_id": asset_id, "backend_name": first.name},
                )
                provenance = self._provenance(
                    tool_name,
                    request,
                    first,
                    duration_ms=0,
                    cache_hit=True,
                )
                warnings = (
                    *result.warnings,
                    *self._empty_result_warnings(request, result),
                )
                return _PerceptionOutcome(asset_id, result, provenance, warnings)
        route_result: RouteResult | None = None

        async def populate() -> bytes:
            nonlocal route_result

            async def before_attempt(descriptor: BackendDescriptor) -> None:
                await budget.reserve_backend_call()
                await trace.emit(
                    "backend_call_started",
                    {
                        "asset_id": asset_id,
                        "backend_name": descriptor.name,
                        "capability": request.capability.value,
                        "request_hash": self._request_hash(request),
                    },
                )

            route_result = await self._router.analyze(
                image,
                request,
                backend_name=backend_name,
                timeout_s=budget.component_timeout(self._settings.run.backend_timeout_s),
                before_attempt=before_attempt,
            )
            normalized = normalize_backend_result(
                route_result.result,
                request,
                limits=self._normalization_limits(),
            )
            encoded = normalized.model_dump_json(exclude_none=True).encode("utf-8")
            actual_key = build_perception_cache_key(
                asset_digest_sha256=image.asset.digest_sha256,
                request=request,
                backend=route_result.descriptor,
                preprocessing_version=_PREPROCESSING_VERSION,
            )
            await self._safe_cache_set(actual_key, encoded)
            return encoded

        encoded = await self._singleflight.run(cache_key, populate)
        result = normalize_backend_result(
            VisionResult.model_validate_json(encoded, strict=True),
            request,
            limits=self._normalization_limits(),
        )
        if route_result is None:
            descriptor = first
            duration_ms = 0
            route_warnings: tuple[WarningInfo, ...] = ()
            cache_hit = True
            await trace.emit("cache_hit", {"asset_id": asset_id, "backend_name": first.name})
        else:
            descriptor = route_result.descriptor
            duration_ms = sum(attempt.duration_ms for attempt in route_result.attempts)
            route_warnings = route_result.warnings
            cache_hit = False
            for attempt in route_result.attempts:
                await trace.emit(
                    "backend_call_finished",
                    {
                        "backend_name": attempt.backend_name,
                        "outcome": attempt.outcome,
                        "error_code": attempt.error_code,
                    },
                    duration_ms=attempt.duration_ms,
                )
        warnings = (
            *((cache_warning,) if cache_warning is not None else ()),
            *route_warnings,
            *result.warnings,
            *self._empty_result_warnings(request, result),
        )
        provenance = self._provenance(
            tool_name,
            request,
            descriptor,
            duration_ms=duration_ms,
            cache_hit=cache_hit,
        )
        return _PerceptionOutcome(asset_id, result, provenance, warnings)

    async def _safe_cache_get(self, key: str) -> bytes | None:
        try:
            return await self._cache.get(key)
        except asyncio.CancelledError:
            raise
        except Exception:
            return None

    async def _safe_cache_set(self, key: str, value: bytes) -> None:
        try:
            await self._cache.set(key, value, size=len(value))
        except asyncio.CancelledError:
            raise
        except Exception:
            return

    def _provenance(
        self,
        tool_name: str,
        request: VisionRequest,
        descriptor: BackendDescriptor,
        *,
        duration_ms: int,
        cache_hit: bool,
    ) -> ProvenanceSpec:
        return ProvenanceSpec(
            tool=tool_name,
            capability=request.capability,
            backend_name=descriptor.name,
            backend_version=descriptor.version,
            model_id=descriptor.model_id,
            model_revision=descriptor.model_revision,
            request_hash=self._request_hash(request),
            cache_hit=cache_hit,
            duration_ms=duration_ms,
        )

    def _normalization_limits(self) -> NormalizationLimits:
        return NormalizationLimits(
            max_ocr_chars_per_observation=(self._settings.run.max_ocr_chars_per_observation)
        )

    @staticmethod
    def _empty_result_warnings(
        request: VisionRequest,
        result: VisionResult,
    ) -> tuple[WarningInfo, ...]:
        if result.observations:
            return ()
        if request.capability is Capability.OCR:
            code = "no_text_detected"
            message = "No text was detected in the requested image region."
        elif request.capability is Capability.DETECT:
            code = "no_objects_detected"
            message = "No objects were detected in the requested image region."
        else:
            code = "no_observations"
            message = "The requested perception call returned no observations."
        return (WarningInfo(code=code, message=message),)

    @staticmethod
    def _request_hash(request: VisionRequest) -> str:
        return hashlib.sha256(canonical_request_json(request)).hexdigest()

    @staticmethod
    def _tool_name(capability: Capability) -> str:
        return {
            Capability.METADATA: "get_metadata",
            Capability.COLORS: "get_colors",
            Capability.CAPTION: "describe_image",
            Capability.OCR: "read_text",
            Capability.DETECT: "detect_objects",
            Capability.SEGMENT: "segment_objects",
        }[capability]

    def _has_root_metadata(self) -> bool:
        return any(
            observation.asset_id == self._assets.root_id and observation.payload.type == "metadata"
            for observation in self._observations.snapshots()
        )

    async def _close_owned(self) -> None:
        async with self._operation_lock:
            self._observations.close()
            self._assets.close()
            self._closed = True
            if self._on_close is not None:
                self._on_close(self)

    def _require_open(self) -> None:
        if self._closing or self._closed:
            raise SessionClosedError()


__all__ = ["AsyncVisionSession"]
