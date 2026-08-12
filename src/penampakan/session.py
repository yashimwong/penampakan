"""Reusable asynchronous image session orchestration."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from PIL.Image import Image as PillowImage

from penampakan.config import Settings, validate_timeout_s
from penampakan.errors import (
    BudgetExceededError,
    CapabilityUnavailableError,
    EvidenceValidationError,
    InspectionFailedError,
    InvalidModelActionError,
    LLMCallLimitExceededError,
    LLMError,
    LLMNotConfiguredError,
    OperationTimeoutError,
    PenampakanError,
    SessionClosedError,
    ToolLimitExceededError,
)
from penampakan.image.assets import AssetCommit, AssetStore
from penampakan.models import (
    AnswerAction,
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
    ObservationDraft,
    OCRRequest,
    PolicyAction,
    PolicyInput,
    ToolAction,
    TransformPayload,
    VisionAnswer,
    VisionRequest,
    VisionResult,
    WarningInfo,
    WarningPayload,
)
from penampakan.perception.cache import (
    SingleFlightCoordinator,
    build_perception_cache_key,
    canonical_request_json,
    is_durable_cache,
)
from penampakan.perception.normalize import NormalizationLimits, normalize_backend_result
from penampakan.perception.registry import ToolRegistry, ToolResult
from penampakan.perception.router import BackendRouter, RouteResult
from penampakan.perception.store import ObservationStore, ProvenanceSpec
from penampakan.protocols import ActionPolicy, Cache, TraceSink
from penampakan.reasoning.actions import ActionParseError
from penampakan.reasoning.answer import materialize_answer, validate_evidence
from penampakan.reasoning.budget import RunBudget
from penampakan.reasoning.context import CompiledContext, ContextCompiler
from penampakan.tracing import TraceBuilder

_PREPROCESSING_VERSION = "normalize-v2"


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
        self._previous_answer_observation_ids: tuple[str, ...] = ()

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
        if self._policy is None:
            raise LLMNotConfiguredError()
        normalized_question = self._validate_question(question)
        timeout = validate_timeout_s(timeout_s)
        self._require_open()
        async with self._operation_lock:
            self._require_open()
            return await self._run_ask(normalized_question, timeout)

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

    async def _run_ask(
        self,
        question: str,
        timeout_s: float | None,
    ) -> VisionAnswer:
        budget = RunBudget(self._settings.run, timeout_s=timeout_s)
        trace = TraceBuilder(
            content_policy=self._settings.trace_content,
            sinks=self._trace_sinks,
        )
        await trace.start(
            {
                "operation": "ask",
                "asset_id": self._assets.root_id,
                "question": question,
            }
        )
        await trace.emit("image_loaded", {"asset_id": self._assets.root_id})
        try:
            return await asyncio.wait_for(
                self._ask_body(question, budget, trace),
                timeout=budget.remaining_time_s(),
            )
        except asyncio.CancelledError:
            await trace.cancel()
            raise
        except asyncio.TimeoutError as error:
            timeout_error = OperationTimeoutError(trace_id=trace.trace_id, cause=error)
            await trace.fail(timeout_error)
            raise timeout_error from error
        except Exception as error:
            if isinstance(error, PenampakanError):
                error.trace_id = trace.trace_id
            if not trace.finalized:
                await trace.fail(error)
            raise

    async def _ask_body(
        self,
        question: str,
        budget: RunBudget,
        trace: TraceBuilder,
    ) -> VisionAnswer:
        warnings = list(self._load_warnings)
        initial_ids, initial_warnings = await self._ensure_initial_observations(
            question,
            budget,
            trace,
        )
        warnings.extend(initial_warnings)
        actions: list[PolicyAction] = []
        action_counts: dict[str, int] = {}
        previous_action_ids = initial_ids
        recent_lineage: tuple[str, ...] = (self._assets.root_id,)
        while budget.can_start_interactive_step():
            context = self._compile_context(
                question,
                previous_action_ids=previous_action_ids,
                recent_lineage=recent_lineage,
            )
            action = await self._policy_action(
                question,
                context,
                tuple(actions),
                budget,
                trace,
                answer_only=False,
            )
            await budget.reserve_step()
            actions.append(action)
            if isinstance(action, AnswerAction):
                return await self._complete_answer(
                    action,
                    question,
                    context,
                    tuple(actions),
                    budget,
                    trace,
                    warnings,
                    final_call=False,
                )
            canonical_call = self._canonical_tool_call(action)
            count = action_counts.get(canonical_call, 0)
            if count >= self._settings.agent.max_identical_actions:
                warning_observation = await self._commit_reasoning_warning(
                    "repeated_action_cycle",
                    "A repeated identical tool action was blocked by the cycle limit.",
                    trace,
                    parent_observation_ids=previous_action_ids,
                )
                warnings.append(
                    WarningInfo(
                        code="repeated_action_cycle",
                        message="A repeated identical tool action was blocked.",
                    )
                )
                previous_action_ids = (warning_observation.id,)
                continue
            action_counts[canonical_call] = count + 1
            try:
                tool_observations, tool_warnings, lineage = await self._execute_action(
                    action,
                    context,
                    budget,
                    trace,
                    parent_observation_ids=previous_action_ids,
                )
            except BudgetExceededError as error:
                warnings.append(
                    WarningInfo(
                        code="budget_stop",
                        message="A tool action was stopped by a configured run budget.",
                        details={"reason": error.code},
                    )
                )
                break
            previous_action_ids = tuple(item.id for item in tool_observations)
            warnings.extend(tool_warnings)
            if lineage:
                recent_lineage = lineage
        stop_reason = budget.soft_stop_reason() or "step_limit"
        await trace.emit("budget_stop", {"reason": stop_reason})
        context = self._compile_context(
            question,
            previous_action_ids=previous_action_ids,
            recent_lineage=recent_lineage,
        )
        action = await self._policy_action(
            question,
            context,
            tuple(actions),
            budget,
            trace,
            answer_only=True,
        )
        await budget.reserve_step(answer_only=True)
        if not isinstance(action, AnswerAction):
            raise InvalidModelActionError(code="final_action_must_answer")
        actions.append(action)
        return await self._complete_answer(
            action,
            question,
            context,
            tuple(actions),
            budget,
            trace,
            warnings,
            final_call=True,
        )

    async def _ensure_initial_observations(
        self,
        question: str,
        budget: RunBudget,
        trace: TraceBuilder,
    ) -> tuple[tuple[str, ...], tuple[WarningInfo, ...]]:
        await trace.emit("initial_plan_started", {"asset_id": self._assets.root_id})
        new_observations: list[Observation] = []
        warnings: list[WarningInfo] = []
        existing = self._observations.snapshots()
        existing_capabilities = {
            observation.provenance.capability
            for observation in existing
            if observation.asset_id == self._assets.root_id
        }
        for capability in self._settings.agent.initial_capabilities:
            if capability in existing_capabilities:
                continue
            request = self._initial_request(capability, question)
            if request is None or not self._router.supports(request):
                warnings.append(
                    WarningInfo(
                        code="capability_unavailable",
                        message="An initial perception capability is unavailable.",
                        details={"capability": capability.value},
                    )
                )
                continue
            try:
                await budget.reserve_tool_call()
            except ToolLimitExceededError:
                warnings.append(
                    WarningInfo(
                        code="initial_plan_truncated",
                        message="The initial perception plan was truncated by the tool budget.",
                    )
                )
                break
            tool_name = self._tool_name(capability)
            await trace.emit(
                "tool_call_started",
                {
                    "tool_name": tool_name,
                    "asset_id": self._assets.root_id,
                    "request_hash": self._request_hash(request),
                },
            )
            outcome = await self._perceive(
                self._assets.root_id,
                request,
                tool_name=tool_name,
                budget=budget,
                trace=trace,
            )
            committed = self._observations.commit_result(
                self._assets.root_id,
                outcome.result,
                outcome.provenance,
            )
            new_observations.extend(committed)
            warning_observations = self._commit_warning_infos(
                self._assets.root_id,
                outcome.warnings,
                parent_observation_ids=tuple(item.id for item in committed),
            )
            new_observations.extend(warning_observations)
            warnings.extend(outcome.warnings)
            await self._trace_committed(
                trace,
                self._assets.root_id,
                (*committed, *warning_observations),
            )
        return tuple(item.id for item in new_observations), tuple(warnings)

    def _initial_request(
        self,
        capability: Capability,
        question: str,
    ) -> VisionRequest | None:
        if capability is Capability.METADATA:
            return MetadataRequest()
        if capability is Capability.COLORS:
            return ColorsRequest()
        if capability is Capability.CAPTION:
            focused = CaptionRequest(focus=question)
            return focused if self._router.supports(focused) else CaptionRequest()
        if capability is Capability.OCR:
            return OCRRequest()
        return None

    def _compile_context(
        self,
        question: str,
        *,
        previous_action_ids: tuple[str, ...],
        recent_lineage: tuple[str, ...],
    ) -> CompiledContext:
        compiler = ContextCompiler(self._settings.run.max_context_chars)
        compiled = compiler.compile(
            question,
            self._observations.snapshots(),
            root_asset_id=self._assets.root_id,
            relevant_asset_ids=tuple(asset.id for asset in self._assets.snapshots()),
            most_recent_asset_lineage=recent_lineage,
            previous_action_observation_ids=previous_action_ids,
            previous_answer_observation_ids=self._previous_answer_observation_ids,
        )
        return compiled

    async def _policy_action(
        self,
        question: str,
        context: CompiledContext,
        prior_actions: tuple[PolicyAction, ...],
        budget: RunBudget,
        trace: TraceBuilder,
        *,
        answer_only: bool,
        validation_feedback: tuple[WarningInfo, ...] = (),
        invalid_model_output: str | None = None,
        repair: bool = False,
        budget_final: bool | None = None,
    ) -> PolicyAction:
        if self._policy is None:
            raise LLMNotConfiguredError()
        final_reservation = answer_only if budget_final is None else budget_final
        await budget.reserve_llm_call(final=final_reservation, repair=repair)
        await trace.emit(
            "policy_call_started",
            {
                "answer_only": answer_only,
                "repair": repair,
                "included_observation_ids": list(context.visible_observation_ids),
                "omitted_observation_ids": list(context.omitted_observation_ids),
            },
        )
        policy_input = PolicyInput(
            question=question,
            context=context.text,
            tools=self._tools.specs,
            prior_actions=prior_actions,
            remaining=budget.remaining(current_depth=self._maximum_depth()),
            answer_only=answer_only,
            validation_feedback=validation_feedback,
            invalid_model_output=invalid_model_output,
        )
        try:
            action = await asyncio.wait_for(
                self._policy.next_action(policy_input),
                timeout=budget.component_timeout(self._settings.run.llm_timeout_s),
            )
        except asyncio.TimeoutError as error:
            raise LLMError(code="llm_timeout", cause=error) from error
        except ActionParseError as error:
            await trace.emit(
                "invalid_action",
                {"code": "invalid_model_action", "repair": repair},
            )
            if repair:
                raise InvalidModelActionError(cause=error) from error
            try:
                return await self._policy_action(
                    question,
                    context,
                    prior_actions,
                    budget,
                    trace,
                    answer_only=answer_only,
                    validation_feedback=error.feedback,
                    invalid_model_output=error.invalid_model_output,
                    repair=True,
                    budget_final=final_reservation,
                )
            except LLMCallLimitExceededError as limit_error:
                raise InvalidModelActionError(cause=limit_error) from limit_error
        if not isinstance(action, (ToolAction, AnswerAction)):
            raise InvalidModelActionError(code="invalid_policy_action")
        await trace.emit(
            "policy_call_finished",
            {"action_type": action.type, "repair": repair},
        )
        return action

    async def _complete_answer(
        self,
        action: AnswerAction,
        question: str,
        context: CompiledContext,
        prior_actions: tuple[PolicyAction, ...],
        budget: RunBudget,
        trace: TraceBuilder,
        warnings: list[WarningInfo],
        *,
        final_call: bool,
    ) -> VisionAnswer:
        try:
            validate_evidence(
                action,
                self._observations.snapshots(),
                visible_observation_ids=context.visible_observation_ids,
                root_asset_id=self._assets.root_id,
                asset_root_ids=self._asset_root_ids(),
            )
        except EvidenceValidationError as error:
            feedback = (
                WarningInfo(
                    code="evidence_validation",
                    message="The answer contains invalid evidence references.",
                ),
            )
            try:
                repaired = await self._policy_action(
                    question,
                    context,
                    prior_actions,
                    budget,
                    trace,
                    answer_only=True,
                    validation_feedback=feedback,
                    invalid_model_output=action.model_dump_json(exclude_none=True),
                    repair=True,
                    budget_final=final_call,
                )
            except (InvalidModelActionError, LLMCallLimitExceededError):
                raise EvidenceValidationError(cause=error) from error
            if not isinstance(repaired, AnswerAction):
                raise EvidenceValidationError(cause=error) from error
            try:
                validate_evidence(
                    repaired,
                    self._observations.snapshots(),
                    visible_observation_ids=context.visible_observation_ids,
                    root_asset_id=self._assets.root_id,
                    asset_root_ids=self._asset_root_ids(),
                )
            except EvidenceValidationError as second_error:
                raise EvidenceValidationError(cause=second_error) from second_error
            action = repaired
        evidence_ids = tuple(item.observation_id for item in action.evidence)
        self._previous_answer_observation_ids = tuple(dict.fromkeys(evidence_ids))
        await trace.emit(
            "answer_validated",
            {"status": action.status, "evidence_ids": list(evidence_ids)},
        )
        if action.status == "answered":
            completed_trace = await trace.finish("completed")
        else:
            completed_trace = await trace.finish("insufficient_evidence")
        return materialize_answer(
            action,
            self._observations.snapshots(),
            visible_observation_ids=context.visible_observation_ids,
            root_asset_id=self._assets.root_id,
            asset_root_ids=self._asset_root_ids(),
            warnings=(*warnings, *trace.warnings),
            trace=completed_trace,
        )

    async def _execute_action(
        self,
        action: ToolAction,
        context: CompiledContext,
        budget: RunBudget,
        trace: TraceBuilder,
        *,
        parent_observation_ids: tuple[str, ...],
    ) -> tuple[tuple[Observation, ...], tuple[WarningInfo, ...], tuple[str, ...]]:
        try:
            validated = self._tools.validate_arguments(action.tool, action.arguments)
        except Exception:
            warning = await self._commit_reasoning_warning(
                "invalid_tool_arguments",
                "A tool action contained invalid or undeclared arguments.",
                trace,
                parent_observation_ids=parent_observation_ids,
            )
            return (
                (warning,),
                (
                    WarningInfo(
                        code="invalid_tool_arguments",
                        message="A tool action contained invalid arguments.",
                    ),
                ),
                (),
            )
        asset_id = getattr(validated, "asset_id", None)
        if not isinstance(asset_id, str) or asset_id not in context.visible_asset_ids:
            warning = await self._commit_reasoning_warning(
                "tool_asset_not_visible",
                "A tool action referenced an asset outside the latest policy context.",
                trace,
                parent_observation_ids=parent_observation_ids,
            )
            return (
                (warning,),
                (
                    WarningInfo(
                        code="tool_asset_not_visible",
                        message="A tool action referenced an unavailable asset.",
                    ),
                ),
                (),
            )
        await budget.reserve_tool_call()
        reserved_assets = self._transform_asset_count(action.tool, validated)
        if reserved_assets:
            parent = self._assets.snapshot(asset_id)
            await budget.reserve_derived_assets(
                reserved_assets,
                parent_depth=parent.derivation_depth,
            )
        await trace.emit(
            "tool_call_started",
            {
                "tool_name": action.tool,
                "asset_id": asset_id,
                "action_hash": self._canonical_tool_call(action),
            },
        )
        self._active_budget = budget
        self._active_trace = trace
        self._active_tool_name = action.tool
        self._last_perception = None
        try:
            result = await self._tools.execute(self, action.tool, action.arguments)
        except asyncio.CancelledError:
            if reserved_assets:
                await budget.refund_reused_assets(reserved_assets)
            raise
        except Exception:
            if reserved_assets:
                await budget.refund_reused_assets(reserved_assets)
            warning_observation = await self._commit_reasoning_warning(
                "tool_execution_failed",
                "A visual tool could not complete its requested operation.",
                trace,
                parent_observation_ids=parent_observation_ids,
            )
            warning_info = WarningInfo(
                code="tool_execution_failed",
                message="A requested visual tool failed safely.",
                details={"tool_name": action.tool},
            )
            return (warning_observation,), (warning_info,), ()
        finally:
            self._active_budget = None
            self._active_trace = None
            self._active_tool_name = None
        if result.assets:
            commits = self._assets.commit(asset_id, result.assets)
            reused = sum(commit.reused for commit in commits)
            if reused:
                await budget.refund_reused_assets(reused)
            observations = self._commit_transform_observations(
                action,
                commits,
                parent_observation_ids,
            )
            for commit in commits:
                if commit.reused:
                    continue
                await trace.emit(
                    "asset_created",
                    {
                        "asset_id": commit.asset.id,
                        "parent_asset_id": commit.parent_id,
                        "reused": commit.reused,
                    },
                )
            warning_observations = self._commit_warning_infos(
                asset_id,
                result.warnings,
                parent_observation_ids=tuple(item.id for item in observations),
            )
            observations = (*observations, *warning_observations)
            await self._trace_committed_many(trace, observations)
            lineage = self._lineage_for_asset(commits[-1].asset.id) if commits else ()
            return observations, result.warnings, lineage
        perception = self._last_perception
        if perception is None:
            return (), result.warnings, self._lineage_for_asset(asset_id)
        observations = self._observations.commit_result(
            perception.asset_id,
            perception.result,
            ProvenanceSpec(
                tool=perception.provenance.tool,
                capability=perception.provenance.capability,
                backend_name=perception.provenance.backend_name,
                backend_version=perception.provenance.backend_version,
                model_id=perception.provenance.model_id,
                model_revision=perception.provenance.model_revision,
                request_hash=perception.provenance.request_hash,
                cache_hit=perception.provenance.cache_hit,
                duration_ms=perception.provenance.duration_ms,
                parent_observation_ids=parent_observation_ids,
            ),
        )
        warning_observations = self._commit_warning_infos(
            perception.asset_id,
            result.warnings,
            parent_observation_ids=tuple(item.id for item in observations),
        )
        observations = (*observations, *warning_observations)
        await self._trace_committed(trace, perception.asset_id, observations)
        return observations, result.warnings, self._lineage_for_asset(asset_id)

    def _commit_transform_observations(
        self,
        action: ToolAction,
        commits: tuple[AssetCommit, ...],
        parent_observation_ids: tuple[str, ...],
    ) -> tuple[Observation, ...]:
        observations: list[Observation] = []
        provenance = ProvenanceSpec(
            tool=action.tool,
            capability=None,
            backend_name="penampakan.transforms",
            backend_version="1",
            request_hash=self._canonical_tool_call(action),
            duration_ms=0,
            parent_observation_ids=parent_observation_ids,
        )
        for commit in commits:
            draft = ObservationDraft(
                payload=TransformPayload(
                    derived_asset_id=commit.asset.id,
                    parent_asset_id=commit.parent_id,
                    transform=commit.transform,
                )
            )
            observations.extend(
                self._observations.commit_drafts(
                    commit.asset.id,
                    (draft,),
                    provenance,
                )
            )
        return tuple(observations)

    async def _commit_reasoning_warning(
        self,
        code: str,
        message: str,
        trace: TraceBuilder,
        *,
        parent_observation_ids: tuple[str, ...],
    ) -> Observation:
        safe_code = code if code.replace("_", "a").isalnum() else "reasoning_warning"
        draft = ObservationDraft(payload=WarningPayload(code=safe_code, message=message))
        committed = self._observations.commit_drafts(
            self._assets.root_id,
            (draft,),
            ProvenanceSpec(
                tool="reasoning_warning",
                capability=None,
                backend_name="penampakan.core",
                backend_version="1",
                request_hash=hashlib.sha256(code.encode("utf-8")).hexdigest(),
                duration_ms=0,
                parent_observation_ids=parent_observation_ids,
            ),
        )
        await self._trace_committed(trace, self._assets.root_id, committed)
        return committed[0]

    def _commit_warning_infos(
        self,
        asset_id: str,
        warnings: tuple[WarningInfo, ...],
        *,
        parent_observation_ids: tuple[str, ...],
    ) -> tuple[Observation, ...]:
        if not warnings:
            return ()
        drafts = tuple(
            ObservationDraft(payload=WarningPayload(code=warning.code, message=warning.message))
            for warning in warnings
        )
        request_material = "\n".join(warning.code for warning in warnings).encode("utf-8")
        return self._observations.commit_drafts(
            asset_id,
            drafts,
            ProvenanceSpec(
                tool="perception_warning",
                capability=None,
                backend_name="penampakan.core",
                backend_version="1",
                request_hash=hashlib.sha256(request_material).hexdigest(),
                duration_ms=0,
                parent_observation_ids=parent_observation_ids,
            ),
        )

    @staticmethod
    def _transform_asset_count(tool_name: str, validated: object) -> int:
        if tool_name == "tile":
            rows = getattr(validated, "rows", 0)
            columns = getattr(validated, "columns", 0)
            return rows * columns if isinstance(rows, int) and isinstance(columns, int) else 0
        if tool_name in {
            "crop",
            "rotate",
            "enhance_contrast",
            "to_grayscale",
            "add_coordinate_grid",
        }:
            return 1
        return 0

    @staticmethod
    def _canonical_tool_call(action: ToolAction) -> str:
        encoded = json.dumps(
            {"arguments": action.arguments, "tool": action.tool},
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    async def _trace_committed(
        self,
        trace: TraceBuilder,
        asset_id: str,
        observations: tuple[Observation, ...],
    ) -> None:
        await trace.emit(
            "observations_committed",
            {
                "asset_id": asset_id,
                "observation_ids": [item.id for item in observations],
            },
        )

    async def _trace_committed_many(
        self,
        trace: TraceBuilder,
        observations: tuple[Observation, ...],
    ) -> None:
        by_asset: dict[str, list[str]] = {}
        for observation in observations:
            by_asset.setdefault(observation.asset_id, []).append(observation.id)
        for asset_id, observation_ids in by_asset.items():
            await trace.emit(
                "observations_committed",
                {"asset_id": asset_id, "observation_ids": observation_ids},
            )

    def _asset_root_ids(self) -> dict[str, str]:
        return {asset.id: self._assets.root_id for asset in self._assets.snapshots()}

    def _lineage_for_asset(self, asset_id: str) -> tuple[str, ...]:
        snapshots = {asset.id: asset for asset in self._assets.snapshots()}
        lineage: list[str] = []
        current = snapshots.get(asset_id)
        while current is not None:
            lineage.append(current.id)
            current = snapshots.get(current.parent_id) if current.parent_id is not None else None
        return tuple(reversed(lineage))

    def _maximum_depth(self) -> int:
        return max((asset.derivation_depth for asset in self._assets.snapshots()), default=0)

    @staticmethod
    def _validate_question(question: str) -> str:
        if not isinstance(question, str):
            raise TypeError("question must be text")
        normalized = question.strip()
        if not normalized or "\x00" in normalized:
            raise ValueError("question must be non-blank and NUL-free")
        return normalized

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
        durable = is_durable_cache(self._cache)
        allowed = self._cache_allowed(durable, first)
        cached = await self._safe_cache_get(cache_key) if allowed else None
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
            if self._cache_allowed(durable, route_result.descriptor):
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

    @staticmethod
    def _cache_allowed(durable: bool, descriptor: BackendDescriptor) -> bool:
        return not durable or descriptor.durable_cache_eligible

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
