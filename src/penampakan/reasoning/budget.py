"""Atomic bounded-run counters and monotonic deadline enforcement."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from penampakan.config import RunLimits, validate_timeout_s
from penampakan.errors import (
    AssetLimitExceededError,
    BackendCallLimitExceededError,
    ContextLimitExceededError,
    DerivationDepthLimitExceededError,
    LLMCallLimitExceededError,
    OperationTimeoutError,
    StepLimitExceededError,
    ToolLimitExceededError,
)
from penampakan.models import RemainingBudget

BudgetStopReason = Literal[
    "step_limit",
    "llm_limit",
    "tool_limit",
    "backend_limit",
    "asset_limit",
    "depth_limit",
    "context_limit",
]


@dataclass(frozen=True, slots=True)
class RunBudgetUsage:
    """Immutable counters for work reserved or completed by one run."""

    steps: int
    llm_calls: int
    tool_calls: int
    backend_calls: int
    derived_assets: int


class RunBudget:
    """Atomically reserve bounded work before execution under one deadline."""

    def __init__(
        self,
        limits: RunLimits,
        *,
        timeout_s: float | None = None,
        monotonic_clock: Callable[[], float] | None = None,
    ) -> None:
        self._limits = limits
        self._clock = monotonic_clock or time.monotonic
        self._started = self._read_clock()
        selected_timeout = validate_timeout_s(timeout_s)
        self._timeout_s = selected_timeout or limits.default_timeout_s
        self._deadline = self._started + self._timeout_s
        self._steps = 0
        self._llm_calls = 0
        self._tool_calls = 0
        self._backend_calls = 0
        self._derived_assets = 0
        self._lock = asyncio.Lock()

    @property
    def limits(self) -> RunLimits:
        """Return immutable configured run limits."""

        return self._limits

    @property
    def deadline(self) -> float:
        """Return the absolute monotonic deadline."""

        return self._deadline

    @property
    def usage(self) -> RunBudgetUsage:
        """Return an immutable atomic-value snapshot of reserved work."""

        return RunBudgetUsage(
            steps=self._steps,
            llm_calls=self._llm_calls,
            tool_calls=self._tool_calls,
            backend_calls=self._backend_calls,
            derived_assets=self._derived_assets,
        )

    def remaining_time_s(self) -> float:
        """Return finite non-negative time remaining on the overall deadline."""

        return max(0.0, self._deadline - self._read_clock())

    def check_deadline(self) -> None:
        """Raise immediately when the overall monotonic deadline is exhausted."""

        if self.remaining_time_s() <= 0.0:
            raise OperationTimeoutError()

    def component_timeout(self, configured_timeout_s: float) -> float:
        """Return the smaller positive component or overall remaining timeout."""

        if (
            isinstance(configured_timeout_s, bool)
            or not isinstance(configured_timeout_s, (int, float))
            or not math.isfinite(configured_timeout_s)
            or configured_timeout_s <= 0.0
        ):
            raise ValueError("component timeout must be finite and positive")
        self.check_deadline()
        return min(float(configured_timeout_s), self.remaining_time_s())

    def can_start_interactive_step(self) -> bool:
        """Return whether one decision can start while preserving the final call."""

        return (
            self.remaining_time_s() > 0.0
            and self._steps < self._limits.max_steps - 1
            and self._llm_calls < self._limits.max_llm_calls - 1
        )

    def soft_stop_reason(self) -> BudgetStopReason | None:
        """Return the first deterministic soft limit preventing another decision."""

        if self._steps >= self._limits.max_steps - 1:
            return "step_limit"
        if self._llm_calls >= self._limits.max_llm_calls - 1:
            return "llm_limit"
        return None

    async def reserve_step(self, *, answer_only: bool = False) -> None:
        """Reserve one valid policy decision, preserving the final step when interactive."""

        async with self._lock:
            self.check_deadline()
            maximum = self._limits.max_steps if answer_only else self._limits.max_steps - 1
            if self._steps >= maximum:
                raise StepLimitExceededError()
            self._steps += 1

    async def reserve_llm_call(
        self,
        *,
        final: bool = False,
        repair: bool = False,
    ) -> None:
        """Reserve one policy invocation under final-call and repair semantics."""

        async with self._lock:
            self.check_deadline()
            preserve_final = not final
            maximum = (
                self._limits.max_llm_calls - 1 if preserve_final else self._limits.max_llm_calls
            )
            if self._llm_calls >= maximum:
                raise LLMCallLimitExceededError()
            if repair and not final and self._llm_calls + 1 >= self._limits.max_llm_calls:
                raise LLMCallLimitExceededError()
            self._llm_calls += 1

    async def reserve_policy_decision(self, *, answer_only: bool = False) -> None:
        """Atomically reserve an LLM call and valid decision step."""

        async with self._lock:
            self.check_deadline()
            maximum_steps = self._limits.max_steps if answer_only else self._limits.max_steps - 1
            maximum_calls = (
                self._limits.max_llm_calls if answer_only else self._limits.max_llm_calls - 1
            )
            if self._steps >= maximum_steps:
                raise StepLimitExceededError()
            if self._llm_calls >= maximum_calls:
                raise LLMCallLimitExceededError()
            self._steps += 1
            self._llm_calls += 1

    async def reserve_tool_call(self, count: int = 1) -> None:
        """Reserve tool calls before validation-independent execution starts."""

        amount = self._positive_count(count)
        async with self._lock:
            self.check_deadline()
            if self._tool_calls + amount > self._limits.max_tool_calls:
                raise ToolLimitExceededError()
            self._tool_calls += amount

    async def reserve_backend_call(self, count: int = 1) -> None:
        """Reserve actual backend attempts, including every fallback attempt."""

        amount = self._positive_count(count)
        async with self._lock:
            self.check_deadline()
            if self._backend_calls + amount > self._limits.max_backend_calls:
                raise BackendCallLimitExceededError()
            self._backend_calls += amount

    async def refund_unstarted_backend_call(self, count: int = 1) -> None:
        """Refund reservations only when backend work never started."""

        amount = self._positive_count(count)
        async with self._lock:
            if amount > self._backend_calls:
                raise ValueError("backend refund exceeds reserved calls")
            self._backend_calls -= amount

    async def reserve_derived_assets(self, count: int, *, parent_depth: int) -> None:
        """Reserve complete transform fanout and validate the next lineage depth."""

        amount = self._positive_count(count)
        if isinstance(parent_depth, bool) or not isinstance(parent_depth, int) or parent_depth < 0:
            raise ValueError("parent_depth must be a non-negative integer")
        async with self._lock:
            self.check_deadline()
            if parent_depth + 1 > self._limits.max_derivation_depth:
                raise DerivationDepthLimitExceededError()
            if self._derived_assets + amount > self._limits.max_derived_assets:
                raise AssetLimitExceededError()
            self._derived_assets += amount

    async def refund_reused_assets(self, count: int) -> None:
        """Reconcile reservations for derivatives that reused existing pixels."""

        amount = self._positive_count(count)
        async with self._lock:
            if amount > self._derived_assets:
                raise ValueError("asset refund exceeds reserved derivatives")
            self._derived_assets -= amount

    def validate_context_size(self, characters: int) -> None:
        """Reject context text exceeding the configured Unicode character budget."""

        if isinstance(characters, bool) or not isinstance(characters, int) or characters < 0:
            raise ValueError("context characters must be a non-negative integer")
        if characters > self._limits.max_context_chars:
            raise ContextLimitExceededError()

    def remaining(self, *, current_depth: int = 0) -> RemainingBudget:
        """Return a non-negative public snapshot for a policy invocation."""

        if (
            isinstance(current_depth, bool)
            or not isinstance(current_depth, int)
            or current_depth < 0
        ):
            raise ValueError("current_depth must be a non-negative integer")
        return RemainingBudget(
            steps=max(0, self._limits.max_steps - self._steps),
            llm_calls=max(0, self._limits.max_llm_calls - self._llm_calls),
            tool_calls=max(0, self._limits.max_tool_calls - self._tool_calls),
            backend_calls=max(0, self._limits.max_backend_calls - self._backend_calls),
            derived_assets=max(0, self._limits.max_derived_assets - self._derived_assets),
            derivation_depth=max(0, self._limits.max_derivation_depth - current_depth),
            context_chars=self._limits.max_context_chars,
            remaining_time_s=self.remaining_time_s(),
        )

    def _read_clock(self) -> float:
        value = self._clock()
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("monotonic clock must return a finite number")
        return float(value)

    @staticmethod
    def _positive_count(count: int) -> int:
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("reservation count must be a positive integer")
        return count


__all__ = ["BudgetStopReason", "RunBudget", "RunBudgetUsage"]
