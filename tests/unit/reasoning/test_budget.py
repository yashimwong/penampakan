import asyncio

import pytest

from penampakan.config import RunLimits
from penampakan.errors import (
    AssetLimitExceededError,
    BackendCallLimitExceededError,
    LLMCallLimitExceededError,
    OperationTimeoutError,
    StepLimitExceededError,
    ToolLimitExceededError,
)
from penampakan.reasoning.budget import RunBudget


class ManualClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


async def test_concurrent_backend_reservations_are_atomic() -> None:
    budget = RunBudget(RunLimits())

    async def reserve() -> bool:
        try:
            await budget.reserve_backend_call()
        except BackendCallLimitExceededError:
            return False
        return True

    results = await asyncio.gather(*(reserve() for _ in range(100)))

    assert sum(results) == budget.limits.max_backend_calls
    assert budget.usage.backend_calls == budget.limits.max_backend_calls
    assert budget.remaining().backend_calls == 0


async def test_concurrent_policy_reservations_never_partially_increment() -> None:
    limits = RunLimits(
        max_steps=4,
        max_llm_calls=5,
        max_tool_calls=4,
        max_parallel_tools=1,
    )
    budget = RunBudget(limits)

    async def reserve() -> str:
        try:
            await budget.reserve_policy_decision()
        except StepLimitExceededError:
            return "step_limit"
        except LLMCallLimitExceededError:
            return "llm_limit"
        return "reserved"

    results = await asyncio.gather(*(reserve() for _ in range(20)))

    assert results.count("reserved") == 3
    assert results.count("step_limit") == 17
    assert budget.usage.steps == 3
    assert budget.usage.llm_calls == 3


async def test_multi_count_reservation_failure_is_transactional() -> None:
    limits = RunLimits(max_tool_calls=5, max_parallel_tools=1)
    budget = RunBudget(limits)

    await budget.reserve_tool_call(4)

    with pytest.raises(ToolLimitExceededError):
        await budget.reserve_tool_call(2)

    assert budget.usage.tool_calls == 4


async def test_cache_fallback_repair_initial_and_final_counts_are_exact() -> None:
    budget = RunBudget(RunLimits())

    await budget.reserve_tool_call()
    await budget.reserve_tool_call()
    await budget.reserve_backend_call()
    await budget.reserve_tool_call()
    await budget.reserve_backend_call(2)
    await budget.reserve_policy_decision()
    await budget.reserve_tool_call()
    await budget.reserve_backend_call()
    await budget.reserve_llm_call(repair=True)
    await budget.reserve_policy_decision(answer_only=True)

    assert budget.usage.steps == 2
    assert budget.usage.llm_calls == 3
    assert budget.usage.tool_calls == 4
    assert budget.usage.backend_calls == 4
    assert budget.usage.derived_assets == 0


async def test_cache_hit_consumes_tool_but_not_backend_budget() -> None:
    budget = RunBudget(RunLimits())

    await budget.reserve_tool_call()

    assert budget.usage.tool_calls == 1
    assert budget.usage.backend_calls == 0


async def test_each_fallback_attempt_consumes_backend_budget() -> None:
    budget = RunBudget(RunLimits())

    await budget.reserve_backend_call()
    await budget.reserve_backend_call()
    await budget.reserve_backend_call()

    assert budget.usage.backend_calls == 3


async def test_repair_preserves_final_call_and_does_not_consume_step() -> None:
    limits = RunLimits(
        max_steps=2,
        max_llm_calls=3,
        max_parallel_tools=1,
    )
    budget = RunBudget(limits)

    await budget.reserve_policy_decision()
    await budget.reserve_llm_call(repair=True)

    assert budget.usage.steps == 1
    assert budget.usage.llm_calls == 2

    await budget.reserve_policy_decision(answer_only=True)

    assert budget.usage.steps == 2
    assert budget.usage.llm_calls == 3


async def test_repair_is_rejected_when_only_reserved_final_call_remains() -> None:
    limits = RunLimits(
        max_steps=3,
        max_llm_calls=4,
        max_parallel_tools=1,
    )
    budget = RunBudget(limits)

    await budget.reserve_llm_call()
    await budget.reserve_llm_call()
    await budget.reserve_llm_call()

    with pytest.raises(LLMCallLimitExceededError):
        await budget.reserve_llm_call(repair=True)

    assert budget.usage.llm_calls == 3


async def test_soft_step_stop_preserves_answer_only_reservation() -> None:
    limits = RunLimits(
        max_steps=3,
        max_llm_calls=4,
        max_parallel_tools=1,
    )
    budget = RunBudget(limits)

    await budget.reserve_policy_decision()
    await budget.reserve_policy_decision()

    assert not budget.can_start_interactive_step()
    assert budget.soft_stop_reason() == "step_limit"

    with pytest.raises(StepLimitExceededError):
        await budget.reserve_policy_decision()

    await budget.reserve_policy_decision(answer_only=True)

    assert budget.usage.steps == 3
    assert budget.usage.llm_calls == 3


async def test_soft_llm_stop_preserves_final_call() -> None:
    limits = RunLimits(
        max_steps=3,
        max_llm_calls=4,
        max_parallel_tools=1,
    )
    budget = RunBudget(limits)

    await budget.reserve_llm_call()
    await budget.reserve_llm_call()
    await budget.reserve_llm_call()

    assert not budget.can_start_interactive_step()
    assert budget.soft_stop_reason() == "llm_limit"

    await budget.reserve_llm_call(final=True)

    assert budget.usage.llm_calls == 4


async def test_hard_deadline_preempts_all_reservations() -> None:
    clock = ManualClock()
    budget = RunBudget(RunLimits(), timeout_s=2.0, monotonic_clock=clock)

    assert budget.remaining_time_s() == 2.0
    assert budget.component_timeout(5.0) == 2.0

    clock.advance(2.0)

    assert budget.remaining_time_s() == 0.0
    assert not budget.can_start_interactive_step()
    assert budget.soft_stop_reason() is None

    with pytest.raises(OperationTimeoutError):
        budget.check_deadline()
    with pytest.raises(OperationTimeoutError):
        budget.component_timeout(1.0)
    with pytest.raises(OperationTimeoutError):
        await budget.reserve_tool_call()

    assert budget.usage.tool_calls == 0


async def test_backend_refund_applies_only_to_unstarted_work() -> None:
    budget = RunBudget(RunLimits())

    await budget.reserve_backend_call(3)
    await budget.refund_unstarted_backend_call()

    assert budget.usage.backend_calls == 2

    with pytest.raises(ValueError):
        await budget.refund_unstarted_backend_call(3)

    assert budget.usage.backend_calls == 2


async def test_asset_fanout_and_reuse_accounting_are_transactional() -> None:
    limits = RunLimits(
        max_derived_assets=3,
        max_derivation_depth=2,
        max_parallel_tools=1,
    )
    budget = RunBudget(limits)

    await budget.reserve_derived_assets(3, parent_depth=0)

    with pytest.raises(AssetLimitExceededError):
        await budget.reserve_derived_assets(1, parent_depth=0)

    assert budget.usage.derived_assets == 3

    await budget.refund_reused_assets(2)

    assert budget.usage.derived_assets == 1
    assert budget.remaining(current_depth=1).derived_assets == 2
    assert budget.remaining(current_depth=1).derivation_depth == 1
