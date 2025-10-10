from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Dict, Optional

from app.context_engine.conversation_memory import ConversationMemory
from app.planner.models import Plan, Step
from app.planner import PlanValidationError
from app.policy import PolicyEngine
from app.executor import Executor, ToolExecutionError
from app.tooling import ToolRegistry
from app.verifier import Verifier
from app.reasoning import ReasoningEngine

from .models import AgentContext, AgentTask, OrchestrationResult
from .strategy import StrategySelector
from .task_tracker import TaskTracker
from app.verifier.safe_eval import safe_eval


class AgentOrchestrator:
    def __init__(
        self,
        reasoning_engine: ReasoningEngine,
        executor: Executor,
        tool_registry: ToolRegistry,
        strategy_selector: StrategySelector,
        context_builder,
        memory_provider,
        learning_system,
        policy_engine: PolicyEngine,
    ) -> None:
        self.reasoning_engine = reasoning_engine
        self.executor = executor
        self.tool_registry = tool_registry
        self.strategy_selector = strategy_selector
        self.context_builder = context_builder
        self.memory_provider = memory_provider
        self.learning_system = learning_system
        self.policy_engine = policy_engine

    async def run_task(self, context: AgentContext, task: AgentTask) -> OrchestrationResult:
        conversation_memory = await self._load_conversation_memory(context)
        strategy_decision = await self.strategy_selector.select(task, context)

        cache_context = await self.context_builder.build(task, context, conversation_memory)
        plan = await self.reasoning_engine.generate_plan(
            strategy_name=strategy_decision.name,
            task=task,
            context=context,
            cache_context=cache_context,
            conversation_memory=conversation_memory,
        )
        tracker = TaskTracker(plan)

        shared_state: Dict[str, Any] = {}
        try:
            while tracker.has_pending():
                runnable = tracker.runnable_steps()
                if not runnable:
                    raise PlanValidationError("Plan deadlock detected")
                step = runnable[0]
                resolved_step = self._resolve_step(step, tracker)
                tracker.mark_running(step.id)
                try:
                    result = await self.executor.execute(
                        resolved_step,
                        user=context.user,
                        context=shared_state,
                    )
                    tracker.mark_completed(step.id, resolved_step.args, result)
                    shared_state[step.id] = {"result": result, "args": resolved_step.args}
                    if conversation_memory:
                        conversation_memory.add_turn(
                            role="assistant",
                            content=f"Executed {step.id}",
                            metadata={"action": step.action},
                        )
                except ToolExecutionError as exc:
                    tracker.mark_failed(step.id, resolved_step.args, str(exc))
                    await self.learning_system.record_execution(plan, success=False)
                    return OrchestrationResult(
                        status="failed",
                        plan=plan,
                        step_results={sid: record.__dict__ for sid, record in tracker.records.items()},
                        error=str(exc),
                    )
        finally:
            if conversation_memory and hasattr(self.memory_provider, "save"):
                await self.memory_provider.save(context, conversation_memory)

        await self.learning_system.record_execution(plan, success=True)
        return OrchestrationResult(
            status="completed",
            plan=plan,
            step_results={sid: record.__dict__ for sid, record in tracker.records.items()},
        )

    async def _load_conversation_memory(self, context: AgentContext) -> Optional[ConversationMemory]:
        if not self.memory_provider or not hasattr(self.memory_provider, "load"):
            return None
        memory = await self.memory_provider.load(context)
        if memory is None:
            memory = ConversationMemory()
        return memory

    def _resolve_step(self, step: Step, tracker: TaskTracker) -> Step:
        resolved_args = self._resolve_args(step.args, tracker)
        return replace(step, args=resolved_args)

    def _resolve_args(self, args: Dict[str, Any], tracker: TaskTracker) -> Dict[str, Any]:
        resolved: Dict[str, Any] = {}
        context = {
            step_id: SimpleNamespace(args=record.args, result=record.result)
            for step_id, record in tracker.records.items()
        }

        for key, value in args.items():
            resolved[key] = self._resolve_value(value, context)
        return resolved

    def _resolve_value(self, value: Any, context: Dict[str, Any]) -> Any:
        if isinstance(value, str) and value.startswith("$"):
            expression = value[1:]
            return safe_eval(expression, context)
        if isinstance(value, dict):
            return {k: self._resolve_value(v, context) for k, v in value.items()}
        if isinstance(value, list):
            return [self._resolve_value(item, context) for item in value]
        return value
