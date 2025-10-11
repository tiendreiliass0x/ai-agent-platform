from __future__ import annotations

import inspect
import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from app.context_engine.conversation_memory import ConversationMemory
from app.planner.models import Plan, Step
from app.planner import PlanValidationError
from app.policy import PolicyEngine
from app.executor import Executor, ToolExecutionError
from app.tooling import ToolRegistry
from app.verifier import Verifier
from app.reasoning import ReasoningEngine, PlanExecuteReasoner
from app.reasoning.factory import build_default_reasoning_engine

from .models import AgentContext, AgentTask, OrchestrationResult
from .strategy import StrategySelector
from .task_tracker import TaskTracker
from .working_memory import WorkingMemory
from app.verifier.safe_eval import safe_eval


class AgentOrchestrator:
    def __init__(
        self,
        *,
        planner: Optional[Any] = None,
        llm_service: Optional[Any] = None,
        reasoning_engine: Optional[ReasoningEngine] = None,
        executor: Executor,
        tool_registry: ToolRegistry,
        strategy_selector: StrategySelector,
        context_builder,
        memory_provider,
        learning_system,
        policy_engine: PolicyEngine,
        working_memory_factory: Optional[
            Callable[[AgentContext, AgentTask, Plan], Awaitable[WorkingMemory] | WorkingMemory]
        ] = None,
    ) -> None:
        if reasoning_engine is None:
            if planner is None:
                raise ValueError("planner is required when reasoning_engine is not provided")
            if llm_service is not None:
                reasoning_engine = build_default_reasoning_engine(planner, tool_registry, llm_service)
            else:
                reasoning_engine = ReasoningEngine({"plan_execute": PlanExecuteReasoner(planner)})
        self.reasoning_engine = reasoning_engine
        self.executor = executor
        self.tool_registry = tool_registry
        self.strategy_selector = strategy_selector
        self.context_builder = context_builder
        self.memory_provider = memory_provider
        self.learning_system = learning_system
        self.policy_engine = policy_engine
        self.working_memory_factory = working_memory_factory

    async def run_task(self, context: AgentContext, task: AgentTask) -> OrchestrationResult:
        conversation_memory = await self._load_conversation_memory(context)
        strategy_decision = await self.strategy_selector.select(task, context)
        strategy_name = strategy_decision.name

        max_attempts = self._get_max_attempts(strategy_name)
        attempt = 0
        feedback: Optional[str] = None
        previous_plan: Optional[Plan] = None
        last_failure: Optional[Tuple[str, Dict[str, Any], str]] = None
        working_memory: Optional[WorkingMemory] = None

        while attempt < max_attempts:
            attempt += 1
            cache_context = await self.context_builder.build(task, context, conversation_memory)
            feedback_to_send = feedback
            failure_context = last_failure
            feedback = None
            last_failure = None

            plan = await self.reasoning_engine.generate_plan(
                strategy_name=strategy_name,
                task=task,
                context=context,
                cache_context=cache_context,
                conversation_memory=conversation_memory,
                feedback=feedback_to_send,
                previous_plan=previous_plan,
                error=failure_context[2] if failure_context else None,
            )
            tracker = TaskTracker(plan)

            if working_memory is None:
                working_memory = await self._create_working_memory(context, task, plan)
            elif attempt > 1 and working_memory is not None:
                working_memory.append_context(f"Retry attempt {attempt}: regenerating plan.")

            self._update_progress(working_memory, plan, tracker)
            shared_state: Dict[str, Any] = {"_agent_metadata": context.metadata or {}}

            try:
                while tracker.has_pending():
                    runnable = tracker.runnable_steps()
                    if not runnable:
                        raise PlanValidationError("Plan deadlock detected")
                    step = runnable[0]
                    resolved_step = self._resolve_step(step, tracker)
                    tracker.mark_running(step.id)
                    self._update_progress(working_memory, plan, tracker)
                    try:
                        result = await self.executor.execute(
                            resolved_step,
                            user=context.user,
                            context=shared_state,
                        )
                        tracker.mark_completed(step.id, resolved_step.args, result)
                        self._update_progress(working_memory, plan, tracker)
                        shared_state[step.id] = {"result": result, "args": resolved_step.args}
                        if conversation_memory:
                            await conversation_memory.add_turn_async(
                                role="assistant",
                                content=f"Executed {step.id}",
                                metadata={"action": step.action},
                            )
                    except ToolExecutionError as exc:
                        last_failure = (step.id, resolved_step.args, str(exc))
                        tracker.mark_failed(step.id, resolved_step.args, str(exc))
                        self._update_progress(working_memory, plan, tracker)
                        if working_memory is not None:
                            working_memory.save_error(
                                step.id,
                                str(exc),
                                lesson="tool_execution_failed",
                            )
                        if conversation_memory:
                            await conversation_memory.add_turn_async(
                                role="assistant",
                                content=f"Failed step {step.id}: {exc}",
                                metadata={"action": step.action, "status": "error"},
                            )
                        raise
            except ToolExecutionError as exc:
                await self.learning_system.record_execution(plan, success=False)
                error_message = last_failure[2] if last_failure else str(exc)
                if conversation_memory and hasattr(self.memory_provider, "save"):
                    await self.memory_provider.save(context, conversation_memory)
                if not self._should_retry(strategy_name, attempt, max_attempts):
                    step_results = self._build_step_results(plan, tracker)
                    return OrchestrationResult(
                        status="failed",
                        plan=plan,
                        step_results=step_results,
                        error=error_message,
                    )

                feedback = self._build_retry_feedback(plan, last_failure, attempt, strategy_name)
                if working_memory is not None and feedback:
                    working_memory.append_context(feedback)
                previous_plan = plan
                continue
            else:
                await self.learning_system.record_execution(plan, success=True)
                self._update_progress(working_memory, plan, tracker)
                if conversation_memory and hasattr(self.memory_provider, "save"):
                    await self.memory_provider.save(context, conversation_memory)
                step_results = self._build_step_results(plan, tracker)
                return OrchestrationResult(
                    status="completed",
                    plan=plan,
                    step_results=step_results,
                )

        # If loop exits without returning, treat as failure using the last captured state.
        failure_message = (
            last_failure[2]
            if last_failure
            else "Plan execution failed after maximum retries"
        )
        step_results = self._build_step_results(plan, tracker)
        return OrchestrationResult(
            status="failed",
            plan=plan,
            step_results=step_results,
            error=failure_message,
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

    async def _create_working_memory(
        self,
        context: AgentContext,
        task: AgentTask,
        plan: Plan,
    ) -> Optional[WorkingMemory]:
        if self.working_memory_factory is None:
            return None
        candidate = self.working_memory_factory(context, task, plan)
        if inspect.isawaitable(candidate):
            return await candidate
        return candidate

    def _update_progress(
        self,
        working_memory: Optional[WorkingMemory],
        plan: Plan,
        tracker: TaskTracker,
    ) -> None:
        if working_memory is None:
            return
        working_memory.create_progress_file(plan, tracker)

    def _get_max_attempts(self, strategy_name: str) -> int:
        if strategy_name == "reflexion":
            return 3
        if strategy_name == "react":
            return 2
        return 1

    def _should_retry(self, strategy_name: str, attempt: int, max_attempts: int) -> bool:
        return attempt < max_attempts and strategy_name in {"react", "reflexion"}

    def _build_retry_feedback(
        self,
        plan: Plan,
        failure_info: Optional[Tuple[str, Dict[str, Any], str]],
        attempt: int,
        strategy_name: str,
    ) -> str:
        if not failure_info:
            return ""

        step_id, args, error_message = failure_info
        try:
            args_text = json.dumps(args, sort_keys=True, default=str)
        except TypeError:
            args_text = str(args)

        header = "Reflexion feedback" if strategy_name == "reflexion" else "ReAct feedback"
        return (
            f"{header} for attempt {attempt} on plan '{plan.goal}':\n"
            f"- Failing step: {step_id}\n"
            f"- Arguments: {args_text}\n"
            f"- Error: {error_message}\n"
            "Regenerate the plan incorporating this feedback to avoid repeating the failure."
        )

    def _build_step_results(self, plan: Plan, tracker: TaskTracker) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        for step in plan.steps:
            record = tracker.get_record(step.id)
            status = tracker.get_status(step.id)
            entry: Dict[str, Any] = {
                "status": status,
                "args": record.args if record else {},
                "result": record.result if record else {},
            }
            if record and record.error:
                entry["error"] = record.error
            results[step.id] = entry
        return results
