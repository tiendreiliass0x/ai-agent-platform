from __future__ import annotations

import json
from typing import Iterable

from app.planner.models import Plan, Step
from app.planner import PlanValidationError
from app.tooling.models import ToolManifest
from app.tooling.registry import ToolRegistry

from .engine import BaseReasoner


class PlanExecuteReasoner(BaseReasoner):
    def __init__(self, planner):
        self._planner = planner

    async def generate_plan(self, **kwargs) -> Plan:
        task = kwargs["task"]
        strategy = "plan_execute"
        cache_context = kwargs.get("cache_context")
        conversation_memory = kwargs.get("conversation_memory")

        if hasattr(self._planner, "build_plan"):
            return await self._planner.build_plan(task, strategy, cache_context)
        return await self._planner.plan(task, strategy, cache_context)


class ReactReasoner(BaseReasoner):
    def __init__(self, llm_service, tool_registry: ToolRegistry, max_iterations: int = 5):
        self.llm = llm_service
        self.tool_registry = tool_registry
        self.max_iterations = max_iterations

    async def generate_plan(self, **kwargs) -> Plan:
        task = kwargs["task"]
        cache_context = kwargs.get("cache_context")

        tools = await self.tool_registry.list_tools()
        prompt = self._build_prompt(task.description, cache_context, tools)
        response = await self.llm.generate_response(prompt=prompt, temperature=0.2, max_tokens=2000)
        try:
            plan_data = json.loads(response)
        except json.JSONDecodeError as exc:
            raise PlanValidationError("ReAct strategy returned invalid JSON") from exc

        return self._plan_from_data(plan_data)

    def _build_prompt(self, description: str, context, tools: Iterable[ToolManifest]) -> str:
        lines = [f"Task: {description}", "", "Context:", str(context), "", "Available tools:"]
        for tool in tools:
            lines.append(f"- {tool.name}: {tool.description}")
            for op in tool.operations.values():
                lines.append(f"  * {op.op_id}: {op.description}")
        lines.append(
            "\nReturn JSON with fields goal, steps (with id, kind, action, args, depends_on), and optional parallel_groups."
        )
        return "\n".join(lines)

    def _plan_from_data(self, data: dict) -> Plan:
        steps_data = data.get("steps", [])
        if not steps_data:
            raise PlanValidationError("ReAct strategy produced empty plan")

        steps = []
        for raw in steps_data:
            if "id" not in raw or "kind" not in raw:
                raise PlanValidationError("Step missing required fields")
            steps.append(
                Step(
                    id=raw["id"],
                    kind=raw.get("kind", "tool"),
                    action=raw.get("action"),
                    args=raw.get("args", {}),
                    depends_on=raw.get("depends_on", []),
                    reasoning=raw.get("reasoning", ""),
                )
            )

        return Plan(
            goal=data.get("goal", ""),
            strategy="react",
            steps=steps,
            parallel_groups=data.get("parallel_groups", []),
        )
