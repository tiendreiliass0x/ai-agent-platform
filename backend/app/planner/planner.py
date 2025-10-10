from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Optional

from app.tooling.models import ToolManifest
from app.tooling.registry import ToolRegistry
from .models import Plan, Step
from .exceptions import PlanValidationError


class Planner:
    """LLM-driven structured planner with validation."""

    def __init__(self, llm_service, tool_registry: ToolRegistry) -> None:
        self._llm = llm_service
        self._registry = tool_registry

    async def plan(self, task: Any, strategy: str, context: str) -> Plan:
        description = getattr(task, "description", None)
        if not description:
            raise ValueError("Task must expose a description")

        tools = await self._registry.search_tools(description, top_k=10)
        prompt = self._build_prompt(description, strategy, context, tools)
        response = await self._llm.generate_response(prompt=prompt, temperature=0.2, max_tokens=2000)

        try:
            plan_data = json.loads(response)
        except json.JSONDecodeError as exc:
            raise PlanValidationError("Planner returned invalid JSON") from exc

        await self._validate_plan(plan_data, tools)
        steps = [self._build_step(step_data) for step_data in plan_data.get("steps", [])]
        return Plan(
            goal=plan_data.get("goal", description),
            strategy=strategy,
            steps=steps,
            parallel_groups=plan_data.get("parallel_groups", []),
        )

    def _build_prompt(
        self,
        task_description: str,
        strategy: str,
        context: str,
        tools: Iterable[ToolManifest],
    ) -> str:
        tools_description = self._format_tools(tools)
        return (
            f"Task: {task_description}\n\n"
            f"Strategy: {strategy}\n\n"
            f"Available tools:\n{tools_description}\n"
            f"Context:\n{context}\n\n"
            "Generate a complete execution plan following the JSON schema provided."
        )

    def _format_tools(self, tools: Iterable[ToolManifest]) -> str:
        lines: List[str] = []
        for tool in tools:
            lines.append(f"## {tool.display_name}\n{tool.description}\n")
            for op in tool.operations.values():
                lines.append(f"### {op.op_id}\n")
                lines.append(f"Description: {op.description}\n")
                lines.append(f"Side Effect: {op.side_effect.value}\n")
                lines.append(f"Args Schema: {json.dumps(op.args_schema, indent=2)}\n")
        return "\n".join(lines)

    async def _validate_plan(self, plan_data: Dict[str, Any], tools: Iterable[ToolManifest]) -> None:
        if "steps" not in plan_data:
            raise PlanValidationError("Plan missing steps field")

        ids = set()
        tool_operations = {
            op.op_id
            for manifest in tools
            for op in manifest.operations.values()
        }
        dependency_graph: Dict[str, List[str]] = {}

        for step in plan_data["steps"]:
            step_id = step.get("id")
            if not step_id:
                raise PlanValidationError("Step missing id")
            if step_id in ids:
                raise PlanValidationError(f"Duplicate step id: {step_id}")
            ids.add(step_id)

            kind = step.get("kind")
            if kind not in {"tool", "check"}:
                raise PlanValidationError(f"Invalid step kind: {kind}")

            if kind == "tool":
                action = step.get("action")
                if action not in tool_operations:
                    raise PlanValidationError(f"Unknown tool action: {action}")

            depends = step.get("depends_on", [])
            if not isinstance(depends, list):
                raise PlanValidationError(f"depends_on must be list for step {step_id}")
            dependency_graph[step_id] = depends

        self._ensure_dependencies_exist(dependency_graph, ids)
        if self._has_cycle(dependency_graph):
            raise PlanValidationError("Plan contains circular dependencies")

    def _ensure_dependencies_exist(self, graph: Dict[str, List[str]], ids: set[str]) -> None:
        for step_id, deps in graph.items():
            for dep in deps:
                if dep not in ids:
                    raise PlanValidationError(f"Step '{step_id}' depends on unknown step '{dep}'")

    def _build_step(self, step_data: Dict[str, Any]) -> Step:
        kind = step_data.get("kind", "tool")
        return Step(
            id=step_data["id"],
            kind=kind,
            action=step_data.get("action"),
            args=step_data.get("args", {}),
            depends_on=step_data.get("depends_on", []),
            reasoning=step_data.get("reasoning", ""),
            assert_condition=step_data.get("assert"),
        )

    def _has_cycle(self, graph: Dict[str, List[str]]) -> bool:
        visiting: set[str] = set()
        visited: set[str] = set()

        def dfs(node: str) -> bool:
            if node in visiting:
                return True
            if node in visited:
                return False
            visiting.add(node)
            for neighbor in graph.get(node, []):
                if dfs(neighbor):
                    return True
            visiting.remove(node)
            visited.add(node)
            return False

        return any(dfs(node) for node in graph)
