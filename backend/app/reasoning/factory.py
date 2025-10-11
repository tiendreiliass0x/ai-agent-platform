from __future__ import annotations

from typing import Dict

from app.planner import Planner
from app.reasoning.engine import ReasoningEngine, BaseReasoner
from app.reasoning.strategies import PlanExecuteReasoner, ReactReasoner, ReflexionReasoner
from app.tooling import ToolRegistry


def build_default_reasoning_engine(
    planner: Planner,
    tool_registry: ToolRegistry,
    llm_service,
) -> ReasoningEngine:
    plan_reasoner = PlanExecuteReasoner(planner)
    react_reasoner = ReactReasoner(llm_service=llm_service, tool_registry=tool_registry)
    reflexion_reasoner = ReflexionReasoner(plan_reasoner)

    reasoners: Dict[str, BaseReasoner] = {
        "plan_execute": plan_reasoner,
        "react": react_reasoner,
        "reflexion": reflexion_reasoner,
    }
    return ReasoningEngine(reasoners)
