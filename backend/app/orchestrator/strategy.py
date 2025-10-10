from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .models import AgentTask, AgentContext

StrategyName = Literal["plan_execute", "react"]


@dataclass
class StrategyDecision:
    name: StrategyName
    reasoning: str


class StrategySelector:
    async def select(self, task: AgentTask, context: AgentContext) -> StrategyDecision:
        description = task.description.lower()
        if any(word in description for word in ["investigate", "diagnose", "why"]):
            return StrategyDecision(name="react", reasoning="Exploratory task detected")
        return StrategyDecision(name="plan_execute", reasoning="Deterministic workflow")
