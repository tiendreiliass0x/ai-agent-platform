from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

from app.planner.models import Plan
from app.planner import PlanValidationError


@dataclass
class StrategyDecision:
    name: str
    reasoning: str


class BaseReasoner:
    async def generate_plan(self, *args, **kwargs) -> Plan:  # pragma: no cover - interface
        raise NotImplementedError


class ReasoningEngine:
    def __init__(self, reasoners: Mapping[str, BaseReasoner]):
        self._reasoners: Dict[str, BaseReasoner] = dict(reasoners)

    def register_reasoner(self, name: str, reasoner: BaseReasoner) -> None:
        self._reasoners[name] = reasoner

    async def generate_plan(
        self,
        strategy_name: str,
        task,
        context,
        cache_context,
        conversation_memory,
    ) -> Plan:
        if strategy_name not in self._reasoners:
            raise PlanValidationError(f"No reasoner registered for strategy '{strategy_name}'")

        reasoner = self._reasoners[strategy_name]
        return await reasoner.generate_plan(
            task=task,
            context=context,
            cache_context=cache_context,
            conversation_memory=conversation_memory,
        )
