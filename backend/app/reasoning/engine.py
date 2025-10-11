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
        *,
        feedback=None,
        previous_plan=None,
        error=None,
        shared_state=None,
    ) -> Plan:
        if strategy_name not in self._reasoners:
            raise PlanValidationError(f"No reasoner registered for strategy '{strategy_name}'")

        reasoner = self._reasoners[strategy_name]
        augmented_context = cache_context
        if feedback:
            base_context = cache_context or ""
            augmented_context = f"{base_context}\n\n[Retry Feedback]\n{feedback}"
        return await reasoner.generate_plan(
            task=task,
            context=context,
            cache_context=augmented_context,
            conversation_memory=conversation_memory,
            feedback=feedback,
            previous_plan=previous_plan,
            error=error,
            shared_state=shared_state,
        )
