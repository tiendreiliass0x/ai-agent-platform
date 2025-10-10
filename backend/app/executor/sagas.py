from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List


@dataclass
class SagaStep:
    rollback_operation: str
    args: Dict[str, Any]


class SagaManager:
    def __init__(self) -> None:
        self._steps: List[SagaStep] = []

    def add_step(self, step: SagaStep) -> None:
        self._steps.append(step)

    def pop_last(self) -> SagaStep | None:
        return self._steps.pop() if self._steps else None

    async def rollback(self, executor: Callable[[SagaStep], Any]) -> None:
        while self._steps:
            step = self._steps.pop()
            await executor(step)
