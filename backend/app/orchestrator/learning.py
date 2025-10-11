from __future__ import annotations

import asyncio
from collections import deque
from typing import Deque, Dict, Optional, Protocol


class ExecutionRecorder(Protocol):
    async def __call__(self, payload: Dict[str, object]) -> None: ...


class LearningSystem:
    """Lightweight learning system that records plan executions."""

    def __init__(
        self,
        max_records: int = 100,
        recorder: Optional[ExecutionRecorder] = None,
    ) -> None:
        self._history: Deque[Dict[str, object]] = deque(maxlen=max_records)
        self._recorder = recorder

    async def record_execution(self, plan, success: bool) -> None:
        entry = {
            "plan_goal": getattr(plan, "goal", None),
            "strategy": getattr(plan, "strategy", None),
            "success": success,
        }
        self._history.append(entry)
        if self._recorder is not None:
            await self._invoke_recorder(entry)

    async def _invoke_recorder(self, entry: Dict[str, object]) -> None:
        assert self._recorder is not None
        maybe_coro = self._recorder(entry)
        if asyncio.iscoroutine(maybe_coro):
            await maybe_coro

    @property
    def history(self) -> Deque[Dict[str, object]]:
        return self._history
