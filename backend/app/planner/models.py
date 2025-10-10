from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class Step:
    id: str
    kind: str
    action: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    reasoning: str = ""
    assert_condition: Optional[str] = None  # for check steps


@dataclass
class Plan:
    goal: str
    strategy: str
    steps: List[Step]
    parallel_groups: List[List[str]] = field(default_factory=list)

    def get_step(self, step_id: str) -> Step:
        for step in self.steps:
            if step.id == step_id:
                return step
        raise KeyError(step_id)
