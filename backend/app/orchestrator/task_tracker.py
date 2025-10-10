from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from app.planner.models import Plan, Step


@dataclass
class StepRecord:
    args: Dict
    result: Dict


class TaskTracker:
    def __init__(self, plan: Plan) -> None:
        self.plan = plan
        self.status: Dict[str, str] = {step.id: "pending" for step in plan.steps}
        self.records: Dict[str, StepRecord] = {}

    def runnable_steps(self) -> List[Step]:
        runnable = []
        for step in self.plan.steps:
            if self.status[step.id] != "pending":
                continue
            if all(self.status.get(dep) == "completed" for dep in step.depends_on):
                runnable.append(step)
        return runnable

    def mark_running(self, step_id: str) -> None:
        self.status[step_id] = "running"

    def mark_completed(self, step_id: str, args: Dict, result: Dict) -> None:
        self.status[step_id] = "completed"
        self.records[step_id] = StepRecord(args=args, result=result)

    def mark_failed(self, step_id: str, args: Dict, error: str) -> None:
        self.status[step_id] = "failed"
        self.records[step_id] = StepRecord(args=args, result={"error": error})

    def all_completed(self) -> bool:
        return all(status == "completed" for status in self.status.values())

    def has_pending(self) -> bool:
        return any(status == "pending" for status in self.status.values())
