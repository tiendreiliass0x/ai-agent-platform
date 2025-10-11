from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Optional

from app.planner.models import Plan

from .task_tracker import TaskTracker, StepRecord


@dataclass
class WorkingMemoryFiles:
    """Convenience container that exposes key file locations."""

    workspace: str
    progress_path: str
    context_path: str
    errors_path: str


class WorkingMemory:
    """File-backed working memory inspired by Manus-style append-only context."""

    def __init__(self, task_id: str, base_path: str | None = None) -> None:
        self.task_id = task_id
        root = base_path or "/tmp/agent_workspace"
        workspace = os.path.join(root, task_id)
        os.makedirs(workspace, exist_ok=True)

        self.files = WorkingMemoryFiles(
            workspace=workspace,
            progress_path=os.path.join(workspace, "PROGRESS.md"),
            context_path=os.path.join(workspace, "context.txt"),
            errors_path=os.path.join(workspace, "errors.jsonl"),
        )

    def create_progress_file(self, plan: Plan, tracker: TaskTracker) -> str:
        """Write a human-readable progress file that mirrors current execution state."""
        lines = [
            "# Task Progress",
            "",
            f"Goal: {plan.goal}",
            "",
            "## Steps",
            "",
        ]

        status_symbols = {
            "completed": "[done]",
            "running": "[run]",
            "failed": "[fail]",
            "pending": "[ ]",
        }

        for step in plan.steps:
            status = tracker.get_status(step.id)
            symbol = status_symbols.get(status, "[ ]")
            action = step.action or step.kind
            lines.append(f"{symbol} {step.id}: {action}")

            record = tracker.get_record(step.id)
            if record:
                self._append_step_details(lines, record)

            lines.append("")

        content = "\n".join(lines).rstrip() + "\n"
        with open(self.files.progress_path, "w", encoding="utf-8") as handle:
            handle.write(content)
        return content

    def append_context(self, content: str) -> None:
        """Append contextual information using an append-only pattern."""
        with open(self.files.context_path, "a", encoding="utf-8") as handle:
            handle.write(f"{content}\n")

    def save_error(self, step_id: str, error: str, lesson: Optional[str] = None) -> None:
        """Persist error details for later learning passes."""
        payload = {
            "step_id": step_id,
            "error": error,
            "lesson": lesson or "unspecified",
            "timestamp": time.time(),
        }
        with open(self.files.errors_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    @staticmethod
    def _append_step_details(lines: list[str], record: StepRecord) -> None:
        if record.error:
            lines.append(f"    -> Error: {record.error}")
        if record.result:
            encoded = json.dumps(record.result, sort_keys=True)
            lines.append(f"    -> Result: {encoded}")
        if record.args:
            encoded_args = json.dumps(record.args, sort_keys=True)
            lines.append(f"    -> Args: {encoded_args}")
