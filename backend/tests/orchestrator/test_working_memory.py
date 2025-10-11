import json

from app.orchestrator.task_tracker import TaskTracker
from app.orchestrator.working_memory import WorkingMemory
from app.planner.models import Plan, Step


def _build_plan() -> Plan:
    return Plan(
        goal="Verify working memory",
        strategy="plan_execute",
        steps=[
            Step(id="step_one", kind="tool", action="crm.lookup", args={"q": "acme"}),
            Step(id="step_two", kind="tool", action="crm.create", args={"id": "$step_one.result['id']"}),
        ],
    )


def test_working_memory_tracks_progress(tmp_path):
    plan = _build_plan()
    tracker = TaskTracker(plan)
    workspace = WorkingMemory("task-123", base_path=str(tmp_path))

    # Initial state: all pending
    workspace.create_progress_file(plan, tracker)
    progress_contents = (tmp_path / "task-123" / "PROGRESS.md").read_text(encoding="utf-8")
    assert "[ ] step_one" in progress_contents
    assert "[ ] step_two" in progress_contents

    # Mark running and completed to ensure status transitions appear
    tracker.mark_running("step_one")
    workspace.create_progress_file(plan, tracker)
    progress_contents = (tmp_path / "task-123" / "PROGRESS.md").read_text(encoding="utf-8")
    assert "[run] step_one" in progress_contents

    tracker.mark_completed("step_one", {"q": "acme"}, {"id": "ACC-1"})
    workspace.create_progress_file(plan, tracker)
    progress_contents = (tmp_path / "task-123" / "PROGRESS.md").read_text(encoding="utf-8")
    assert "[done] step_one" in progress_contents
    assert '"id": "ACC-1"' in progress_contents
    assert '"q": "acme"' in progress_contents

    tracker.mark_failed("step_two", {"id": "ACC-1"}, "permission denied")
    workspace.create_progress_file(plan, tracker)
    progress_contents = (tmp_path / "task-123" / "PROGRESS.md").read_text(encoding="utf-8")
    assert "[fail] step_two" in progress_contents
    assert "permission denied" in progress_contents


def test_working_memory_append_and_error_logs(tmp_path):
    plan = _build_plan()
    tracker = TaskTracker(plan)
    workspace = WorkingMemory("task-456", base_path=str(tmp_path))

    workspace.append_context("Initial context snippet")
    workspace.append_context("Follow-up note")
    context_path = tmp_path / "task-456" / "context.txt"
    context_contents = context_path.read_text(encoding="utf-8")
    assert "Initial context snippet" in context_contents
    assert "Follow-up note" in context_contents

    workspace.save_error("step_two", "rate limited", lesson="retry later")
    error_path = tmp_path / "task-456" / "errors.jsonl"
    error_line = error_path.read_text(encoding="utf-8").strip()
    payload = json.loads(error_line)
    assert payload["step_id"] == "step_two"
    assert payload["error"] == "rate limited"
    assert payload["lesson"] == "retry later"
    assert isinstance(payload["timestamp"], float)
