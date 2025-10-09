"""Tests for task status and cancellation endpoints."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.v1 import tasks as tasks_module
from app.core.auth import SimpleUser, get_current_user
from main import app


class StubAsyncResult:
    """Deterministic Celery AsyncResult stub for testing."""

    last_instance: "StubAsyncResult | None" = None

    def __init__(self, task_id: str, app=None):  # noqa: D401 - mimic signature
        self.task_id = task_id
        self.app = app
        self.state = "PROGRESS"
        self.info = {
            "status": "Processing...",
            "current": 25,
            "total": 100,
            "document_id": 1,
        }
        self.result = {"document_id": 1}
        self.revoked = False
        StubAsyncResult.last_instance = self

    def revoke(self, terminate: bool = False) -> None:
        self.revoked = True


@pytest.fixture(autouse=True)
def reset_dependency_overrides():
    """Ensure dependency overrides are cleared between tests."""
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


def _override_user(user: SimpleUser) -> None:
    app.dependency_overrides[get_current_user] = lambda: user


def test_get_task_status_authorized_returns_progress(monkeypatch):
    user = SimpleUser(
        id=42,
        email="user@example.com",
        name="Test User",
        organization_id=7,
    )

    async def fake_get_document_by_task_id(task_id: str):
        return SimpleNamespace(id=1, agent_id=55)

    async def fake_get_agent_by_id(agent_id: int):
        return SimpleNamespace(id=agent_id, user_id=user.id, organization_id=user.organization_id)

    monkeypatch.setattr(tasks_module.db_service, "get_document_by_task_id", fake_get_document_by_task_id)
    monkeypatch.setattr(tasks_module.db_service, "get_agent_by_id", fake_get_agent_by_id)
    monkeypatch.setattr(tasks_module, "AsyncResult", StubAsyncResult)

    _override_user(user)

    with TestClient(app) as client:
        response = client.get(f"/api/v1/tasks/{VALID_TASK_ID}")

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == 1
    assert body["status"] == "Processing..."
    assert body["state"] == "PROGRESS"


def test_get_task_status_unauthorized_returns_404(monkeypatch):
    user = SimpleUser(
        id=1,
        email="user@example.com",
        name="Unauthorized User",
        organization_id=1,
    )

    async def fake_get_document_by_task_id(task_id: str):
        return SimpleNamespace(id=99, agent_id=200)

    async def fake_get_agent_by_id(agent_id: int):
        return SimpleNamespace(id=agent_id, user_id=2, organization_id=999)

    monkeypatch.setattr(tasks_module.db_service, "get_document_by_task_id", fake_get_document_by_task_id)
    monkeypatch.setattr(tasks_module.db_service, "get_agent_by_id", fake_get_agent_by_id)

    _override_user(user)

    with TestClient(app) as client:
        response = client.get(f"/api/v1/tasks/{VALID_TASK_ID}")

    assert response.status_code == 404


def test_cancel_task_revokes_and_marks_cancelled(monkeypatch):
    user = SimpleUser(
        id=10,
        email="owner@example.com",
        name="Owner",
        organization_id=5,
    )
    updates = []

    async def fake_get_document_by_task_id(task_id: str):
        return SimpleNamespace(id=11, agent_id=33)

    async def fake_get_agent_by_id(agent_id: int):
        return SimpleNamespace(id=agent_id, user_id=user.id, organization_id=user.organization_id)

    async def fake_update_document(document_id: int, **kwargs):
        updates.append({"document_id": document_id, **kwargs})
        return None

    class CancelAsyncResult(StubAsyncResult):
        def __init__(self, task_id: str, app=None):
            super().__init__(task_id, app)
            self.state = "PROGRESS"

    monkeypatch.setattr(tasks_module.db_service, "get_document_by_task_id", fake_get_document_by_task_id)
    monkeypatch.setattr(tasks_module.db_service, "get_agent_by_id", fake_get_agent_by_id)
    monkeypatch.setattr(tasks_module.db_service, "update_document", fake_update_document)
    monkeypatch.setattr(tasks_module, "AsyncResult", CancelAsyncResult)

    _override_user(user)

    with TestClient(app) as client:
        response = client.delete(f"/api/v1/tasks/{VALID_TASK_ID}")

    assert response.status_code == 200
    assert CancelAsyncResult.last_instance is not None
    assert CancelAsyncResult.last_instance.revoked is True
    assert updates and updates[-1]["status"] == "cancelled"


def test_get_task_status_invalid_task_id_returns_400(monkeypatch):
    user = SimpleUser(
        id=42,
        email="user@example.com",
        name="Test User",
        organization_id=7,
    )

    called = {"get_document": False}

    async def fake_get_document_by_task_id(task_id: str):  # pragma: no cover - should not be hit
        called["get_document"] = True
        return SimpleNamespace(id=1, agent_id=55)

    monkeypatch.setattr(tasks_module.db_service, "get_document_by_task_id", fake_get_document_by_task_id)

    _override_user(user)

    with TestClient(app) as client:
        response = client.get("/api/v1/tasks/not-a-valid-id")

    assert response.status_code == 400
    assert "Invalid task ID format" in response.text
    assert called["get_document"] is False
VALID_TASK_ID = "123e4567-e89b-12d3-a456-426614174000"
