"""Task status and control endpoints for background Celery jobs."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel

from ...celery_app import celery_app
from ...core.auth import SimpleUser, get_current_user
from ...services.database_service import db_service

logger = logging.getLogger(__name__)

router = APIRouter()

CELERY_TASK_ID_PATTERN = re.compile(
    r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$"
)


class TaskStatusResponse(BaseModel):
    """Response model for task status queries."""

    task_id: str
    state: str
    status: Optional[str] = None
    current: Optional[int] = None
    total: Optional[int] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    document_id: Optional[int] = None
    chunk_count: Optional[int] = None
    vector_count: Optional[int] = None


async def _get_authorized_task_document(
    task_id: str,
    current_user: SimpleUser,
) -> Any:
    """Return the document linked to the task after access validation."""

    if not CELERY_TASK_ID_PATTERN.match(task_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid task ID format",
        )

    document = await db_service.get_document_by_task_id(task_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found",
        )

    agent = await db_service.get_agent_by_id(document.agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found",
        )

    user_id = getattr(current_user, "id", None)
    user_org = getattr(current_user, "organization_id", None)
    is_superuser = bool(getattr(current_user, "is_superuser", False))

    if not (
        is_superuser
        or agent.user_id == user_id
        or (user_org is not None and agent.organization_id == user_org)
    ):
        # Return 404 to avoid leaking task existence
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found",
        )

    return document


@router.get("/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str,
    current_user: SimpleUser = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get the status and progress of a Celery task."""

    document = await _get_authorized_task_document(task_id, current_user)

    try:
        result = AsyncResult(task_id, app=celery_app)

        response: Dict[str, Any] = {
            "task_id": task_id,
            "state": result.state,
            "document_id": document.id,
        }

        if result.state == "PENDING":
            response.update(
                {
                    "status": "Task is queued and waiting to be processed",
                    "current": 0,
                    "total": 100,
                }
            )

        elif result.state == "PROGRESS":
            info = result.info if isinstance(result.info, dict) else {}
            response.update(
                {
                    "status": info.get("status", "Processing..."),
                    "current": info.get("current", 0),
                    "total": info.get("total", 100),
                    "chunk_count": info.get("chunk_count"),
                    "vector_count": info.get("vector_count"),
                    "document_id": info.get("document_id", document.id),
                }
            )

        elif result.state == "SUCCESS":
            payload = result.result
            response.update(
                {
                    "status": "Task completed successfully",
                    "current": 100,
                    "total": 100,
                    "result": payload if isinstance(payload, dict) else {"value": payload},
                }
            )

        elif result.state == "FAILURE":
            error_message = str(result.info) if result.info else "Unknown error"
            response.update(
                {
                    "status": "Task failed",
                    "error": error_message,
                }
            )
            logger.error(
                "Task failed",
                extra={"task_id": task_id, "document_id": document.id, "error": error_message},
            )

        elif result.state == "RETRY":
            response.update(
                {
                    "status": "Task failed and is being retried",
                    "error": str(result.info) if result.info else None,
                }
            )

        elif result.state == "REVOKED":
            response.update({"status": "Task was cancelled"})

        else:
            response.update({"status": f"Unknown task state: {result.state}"})

        return response

    except HTTPException:
        raise
    except SQLAlchemyError as exc:
        logger.error(
            "Error retrieving task status",
            extra={"task_id": task_id, "document_id": document.id, "error": str(exc)},
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve task status",
        )
    except Exception as exc:  # noqa: BLE001
        logger.critical(
            "Unexpected error retrieving task status",
            extra={"task_id": task_id, "document_id": document.id, "error": str(exc)},
            exc_info=True,
        )
        raise


@router.delete("/{task_id}")
async def cancel_task(
    task_id: str,
    current_user: SimpleUser = Depends(get_current_user),
) -> Dict[str, Any]:
    """Cancel a running or pending Celery task."""

    document = await _get_authorized_task_document(task_id, current_user)

    try:
        result = AsyncResult(task_id, app=celery_app)

        if result.state in {"SUCCESS", "FAILURE", "REVOKED"}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot cancel task in state {result.state}",
            )

        result.revoke(terminate=True)
        logger.info(
            "Task cancelled",
            extra={
                "task_id": task_id,
                "document_id": document.id,
                "requested_by": getattr(current_user, "id", None),
            },
        )

        terminal_states = {"cancelled", "failed", "completed"}
        if getattr(document, "status", None) not in terminal_states:
            await db_service.update_document(
                document.id,
                status="cancelled",
                error_message="Task cancelled by user",
            )

        return {
            "task_id": task_id,
            "status": "cancelled",
            "message": "Task has been cancelled",
        }

    except HTTPException:
        raise
    except SQLAlchemyError as exc:
        logger.error(
            "Error cancelling task",
            extra={"task_id": task_id, "document_id": document.id, "error": str(exc)},
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel task",
        )
    except Exception as exc:  # noqa: BLE001
        logger.critical(
            "Unexpected error cancelling task",
            extra={"task_id": task_id, "document_id": document.id, "error": str(exc)},
            exc_info=True,
        )
        raise
