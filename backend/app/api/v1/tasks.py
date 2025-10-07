"""
Task status tracking endpoints.
Provides status and progress information for Celery background tasks.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from celery.result import AsyncResult
from pydantic import BaseModel

from ...celery_app import celery_app
from ...core.auth import get_current_user
from ...models.user import User

logger = logging.getLogger(__name__)

router = APIRouter()


class TaskStatusResponse(BaseModel):
    """Response model for task status"""
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


@router.get("/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get the status and progress of a Celery task.

    Task States:
    - PENDING: Task is waiting to be executed
    - PROGRESS: Task is currently running (includes progress info)
    - SUCCESS: Task completed successfully
    - FAILURE: Task failed with an error
    - RETRY: Task is being retried
    - REVOKED: Task was cancelled

    Args:
        task_id: Celery task ID
        current_user: Authenticated user (required for security)

    Returns:
        TaskStatusResponse with current task state and progress
    """
    try:
        # Get task result from Celery
        result = AsyncResult(task_id, app=celery_app)

        response = {
            "task_id": task_id,
            "state": result.state,
        }

        # Handle different task states
        if result.state == 'PENDING':
            response.update({
                "status": "Task is queued and waiting to be processed",
                "current": 0,
                "total": 100
            })

        elif result.state == 'PROGRESS':
            # Task is running - extract progress info
            if isinstance(result.info, dict):
                response.update({
                    "status": result.info.get('status', 'Processing...'),
                    "current": result.info.get('current', 0),
                    "total": result.info.get('total', 100),
                    "document_id": result.info.get('document_id'),
                    "chunk_count": result.info.get('chunk_count'),
                    "vector_count": result.info.get('vector_count')
                })
            else:
                response.update({
                    "status": "Processing...",
                    "current": 50,
                    "total": 100
                })

        elif result.state == 'SUCCESS':
            # Task completed successfully
            response.update({
                "status": "Task completed successfully",
                "current": 100,
                "total": 100,
                "result": result.result if isinstance(result.result, dict) else {"value": result.result}
            })

        elif result.state == 'FAILURE':
            # Task failed
            error_message = str(result.info) if result.info else "Unknown error"
            response.update({
                "status": "Task failed",
                "error": error_message
            })
            logger.error(f"Task {task_id} failed: {error_message}")

        elif result.state == 'RETRY':
            # Task is being retried
            response.update({
                "status": "Task failed and is being retried",
                "error": str(result.info) if result.info else None
            })

        elif result.state == 'REVOKED':
            # Task was cancelled
            response.update({
                "status": "Task was cancelled"
            })

        else:
            # Unknown state
            response.update({
                "status": f"Unknown task state: {result.state}"
            })

        return response

    except Exception as e:
        logger.error(f"Error retrieving task status for {task_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve task status: {str(e)}"
        )


@router.delete("/{task_id}")
async def cancel_task(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Cancel a running or pending Celery task.

    Args:
        task_id: Celery task ID
        current_user: Authenticated user (required for security)

    Returns:
        Confirmation message
    """
    try:
        result = AsyncResult(task_id, app=celery_app)

        if result.state in ['SUCCESS', 'FAILURE', 'REVOKED']:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel task in state {result.state}"
            )

        # Revoke the task
        result.revoke(terminate=True)

        logger.info(f"Task {task_id} cancelled by user {current_user.id}")

        return {
            "task_id": task_id,
            "status": "cancelled",
            "message": "Task has been cancelled"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling task {task_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel task: {str(e)}"
        )
