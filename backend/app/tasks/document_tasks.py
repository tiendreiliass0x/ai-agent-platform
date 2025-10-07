
"""Celery tasks for document processing."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Coroutine, TypeVar

from celery import Task

from app.celery_app import celery_app
from app.services.database_service import db_service
from app.services.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

document_processor = DocumentProcessor()

T = TypeVar("T")

# Progress milestones (avoid magic numbers scattered through the task)
PROGRESS_INIT = 0
PROGRESS_EXTRACTION = 20
PROGRESS_EMBEDDING = 50
PROGRESS_VECTOR_WRITE = 80
PROGRESS_COMPLETE = 100
PROGRESS_WEB_FETCH = 30
PROGRESS_WEB_EMBEDDING = 70


def _run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine in a synchronous Celery task without leaking loops."""
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except (RuntimeError, ValueError):
            # Loop might already be closed or not started; ignore to keep cleanup robust
            pass
        finally:
            asyncio.set_event_loop(None)
            loop.close()


async def _mark_document_failed(document_id: int, error_message: str) -> None:
    """Persist failure state for a document."""
    await db_service.update_document(
        document_id,
        status="failed",
        error_message=error_message,
    )


async def _process_document_workflow(
    task: Task,
    *,
    document_id: int,
    file_path: str,
    agent_id: int,
    filename: str,
    content_type: str,
    organization_id: int,
) -> Dict[str, Any]:
    """Async workflow executed within the Celery task context."""

    task.update_state(
        state="PROGRESS",
        meta={
            "current": PROGRESS_INIT,
            "total": PROGRESS_COMPLETE,
            "status": "Initializing document processing...",
            "document_id": document_id,
        },
    )

    await db_service.update_document(document_id, status="processing")

    task.update_state(
        state="PROGRESS",
        meta={
            "current": PROGRESS_EXTRACTION,
            "total": PROGRESS_COMPLETE,
            "status": f"Extracting text from {filename}...",
            "document_id": document_id,
        },
    )

    processing_result = await document_processor.process_file(
        file_path,
        agent_id=agent_id,
        filename=filename,
        file_type=content_type,
        document_id=document_id,
        extra_metadata={"organization_id": organization_id},
    )

    chunk_count = processing_result.get("chunk_count", 0)
    vector_ids = processing_result.get("vector_ids", [])

    task.update_state(
        state="PROGRESS",
        meta={
            "current": PROGRESS_EMBEDDING,
            "total": PROGRESS_COMPLETE,
            "status": f"Generated {chunk_count} text chunks, creating embeddings...",
            "document_id": document_id,
            "chunk_count": chunk_count,
        },
    )

    task.update_state(
        state="PROGRESS",
        meta={
            "current": PROGRESS_VECTOR_WRITE,
            "total": PROGRESS_COMPLETE,
            "status": "Storing embeddings in vector database...",
            "document_id": document_id,
            "chunk_count": chunk_count,
        },
    )

    metadata_updates: Dict[str, Any] = {
        "preview": processing_result.get("preview", ""),
        "processing_completed_at": datetime.utcnow().isoformat(),
    }
    if processing_result.get("metadata"):
        metadata_updates.update(processing_result["metadata"])
    if processing_result.get("keywords"):
        metadata_updates["keywords"] = processing_result["keywords"]

    await db_service.update_document_processing(
        document_id,
        status=processing_result.get("status", "completed"),
        chunk_count=chunk_count,
        vector_ids=vector_ids,
        error_message=processing_result.get("error_message"),
        content=processing_result.get("extracted_text", ""),
        doc_metadata_updates=metadata_updates,
    )

    task.update_state(
        state="PROGRESS",
        meta={
            "current": PROGRESS_COMPLETE,
            "total": PROGRESS_COMPLETE,
            "status": "Processing complete!",
            "document_id": document_id,
            "chunk_count": chunk_count,
            "vector_count": len(vector_ids),
        },
    )

    return {
        "document_id": document_id,
        "status": "completed",
        "chunk_count": chunk_count,
        "vector_count": len(vector_ids),
        "filename": filename,
    }

@celery_app.task(
    name="app.tasks.document_tasks.process_document",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=600,
    retry_jitter=True,
)
def process_document(
    self: Task,
    document_id: int,
    file_path: str,
    agent_id: int,
    filename: str,
    content_type: str,
    organization_id: int,
) -> Dict[str, Any]:
    """Process an uploaded document end-to-end."""

    cleanup_temp_file = True

    try:
        logger.info(
            "Starting document processing task",
            extra={"document_id": document_id, "filename": filename},
        )

        result = _run_sync(
            _process_document_workflow(
                self,
                document_id=document_id,
                file_path=file_path,
                agent_id=agent_id,
                filename=filename,
                content_type=content_type,
                organization_id=organization_id,
            )
        )

        logger.info(
            "Completed document processing",
            extra={
                "document_id": document_id,
                "chunk_count": result.get("chunk_count"),
                "vector_count": result.get("vector_count"),
            },
        )
        return result

    except Exception as exc:  # noqa: BLE001 - we re-raise for Celery autoretry
        logger.error(
            "Error processing document",
            extra={"document_id": document_id, "error": str(exc)},
            exc_info=True,
        )

        _run_sync(_mark_document_failed(document_id, str(exc)))

        # Preserve temp file when Celery will retry so the next attempt can reuse it
        cleanup_temp_file = self.request.retries >= self.max_retries

        # Surface exception to trigger Celery retry/backoff or final failure
        raise

    finally:
        if cleanup_temp_file:
            try:
                Path(file_path).unlink(missing_ok=True)
                logger.debug("Cleaned up temporary file", extra={"file_path": file_path})
            except Exception as cleanup_error:  # noqa: BLE001 - log and continue
                logger.warning(
                    "Failed to clean up temporary file",
                    extra={"file_path": file_path, "error": str(cleanup_error)},
                )


async def _process_webpage_workflow(
    task: Task,
    *,
    document_id: int,
    url: str,
    agent_id: int,
    organization_id: int,
) -> Dict[str, Any]:
    """Async workflow for processing webpages."""

    task.update_state(
        state="PROGRESS",
        meta={
            "current": PROGRESS_INIT,
            "total": PROGRESS_COMPLETE,
            "status": f"Fetching webpage {url}...",
            "document_id": document_id,
        },
    )

    await db_service.update_document(document_id, status="processing")

    task.update_state(
        state="PROGRESS",
        meta={
            "current": PROGRESS_WEB_FETCH,
            "total": PROGRESS_COMPLETE,
            "status": "Extracting content from webpage...",
            "document_id": document_id,
        },
    )

    processing_result = await document_processor.process_url(
        url,
        agent_id=agent_id,
        document_id=document_id,
        extra_metadata={"organization_id": organization_id},
    )

    chunk_count = processing_result.get("chunk_count", 0)
    vector_ids = processing_result.get("vector_ids", [])

    task.update_state(
        state="PROGRESS",
        meta={
            "current": PROGRESS_WEB_EMBEDDING,
            "total": PROGRESS_COMPLETE,
            "status": f"Created {chunk_count} chunks, generating embeddings...",
            "document_id": document_id,
            "chunk_count": chunk_count,
        },
    )

    metadata_updates: Dict[str, Any] = {
        "preview": processing_result.get("preview", ""),
        "processing_completed_at": datetime.utcnow().isoformat(),
        "source_url": url,
    }
    if processing_result.get("metadata"):
        metadata_updates.update(processing_result["metadata"])

    await db_service.update_document_processing(
        document_id,
        status=processing_result.get("status", "completed"),
        chunk_count=chunk_count,
        vector_ids=vector_ids,
        error_message=processing_result.get("error_message"),
        content=processing_result.get("extracted_text", ""),
        doc_metadata_updates=metadata_updates,
    )

    task.update_state(
        state="PROGRESS",
        meta={
            "current": PROGRESS_COMPLETE,
            "total": PROGRESS_COMPLETE,
            "status": "Webpage processing complete!",
            "document_id": document_id,
            "chunk_count": chunk_count,
            "vector_count": len(vector_ids),
        },
    )

    return {
        "document_id": document_id,
        "status": "completed",
        "chunk_count": chunk_count,
        "vector_count": len(vector_ids),
        "url": url,
    }


@celery_app.task(
    name="app.tasks.document_tasks.process_webpage",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(Exception,),
    retry_backoff=True,
)
def process_webpage(
    self: Task,
    document_id: int,
    url: str,
    agent_id: int,
    organization_id: int,
) -> Dict[str, Any]:
    """Process a webpage URL by crawling, chunking, and embedding content."""

    try:
        logger.info(
            "Starting webpage processing task",
            extra={"document_id": document_id, "url": url},
        )

        result = _run_sync(
            _process_webpage_workflow(
                self,
                document_id=document_id,
                url=url,
                agent_id=agent_id,
                organization_id=organization_id,
            )
        )

        logger.info(
            "Completed webpage processing",
            extra={
                "document_id": document_id,
                "chunk_count": result.get("chunk_count"),
                "vector_count": result.get("vector_count"),
            },
        )
        return result

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Error processing webpage",
            extra={"document_id": document_id, "error": str(exc)},
            exc_info=True,
        )

        _run_sync(_mark_document_failed(document_id, str(exc)))

        raise
