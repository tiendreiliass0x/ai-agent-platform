"""
Document processing Celery tasks.
Handles asynchronous document uploads, processing, embedding generation, and vector storage.
"""

import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from app.celery_app import celery_app
from app.services.document_processor import DocumentProcessor
from app.services.database_service import db_service
from celery.exceptions import Retry

logger = logging.getLogger(__name__)

# Initialize document processor (will be reused across tasks)
document_processor = DocumentProcessor()


@celery_app.task(
    name="app.tasks.document_tasks.process_document",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=600,
    retry_jitter=True
)
def process_document(
    self,
    document_id: int,
    file_path: str,
    agent_id: int,
    filename: str,
    content_type: str,
    organization_id: int
) -> Dict[str, Any]:
    """
    Process uploaded document: extract text, generate embeddings, store vectors.

    Args:
        self: Celery task instance (bind=True)
        document_id: Database document ID
        file_path: Temporary file path
        agent_id: Associated agent ID
        filename: Original filename
        content_type: MIME type
        organization_id: Organization ID for access control

    Returns:
        Dict with processing results

    Raises:
        Retry: If processing fails (will retry with backoff)
    """
    try:
        logger.info(f"Starting document processing task for document_id={document_id}, filename={filename}")

        # Update task state to PROGRESS (0%)
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 0,
                'total': 100,
                'status': 'Initializing document processing...',
                'document_id': document_id
            }
        )

        # Update database status to processing
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.run_until_complete(
            db_service.update_document(document_id, status="processing")
        )

        # Update progress: File extraction starting (20%)
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 20,
                'total': 100,
                'status': f'Extracting text from {filename}...',
                'document_id': document_id
            }
        )

        # Process the file
        processing_result = loop.run_until_complete(
            document_processor.process_file(
                file_path,
                agent_id=agent_id,
                filename=filename,
                file_type=content_type,
                document_id=document_id,
                extra_metadata={"organization_id": organization_id}
            )
        )

        # Update progress: Text chunking (50%)
        chunk_count = processing_result.get("chunk_count", 0)
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 50,
                'total': 100,
                'status': f'Generated {chunk_count} text chunks, creating embeddings...',
                'document_id': document_id,
                'chunk_count': chunk_count
            }
        )

        # Update progress: Storing vectors (80%)
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 80,
                'total': 100,
                'status': 'Storing embeddings in vector database...',
                'document_id': document_id,
                'chunk_count': chunk_count
            }
        )

        # Update document with results
        metadata_updates = {
            "preview": processing_result.get("preview", ""),
            "processing_completed_at": datetime.utcnow().isoformat(),
        }
        if processing_result.get("metadata"):
            metadata_updates.update(processing_result["metadata"])
        if processing_result.get("keywords"):
            metadata_updates["keywords"] = processing_result["keywords"]

        loop.run_until_complete(
            db_service.update_document_processing(
                document_id,
                status=processing_result.get("status", "completed"),
                chunk_count=processing_result.get("chunk_count", 0),
                vector_ids=processing_result.get("vector_ids", []),
                error_message=processing_result.get("error_message"),
                content=processing_result.get("extracted_text", ""),
                doc_metadata_updates=metadata_updates
            )
        )

        # Update progress: Completed (100%)
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 100,
                'total': 100,
                'status': 'Processing complete!',
                'document_id': document_id,
                'chunk_count': chunk_count,
                'vector_count': len(processing_result.get("vector_ids", []))
            }
        )

        logger.info(
            f"Successfully completed processing for document_id={document_id}, "
            f"chunks={chunk_count}, vectors={len(processing_result.get('vector_ids', []))}"
        )

        return {
            "document_id": document_id,
            "status": "completed",
            "chunk_count": processing_result.get("chunk_count", 0),
            "vector_count": len(processing_result.get("vector_ids", [])),
            "filename": filename
        }

    except Exception as e:
        logger.error(f"Error processing document_id={document_id}: {str(e)}", exc_info=True)

        # Update document with error status
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            loop.run_until_complete(
                db_service.update_document(
                    document_id,
                    status="failed",
                    error_message=str(e)
                )
            )
        except Exception as db_error:
            logger.error(f"Failed to update document status: {str(db_error)}")

        # Raise to trigger retry (unless max retries reached)
        raise

    finally:
        # Clean up temporary file
        try:
            Path(file_path).unlink(missing_ok=True)
            logger.debug(f"Cleaned up temporary file: {file_path}")
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup temp file {file_path}: {str(cleanup_error)}")


@celery_app.task(
    name="app.tasks.document_tasks.process_webpage",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(Exception,),
    retry_backoff=True
)
def process_webpage(
    self,
    document_id: int,
    url: str,
    agent_id: int,
    organization_id: int
) -> Dict[str, Any]:
    """
    Process webpage URL: crawl, extract text, generate embeddings, store vectors.

    Args:
        self: Celery task instance
        document_id: Database document ID
        url: Webpage URL to process
        agent_id: Associated agent ID
        organization_id: Organization ID

    Returns:
        Dict with processing results
    """
    try:
        logger.info(f"Starting webpage processing task for document_id={document_id}, url={url}")

        self.update_state(
            state='PROGRESS',
            meta={
                'current': 0,
                'total': 100,
                'status': f'Fetching webpage {url}...',
                'document_id': document_id
            }
        )

        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.run_until_complete(
            db_service.update_document(document_id, status="processing")
        )

        self.update_state(
            state='PROGRESS',
            meta={
                'current': 30,
                'total': 100,
                'status': 'Extracting content from webpage...',
                'document_id': document_id
            }
        )

        # Process the webpage
        processing_result = loop.run_until_complete(
            document_processor.process_url(
                url,
                agent_id=agent_id,
                document_id=document_id,
                extra_metadata={"organization_id": organization_id}
            )
        )

        chunk_count = processing_result.get("chunk_count", 0)
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 70,
                'total': 100,
                'status': f'Created {chunk_count} chunks, generating embeddings...',
                'document_id': document_id,
                'chunk_count': chunk_count
            }
        )

        # Update document with results
        metadata_updates = {
            "preview": processing_result.get("preview", ""),
            "processing_completed_at": datetime.utcnow().isoformat(),
            "source_url": url
        }
        if processing_result.get("metadata"):
            metadata_updates.update(processing_result["metadata"])

        loop.run_until_complete(
            db_service.update_document_processing(
                document_id,
                status=processing_result.get("status", "completed"),
                chunk_count=processing_result.get("chunk_count", 0),
                vector_ids=processing_result.get("vector_ids", []),
                error_message=processing_result.get("error_message"),
                content=processing_result.get("extracted_text", ""),
                doc_metadata_updates=metadata_updates
            )
        )

        self.update_state(
            state='PROGRESS',
            meta={
                'current': 100,
                'total': 100,
                'status': 'Webpage processing complete!',
                'document_id': document_id,
                'chunk_count': chunk_count
            }
        )

        logger.info(f"Successfully completed webpage processing for document_id={document_id}")

        return {
            "document_id": document_id,
            "status": "completed",
            "chunk_count": chunk_count,
            "url": url
        }

    except Exception as e:
        logger.error(f"Error processing webpage document_id={document_id}: {str(e)}", exc_info=True)

        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            loop.run_until_complete(
                db_service.update_document(
                    document_id,
                    status="failed",
                    error_message=str(e)
                )
            )
        except Exception as db_error:
            logger.error(f"Failed to update document status: {str(db_error)}")

        raise
