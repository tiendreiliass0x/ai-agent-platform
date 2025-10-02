"""
API endpoints for document management.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
import tempfile
import logging
from pathlib import Path
from datetime import datetime
import re
from uuid import uuid4

from ...core.auth import get_current_user
from ...core.config import settings
from ...services.database_service import db_service
from ...models.user import User
from ...services.document_processor import DocumentProcessor
from ...services.document_security import document_security


document_processor = DocumentProcessor()
logger = logging.getLogger(__name__)


def _sanitize_filename(filename: str) -> str:
    """Normalize filenames to prevent traversal and unsafe characters."""

    if not filename:
        return f"upload_{uuid4().hex}"

    basename = Path(filename).name
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", basename)
    sanitized = sanitized.lstrip(".") or f"upload_{uuid4().hex}"
    return sanitized[:120]

router = APIRouter()


async def process_document_background(
    document_id: int,
    file_path: str,
    agent_id: int,
    filename: str,
    content_type: str,
    organization_id: int
):
    """
    Background task to process uploaded document.
    This runs asynchronously and doesn't block the HTTP response.
    """
    try:
        logger.info(f"Starting background processing for document {document_id}")

        # Update status to processing
        await db_service.update_document(document_id, status="processing")

        # Process the file (this can take time)
        processing_result = await document_processor.process_file(
            file_path,
            agent_id=agent_id,
            filename=filename,
            file_type=content_type,
            document_id=document_id,
            extra_metadata={"organization_id": organization_id}
        )

        # Update document with results
        metadata_updates = {
            "preview": processing_result.get("preview", ""),
            "processing_completed_at": str(datetime.utcnow()),
        }
        if processing_result.get("metadata"):
            metadata_updates.update(processing_result["metadata"])
        if processing_result.get("keywords"):
            metadata_updates["keywords"] = processing_result["keywords"]

        await db_service.update_document_processing(
            document_id,
            status=processing_result.get("status"),
            chunk_count=processing_result.get("chunk_count"),
            vector_ids=processing_result.get("vector_ids"),
            error_message=processing_result.get("error_message"),
            content=processing_result.get("extracted_text", ""),
            doc_metadata_updates=metadata_updates
        )

        logger.info(f"Successfully completed processing for document {document_id}")

    except Exception as e:
        logger.error(f"Error processing document {document_id}: {e}")
        # Update document with error status
        await db_service.update_document(
            document_id,
            status="failed",
            error_message=str(e)
        )
    finally:
        # Clean up temporary file
        try:
            Path(file_path).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {file_path}: {e}")


class DocumentResponse(BaseModel):
    id: int
    filename: str
    content_type: str
    size: int
    content_hash: str
    status: str
    doc_metadata: Dict[str, Any]
    agent_id: int
    created_at: str
    updated_at: Optional[str]

    model_config = {
        "from_attributes": True,
    }


class DocumentCreate(BaseModel):
    filename: str
    content: str
    content_type: str
    doc_metadata: Dict[str, Any] = {}


@router.get("/agent/{agent_id}", response_model=List[DocumentResponse])
async def get_agent_documents(
    agent_id: int,
    current_user: User = Depends(get_current_user)
):
    """Get all documents for an agent"""
    try:
        # Verify agent exists and user owns it
        agent = await db_service.get_agent_by_id(agent_id)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent not found"
            )

        if agent.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )

        documents = await db_service.get_agent_documents(agent_id)
        return [
            DocumentResponse(
                id=doc.id,
                filename=doc.filename,
                content_type=doc.content_type,
                size=doc.size,
                content_hash=doc.content_hash or "",
                status=doc.status,
                doc_metadata=doc.doc_metadata or {},
                agent_id=doc.agent_id,
                created_at=doc.created_at.isoformat(),
                updated_at=doc.updated_at.isoformat() if doc.updated_at else None
            )
            for doc in documents
        ]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching documents: {str(e)}"
        )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    current_user: User = Depends(get_current_user)
):
    """Get a specific document by ID"""
    try:
        document = await db_service.get_document_by_id(document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

        # Verify user owns the agent that owns this document
        agent = await db_service.get_agent_by_id(document.agent_id)
        if not agent or agent.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )

        return DocumentResponse(
            id=document.id,
            filename=document.filename,
            content_type=document.content_type,
            size=document.size,
            content_hash=document.content_hash or "",
            status=document.status,
            doc_metadata=document.doc_metadata or {},
            agent_id=document.agent_id,
            created_at=document.created_at.isoformat(),
            updated_at=document.updated_at.isoformat() if document.updated_at else None
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching document: {str(e)}"
        )


@router.post("/agent/{agent_id}", response_model=DocumentResponse)
async def create_document(
    agent_id: int,
    document_data: DocumentCreate,
    current_user: User = Depends(get_current_user)
):
    """Create a new document for an agent"""
    try:
        # Verify agent exists and user owns it
        agent = await db_service.get_agent_by_id(agent_id)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent not found"
            )

        if agent.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )

        # Create document in processing state so we can attach vectors
        document = await db_service.create_document(
            agent_id=agent_id,
            filename=document_data.filename,
            content=document_data.content,
            content_type=document_data.content_type,
            doc_metadata=document_data.doc_metadata,
            status="processing",
            chunk_count=0,
            vector_ids=[]
        )

        processing_result = await document_processor.process_text_content(
            document_data.content,
            agent_id=agent_id,
            source=document_data.filename,
            document_id=document.id,
            extra_metadata={"organization_id": agent.organization_id}
        )

        metadata_updates = {
            **(document_data.doc_metadata or {}),
            "preview": processing_result.get("preview", "")
        }
        if processing_result.get("metadata"):
            metadata_updates.update(processing_result["metadata"])
        if processing_result.get("keywords"):
            metadata_updates["keywords"] = processing_result["keywords"]

        updated_document = await db_service.update_document_processing(
            document.id,
            status=processing_result.get("status"),
            chunk_count=processing_result.get("chunk_count"),
            vector_ids=processing_result.get("vector_ids"),
            error_message=processing_result.get("error_message"),
            content=processing_result.get("extracted_text") or document_data.content,
            doc_metadata_updates=metadata_updates
        )

        return DocumentResponse(
            id=updated_document.id,
            filename=updated_document.filename,
            content_type=updated_document.content_type,
            size=updated_document.size,
            content_hash=updated_document.content_hash or "",
            status=updated_document.status,
            doc_metadata=updated_document.doc_metadata or {},
            agent_id=updated_document.agent_id,
            created_at=updated_document.created_at.isoformat(),
            updated_at=updated_document.updated_at.isoformat() if updated_document.updated_at else None
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating document: {str(e)}"
        )


@router.post("/agent/{agent_id}/upload")
async def upload_document(
    agent_id: int,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """
    Upload a file and queue it for background processing.

    Returns immediately with document ID and 'queued' status.
    Processing happens asynchronously in the background.
    Use GET /documents/{document_id}/status to check processing progress.
    """
    try:
        # Verify agent exists and user owns it
        agent = await db_service.get_agent_by_id(agent_id)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent not found"
            )

        if agent.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )

        # Read file content immediately (while request is active)
        file_bytes = await file.read()
        if len(file_bytes) > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds maximum allowed size of {settings.MAX_FILE_SIZE} bytes"
            )

        original_filename = file.filename or "uploaded_file"
        filename = _sanitize_filename(original_filename)
        content_type = file.content_type or "application/octet-stream"

        # Comprehensive security validation
        is_safe, security_result = await document_security.validate_upload(
            file_content=file_bytes,
            filename=filename,
            content_type=content_type,
            user_id=current_user.id
        )

        if not is_safe:
            logger.warning(f"Unsafe file upload blocked for user {current_user.id}: {filename}")
            logger.warning(f"Security issues: {security_result.get('issues', [])}")

            # Return detailed security error
            error_detail = {
                "message": "File upload blocked due to security concerns",
                "issues": security_result.get('issues', []),
                "security_score": security_result.get('security_score', 0),
                "quarantined": security_result.get('quarantined', False)
            }

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_detail
            )

        # Log security warnings if any
        if security_result.get('warnings'):
            logger.info(f"File upload warnings for {filename}: {security_result['warnings']}")

        # Additional legacy validation for backward compatibility
        if len(file_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file"
            )

        # Create document entry immediately with 'queued' status
        document = await db_service.create_document(
            agent_id=agent_id,
            filename=filename,
            content="",  # Will be populated during processing
            content_type=content_type,
            doc_metadata={
                "upload_size": len(file_bytes),
                "original_filename": original_filename,
                "content_type": content_type,
                "queued_at": datetime.utcnow().isoformat()
            },
            status="queued",  # Indicates ready for processing
            chunk_count=0,
            vector_ids=[]
        )

        # Save file to temporary location for background processing
        temp_file = tempfile.NamedTemporaryFile(delete=False, prefix=f"doc_{document.id}_")
        temp_file.write(file_bytes)
        temp_file.close()

        # Queue background processing task
        background_tasks.add_task(
            process_document_background,
            document_id=document.id,
            file_path=temp_file.name,
            agent_id=agent_id,
            filename=filename,
            content_type=content_type,
            organization_id=agent.organization_id
        )

        logger.info(f"Document {document.id} queued for background processing")

        # Return immediately with queued status
        return {
            "id": document.id,
            "filename": document.filename,
            "content_type": document.content_type,
            "size": len(file_bytes),
            "status": "queued",
            "agent_id": document.agent_id,
            "created_at": document.created_at.isoformat(),
            "message": "Document uploaded successfully and queued for processing",
            "processing_note": f"Use GET /api/v1/documents/{document.id}/status to check processing progress"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading document: {str(e)}"
        )


@router.get("/{document_id}/status")
async def get_document_processing_status(
    document_id: int,
    current_user: User = Depends(get_current_user)
):
    """
    Get the processing status of a document.

    Returns real-time processing status including:
    - queued: Document uploaded, waiting for processing
    - processing: Currently being processed
    - completed: Successfully processed and embedded
    - failed: Processing failed with error details
    """
    try:
        # Get document
        document = await db_service.get_document_by_id(document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

        # Verify user owns the agent that owns this document
        agent = await db_service.get_agent_by_id(document.agent_id)
        if not agent or agent.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )

        # Return status with detailed information
        response = {
            "id": document.id,
            "filename": document.filename,
            "status": document.status,
            "agent_id": document.agent_id,
            "created_at": document.created_at.isoformat(),
            "updated_at": document.updated_at.isoformat() if document.updated_at else None
        }

        # Add status-specific information
        if document.status == "queued":
            response["message"] = "Document is queued for processing"
            queued_at = document.doc_metadata.get("queued_at") if document.doc_metadata else None
            if queued_at:
                response["queued_at"] = queued_at

        elif document.status == "processing":
            response["message"] = "Document is currently being processed"

        elif document.status == "completed":
            response["message"] = "Document processed successfully"
            response["chunk_count"] = document.chunk_count
            response["vector_ids_count"] = len(document.vector_ids) if document.vector_ids else 0
            if document.doc_metadata:
                response["preview"] = document.doc_metadata.get("preview", "")
                response["processing_completed_at"] = document.doc_metadata.get("processing_completed_at")

        elif document.status == "failed":
            response["message"] = "Document processing failed"
            response["error_message"] = document.error_message

        else:
            response["message"] = f"Unknown status: {document.status}"

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting document status: {str(e)}"
        )


@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    current_user: User = Depends(get_current_user)
):
    """Delete a document"""
    try:
        # Verify user owns the agent that owns this document
        document = await db_service.get_document_by_id(document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

        agent = await db_service.get_agent_by_id(document.agent_id)
        if not agent or agent.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )

        # Remove vectors first (best-effort)
        if document.vector_ids:
            try:
                await document_processor.delete_document_vectors(document.vector_ids)
            except Exception as cleanup_error:
                print(f"Warning: failed to delete document vectors: {cleanup_error}")

        success = await db_service.delete_document(document_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

        return {"message": "Document deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {str(e)}"
        )
