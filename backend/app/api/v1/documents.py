"""
API endpoints for document management.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from pydantic import BaseModel

from ...core.auth import get_current_user
from ...services.database_service import db_service
from ...models.user import User
from ...services.document_processor import DocumentProcessor


document_processor = DocumentProcessor()

router = APIRouter()


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

    class Config:
        from_attributes = True


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

        updated_document = await db_service.update_document_processing(
            document.id,
            status=processing_result.get("status"),
            chunk_count=processing_result.get("chunk_count"),
            vector_ids=processing_result.get("vector_ids"),
            error_message=processing_result.get("error_message"),
            content=processing_result.get("extracted_text") or document_data.content,
            doc_metadata_updates={
                **(document_data.doc_metadata or {}),
                "preview": processing_result.get("preview", "")
            }
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
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Upload a file and create a document"""
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

        # Create placeholder document entry before processing to capture doc_id
        document = await db_service.create_document(
            agent_id=agent_id,
            filename=file.filename or "uploaded_file",
            content="",
            content_type=file.content_type or "application/octet-stream",
            doc_metadata={
                "upload_size": file.spool_max_size if hasattr(file, "spool_max_size") else None,
                "original_filename": file.filename
            },
            status="processing",
            chunk_count=0,
            vector_ids=[]
        )

        import tempfile
        from pathlib import Path

        temp_file = tempfile.NamedTemporaryFile(delete=False)
        try:
            file_bytes = await file.read()
            temp_file.write(file_bytes)
            temp_file.flush()

            processing_result = await document_processor.process_file(
                temp_file.name,
                agent_id=agent_id,
                filename=file.filename or "uploaded_file",
                file_type=file.content_type or "application/octet-stream",
                document_id=document.id,
                extra_metadata={"organization_id": agent.organization_id}
            )
        finally:
            temp_path = Path(temp_file.name)
            temp_file.close()
            if temp_path.exists():
                temp_path.unlink()

        updated_document = await db_service.update_document_processing(
            document.id,
            status=processing_result.get("status"),
            chunk_count=processing_result.get("chunk_count"),
            vector_ids=processing_result.get("vector_ids"),
            error_message=processing_result.get("error_message"),
            content=processing_result.get("extracted_text", ""),
            doc_metadata_updates={
                "preview": processing_result.get("preview", ""),
                "original_filename": file.filename,
                "content_type": file.content_type,
                "upload_size": len(file_bytes)
            }
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
            detail=f"Error uploading document: {str(e)}"
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
