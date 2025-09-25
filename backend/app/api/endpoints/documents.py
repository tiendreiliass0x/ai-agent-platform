from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import List, Optional
from app.models import get_async_session

router = APIRouter()

class DocumentResponse(BaseModel):
    id: int
    filename: str
    file_type: str
    file_size: int
    status: str
    chunk_count: int
    url: Optional[str] = None
    error_message: Optional[str] = None

    class Config:
        from_attributes = True

class URLUpload(BaseModel):
    url: str
    agent_id: int

@router.get("/{agent_id}/documents", response_model=List[DocumentResponse])
async def get_agent_documents(
    agent_id: int,
    db: AsyncSession = Depends(get_async_session)
):
    # TODO: Get documents for agent
    return []

@router.post("/{agent_id}/upload", response_model=DocumentResponse)
async def upload_document(
    agent_id: int,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_async_session)
):
    # TODO: Handle file upload and processing
    return {
        "id": 1,
        "filename": file.filename,
        "file_type": file.content_type,
        "file_size": 1024,
        "status": "processing",
        "chunk_count": 0,
        "url": None,
        "error_message": None
    }

@router.post("/{agent_id}/upload-url", response_model=DocumentResponse)
async def upload_url(
    agent_id: int,
    url_data: URLUpload,
    db: AsyncSession = Depends(get_async_session)
):
    # TODO: Handle URL processing
    return {
        "id": 2,
        "filename": "webpage.html",
        "file_type": "html",
        "file_size": 2048,
        "status": "processing",
        "chunk_count": 0,
        "url": url_data.url,
        "error_message": None
    }

@router.delete("/{agent_id}/documents/{document_id}")
async def delete_document(
    agent_id: int,
    document_id: int,
    db: AsyncSession = Depends(get_async_session)
):
    # TODO: Delete document and its vectors
    return {"message": "Document deleted successfully"}

@router.post("/{agent_id}/documents/{document_id}/reprocess")
async def reprocess_document(
    agent_id: int,
    document_id: int,
    db: AsyncSession = Depends(get_async_session)
):
    # TODO: Reprocess document
    return {"message": "Document reprocessing started"}