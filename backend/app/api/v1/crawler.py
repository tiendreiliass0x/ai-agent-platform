"""
Celery-backed crawler API: enqueue discovery using Firecrawl (default) with basic fallback.
Returns task status and incremental progress from Celery result backend.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, AnyHttpUrl, Field
from typing import List, Optional
from celery.result import AsyncResult

from ...core.auth import get_current_user
from ...models.user import User
from ...tasks.crawl_tasks import discover_urls


router = APIRouter()


class EnqueueCrawlRequest(BaseModel):
    root_url: AnyHttpUrl
    max_pages: int = Field(default=50, ge=1, le=500)
    provider: str = Field(default="firecrawl")  # "firecrawl" | "basic"


class CrawlStatusResponse(BaseModel):
    task_id: str
    status: str
    discovered: int = 0
    visited: int = 0
    urls: List[str] = []
    error: Optional[str] = None


@router.post("/enqueue", response_model=CrawlStatusResponse)
async def enqueue_crawl(payload: EnqueueCrawlRequest, current_user: User = Depends(get_current_user)):
    # Submit Celery task
    async_result = discover_urls.apply_async(args=[str(payload.root_url), payload.max_pages, payload.provider])
    return CrawlStatusResponse(task_id=async_result.id, status="queued")


@router.get("/{task_id}", response_model=CrawlStatusResponse)
async def get_crawl_status(task_id: str, current_user: User = Depends(get_current_user)):
    res = AsyncResult(task_id)
    state = res.state
    meta = res.info or {}
    # Map Celery states to simple statuses
    if state in {"PENDING", "RECEIVED", "STARTED", "PROGRESS"}:
        return CrawlStatusResponse(
            task_id=task_id,
            status="running" if state != "PENDING" else "queued",
            discovered=int(meta.get("discovered", 0)) if isinstance(meta, dict) else 0,
            visited=int(meta.get("visited", 0)) if isinstance(meta, dict) else 0,
            urls=list(meta.get("urls", [])) if isinstance(meta, dict) else [],
        )
    elif state == "SUCCESS":
        data = meta if isinstance(meta, dict) else {}
        return CrawlStatusResponse(
            task_id=task_id,
            status="completed",
            discovered=int(data.get("discovered", len(data.get("urls", [])))),
            visited=int(data.get("visited", 0)),
            urls=list(data.get("urls", [])),
        )
    elif state in {"FAILURE", "REVOKED"}:
        err = str(meta) if not isinstance(meta, dict) else str(meta.get("exc", meta.get("error", "failed")))
        return CrawlStatusResponse(task_id=task_id, status="failed", error=err)
    else:
        return CrawlStatusResponse(task_id=task_id, status=state.lower())
