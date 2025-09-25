"""
API endpoints for system-wide operations and statistics.
"""

from fastapi import APIRouter, HTTPException, status
from ...services.database_service import db_service

router = APIRouter()


@router.get("/stats")
async def get_system_stats():
    """Get system-wide statistics"""
    try:
        stats = await db_service.get_system_stats()
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching system stats: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ai-agent-platform",
        "version": "1.0.0"
    }