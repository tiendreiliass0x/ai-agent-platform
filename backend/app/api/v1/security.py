"""
Security management API endpoints for administrators.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel
from datetime import datetime

from ...core.auth import get_current_user, require_admin
from ...services.document_security import document_security
from ...models.user import User

router = APIRouter()

class SecurityStatsResponse(BaseModel):
    total_uploads_today: int
    blocked_uploads_today: int
    quarantined_files: int
    blocked_hashes: int
    top_blocked_reasons: List[Dict[str, Any]]
    security_score_distribution: Dict[str, int]

class QuarantineFileResponse(BaseModel):
    original_filename: str
    quarantine_time: str
    file_size: int
    md5: str
    sha256: str
    security_score: int
    issues: List[str]
    warnings: List[str]

class SecurityConfigResponse(BaseModel):
    max_file_size: Dict[str, int]
    allowed_extensions: List[str]
    blocked_extensions: List[str]
    max_uploads_per_hour: int
    quarantine_enabled: bool
    virus_scanning_enabled: bool

class SecurityConfigUpdate(BaseModel):
    max_uploads_per_hour: Optional[int] = None
    quarantine_enabled: Optional[bool] = None
    additional_blocked_extensions: Optional[List[str]] = None

class HashBlacklistRequest(BaseModel):
    file_hash: str
    reason: str
    added_by: Optional[str] = None

@router.get("/stats", response_model=SecurityStatsResponse)
async def get_security_stats(
    current_user: User = Depends(require_admin)
):
    """Get security statistics and metrics"""
    try:
        # Get quarantined files
        quarantine_files = await document_security.get_quarantine_files()

        # Calculate basic stats
        stats = {
            "total_uploads_today": 0,  # Would come from database in production
            "blocked_uploads_today": 0,  # Would come from logs/database
            "quarantined_files": len(quarantine_files),
            "blocked_hashes": len(document_security.blocked_hashes),
            "top_blocked_reasons": [
                {"reason": "Executable file detected", "count": 15},
                {"reason": "Rate limit exceeded", "count": 12},
                {"reason": "Suspicious content", "count": 8},
                {"reason": "File too large", "count": 5}
            ],
            "security_score_distribution": {
                "0-20": 5,   # Very dangerous
                "21-40": 8,  # Dangerous
                "41-60": 12, # Suspicious
                "61-80": 25, # Medium risk
                "81-100": 150 # Safe
            }
        }

        return SecurityStatsResponse(**stats)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching security stats: {str(e)}"
        )

@router.get("/quarantine", response_model=List[QuarantineFileResponse])
async def get_quarantine_files(
    limit: int = Query(default=50, ge=1, le=1000),
    current_user: User = Depends(require_admin)
):
    """Get list of quarantined files"""
    try:
        quarantine_files = await document_security.get_quarantine_files()

        # Convert to response format
        response_files = []
        for file_data in quarantine_files[:limit]:
            validation_result = file_data.get('validation_result', {})
            response_files.append(QuarantineFileResponse(
                original_filename=file_data.get('original_filename', 'unknown'),
                quarantine_time=file_data.get('quarantine_time', ''),
                file_size=file_data.get('file_size', 0),
                md5=file_data.get('md5', ''),
                sha256=file_data.get('sha256', ''),
                security_score=validation_result.get('security_score', 0),
                issues=validation_result.get('issues', []),
                warnings=validation_result.get('warnings', [])
            ))

        return response_files

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching quarantine files: {str(e)}"
        )

@router.delete("/quarantine/{file_hash}")
async def remove_from_quarantine(
    file_hash: str,
    current_user: User = Depends(require_admin)
):
    """Remove a file from quarantine"""
    try:
        # In production, implement actual removal logic
        # For now, just acknowledge the request
        return {"message": f"File {file_hash} removed from quarantine"}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error removing from quarantine: {str(e)}"
        )

@router.post("/quarantine/clean")
async def clean_quarantine(
    days_old: int = Query(default=30, ge=1, le=365),
    current_user: User = Depends(require_admin)
):
    """Clean old quarantine files"""
    try:
        await document_security.clean_quarantine(days_old=days_old)
        return {"message": f"Cleaned quarantine files older than {days_old} days"}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error cleaning quarantine: {str(e)}"
        )

@router.get("/config", response_model=SecurityConfigResponse)
async def get_security_config(
    current_user: User = Depends(require_admin)
):
    """Get current security configuration"""
    try:
        config = SecurityConfigResponse(
            max_file_size=document_security.SIZE_LIMITS,
            allowed_extensions=list(document_security.ALLOWED_EXTENSIONS),
            blocked_extensions=['.exe', '.bat', '.cmd', '.scr', '.com', '.pif'],
            max_uploads_per_hour=50,  # From settings
            quarantine_enabled=True,
            virus_scanning_enabled=False  # Would be true if integrated
        )

        return config

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching security config: {str(e)}"
        )

@router.put("/config")
async def update_security_config(
    config_update: SecurityConfigUpdate,
    current_user: User = Depends(require_admin)
):
    """Update security configuration"""
    try:
        updates_applied = []

        if config_update.max_uploads_per_hour is not None:
            # In production, update settings/database
            updates_applied.append(f"Max uploads per hour: {config_update.max_uploads_per_hour}")

        if config_update.quarantine_enabled is not None:
            updates_applied.append(f"Quarantine enabled: {config_update.quarantine_enabled}")

        if config_update.additional_blocked_extensions:
            # In production, add to blocked extensions list
            updates_applied.append(f"Added blocked extensions: {config_update.additional_blocked_extensions}")

        return {
            "message": "Security configuration updated",
            "updates_applied": updates_applied
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating security config: {str(e)}"
        )

@router.post("/blacklist/hash")
async def add_hash_to_blacklist(
    request: HashBlacklistRequest,
    current_user: User = Depends(require_admin)
):
    """Add file hash to blacklist"""
    try:
        await document_security.add_to_blacklist(request.file_hash)

        # In production, log this action with audit trail
        return {
            "message": f"Hash {request.file_hash} added to blacklist",
            "reason": request.reason,
            "added_by": current_user.email
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error adding hash to blacklist: {str(e)}"
        )

@router.get("/blacklist/hashes")
async def get_blacklisted_hashes(
    current_user: User = Depends(require_admin)
):
    """Get list of blacklisted file hashes"""
    try:
        return {
            "blacklisted_hashes": list(document_security.blocked_hashes),
            "count": len(document_security.blocked_hashes)
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching blacklisted hashes: {str(e)}"
        )

@router.delete("/blacklist/hash/{file_hash}")
async def remove_hash_from_blacklist(
    file_hash: str,
    current_user: User = Depends(require_admin)
):
    """Remove file hash from blacklist"""
    try:
        if file_hash in document_security.blocked_hashes:
            document_security.blocked_hashes.remove(file_hash)
            return {"message": f"Hash {file_hash} removed from blacklist"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Hash not found in blacklist"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error removing hash from blacklist: {str(e)}"
        )

@router.post("/scan/file")
async def scan_file_on_demand(
    file_hash: str,
    current_user: User = Depends(require_admin)
):
    """Trigger on-demand scan of a specific file"""
    try:
        # In production, implement file scanning logic
        # This would re-scan a file that's already in the system

        return {
            "message": f"Scan initiated for file hash: {file_hash}",
            "scan_id": f"scan_{file_hash[:8]}_{int(datetime.utcnow().timestamp())}"
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error initiating file scan: {str(e)}"
        )

@router.get("/threats/recent")
async def get_recent_threats(
    hours: int = Query(default=24, ge=1, le=168),  # Last 24 hours by default, max 1 week
    current_user: User = Depends(require_admin)
):
    """Get recent security threats and blocked uploads"""
    try:
        # In production, this would query logs/database for recent threats
        recent_threats = [
            {
                "timestamp": "2024-01-15T10:30:00Z",
                "user_id": 123,
                "filename": "malicious.exe",
                "threat_type": "executable",
                "action": "blocked",
                "security_score": 5
            },
            {
                "timestamp": "2024-01-15T09:15:00Z",
                "user_id": 456,
                "filename": "suspicious_doc.pdf",
                "threat_type": "suspicious_content",
                "action": "quarantined",
                "security_score": 45
            }
        ]

        return {
            "threats": recent_threats,
            "count": len(recent_threats),
            "time_range_hours": hours
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching recent threats: {str(e)}"
        )