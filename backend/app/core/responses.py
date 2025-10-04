"""
Standard response models for consistent API responses.
"""

from typing import Any, Dict, List, Optional, Generic, TypeVar
from pydantic import BaseModel, Field
from enum import Enum

T = TypeVar('T')

class ResponseStatus(str, Enum):
    """Standard response status values"""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"

class StandardResponse(BaseModel, Generic[T]):
    """Standard API response wrapper"""
    status: ResponseStatus = Field(..., description="Response status")
    data: Optional[T] = Field(None, description="Response data")
    message: Optional[str] = Field(None, description="Human-readable message")
    errors: Optional[List[str]] = Field(None, description="List of error messages")
    meta: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    class Config:
        json_encoders = {
            # Add custom encoders if needed
        }

class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response wrapper"""
    items: List[T] = Field(..., description="List of items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Items per page")
    pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_prev: bool = Field(..., description="Whether there are previous pages")

class ErrorResponse(BaseModel):
    """Standard error response"""
    status: ResponseStatus = ResponseStatus.ERROR
    message: str = Field(..., description="Error message")
    errors: List[str] = Field(default_factory=list, description="Detailed error messages")
    code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

class SuccessResponse(BaseModel, Generic[T]):
    """Standard success response"""
    status: ResponseStatus = ResponseStatus.SUCCESS
    data: T = Field(..., description="Response data")
    message: Optional[str] = Field(None, description="Success message")
    meta: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

# Convenience functions
def success_response(data: Any, message: str = None, meta: Dict[str, Any] = None) -> StandardResponse:
    """Create a success response"""
    return StandardResponse(
        status=ResponseStatus.SUCCESS,
        data=data,
        message=message,
        meta=meta
    )

def error_response(message: str, errors: List[str] = None, code: str = None, details: Dict[str, Any] = None) -> StandardResponse:
    """Create an error response"""
    return StandardResponse(
        status=ResponseStatus.ERROR,
        message=message,
        errors=errors or [],
        meta={"code": code, "details": details} if code or details else None
    )

def paginated_response(
    items: List[Any], 
    total: int, 
    page: int, 
    per_page: int
) -> PaginatedResponse:
    """Create a paginated response"""
    pages = (total + per_page - 1) // per_page  # Ceiling division
    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        per_page=per_page,
        pages=pages,
        has_next=page < pages,
        has_prev=page > 1
    )
