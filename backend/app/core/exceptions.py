"""
Enhanced exception handling for production-ready API.
"""

import logging
from typing import Any, Dict, List, Optional
from fastapi import Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .responses import ErrorResponse, ResponseStatus

logger = logging.getLogger(__name__)

class BaseAPIException(Exception):
    """Base exception for API errors"""
    def __init__(self, message: str, code: str = None, details: Dict[str, Any] = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)

class ValidationException(BaseAPIException):
    """Validation error exception"""
    def __init__(self, message: str, field_errors: List[str] = None):
        self.field_errors = field_errors or []
        super().__init__(message, code="VALIDATION_ERROR", details={"field_errors": self.field_errors})

class NotFoundException(BaseAPIException):
    """Resource not found exception"""
    def __init__(self, message: str, resource_type: str = None, resource_id: Any = None):
        details = {}
        if resource_type:
            details["resource_type"] = resource_type
        if resource_id is not None:
            details["resource_id"] = resource_id
        super().__init__(message, code="NOT_FOUND", details=details)

class UnauthorizedException(BaseAPIException):
    """Unauthorized access exception"""
    def __init__(self, message: str = "Unauthorized access"):
        super().__init__(message, code="UNAUTHORIZED")

class ForbiddenException(BaseAPIException):
    """Forbidden access exception"""
    def __init__(self, message: str = "Access forbidden"):
        super().__init__(message, code="FORBIDDEN")

class ConflictException(BaseAPIException):
    """Resource conflict exception"""
    def __init__(self, message: str, conflicting_resource: str = None):
        details = {}
        if conflicting_resource:
            details["conflicting_resource"] = conflicting_resource
        super().__init__(message, code="CONFLICT", details=details)

class RateLimitException(BaseAPIException):
    """Rate limit exceeded exception"""
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = None):
        details = {}
        if retry_after:
            details["retry_after"] = retry_after
        super().__init__(message, code="RATE_LIMIT_EXCEEDED", details=details)

class BusinessLogicException(BaseAPIException):
    """Business logic error exception"""
    def __init__(self, message: str, operation: str = None):
        details = {}
        if operation:
            details["operation"] = operation
        super().__init__(message, code="BUSINESS_LOGIC_ERROR", details=details)

class ExternalServiceException(BaseAPIException):
    """External service error exception"""
    def __init__(self, message: str, service: str = None):
        details = {}
        if service:
            details["service"] = service
        super().__init__(message, code="EXTERNAL_SERVICE_ERROR", details=details)

def setup_exception_handlers(app):
    """Setup exception handlers for the FastAPI app"""
    
    @app.exception_handler(BaseAPIException)
    async def base_api_exception_handler(request: Request, exc: BaseAPIException):
        """Handle custom API exceptions"""
        logger.warning(f"API Exception: {exc.message} (Code: {exc.code})", extra={
            "exception_type": exc.__class__.__name__,
            "code": exc.code,
            "details": exc.details,
            "path": request.url.path,
            "method": request.method
        })
        
        # Map custom exceptions to HTTP status codes
        status_code = 500  # Default
        if isinstance(exc, ValidationException):
            status_code = 400
        elif isinstance(exc, UnauthorizedException):
            status_code = 401
        elif isinstance(exc, ForbiddenException):
            status_code = 403
        elif isinstance(exc, NotFoundException):
            status_code = 404
        elif isinstance(exc, ConflictException):
            status_code = 409
        elif isinstance(exc, RateLimitException):
            status_code = 429
        elif isinstance(exc, BusinessLogicException):
            status_code = 422
        elif isinstance(exc, ExternalServiceException):
            status_code = 502
        
        error_response = ErrorResponse(
            status=ResponseStatus.ERROR,
            message=exc.message,
            code=exc.code,
            details=exc.details
        )
        
        return JSONResponse(
            status_code=status_code,
            content=error_response.dict()
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle Pydantic validation errors"""
        logger.warning(f"Validation Error: {exc.errors()}", extra={
            "path": request.url.path,
            "method": request.method,
            "errors": exc.errors()
        })
        
        # Extract field errors
        field_errors = []
        for error in exc.errors():
            field_path = " -> ".join(str(loc) for loc in error["loc"])
            field_errors.append(f"{field_path}: {error['msg']}")
        
        error_response = ErrorResponse(
            status=ResponseStatus.ERROR,
            message="Validation failed",
            errors=field_errors,
            code="VALIDATION_ERROR",
            details={"validation_errors": exc.errors()}
        )
        
        return JSONResponse(
            status_code=422,
            content=error_response.dict()
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle FastAPI HTTP exceptions"""
        logger.warning(f"HTTP Exception: {exc.detail}", extra={
            "status_code": exc.status_code,
            "path": request.url.path,
            "method": request.method
        })
        
        error_response = ErrorResponse(
            status=ResponseStatus.ERROR,
            message=str(exc.detail),
            code=f"HTTP_{exc.status_code}"
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.dict()
        )

    @app.exception_handler(StarletteHTTPException)
    async def starlette_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle Starlette HTTP exceptions"""
        logger.warning(f"Starlette Exception: {exc.detail}", extra={
            "status_code": exc.status_code,
            "path": request.url.path,
            "method": request.method
        })
        
        error_response = ErrorResponse(
            status=ResponseStatus.ERROR,
            message=str(exc.detail),
            code=f"HTTP_{exc.status_code}"
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.dict()
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions"""
        logger.error(f"Unexpected Exception: {str(exc)}", extra={
            "exception_type": exc.__class__.__name__,
            "path": request.url.path,
            "method": request.method
        }, exc_info=True)
        
        error_response = ErrorResponse(
            status=ResponseStatus.ERROR,
            message="An unexpected error occurred",
            code="INTERNAL_SERVER_ERROR"
        )
        
        return JSONResponse(
            status_code=500,
            content=error_response.dict()
        )

# Convenience functions for raising exceptions
def raise_not_found(message: str, resource_type: str = None, resource_id: Any = None):
    """Raise a not found exception"""
    raise NotFoundException(message, resource_type, resource_id)

def raise_unauthorized(message: str = "Unauthorized access"):
    """Raise an unauthorized exception"""
    raise UnauthorizedException(message)

def raise_forbidden(message: str = "Access forbidden"):
    """Raise a forbidden exception"""
    raise ForbiddenException(message)

def raise_validation_error(message: str, field_errors: List[str] = None):
    """Raise a validation exception"""
    raise ValidationException(message, field_errors)

def raise_conflict(message: str, conflicting_resource: str = None):
    """Raise a conflict exception"""
    raise ConflictException(message, conflicting_resource)

def raise_business_logic_error(message: str, operation: str = None):
    """Raise a business logic exception"""
    raise BusinessLogicException(message, operation)