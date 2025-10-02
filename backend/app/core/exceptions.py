"""
Secure global exception handlers for the FastAPI application.
Prevents information leakage while providing proper error responses.
"""

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging
import traceback
import uuid
from typing import Dict, Any
from datetime import datetime, timezone

from .config import settings

# Configure logging
logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Raised when security violations are detected"""
    pass


class BusinessLogicError(Exception):
    """Raised for business logic violations"""
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def create_error_response(
    status_code: int,
    message: str,
    details: Dict[str, Any] = None,
    error_id: str = None
) -> JSONResponse:
    """Create standardized error response"""

    if not error_id:
        error_id = str(uuid.uuid4())

    response_data = {
        "error": True,
        "message": message,
        "error_id": error_id,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    # Only include details in development mode
    if settings.ENVIRONMENT == "development" and details:
        response_data["details"] = details

    return JSONResponse(
        status_code=status_code,
        content=response_data
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTPException with security considerations"""

    error_id = str(uuid.uuid4())

    # Log the error
    logger.warning(
        f"HTTP Exception: {exc.status_code} - {exc.detail} "
        f"(Error ID: {error_id}, Path: {request.url.path})"
    )

    # Don't expose internal details in production
    if settings.ENVIRONMENT == "production" and exc.status_code >= 500:
        message = "Internal server error"
    else:
        message = exc.detail

    return create_error_response(
        status_code=exc.status_code,
        message=message,
        error_id=error_id
    )


async def starlette_http_exception_handler(
    request: Request,
    exc: StarletteHTTPException
) -> JSONResponse:
    """Handle Starlette HTTPException"""

    error_id = str(uuid.uuid4())

    logger.warning(
        f"Starlette HTTP Exception: {exc.status_code} - {exc.detail} "
        f"(Error ID: {error_id}, Path: {request.url.path})"
    )

    return create_error_response(
        status_code=exc.status_code,
        message=exc.detail,
        error_id=error_id
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """Handle request validation errors"""

    error_id = str(uuid.uuid4())

    # Log validation error details
    logger.info(
        f"Validation Error: {exc.errors()} "
        f"(Error ID: {error_id}, Path: {request.url.path})"
    )

    # Sanitize validation errors to prevent information leakage
    sanitized_errors = []
    for error in exc.errors():
        sanitized_error = {
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        }
        sanitized_errors.append(sanitized_error)

    return create_error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        message="Validation error",
        details={"validation_errors": sanitized_errors} if settings.ENVIRONMENT == "development" else None,
        error_id=error_id
    )


async def security_exception_handler(request: Request, exc: SecurityError) -> JSONResponse:
    """Handle security violations"""

    error_id = str(uuid.uuid4())

    # Log security violations with high priority
    logger.error(
        f"SECURITY VIOLATION: {str(exc)} "
        f"(Error ID: {error_id}, Path: {request.url.path}, "
        f"Client: {request.client.host if request.client else 'unknown'})"
    )

    # Never expose security error details
    return create_error_response(
        status_code=status.HTTP_403_FORBIDDEN,
        message="Access denied",
        error_id=error_id
    )


async def business_logic_exception_handler(
    request: Request,
    exc: BusinessLogicError
) -> JSONResponse:
    """Handle business logic errors"""

    error_id = str(uuid.uuid4())

    logger.info(
        f"Business Logic Error: {exc.message} "
        f"(Error ID: {error_id}, Path: {request.url.path})"
    )

    return create_error_response(
        status_code=exc.status_code,
        message=exc.message,
        error_id=error_id
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions"""

    error_id = str(uuid.uuid4())

    # Log the full exception with stack trace
    logger.error(
        f"Unhandled Exception: {type(exc).__name__}: {str(exc)} "
        f"(Error ID: {error_id}, Path: {request.url.path})",
        exc_info=True
    )

    # In production, never expose internal error details
    if settings.ENVIRONMENT == "production":
        message = "Internal server error"
        details = None
    else:
        message = f"{type(exc).__name__}: {str(exc)}"
        details = {
            "exception_type": type(exc).__name__,
            "traceback": traceback.format_exc().split('\n')
        }

    return create_error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        message=message,
        details=details,
        error_id=error_id
    )


def setup_exception_handlers(app):
    """Setup all exception handlers for the FastAPI app"""

    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, starlette_http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(SecurityError, security_exception_handler)
    app.add_exception_handler(BusinessLogicError, business_logic_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)