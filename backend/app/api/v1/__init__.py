"""
API v1 module initialization.
"""

from fastapi import APIRouter
from .agents import router as agents_router
from .documents import router as documents_router
from .system import router as system_router
from .auth import router as auth_router
from .organizations import router as organizations_router
from .conversations import router as conversations_router

api_router = APIRouter()

# Include all v1 routers
api_router.include_router(auth_router, prefix="/auth", tags=["authentication"])
api_router.include_router(organizations_router, prefix="/organizations", tags=["organizations"])
api_router.include_router(agents_router, prefix="/agents", tags=["agents"])
api_router.include_router(documents_router, prefix="/documents", tags=["documents"])
api_router.include_router(system_router, prefix="/system", tags=["system"])
api_router.include_router(conversations_router, prefix="/conversations", tags=["conversations"])
