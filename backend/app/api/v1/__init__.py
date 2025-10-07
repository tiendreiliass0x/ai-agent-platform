"""
API v1 module initialization.
"""

from fastapi import APIRouter
from .agents import router as agents_router
from .documents import router as documents_router
from .system import router as system_router
from .auth import router as auth_router
from .users import router as users_router
from .organizations import router as organizations_router
from .conversations import router as conversations_router
from .governance_test import router as governance_test_router
from .integrations import router as integrations_router
from ..endpoints.chat import router as public_chat_router
from .crawler import router as crawler_router
from .tasks import router as tasks_router

api_router = APIRouter()

# Include all v1 routers
api_router.include_router(auth_router, prefix="/auth", tags=["authentication"])
api_router.include_router(users_router, prefix="/users", tags=["users"])
api_router.include_router(organizations_router, prefix="/organizations", tags=["organizations"])
api_router.include_router(agents_router, prefix="/agents", tags=["agents"])
api_router.include_router(documents_router, prefix="/documents", tags=["documents"])
api_router.include_router(system_router, prefix="/system", tags=["system"])
api_router.include_router(conversations_router, prefix="/conversations", tags=["conversations"])
api_router.include_router(governance_test_router, prefix="/governance", tags=["governance-testing"])
api_router.include_router(integrations_router, prefix="/integrations", tags=["integrations"])
api_router.include_router(public_chat_router, prefix="/chat", tags=["chat-public"])
api_router.include_router(crawler_router, prefix="/crawl", tags=["crawl"])
api_router.include_router(tasks_router, prefix="/tasks", tags=["tasks"])
