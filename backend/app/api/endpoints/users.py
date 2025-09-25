from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from app.models import get_async_session

router = APIRouter()

class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str = None
    is_active: bool

    class Config:
        from_attributes = True

@router.get("/me", response_model=UserResponse)
async def get_current_user(
    db: AsyncSession = Depends(get_async_session)
):
    # TODO: Get current user from token
    return {"id": 1, "email": "user@example.com", "full_name": "Test User", "is_active": True}

@router.put("/me", response_model=UserResponse)
async def update_current_user(
    updates: dict,
    db: AsyncSession = Depends(get_async_session)
):
    # TODO: Update current user
    return {"id": 1, "email": "user@example.com", "full_name": "Test User", "is_active": True}