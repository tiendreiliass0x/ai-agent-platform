"""Conversation endpoints"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.database import get_db
from ...core.auth import get_current_user
from ...services.database_service import db_service
from ...models.user import User

router = APIRouter()


class ConversationMessageResponse(Dict[str, Any]):
    pass


@router.get("/{conversation_id}")
async def get_conversation_detail(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Return conversation details (messages, customer profile) for owners"""

    conversation = await db_service.get_conversation_by_id(conversation_id)
    if not conversation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    agent = conversation.agent
    if not agent or agent.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    customer_profile = conversation.customer_profile

    messages = []
    for msg in sorted(conversation.messages, key=lambda m: m.created_at or conversation.created_at):
        messages.append({
            "id": msg.id,
            "role": msg.role,
            "content": msg.content,
            "created_at": msg.created_at.isoformat() if msg.created_at else None,
            "metadata": msg.msg_metadata or {},
        })

    profile_data: Optional[Dict[str, Any]] = None
    if customer_profile:
        profile_data = {
            "id": customer_profile.id,
            "name": customer_profile.display_name,
            "visitor_id": customer_profile.visitor_id,
            "technical_level": customer_profile.technical_level,
            "communication_style": customer_profile.communication_style,
            "engagement_level": customer_profile.engagement_level,
            "is_vip": customer_profile.is_vip,
            "total_conversations": customer_profile.total_conversations,
            "primary_interests": customer_profile.primary_interests or [],
            "pain_points": customer_profile.pain_points or [],
            "current_journey_stage": customer_profile.current_journey_stage,
        }

    return {
        "id": conversation.id,
        "agent": {
            "id": agent.id,
            "name": agent.name,
        },
        "session_id": conversation.session_id,
        "created_at": conversation.created_at.isoformat() if conversation.created_at else None,
        "messages": messages,
        "customer_profile": profile_data,
    }

