"""
Escalation Service
Creates escalation events and builds rich context summaries so users never have to repeat themselves.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import and_, desc

from ..models.escalation import Escalation
from ..models.conversation import Conversation
from ..models.message import Message
from ..models.customer_profile import CustomerProfile
from .database_service import db_service
from .memory_service import memory_service


class EscalationService:
    def __init__(self):
        pass

    async def get_session(self) -> AsyncSession:
        from ..core.database import get_db_session
        return await get_db_session()

    async def create_escalation_event(
        self,
        agent_id: int,
        conversation_id: Optional[int],
        customer_profile_id: Optional[int],
        priority: str,
        reason: str,
        summary: str,
        details: Dict[str, Any]
    ) -> Escalation:
        async with await self.get_session() as db:
            esc = Escalation(
                agent_id=agent_id,
                conversation_id=conversation_id,
                customer_profile_id=customer_profile_id,
                priority=priority,
                status="open",
                reason=reason,
                summary=summary,
                details=details or {}
            )
            db.add(esc)
            await db.commit()
            await db.refresh(esc)
            return esc

    async def build_context_summary(
        self,
        conversation_id: Optional[int],
        customer_profile_id: Optional[int],
        last_user_message: str = "",
        last_agent_response: str = ""
    ) -> Dict[str, Any]:
        """Assemble a concise summary packet with history and key facts."""

        summary_lines: List[str] = []
        convo_snippets: List[Dict[str, Any]] = []
        profile_data: Dict[str, Any] = {}

        if customer_profile_id:
            ctx = await memory_service.get_contextual_memory(customer_profile_id=customer_profile_id, query_text="")
            profile = ctx.get("customer_profile", {})
            profile_data = profile
            summary_lines.append(
                f"Customer: {profile.get('name', 'Unknown')} (Engagement: {profile.get('engagement_level', 'n/a')}, Journey: {profile.get('journey_stage', 'n/a')})"
            )
            if profile.get("primary_interests"):
                summary_lines.append("Primary interests: " + ", ".join(profile.get("primary_interests")[:3]))
            if profile.get("pain_points"):
                summary_lines.append("Known pain points: " + ", ".join(profile.get("pain_points")[:3]))

        if conversation_id:
            # Grab last few messages for context
            msgs = await db_service.get_conversation_messages(conversation_id=conversation_id, limit=20)
            for m in msgs[-10:]:  # last 10
                convo_snippets.append({
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.created_at.isoformat() if m.created_at else None,
                })
            if msgs:
                first_ts = msgs[0].created_at.isoformat() if msgs[0].created_at else ""
                summary_lines.append(f"Conversation since: {first_ts}; messages: {len(msgs)}")

        if last_user_message:
            summary_lines.append(f"Last user message: {last_user_message[:200]}")
        if last_agent_response:
            summary_lines.append(f"Last agent response: {last_agent_response[:200]}")

        return {
            "summary": " | ".join(summary_lines) if summary_lines else "Escalation requested",
            "conversation_history": convo_snippets,
            "customer_profile": profile_data,
        }

    async def list_escalations(
        self,
        agent_id: int,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[Escalation]:
        async with await self.get_session() as db:
            query = select(Escalation).where(Escalation.agent_id == agent_id).order_by(desc(Escalation.created_at)).limit(limit)
            if status:
                query = select(Escalation).where(and_(Escalation.agent_id == agent_id, Escalation.status == status)).order_by(desc(Escalation.created_at)).limit(limit)
            result = await db.execute(query)
            return result.scalars().all()

    async def resolve_escalation(self, escalation_id: int) -> bool:
        async with await self.get_session() as db:
            result = await db.execute(select(Escalation).where(Escalation.id == escalation_id))
            esc = result.scalar_one_or_none()
            if not esc:
                return False
            esc.status = "resolved"
            await db.commit()
            return True


# Global instance
escalation_service = EscalationService()

