"""
Persistent storage utilities for conversation memory.
"""

from __future__ import annotations

from typing import Callable, Awaitable, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .conversation_memory import ConversationTurn
from ..models.conversation import Conversation
from ..models.message import Message
from ..core.database import get_db_session


class ConversationMemoryStore:
    """Persist conversation turns to the database."""

    def __init__(
        self,
        session_factory: Callable[[], Awaitable[AsyncSession]] = get_db_session
    ) -> None:
        self._session_factory = session_factory
        self._resolved_factory: Optional[Callable[[], Awaitable[AsyncSession]]] = None

    async def _get_session(self) -> AsyncSession:
        factory = await self._resolve_factory()
        maybe_session = await factory()
        if callable(maybe_session):
            return await maybe_session()
        return maybe_session

    async def _resolve_factory(self) -> Callable[[], Awaitable[AsyncSession]]:
        if self._resolved_factory is not None:
            return self._resolved_factory

        factory = self._session_factory
        if hasattr(factory, "__anext__"):
            factory = await factory.__anext__()
        self._resolved_factory = factory
        return factory

    async def get_or_create_conversation(
        self,
        session_id: str,
        agent_id: int,
        user_id: Optional[int] = None,
        customer_profile_id: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> int:
        session = await self._get_session()
        try:
            result = await session.execute(
                select(Conversation).where(Conversation.session_id == session_id)
            )
            conversation = result.scalar_one_or_none()

            if conversation:
                return conversation.id

            conversation = Conversation(
                session_id=session_id,
                agent_id=agent_id,
                user_id=user_id,
                customer_profile_id=customer_profile_id,
                conv_metadata=metadata or {},
            )
            session.add(conversation)
            await session.commit()
            await session.refresh(conversation)
            return conversation.id
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def append_turn(
        self,
        conversation_id: int,
        turn: ConversationTurn
    ) -> None:
        session = await self._get_session()
        try:
            message = Message(
                conversation_id=conversation_id,
                role=turn.role,
                content=turn.content,
                msg_metadata=turn.metadata or {},
            )
            session.add(message)
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def load_history(
        self,
        conversation_id: int,
        limit: Optional[int] = None
    ) -> List[ConversationTurn]:
        session = await self._get_session()
        try:
            stmt = (
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at.asc())
            )
            if limit:
                stmt = stmt.limit(limit)

            result = await session.execute(stmt)
            messages = result.scalars().all()
            return [
                ConversationTurn(
                    role=message.role,
                    content=message.content,
                    metadata=message.msg_metadata or {},
                )
                for message in messages
            ]
        finally:
            await session.close()
