from __future__ import annotations

import logging
from typing import Optional
from uuid import uuid4

from app.context_engine.conversation_memory import ConversationMemory
from app.context_engine.conversation_store import ConversationMemoryStore

logger = logging.getLogger(__name__)


class EphemeralMemoryProvider:
    """In-memory conversation history provider used until persistence arrives."""

    def __init__(self, max_turns: int = 50) -> None:
        self._max_turns = max_turns

    async def load(self, context) -> ConversationMemory:
        session_label = context.session_id or context.conversation_id or uuid4().hex
        memory = ConversationMemory(max_turns=self._max_turns, conversation_id=session_label)
        return memory

    async def save(self, context, memory: Optional[ConversationMemory]) -> None:
        return None


class PersistentMemoryProvider:
    """Conversation memory provider backed by the ConversationMemoryStore."""

    def __init__(
        self,
        store: Optional[ConversationMemoryStore] = None,
        max_turns: int = 50,
        fallback: Optional[EphemeralMemoryProvider] = None,
    ) -> None:
        self.store = store or ConversationMemoryStore()
        self.max_turns = max_turns
        self.fallback = fallback or EphemeralMemoryProvider(max_turns=max_turns)

    async def load(self, context) -> ConversationMemory:
        agent_id = self._extract_agent_id(context)
        if agent_id is None:
            return await self.fallback.load(context)

        try:
            conversation_id = await self._resolve_conversation_id(context, agent_id)
            memory = await ConversationMemory.from_store(
                store=self.store,
                conversation_id=conversation_id,
                max_turns=self.max_turns,
            )
            context.conversation_id = str(conversation_id)
            return memory
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Falling back to ephemeral memory provider due to persistence error: %s",
                exc,
            )
            return await self.fallback.load(context)

    async def save(self, context, memory: Optional[ConversationMemory]) -> None:
        if memory is None or memory.store is None:
            return None
        # ConversationMemory.add_turn_async already persists individual turns.
        return None

    async def _resolve_conversation_id(self, context, agent_id: int) -> int:
        existing = self._parse_conversation_id(context.conversation_id)
        if existing is not None:
            return existing

        session_id = context.session_id or f"session-{uuid4().hex}"
        user_id = getattr(context.user, "id", None)
        customer_profile_id = context.metadata.get("customer_profile_id")
        metadata = context.metadata.get("conversation_metadata")

        conversation_id = await self.store.get_or_create_conversation(
            session_id=session_id,
            agent_id=agent_id,
            user_id=user_id,
            customer_profile_id=customer_profile_id,
            metadata=metadata,
        )
        return conversation_id

    @staticmethod
    def _parse_conversation_id(value: Optional[str]) -> Optional[int]:
        if not value:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_agent_id(context) -> Optional[int]:
        agent_id = context.metadata.get("agent_id") if context.metadata else None
        if agent_id is None:
            return None
        try:
            return int(agent_id)
        except (TypeError, ValueError):
            return None
