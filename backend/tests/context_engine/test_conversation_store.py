import asyncio

import pytest
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

from app.core.database import Base
from app.context_engine.conversation_store import ConversationMemoryStore
from app.context_engine.conversation_memory import ConversationMemory


@pytest.fixture
async def session_factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)

    async with engine.begin() as conn:
        await conn.execute(text("PRAGMA foreign_keys=OFF"))
        await conn.run_sync(Base.metadata.create_all)

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def factory() -> AsyncSession:
        return async_session()

    yield factory

    await engine.dispose()


@pytest.mark.asyncio
async def test_persistent_conversation_memory(session_factory):
    store = ConversationMemoryStore(session_factory=session_factory)

    conversation_id = await store.get_or_create_conversation(
        session_id="session-123",
        agent_id=1,
        metadata={"channel": "web"}
    )

    memory = await ConversationMemory.from_store(
        store=store,
        conversation_id=conversation_id,
        max_turns=5
    )

    await memory.add_turn_async("user", "Hello there")
    await memory.add_turn_async("assistant", "Hi! How can I help?", metadata={"source": "llm"})

    # Reload to ensure persistence
    reloaded = await ConversationMemory.from_store(
        store=store,
        conversation_id=conversation_id,
        max_turns=5
    )

    assert len(reloaded.history) == 2
    assert reloaded.history[0].role == "user"
    assert reloaded.history[1].metadata.get("source") == "llm"
