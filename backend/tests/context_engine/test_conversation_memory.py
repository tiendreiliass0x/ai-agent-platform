"""
Tests for ConversationMemory
"""

from app.context_engine.conversation_memory import ConversationMemory


def test_add_turn_and_window():
    memory = ConversationMemory(max_turns=3)
    memory.add_turn("user", "Hello there")
    memory.add_turn("assistant", "Hi! How can I help?")
    memory.add_turn("user", "Tell me about Pinecone vector database.")
    memory.add_turn("assistant", "Pinecone is a managed vector store.")

    recent = memory.get_recent_context(window=2)
    assert len(recent) == 2
    assert recent[0].role == "user"
    assert "Pinecone vector database" in recent[0].content


def test_reference_resolution():
    memory = ConversationMemory()
    memory.add_turn("user", "We integrated HubSpot yesterday.")
    memory.add_turn("assistant", "Great! HubSpot CRM works well for marketing.")

    result = memory.resolve_references("How does it sync contacts?")
    assert result.get("it") in {"HubSpot", "HubSpot CRM"}


def test_history_compression():
    memory = ConversationMemory()
    for i in range(10):
        memory.add_turn("user" if i % 2 == 0 else "assistant", f"Message {i}")

    summary = memory.compress_history(max_tokens=15)
    assert "USER: Message" in summary
    assert len(summary.split()) <= 15 + 5  # allow minor buffer for role prefixes
