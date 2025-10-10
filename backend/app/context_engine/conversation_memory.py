"""
Conversation Memory - Multi-turn Context Tracking with DB Persistence

Responsibilities:
    - Maintain structured conversation history (in-memory + database)
    - Resolve anaphora/pronoun references with lightweight heuristics
    - Support history compression for long-running conversations
    - LLM-powered summarization for context management
    - Persistent storage with ConversationMemoryStore
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import re

if TYPE_CHECKING:
    from .conversation_store import ConversationMemoryStore
    from ..services.llm_service_interface import LLMServiceInterface


@dataclass
class ConversationTurn:
    role: str
    content: str
    metadata: Dict[str, str] = field(default_factory=dict)


class ConversationMemory:
    """Track conversation state with database persistence and LLM summarization."""

    def __init__(
        self,
        max_turns: int = 50,
        store: Optional["ConversationMemoryStore"] = None,
        conversation_id: Optional[int] = None,
        llm_service: Optional["LLMServiceInterface"] = None
    ) -> None:
        """
        Initialize conversation memory.

        Args:
            max_turns: Maximum number of turns to keep in memory
            store: Database store for persistence
            conversation_id: Existing conversation ID (if loading from DB)
            llm_service: LLM service for summarization (optional)
        """
        self.max_turns = max_turns
        self.history: List[ConversationTurn] = []
        self._entity_pattern = re.compile(r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)\b")
        self.store = store
        self.conversation_id = conversation_id
        self.llm_service = llm_service
        self._summary_cache: Optional[str] = None

    @classmethod
    async def from_store(
        cls,
        store: "ConversationMemoryStore",
        conversation_id: int,
        max_turns: int = 50,
        llm_service: Optional["LLMServiceInterface"] = None
    ) -> "ConversationMemory":
        """
        Load conversation memory from database.

        Args:
            store: Database store
            conversation_id: Conversation ID to load
            max_turns: Maximum turns to keep in memory
            llm_service: Optional LLM service for summarization

        Returns:
            ConversationMemory instance with loaded history
        """
        memory = cls(
            max_turns=max_turns,
            store=store,
            conversation_id=conversation_id,
            llm_service=llm_service
        )
        stored_turns = await store.load_history(conversation_id, limit=max_turns)
        for turn in stored_turns:
            memory._append_turn(turn)
        return memory

    def _append_turn(self, turn: ConversationTurn) -> None:
        self.history.append(turn)
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns :]

    def add_turn(self, role: str, content: str, metadata: Optional[Dict[str, str]] = None) -> ConversationTurn:
        turn = ConversationTurn(role=role, content=content.strip(), metadata=metadata or {})
        self._append_turn(turn)
        return turn

    async def add_turn_async(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> ConversationTurn:
        turn = self.add_turn(role, content, metadata)
        if self.store and self.conversation_id is not None:
            await self.store.append_turn(self.conversation_id, turn)
        return turn

    def get_recent_context(self, window: int = 6) -> List[ConversationTurn]:
        return self.history[-window:]

    def resolve_references(self, query: str) -> Dict[str, Optional[str]]:
        """
        Resolve pronouns like "it", "they", "that" using recent entities.

        Returns:
            Mapping from pronoun to inferred entity (or None if unresolved).
        """
        pronouns = {"it", "they", "them", "that", "this"}
        words = query.lower().split()
        present_pronouns = [word for word in words if word in pronouns]

        if not present_pronouns:
            return {}

        recent_entities = self._extract_recent_entities()
        if not recent_entities:
            return {pronoun: None for pronoun in present_pronouns}

        resolved = {}
        fallback_entity = recent_entities[-1]
        for pronoun in present_pronouns:
            resolved[pronoun] = fallback_entity
        return resolved

    def compress_history(self, max_tokens: int = 600) -> str:
        """
        Provide a condensed summary of the conversation history (template-based).

        Args:
            max_tokens: Rough token target for compression.

        Returns:
            Concatenated string representation of recent turns.
        """
        summary: List[str] = []
        token_budget = max_tokens

        for turn in reversed(self.history):
            turn_tokens = len(turn.content.split())
            role_tokens = 1  # Account for role prefix
            if token_budget - (turn_tokens + role_tokens) < 0:
                break
            summary.append(f"{turn.role.upper()}: {turn.content}")
            token_budget -= (turn_tokens + role_tokens)

        return "\n".join(reversed(summary))

    async def generate_summary(self, force_refresh: bool = False) -> str:
        """
        Generate an LLM-powered summary of the conversation.

        This is useful for:
        - Long conversations that exceed context windows
        - Quick context loading for agents
        - Analytics and conversation understanding

        Args:
            force_refresh: Force regeneration even if cached summary exists

        Returns:
            2-3 sentence summary of the conversation
        """
        # Return cached summary if available and not forcing refresh
        if self._summary_cache and not force_refresh:
            return self._summary_cache

        # If no LLM or conversation is too short, use template
        if not self.llm_service or len(self.history) < 3:
            return self._generate_template_summary()

        # Build conversation text
        history_text = "\n".join([
            f"{turn.role.upper()}: {turn.content}"
            for turn in self.history
        ])

        prompt = f"""
Summarize this conversation in 2-3 sentences. Focus on:
1. What the user wanted to know or accomplish
2. Key information provided
3. Current conversation status (resolved, ongoing, needs escalation)

Conversation:
{history_text}

Summary (2-3 sentences):
"""

        try:
            summary = await self.llm_service.generate_response(
                prompt=prompt,
                temperature=0.2,  # Deterministic summary
                max_tokens=150
            )

            # Cache the summary
            self._summary_cache = summary.strip()
            return self._summary_cache

        except Exception as e:
            print(f"LLM summarization failed: {e}")
            return self._generate_template_summary()

    def _generate_template_summary(self) -> str:
        """Generate a simple template-based summary (fallback)"""
        if not self.history:
            return "Empty conversation - no turns recorded yet."

        if len(self.history) < 3:
            return "Conversation just started - not enough context for summary."

        # Count user and assistant turns
        user_turns = sum(1 for t in self.history if t.role == "user")
        assistant_turns = sum(1 for t in self.history if t.role == "assistant")

        # Get first user message
        first_user = next((t.content for t in self.history if t.role == "user"), "")
        first_question = first_user[:100] + "..." if len(first_user) > 100 else first_user

        return (
            f"Conversation with {user_turns} user messages and {assistant_turns} responses. "
            f"Initial question: \"{first_question}\""
        )

    async def get_context_for_retrieval(self, window: int = 6) -> Dict[str, Any]:
        """
        Get conversation context formatted for retrieval augmentation.

        Args:
            window: Number of recent turns to include

        Returns:
            Dictionary with conversation context, entities, and summary
        """
        recent = self.get_recent_context(window)

        return {
            "recent_turns": [
                {"role": t.role, "content": t.content}
                for t in recent
            ],
            "entities": self._extract_recent_entities(),
            "summary": await self.generate_summary() if self.llm_service else self._generate_template_summary(),
            "turn_count": len(self.history)
        }

    def _extract_recent_entities(self) -> List[str]:
        entities: List[str] = []
        for turn in reversed(self.history[-10:]):
            for match in self._entity_pattern.finditer(turn.content):
                entity = match.group(1).strip()
                if entity.lower() not in {"what", "how", "why"} and entity not in entities:
                    entities.append(entity)
        return entities
