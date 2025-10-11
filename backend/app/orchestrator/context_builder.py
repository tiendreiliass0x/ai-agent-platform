from __future__ import annotations

import inspect
from typing import Any, Iterable, List, Optional, Sequence

from app.context_engine.conversation_memory import ConversationMemory, ConversationTurn
from app.tooling import ToolRegistry
from app.tooling.models import ToolManifest

from .models import AgentContext, AgentTask


class CacheAwareContextBuilder:
    """Construct deterministic, cache-friendly prompts for the reasoning engine."""

    STABLE_PREFIX = (
        "You are an AI business assistant with access to integrated tools.\n\n"
        "Your capabilities:\n"
        "- CRM operations (Salesforce, HubSpot)\n"
        "- Order management (Shopify, NetSuite)\n"
        "- Support ticketing (Zendesk, Intercom)\n"
        "- Payment processing (Stripe, PayPal)\n"
        "- Communication (Slack, Email)\n\n"
        "Guidelines:\n"
        "1. Always check permissions before suggesting actions\n"
        "2. Show confidence scores for uncertain information\n"
        "3. Cite sources for factual claims\n"
        "4. Ask for approval for destructive operations\n\n"
    )

    def __init__(
        self,
        tool_registry: ToolRegistry,
        knowledge_retriever: Any = None,
        retriever_limit: int = 5,
    ) -> None:
        self.tool_registry = tool_registry
        self.knowledge_retriever = knowledge_retriever
        self.retriever_limit = retriever_limit

    async def build(
        self,
        task: AgentTask,
        context: AgentContext,
        conversation_memory: Optional[ConversationMemory],
    ) -> str:
        query = task.description
        tools = await self.tool_registry.list_tools()
        retrieved = await self._retrieve_context(query, context, conversation_memory)
        history = self._conversation_history(conversation_memory)

        return self._format_context(
            query=query,
            retrieved_contexts=retrieved,
            conversation_history=history,
            tools=tools,
        )

    async def _retrieve_context(
        self,
        query: str,
        agent_context: AgentContext,
        conversation_memory: Optional[ConversationMemory],
    ) -> List[str]:
        if self.knowledge_retriever is None:
            return []

        payload = {
            "query": query,
            "limit": self.retriever_limit,
            "agent_context": agent_context,
            "conversation_memory": conversation_memory,
        }

        retriever = self.knowledge_retriever
        if callable(retriever):
            result = retriever(**payload)
        elif hasattr(retriever, "retrieve"):
            result = retriever.retrieve(**payload)
        else:
            return []

        if inspect.isawaitable(result):
            result = await result

        if result is None:
            return []
        if isinstance(result, (list, tuple)):
            return [str(item) for item in result[: self.retriever_limit]]
        return [str(result)]

    def _conversation_history(
        self,
        conversation_memory: Optional[ConversationMemory],
        window: int = 6,
    ) -> List[dict]:
        if not conversation_memory:
            return []
        turns: Sequence[ConversationTurn] = conversation_memory.get_recent_context(window)
        history: List[dict] = []
        for turn in turns:
            history.append({"role": turn.role, "content": turn.content})
        return history

    def _format_context(
        self,
        query: str,
        retrieved_contexts: Iterable[str],
        conversation_history: Iterable[dict],
        tools: Iterable[ToolManifest],
    ) -> str:
        parts: List[str] = [self.STABLE_PREFIX]

        parts.append("## Available Tools\n\n")
        for tool in sorted(tools, key=lambda manifest: manifest.name):
            parts.append(f"### {tool.name}\n")
            description = tool.description or "No description provided."
            parts.append(f"{description}\n")
            if tool.operations:
                for op_id in sorted(tool.operations.keys()):
                    operation = tool.operations[op_id]
                    summary = operation.description or "No description."
                    parts.append(
                        f"- {operation.op_id} [{operation.side_effect.value}]: {summary}\n"
                    )
            parts.append("\n")

        retrieved_list = [entry for entry in retrieved_contexts if entry]
        if retrieved_list:
            parts.append("## Relevant Knowledge\n\n")
            for index, entry in enumerate(retrieved_list, start=1):
                parts.append(f"[Context {index}]\n{entry}\n\n")

        history_list = [turn for turn in conversation_history if turn.get("content")]
        if history_list:
            parts.append("## Conversation History\n\n")
            for turn in history_list:
                role = (turn.get("role") or "unknown").upper()
                content = turn.get("content", "")
                parts.append(f"{role}: {content}\n\n")

        parts.append("## Current Query\n\n")
        parts.append(f"{query}\n")
        return "".join(parts)

    def get_cache_stats(self) -> dict:
        """Placeholder for future cache metric integration."""
        return {
            "stable_prefix_hit_rate": 0.95,
            "tool_catalog_hit_rate": 0.92,
            "avg_cache_savings_ms": 450,
        }
