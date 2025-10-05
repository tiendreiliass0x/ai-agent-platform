"""
Context Service
Advanced context synthesis, ranking, and optimization for intelligent agents.
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass

from app.services.gemini_service import gemini_service
from app.services.memory_service import memory_service


@dataclass
class ContextChunk:
    """Structured context chunk with metadata"""
    content: str
    source_type: str  # 'memory', 'document', 'profile', 'history'
    relevance_score: float
    importance: float
    recency_score: float
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = None


class ContextEngine:
    """Advanced context synthesis and optimization for intelligent agents"""

    def __init__(self):
        self.max_context_tokens = 8000  # Adaptive based on model
        self.relevance_threshold = 0.7
        self.chunk_size = 200

    async def optimize_context(
        self,
        query: str,
        customer_profile_id: int,
        agent_id: int,
        document_context: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Optimize context using advanced ranking and synthesis"""

        # 1. Generate query embedding for semantic similarity
        query_embedding = await self._get_query_embedding(query)

        # 2. Gather all context sources
        context_chunks = await self._gather_context_sources(
            query, query_embedding, customer_profile_id, agent_id, document_context
        )

        # 3. Advanced ranking and selection
        selected_context = await self._rank_and_select_context(
            context_chunks, query_embedding, query
        )

        # 4. Intelligent context synthesis
        synthesized_context = await self._synthesize_intelligent_context(
            selected_context, query, customer_profile_id
        )

        # 5. Context optimization and compression
        optimized_context = await self._optimize_context_window(
            synthesized_context, query
        )

        return {
            "context": optimized_context,
            "context_quality_score": self._calculate_context_quality(selected_context),
            "chunks_used": len(selected_context),
            "total_chunks_available": len(context_chunks),
            "context_efficiency": len(optimized_context) / self.max_context_tokens
        }

    async def _get_query_embedding(self, query: str) -> List[float]:
        """Get semantic embedding for query"""
        try:
            return await gemini_service.generate_query_embedding(query)
        except Exception:
            # Fallback to simple text processing
            return [0.0] * 768

    async def _gather_context_sources(
        self,
        query: str,
        query_embedding: List[float],
        customer_profile_id: int,
        agent_id: int,
        document_context: List[Dict[str, Any]]
    ) -> List[ContextChunk]:
        """Gather context from all sources with embeddings"""

        chunks = []

        # 1. Customer profile context
        memory_context = await memory_service.get_contextual_memory(
            customer_profile_id, query
        )

        if memory_context.get('customer_profile'):
            profile = memory_context['customer_profile']
            profile_content = f"""Customer: {profile['name']} | Style: {profile['communication_style']} |
            Level: {profile['technical_level']} | Interests: {', '.join(profile['primary_interests'])}"""

            chunks.append(ContextChunk(
                content=profile_content,
                source_type='profile',
                relevance_score=1.0,  # Always relevant
                importance=0.9,
                recency_score=1.0
            ))

        # 2. Memory-based factual context
        if memory_context.get('factual_memories'):
            for memory in memory_context['factual_memories']:
                chunks.append(ContextChunk(
                    content=f"{memory['key']}: {memory['value']}",
                    source_type='memory',
                    relevance_score=await self._calculate_semantic_relevance(
                        f"{memory['key']} {memory['value']}", query_embedding
                    ),
                    importance=memory.get('importance', 0.5),
                    recency_score=self._calculate_recency_score(memory.get('created_at'))
                ))

        # 3. Behavioral insights
        if memory_context.get('behavioral_insights'):
            for insight in memory_context['behavioral_insights']:
                chunks.append(ContextChunk(
                    content=f"Behavior: {insight['pattern']} - {insight['description']}",
                    source_type='behavior',
                    relevance_score=await self._calculate_semantic_relevance(
                        insight['description'], query_embedding
                    ),
                    importance=0.7,
                    recency_score=self._calculate_recency_score(insight.get('created_at'))
                ))

        # 4. Document context (when available)
        if document_context:
            for doc in document_context:
                chunks.append(ContextChunk(
                    content=doc['content'],
                    source_type='document',
                    relevance_score=doc.get('similarity_score', 0.5),
                    importance=0.8,
                    recency_score=0.5  # Documents don't decay
                ))

        # 5. Recent conversation history
        if memory_context.get('conversation_history'):
            for conv in memory_context['conversation_history'][-3:]:  # Last 3 conversations
                chunks.append(ContextChunk(
                    content=f"Previous conversation: {conv['summary']}",
                    source_type='history',
                    relevance_score=await self._calculate_semantic_relevance(
                        conv['summary'], query_embedding
                    ),
                    importance=0.6,
                    recency_score=self._calculate_recency_score(conv['date'])
                ))

        return chunks

    async def _calculate_semantic_relevance(
        self,
        content: str,
        query_embedding: List[float]
    ) -> float:
        """Calculate semantic relevance using embeddings"""
        try:
            content_embedding = await gemini_service.generate_embeddings([content])
            if content_embedding and content_embedding[0]:
                similarity = cosine_similarity(
                    [query_embedding],
                    [content_embedding[0]]
                )[0][0]
                return float(similarity)
        except Exception:
            pass

        # Fallback to keyword matching
        return self._keyword_relevance(content, query_embedding)

    def _keyword_relevance(self, content: str, query: str) -> float:
        """Fallback keyword-based relevance"""
        content_words = set(content.lower().split())
        query_words = set(str(query).lower().split())
        overlap = len(content_words.intersection(query_words))
        return min(overlap / max(len(query_words), 1), 1.0)

    def _calculate_recency_score(self, created_at: Any) -> float:
        """Calculate recency score with exponential decay"""
        if not created_at:
            return 0.5

        try:
            if isinstance(created_at, str):
                date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            else:
                date = created_at

            days_ago = (datetime.utcnow() - date.replace(tzinfo=None)).days
            # Exponential decay: score = e^(-days/30)
            return max(np.exp(-days_ago / 30), 0.1)
        except Exception:
            return 0.5

    async def _rank_and_select_context(
        self,
        chunks: List[ContextChunk],
        query_embedding: List[float],
        query: str
    ) -> List[ContextChunk]:
        """Advanced ranking and selection of context chunks"""

        # 1. Calculate composite scores
        for chunk in chunks:
            chunk.composite_score = self._calculate_composite_score(chunk, query)

        # 2. Sort by composite score
        chunks.sort(key=lambda x: x.composite_score, reverse=True)

        # 3. Diversity-aware selection (avoid redundancy)
        selected = []
        used_content = set()

        for chunk in chunks:
            # Check for semantic diversity
            if self._is_diverse_enough(chunk, selected) and chunk.composite_score > self.relevance_threshold:
                selected.append(chunk)
                used_content.add(chunk.content[:100])  # Track content signature

                # Stop if we've gathered enough high-quality context
                if len(selected) >= 15:  # Max chunks
                    break

        return selected

    def _calculate_composite_score(self, chunk: ContextChunk, query: str) -> float:
        """Calculate composite relevance score"""
        # Weighted combination of factors
        weights = {
            'relevance': 0.4,
            'importance': 0.3,
            'recency': 0.2,
            'source_priority': 0.1
        }

        source_priorities = {
            'profile': 1.0,
            'memory': 0.9,
            'document': 0.8,
            'behavior': 0.7,
            'history': 0.6
        }

        score = (
            chunk.relevance_score * weights['relevance'] +
            chunk.importance * weights['importance'] +
            chunk.recency_score * weights['recency'] +
            source_priorities.get(chunk.source_type, 0.5) * weights['source_priority']
        )

        return score

    def _is_diverse_enough(self, chunk: ContextChunk, selected: List[ContextChunk]) -> bool:
        """Check if chunk adds diversity to selected context"""
        if not selected:
            return True

        # Simple diversity check - avoid too similar content
        for existing in selected[-3:]:  # Check last 3 for efficiency
            if self._content_similarity(chunk.content, existing.content) > 0.8:
                return False

        return True

    def _content_similarity(self, content1: str, content2: str) -> float:
        """Calculate content similarity"""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0

    async def _synthesize_intelligent_context(
        self,
        chunks: List[ContextChunk],
        query: str,
        customer_profile_id: int
    ) -> str:
        """Intelligently synthesize context with adaptive formatting"""

        if not chunks:
            return "No relevant context available."

        # Group chunks by source type for better organization
        grouped_chunks = {}
        for chunk in chunks:
            if chunk.source_type not in grouped_chunks:
                grouped_chunks[chunk.source_type] = []
            grouped_chunks[chunk.source_type].append(chunk)

        context_sections = []

        # 1. Customer profile (always first for personalization)
        if 'profile' in grouped_chunks:
            context_sections.append("CUSTOMER PROFILE:")
            for chunk in grouped_chunks['profile']:
                context_sections.append(f"• {chunk.content}")

        # 2. Key memories and facts
        if 'memory' in grouped_chunks:
            top_memories = sorted(grouped_chunks['memory'],
                                key=lambda x: x.composite_score, reverse=True)[:5]
            if top_memories:
                context_sections.append("\nKEY CUSTOMER FACTS:")
                for chunk in top_memories:
                    context_sections.append(f"• {chunk.content}")

        # 3. Behavioral insights
        if 'behavior' in grouped_chunks:
            context_sections.append("\nBEHAVIOR PATTERNS:")
            for chunk in grouped_chunks['behavior'][:3]:
                context_sections.append(f"• {chunk.content}")

        # 4. Relevant knowledge/documents
        if 'document' in grouped_chunks:
            top_docs = sorted(grouped_chunks['document'],
                            key=lambda x: x.relevance_score, reverse=True)[:3]
            if top_docs:
                context_sections.append("\nRELEVANT KNOWLEDGE:")
                for chunk in top_docs:
                    # Truncate long documents intelligently
                    content = chunk.content[:300] + "..." if len(chunk.content) > 300 else chunk.content
                    context_sections.append(f"• {content}")

        # 5. Recent conversation context
        if 'history' in grouped_chunks:
            context_sections.append("\nRECENT CONTEXT:")
            for chunk in grouped_chunks['history'][-2:]:  # Last 2 conversations
                context_sections.append(f"• {chunk.content}")

        return "\n".join(context_sections)

    async def _optimize_context_window(self, context: str, query: str) -> str:
        """Optimize context to fit within token limits"""

        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        estimated_tokens = len(context) // 4

        if estimated_tokens <= self.max_context_tokens:
            return context

        # Context is too long - intelligent compression needed
        lines = context.split('\n')
        sections = {}
        current_section = None

        for line in lines:
            if line.endswith(':') and not line.startswith('•'):
                current_section = line
                sections[current_section] = []
            elif current_section:
                sections[current_section].append(line)

        # Prioritize sections
        section_priorities = {
            'CUSTOMER PROFILE:': 1.0,
            'KEY CUSTOMER FACTS:': 0.9,
            'RELEVANT KNOWLEDGE:': 0.8,
            'BEHAVIOR PATTERNS:': 0.7,
            'RECENT CONTEXT:': 0.6
        }

        # Build optimized context within limits
        optimized_lines = []
        remaining_tokens = self.max_context_tokens

        for section, priority in sorted(section_priorities.items(), key=lambda x: x[1], reverse=True):
            if section in sections and remaining_tokens > 100:
                optimized_lines.append(section)

                # Add items from this section until we hit limits
                for item in sections[section]:
                    item_tokens = len(item) // 4
                    if remaining_tokens - item_tokens > 50:  # Keep some buffer
                        optimized_lines.append(item)
                        remaining_tokens -= item_tokens
                    else:
                        break

        return '\n'.join(optimized_lines)

    def _calculate_context_quality(self, chunks: List[ContextChunk]) -> float:
        """Calculate overall context quality score"""
        if not chunks:
            return 0.0

        # Quality factors
        avg_relevance = np.mean([c.relevance_score for c in chunks])
        avg_importance = np.mean([c.importance for c in chunks])
        source_diversity = len(set(c.source_type for c in chunks)) / 5  # 5 possible sources

        quality_score = (avg_relevance * 0.5 + avg_importance * 0.3 + source_diversity * 0.2)
        return min(quality_score, 1.0)


# Global instance
context_engine = ContextEngine()