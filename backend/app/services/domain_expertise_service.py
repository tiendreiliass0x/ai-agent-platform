"""
Domain Expertise Service - Revolutionary concierge intelligence with personas and web search
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from app.services.database_service import db_service
from app.services.document_processor import DocumentProcessor
from app.services.gemini_service import gemini_service
from app.services.web_search_service import web_search_service
from app.services.reranker_service import reranker_service
from app.core.logging_config import get_logger, LoggerMixin

logger = get_logger(__name__)


@dataclass
class RetrievalCandidate:
    """Search result candidate"""
    doc_id: str
    content: str
    score: float
    source_url: Optional[str] = None
    doc_title: Optional[str] = None
    timestamp: Optional[datetime] = None
    source_type: str = "internal"  # "internal", "web", "site"


@dataclass
class GroundedResponse:
    """Response with grounding and confidence information"""
    answer: str
    confidence_score: float
    sources: List[Dict[str, Any]]
    grounding_mode: str
    persona_applied: str
    escalation_suggested: bool = False
    web_search_used: bool = False


class DomainExpertiseService(LoggerMixin):
    """Orchestrates domain expertise with personas, knowledge packs, and web search"""

    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.confidence_thresholds = {
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4
        }

    async def answer_with_domain_expertise(
        self,
        message: str,
        agent: Any,
        organization: Any,
        conversation_context: Dict[str, Any] = None
    ) -> GroundedResponse:
        """Main entry point for domain expertise responses"""

        if not agent.domain_expertise_enabled:
            # Fallback to standard response
            return await self._standard_response(message, agent)

        # Load persona and knowledge pack
        persona = await self._load_persona(agent.persona_id, organization.id)
        knowledge_pack = await self._load_knowledge_pack(agent.knowledge_pack_id, organization.id)

        # 1. Hybrid retrieval from knowledge pack sources
        candidates = await self._hybrid_retrieve(
            message=message,
            organization_id=organization.id,
            knowledge_pack=knowledge_pack,
            limit=50
        )

        # 2. Rerank with semantic cross-encoder + recency adjustments
        ranked_candidates = await self._rerank_candidates(candidates, message, knowledge_pack)

        # 3. Grounding guard - check if we have sufficient support
        has_support = self._has_sufficient_support(ranked_candidates, agent.grounding_mode)

        # 4. Web search if needed and enabled
        if (not has_support or self._needs_fresh_info(message)) and agent.tool_policy.get("web_search"):
            web_results = await self._web_search(
                message=message,
                site_whitelist=agent.tool_policy.get("site_search", []),
                context=conversation_context
            )
            ranked_candidates = self._merge_and_rerank(ranked_candidates, web_results)

        # 5. Final grounding check
        if agent.grounding_mode == "strict" and not self._has_sufficient_support(ranked_candidates):
            return await self._escalation_response(message, agent, persona)

        # 6. Plan answer based on persona tactics
        answer_plan = await self._plan_answer(message, ranked_candidates, persona, conversation_context)

        # 7. Synthesize final response with citations and confidence
        return await self._synthesize_response(
            plan=answer_plan,
            candidates=ranked_candidates,
            persona=persona,
            agent=agent
        )

    async def _hybrid_retrieve(
        self,
        message: str,
        organization_id: int,
        knowledge_pack: Optional[Any],
        limit: int = 50
    ) -> List[RetrievalCandidate]:
        """Combine vector and keyword search"""

        agents = await db_service.get_organization_agents(organization_id)
        if not agents:
            return []

        per_agent_limit = max(3, limit // max(len(agents), 1))
        candidates = []

        allowed_document_ids = None
        if knowledge_pack:
            allowed_document_ids = {
                source.source_id for source in knowledge_pack.sources if getattr(source, "is_active", True)
            }

        for agent in agents:
            search_results = await self.document_processor.search_similar_content(
                query=message,
                agent_id=agent.id,
                top_k=per_agent_limit
            )

            for result in search_results:
                metadata = result.get("metadata", {})
                document_id = metadata.get("document_id")

                if allowed_document_ids is not None and document_id not in allowed_document_ids:
                    continue

                candidates.append(
                    RetrievalCandidate(
                        doc_id=str(document_id or metadata.get("chunk_id") or metadata.get("source", "doc")),
                        content=result.get("text", ""),
                        score=result.get("score", 0.0),
                        source_url=metadata.get("source_url"),
                        doc_title=metadata.get("source") or metadata.get("title"),
                        timestamp=None,
                        source_type="internal"
                    )
                )

        # TODO: Add keyword search (BM25) and merge
        # For now, return vector results
        return candidates

    async def _rerank_candidates(
        self,
        candidates: List[RetrievalCandidate],
        query: str,
        knowledge_pack: Optional[Any]
    ) -> List[RetrievalCandidate]:
        """Rerank with cross-encoder and freshness boost"""

        if not candidates:
            return []

        indexed_candidates = {
            idx: candidate for idx, candidate in enumerate(candidates)
        }

        items = [
            {
                "text": candidate.content,
                "score": candidate.score,
                "metadata": {"candidate_index": idx}
            }
            for idx, candidate in indexed_candidates.items()
            if candidate.content
        ]

        reranked_dicts = await reranker_service.rerank(query, items, top_k=12)

        enriched: List[RetrievalCandidate] = []
        for item in reranked_dicts:
            metadata = item.get("metadata") or {}
            candidate_idx = metadata.get("candidate_index")
            original = indexed_candidates.get(candidate_idx)
            if original is None:
                continue

            combined_score = item.get("combined_score")
            rerank_score = item.get("rerank_score")
            base_score = original.score
            final_score = combined_score or rerank_score or base_score

            if original.timestamp:
                days_old = (datetime.now() - original.timestamp).days
                recency_boost = max(0.0, 1.0 - (days_old / 30)) * 0.1
                final_score += recency_boost

            enriched.append(
                RetrievalCandidate(
                    doc_id=original.doc_id,
                    content=original.content,
                    score=final_score,
                    source_url=original.source_url,
                    doc_title=original.doc_title,
                    timestamp=original.timestamp,
                    source_type=original.source_type,
                )
            )

        enriched.sort(key=lambda cand: cand.score, reverse=True)
        return enriched[:12]

    def _has_sufficient_support(
        self,
        candidates: List[RetrievalCandidate],
        grounding_mode: str,
        min_sources: int = 2,
        min_score: float = 0.35
    ) -> bool:
        """Check if we have sufficient support for grounded response"""

        if grounding_mode == "blended":
            return len(candidates) > 0

        # Strict mode requires multiple high-quality sources
        high_quality = [c for c in candidates if c.score >= min_score]
        unique_docs = {c.doc_id for c in high_quality}
        return len(unique_docs) >= min_sources

    def _needs_fresh_info(self, message: str) -> bool:
        """Determine if query needs fresh web information"""

        fresh_keywords = [
            "latest", "recent", "current", "today", "now", "new", "update",
            "2024", "2025", "this year", "this month", "pricing", "price"
        ]

        message_lower = message.lower()
        return any(keyword in message_lower for keyword in fresh_keywords)

    async def _web_search(
        self,
        message: str,
        site_whitelist: List[str] = None,
        context: Dict[str, Any] = None,
        agent_id: Optional[int] = None
    ) -> List[RetrievalCandidate]:
        """Perform controlled web search"""

        try:
            self.log_info(
                "Starting web search",
                message=message[:100],
                site_whitelist=site_whitelist,
                agent_id=agent_id
            )

            results = await web_search_service.search(
                query=message,
                site_whitelist=site_whitelist,
                max_results=5,
                timeout_seconds=8,
                agent_id=agent_id
            )

            candidates = []
            for result in results:
                candidates.append(RetrievalCandidate(
                    doc_id=f"web_{result.url}",
                    content=result.snippet,
                    score=0.7,  # Web results get medium score
                    source_url=result.url,
                    doc_title=result.title,
                    timestamp=datetime.now(),
                    source_type="web"
                ))

            self.log_info(
                "Web search completed",
                results_count=len(candidates),
                agent_id=agent_id
            )
            return candidates

        except Exception as e:
            self.log_error(
                "Web search failed",
                error=str(e),
                message=message[:100],
                agent_id=agent_id
            )
            return []

    def _merge_and_rerank(
        self,
        internal_candidates: List[RetrievalCandidate],
        web_candidates: List[RetrievalCandidate]
    ) -> List[RetrievalCandidate]:
        """Merge internal and web results, avoiding duplicates"""

        # Simple merge - could be more sophisticated
        all_candidates = internal_candidates + web_candidates

        # Remove duplicates based on content similarity (simplified)
        seen_content = set()
        unique_candidates = []

        for candidate in all_candidates:
            content_hash = hash(candidate.content[:100])  # Simple dedup
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_candidates.append(candidate)

        # Re-sort by score
        unique_candidates.sort(key=lambda c: c.score, reverse=True)
        return unique_candidates[:15]  # Top 15 total

    async def _plan_answer(
        self,
        message: str,
        candidates: List[RetrievalCandidate],
        persona: Optional[Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Plan answer structure based on persona tactics"""

        if not persona:
            return {
                "structure": "direct",
                "steps": ["answer"],
                "original_message": message
            }

        tactics = persona.tactics or {}
        steps = tactics.get("steps", ["answer"])

        return {
            "structure": tactics.get("style", "direct"),
            "steps": steps,
            "communication_style": getattr(persona, "communication_style", {}) or {},
            "response_patterns": getattr(persona, "response_patterns", {}) or {},
            "original_message": message
        }

    async def _synthesize_response(
        self,
        plan: Dict[str, Any],
        candidates: List[RetrievalCandidate],
        persona: Optional[Any],
        agent: Any
    ) -> GroundedResponse:
        """Synthesize final response with persona, grounding, and confidence"""

        # Build context for LLM
        context_parts = []
        sources = []

        for i, candidate in enumerate(candidates[:8]):  # Top 8 sources
            context_parts.append(f"[Source {i+1}] {candidate.content}")
            sources.append({
                "id": candidate.doc_id,
                "title": candidate.doc_title or "Document",
                "url": candidate.source_url,
                "type": candidate.source_type,
                "score": candidate.score
            })

        context_text = "\n\n".join(context_parts)

        # Build system prompt with persona
        system_prompt = persona.system_prompt if persona else "You are a helpful assistant."

        if agent.grounding_mode == "strict":
            system_prompt += "\n\nIMPORTANT: You must only use information from the provided sources. If the sources don't contain enough information to answer fully, say so and suggest next steps."

        # Add persona-specific instructions
        if persona and persona.tactics:
            style = persona.tactics.get("style", "direct")
            steps = persona.tactics.get("steps", [])

            if style == "executive":
                system_prompt += "\n\nUse an executive communication style: concise, strategic, results-focused."
            elif style == "technical":
                system_prompt += "\n\nUse a technical communication style: precise, detailed, with specific examples."

            if steps:
                system_prompt += f"\n\nStructure your response following these steps: {', '.join(steps)}"

        # Generate response
        prompt = f"""Sources:
{context_text}

Question: {plan.get('original_message', 'User question')}

Please provide a comprehensive answer using the sources above. Include citations in the format [Source X] where appropriate."""

        try:
            response_text = await gemini_service.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.6,
                max_tokens=1000
            )

            # Calculate confidence score
            confidence = self._calculate_confidence(candidates, response_text)

            return GroundedResponse(
                answer=response_text,
                confidence_score=confidence,
                sources=sources,
                grounding_mode=agent.grounding_mode,
                persona_applied=persona.name if persona else "None",
                escalation_suggested=confidence < 0.5,
                web_search_used=any(s["type"] == "web" for s in sources)
            )

        except Exception as e:
            print(f"Response synthesis failed: {e}")
            return await self._fallback_response(candidates, agent)

    def _calculate_confidence(self, candidates: List[RetrievalCandidate], response: str) -> float:
        """Calculate confidence score for the response"""

        if not candidates:
            return 0.2

        # Simple confidence calculation based on source quality and quantity
        avg_score = sum(c.score for c in candidates) / len(candidates)
        source_count_factor = min(1.0, len(candidates) / 5.0)  # More sources = higher confidence

        confidence = (avg_score * 0.7) + (source_count_factor * 0.3)
        return min(0.95, confidence)  # Cap at 95%

    async def _escalation_response(self, message: str, agent: Any, persona: Optional[Any]) -> GroundedResponse:
        """Generate escalation response when grounding fails"""

        escalation_text = "I don't have enough specific information in my knowledge base to fully answer your question. "

        if persona:
            if persona.template_name == "sales_rep":
                escalation_text += "Let me connect you with our sales team who can provide detailed information."
            elif persona.template_name == "solutions_engineer":
                escalation_text += "I'd like to connect you with a solutions engineer who can dive deeper into the technical details."
            else:
                escalation_text += "Let me escalate this to a specialist who can help."

        return GroundedResponse(
            answer=escalation_text,
            confidence_score=0.3,
            sources=[],
            grounding_mode=agent.grounding_mode,
            persona_applied=persona.name if persona else "None",
            escalation_suggested=True
        )

    async def _standard_response(self, message: str, agent: Any) -> GroundedResponse:
        """Fallback to standard response when domain expertise is disabled"""

        # Use existing chat service logic
        response = await gemini_service.generate_response(
            prompt=message,
            system_prompt=agent.system_prompt or "You are a helpful assistant.",
            temperature=0.7
        )

        return GroundedResponse(
            answer=response,
            confidence_score=0.7,
            sources=[],
            grounding_mode="blended",
            persona_applied="Standard"
        )

    async def _fallback_response(self, candidates: List[RetrievalCandidate], agent: Any) -> GroundedResponse:
        """Fallback response when synthesis fails"""

        return GroundedResponse(
            answer="I apologize, but I'm having trouble processing your request right now. Please try again or contact support.",
            confidence_score=0.2,
            sources=[],
            grounding_mode=agent.grounding_mode,
            persona_applied="Fallback",
            escalation_suggested=True
        )

    # Helper methods for loading data
    async def _load_persona(self, persona_id: Optional[int], organization_id: int) -> Optional[Any]:
        """Load persona configuration"""
        if not persona_id:
            return None

        # TODO: Implement database query
        # For now, return None - will be implemented when database service is ready
        return None

    async def _load_knowledge_pack(self, pack_id: Optional[int], organization_id: int) -> Optional[Any]:
        """Load knowledge pack configuration"""
        if not pack_id:
            return None

        # TODO: Implement database query
        # For now, return None - will be implemented when database service is ready
        return None


# Global instance
domain_expertise_service = DomainExpertiseService()
