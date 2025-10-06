"""
Domain Expertise Service - Revolutionary concierge intelligence with personas and web search
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta, timezone
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
    metadata: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = None


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

    async def _get_organization_agents(self, organization_id: int) -> List[Any]:
        """Safe wrapper around db_service.get_organization_agents"""

        try:
            agents = await db_service.get_organization_agents(organization_id)
            return list(agents or [])
        except Exception as exc:  # pragma: no cover - defensive logging
            self.log_error(
                "Failed to load organization agents",
                organization_id=organization_id,
                error=str(exc)
            )
            return []

    async def _retrieve_from_agent(
        self,
        query: str,
        agent: Any,
        top_k: int = 5,
        allowed_document_ids: Optional[set] = None
    ) -> List[RetrievalCandidate]:
        """Fetch knowledge for a single agent"""

        agent_id = getattr(agent, "id", None)
        if agent_id is None and isinstance(agent, dict):
            agent_id = agent.get("id")

        if agent_id is None:
            return []

        try:
            results = await self.document_processor.search_similar_content(
                query=query,
                agent_id=agent_id,
                top_k=top_k
            )
        except Exception as exc:
            self.log_warning(
                "Agent retrieval failed",
                agent_id=agent_id,
                error=str(exc)
            )
            raise

        candidates: List[RetrievalCandidate] = []
        for result in results or []:
            metadata = result.get("metadata") or {}
            document_id = (
                metadata.get("document_id")
                or metadata.get("chunk_id")
                or metadata.get("source")
                or f"{agent_id}_doc"
            )

            if allowed_document_ids is not None and document_id not in allowed_document_ids:
                continue

            timestamp = metadata.get("timestamp")
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                except ValueError:
                    timestamp = None

            candidates.append(
                RetrievalCandidate(
                    doc_id=str(document_id),
                    content=result.get("text", ""),
                    score=float(result.get("score", 0.0)),
                    source_url=metadata.get("source_url"),
                    doc_title=metadata.get("title") or metadata.get("source"),
                    timestamp=timestamp,
                    source_type=metadata.get("source_type", "internal"),
                    metadata=metadata
                )
            )

        return candidates

    async def retrieve_multi_agent_knowledge(
        self,
        query: str,
        organization_id: int,
        top_k_per_agent: int = 3,
        max_agents: Optional[int] = None,
        knowledge_pack: Optional[Any] = None
    ) -> List[RetrievalCandidate]:
        """Aggregate knowledge across multiple agents"""

        if not query or not query.strip():
            return []

        agents = await self._get_organization_agents(organization_id)
        if not agents:
            return []

        if max_agents:
            agents = agents[:max_agents]

        allowed_document_ids = None
        if knowledge_pack and getattr(knowledge_pack, "sources", None):
            allowed_document_ids = {
                source.source_id for source in knowledge_pack.sources if getattr(source, "is_active", True)
            }

        retrieval_tasks = [
            self._retrieve_from_agent(query, agent, top_k_per_agent, allowed_document_ids)
            for agent in agents
        ]

        aggregated: List[RetrievalCandidate] = []
        results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)
        for agent, result in zip(agents, results):
            if isinstance(result, Exception):
                agent_id = getattr(agent, "id", None)
                if agent_id is None and isinstance(agent, dict):
                    agent_id = agent.get("id")
                self.log_warning(
                    "Agent retrieval encountered error",
                    agent_id=agent_id,
                    error=str(result)
                )
                continue
            aggregated.extend(result)

        if knowledge_pack:
            aggregated = await self._apply_knowledge_pack_filter(aggregated, knowledge_pack)

        if not aggregated:
            return []

        return await self._calculate_confidence_scores(aggregated, query)

    async def _apply_knowledge_pack_filter(
        self,
        candidates: List[RetrievalCandidate],
        knowledge_pack: Any
    ) -> List[RetrievalCandidate]:
        """Apply knowledge pack-specific filtering while keeping fallbacks"""

        if not candidates or not knowledge_pack:
            return candidates

        filtered = self._filter_by_knowledge_pack(candidates, knowledge_pack)
        return filtered if filtered else candidates

    def _filter_by_knowledge_pack(
        self,
        candidates: List[RetrievalCandidate],
        knowledge_pack: Any
    ) -> List[RetrievalCandidate]:
        """Filter candidates that match knowledge pack themes"""

        if not knowledge_pack:
            return candidates

        terms: List[str] = []
        if isinstance(knowledge_pack, str):
            terms = [knowledge_pack.lower()]
        else:
            for attr in ("name", "slug", "category", "type"):
                value = getattr(knowledge_pack, attr, None)
                if value:
                    terms.append(str(value).lower())
            for keyword in getattr(knowledge_pack, "keywords", []) or []:
                terms.append(str(keyword).lower())

        if not terms:
            return candidates

        filtered: List[RetrievalCandidate] = []
        for candidate in candidates:
            content_lower = candidate.content.lower()
            metadata_text = " ".join(
                str(value).lower() for value in (candidate.metadata or {}).values()
                if isinstance(value, str)
            )
            if any(term in content_lower or term in metadata_text for term in terms):
                filtered.append(candidate)

        return filtered

    async def _calculate_confidence_scores(
        self,
        candidates: List[RetrievalCandidate],
        query: str
    ) -> List[RetrievalCandidate]:
        """Annotate candidates with confidence scores"""

        if not candidates:
            return []

        now = datetime.now(timezone.utc)
        for candidate in candidates:
            length_factor = min(1.0, len(candidate.content) / 400) if candidate.content else 0.2

            recency_factor = 0.7
            timestamp = candidate.timestamp
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                except ValueError:
                    timestamp = None

            if isinstance(timestamp, datetime):
                try:
                    ts = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
                    delta_days = max(0, (now - ts).days)
                    recency_factor = max(0.4, 1 - (delta_days / 365))
                except Exception:  # pragma: no cover - safety
                    recency_factor = 0.7

            base_score = candidate.score if candidate.score is not None else 0.0
            confidence = (base_score * 0.6) + (length_factor * 0.2) + (recency_factor * 0.2)
            candidate.confidence_score = round(min(0.99, max(confidence, 0.0)), 3)

        candidates.sort(key=lambda c: (c.confidence_score or 0.0, c.score), reverse=True)
        return candidates

    async def _enhance_with_persona(
        self,
        context: str,
        persona: Any,
        query: str
    ) -> str:
        """Augment context with persona voice"""

        if not persona:
            return context

        persona_key = persona
        if not isinstance(persona_key, str):
            persona_key = getattr(persona, "template_name", None) or getattr(persona, "name", "")
        persona_key = str(persona_key).lower()

        persona_styles = {
            "technical_expert": "Provide a precise technical breakdown tailored to an engineering audience.",
            "business_consultant": "Frame the insights in terms of ROI, risk mitigation, and strategic alignment.",
            "startup_advisor": "Highlight agile execution tips and rapid experimentation ideas for founders.",
            "sales_rep": "Emphasize customer value, differentiators, and next steps to drive the deal forward.",
            "solutions_engineer": "Detail architecture choices, integrations, and scalability considerations.",
            "default": "Deliver a balanced, user-friendly explanation with actionable steps."
        }

        persona_instruction = persona_styles.get(persona_key, persona_styles["default"])
        enhancement = (
            f"Persona Guidance ({persona_key or 'general'}): {persona_instruction} "
            f"Address the query explicitly: '{query}'."
        )
        return f"{context}\n\n{enhancement}"

    async def _perform_web_search(
        self,
        query: str,
        site_whitelist: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        agent_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Compatibility wrapper returning lightweight dict results"""

        candidates = await self._web_search(
            message=query,
            site_whitelist=site_whitelist or [],
            context=context or {},
            agent_id=agent_id
        )

        return [
            {
                "title": candidate.doc_title or candidate.doc_id,
                "content": candidate.content,
                "url": candidate.source_url,
                "score": candidate.score,
                "source_type": candidate.source_type,
            }
            for candidate in candidates
        ]

    async def _supplement_with_web_search(
        self,
        query: str,
        existing_candidates: List[RetrievalCandidate],
        max_web_results: int = 3,
        site_whitelist: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        agent_id: Optional[int] = None
    ) -> List[RetrievalCandidate]:
        """Blend existing knowledge with additional web results"""

        web_entries = await self._perform_web_search(
            query=query,
            site_whitelist=site_whitelist,
            context=context,
            agent_id=agent_id
        )

        blended = list(existing_candidates or [])
        for entry in (web_entries or [])[:max_web_results]:
            blended.append(
                RetrievalCandidate(
                    doc_id=str(entry.get("url") or f"web_{len(blended)}"),
                    content=entry.get("content", ""),
                    score=float(entry.get("score", 0.65)),
                    source_url=entry.get("url"),
                    doc_title=entry.get("title"),
                    timestamp=datetime.now(timezone.utc),
                    source_type=entry.get("source_type", "web"),
                    metadata=entry
                )
            )

        return blended

    async def _match_specialized_agents(
        self,
        query: str,
        available_agents: List[Dict[str, Any]],
        max_matches: int = 3
    ) -> List[Dict[str, Any]]:
        """Match query intent to agent expertise"""

        if not available_agents:
            return []

        query_terms = {term.strip('"').strip(".,?").lower() for term in query.split() if term}
        scored: List[Tuple[int, Dict[str, Any]]] = []

        for agent in available_agents:
            expertise = ""
            if isinstance(agent, dict):
                expertise = str(agent.get("expertise", "")).lower()
            else:
                expertise = str(getattr(agent, "expertise", "")).lower()

            overlap = sum(1 for term in query_terms if term and term in expertise)
            if overlap:
                scored.append((overlap, agent))

        if not scored:
            return available_agents[:max_matches]

        scored.sort(key=lambda item: item[0], reverse=True)
        return [agent for _, agent in scored[:max_matches]]

    async def _synthesize_cross_domain_knowledge(
        self,
        candidates: List[RetrievalCandidate],
        query: str,
        domains: List[str]
    ) -> str:
        """Produce a narrative that blends multiple domain insights"""

        if not candidates:
            return ""

        sections: List[str] = []
        for domain in domains or []:
            domain_lower = domain.lower()
            domain_snippets = [
                candidate.content
                for candidate in candidates
                if domain_lower in candidate.content.lower()
            ]
            if domain_snippets:
                sections.append(f"{domain.title()} Insights: {domain_snippets[0]}")

        if not sections:
            sections = [candidate.content for candidate in candidates[:3]]

        sections.append(f"Query: {query}")
        return "\n\n".join(sections)

    async def _verify_answer_grounding(
        self,
        answer: str,
        sources: List[RetrievalCandidate],
        query: str
    ) -> Dict[str, Any]:
        """Assess how well the answer is grounded in provided sources"""

        support_count = len(sources)
        strong_support = len([source for source in sources if source.score >= 0.7])

        confidence = 0.5 + (0.1 * strong_support) + (0.05 * support_count)
        confidence = min(0.99, confidence)
        grounding_quality = "high" if confidence >= 0.75 else ("medium" if confidence >= 0.6 else "low")

        return {
            "confidence_score": round(confidence, 3),
            "grounding_quality": grounding_quality,
            "supporting_sources": [source.doc_id for source in sources],
            "answer_excerpt": answer[:200],
            "query": query,
        }

    async def _learn_from_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Incorporate user feedback into lightweight heuristics"""

        rating = float(feedback.get("user_rating", 0))
        adjustment = max(0.0, min(1.0, rating / 5.0))

        updated_weights = {
            "relevance": round(0.5 + (0.2 * adjustment), 3),
            "recency": round(0.3 + (0.2 * adjustment), 3),
            "persona_alignment": round(0.2 + (0.1 * adjustment), 3),
        }

        return {
            "updated_weights": updated_weights,
            "improved_ranking": adjustment > 0.3,
            "captured_feedback": feedback.get("feedback", ""),
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
