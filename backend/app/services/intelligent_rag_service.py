"""
Intelligent RAG Service with advanced memory and personalization.
"""

import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from app.core.config import settings
from app.services.memory_service import memory_service
from app.services.gemini_service import gemini_service
from app.services.context_service import context_engine
from app.services.user_intelligence_service import user_intelligence_service
from app.services.database_service import db_service
from app.services.web_search_service import web_search_service

if TYPE_CHECKING:
    from app.models.agent import Agent


class IntelligentRAGService:
    """Enhanced RAG service with memory, personalization, and context awareness"""

    def __init__(self):
        self.memory_service = memory_service

    async def generate_intelligent_response(
        self,
        query: str,
        agent_id: int,
        visitor_id: str,
        conversation_id: int = None,
        session_context: Dict[str, Any] = None,
        system_prompt: str = None,
        agent_config: Dict[str, Any] = None,
        agent_profile: Optional['Agent'] = None
    ) -> Dict[str, Any]:
        """Generate personalized response using memory and context"""

        try:
            # Step 1: Get or create customer profile
            customer_profile = await self.memory_service.get_or_create_customer_profile(
                visitor_id=visitor_id,
                agent_id=agent_id,
                initial_context=session_context or {}
            )

            # Step 1.5: Advanced user intelligence analysis
            conversation_history = await self._get_recent_conversation_history(customer_profile.id, conversation_id)
            user_analysis = await user_intelligence_service.analyze_user_message(
                message=query,
                customer_profile_id=customer_profile.id,
                conversation_history=conversation_history,
                session_context=session_context
            )

            # Domain expertise configuration
            persona_profile = {}
            domain_enabled = False
            domain_document_ids: List[int] = []
            tool_policy = {}
            grounding_mode = "blended"
            domain_expertise_type = None
            expertise_level = 0.7
            additional_context = None

            if agent_profile is not None:
                domain_enabled = getattr(agent_profile, "domain_expertise_enabled", False)
                persona_profile = getattr(agent_profile, "personality_profile", {}) or {}
                domain_document_ids = getattr(agent_profile, "domain_knowledge_sources", []) or []
                tool_policy = getattr(agent_profile, "tool_policy", {}) or {}
                grounding_mode = getattr(agent_profile, "grounding_mode", "blended") or "blended"
                domain_expertise_type = getattr(agent_profile, "domain_expertise_type", None)
                expertise_level = getattr(agent_profile, "expertise_level", 0.7) or 0.7
                additional_context = getattr(agent_profile, "expert_context", None)
                if getattr(agent_profile, "web_search_enabled", False):
                    tool_policy.setdefault("web_search", True)

            domain_document_context = []
            if domain_enabled and domain_document_ids:
                domain_document_context = await self._build_domain_document_context(domain_document_ids)

            # Step 2 & 3: Use advanced context optimization
            context_result = await context_engine.optimize_context(
                query=query,
                customer_profile_id=customer_profile.id,
                agent_id=agent_id,
                document_context=domain_document_context
            )

            synthesized_context = context_result["context"]
            context_quality = context_result["context_quality_score"]

            # Optional web search augmentation
            allow_web_search = tool_policy.get("web_search")
            site_whitelist = tool_policy.get("site_whitelist", []) if isinstance(tool_policy, dict) else []
            if allow_web_search and await self._should_use_web_search(query, context_result, domain_document_context):
                web_results = await web_search_service.search(query, site_whitelist=site_whitelist)
                if web_results:
                    web_context = self._format_web_results_block(web_results)
                    synthesized_context = f"{synthesized_context}\n\n{web_context}".strip()

            # Domain expertise additional context
            if domain_enabled and additional_context:
                synthesized_context = f"{synthesized_context}\n\nDomain Expert Notes:\n{additional_context}".strip()

            # Step 4.5: Generate intelligent response strategy
            response_strategy = await user_intelligence_service.generate_smart_response_strategy(
                analysis=user_analysis,
                customer_profile_id=customer_profile.id,
                agent_context=agent_config or {}
            )

            response_strategy["grounding_mode"] = grounding_mode
            response_strategy["domain_expertise"] = domain_enabled

            # Step 5: Build personalized conversation messages with intelligence
            adjusted_system_prompt = system_prompt or "You are a helpful assistant."
            if domain_enabled:
                if persona_profile:
                    persona_prompt = persona_profile.get("system_prompt")
                    if persona_prompt:
                        adjusted_system_prompt = f"{persona_prompt}\n\n{adjusted_system_prompt}".strip()
                    tactics = persona_profile.get("tactics")
                    if tactics:
                        tactics_block = self._format_persona_tactics(tactics)
                        if tactics_block:
                            adjusted_system_prompt = f"{adjusted_system_prompt}\n\nPersona Guidance:\n{tactics_block}".strip()
                if grounding_mode == "strict":
                    adjusted_system_prompt = (
                        f"{adjusted_system_prompt}\n\nGrounding Mode: STRICT."
                        " Only answer using the provided context and cite the sources."
                        " If the context is insufficient, acknowledge it and propose next steps or escalation."
                    )
                else:
                    adjusted_system_prompt = (
                        f"{adjusted_system_prompt}\n\nGrounding Mode: BLENDED."
                        " Prefer provided context and cite sources."
                        " If you add insights beyond the context, label them as 'Insight (model)'."
                    )

            messages = self._build_intelligent_messages(
                query=query,
                context=synthesized_context,
                customer_profile=customer_profile,
                user_analysis=user_analysis,
                response_strategy=response_strategy,
                system_prompt=adjusted_system_prompt
            )

            # Step 6: Generate adaptive response with intelligence
            response = await self._generate_intelligent_response(
                messages=messages,
                customer_profile=customer_profile,
                user_analysis=user_analysis,
                response_strategy=response_strategy,
                agent_config=agent_config or {}
            )

            # Step 7: Optional auto-escalation when needed (before learning)
            if response_strategy.get("escalation_needed") and (agent_config.get("escalation_enabled", False)):
                try:
                    from app.services.escalation_service import escalation_service
                    # Build a concise packet so humans have context instantly
                    context_packet = await escalation_service.build_context_summary(
                        conversation_id=conversation_id,
                        customer_profile_id=customer_profile.id,
                        last_user_message=query,
                        last_agent_response=response['content']
                    )

                    priority = "critical" if user_analysis.urgency_level.value in ["critical"] else (
                        "high" if user_analysis.urgency_level.value in ["high"] else "normal"
                    )

                    esc = await escalation_service.create_escalation_event(
                        agent_id=agent_id,
                        conversation_id=conversation_id,
                        customer_profile_id=customer_profile.id,
                        priority=priority,
                        reason="auto_detection",
                        summary=context_packet.get("summary", "Escalation requested"),
                        details=context_packet
                    )

                    # Surface escalation info in response payload
                    response.setdefault("customer_context", {}).setdefault("escalation", {})
                    response["customer_context"]["escalation"].update({
                        "created": True,
                        "id": esc.id,
                        "priority": priority
                    })
                except Exception as esc_err:
                    response.setdefault("customer_context", {}).setdefault("escalation", {})
                    response["customer_context"]["escalation"].update({
                        "created": False,
                        "error": str(esc_err)
                    })

            # Step 8: Learn from interaction
            if conversation_id:
                await self._learn_from_interaction(
                    customer_profile_id=customer_profile.id,
                    conversation_id=conversation_id,
                    query=query,
                    response=response['content'],
                    context_used=synthesized_context
                )

            # Step 9: Update engagement metrics
            await self._update_engagement_metrics(customer_profile.id, response)

            return {
                **response,
                "customer_context": {
                    "visitor_id": visitor_id,
                    "engagement_level": customer_profile.engagement_level,
                    "personalization_applied": True,
                    "context_quality_score": context_quality,
                    "context_chunks_used": context_result.get("chunks_used", 0),
                    "context_efficiency": context_result.get("context_efficiency", 0.0),
                    "returning_customer": customer_profile.is_returning_customer,
                    # Enhanced intelligence data
                    "user_intelligence": {
                        "emotional_state": user_analysis.emotional_state.value,
                        "urgency_level": user_analysis.urgency_level.value,
                        "intent_category": user_analysis.intent_category.value,
                        "confidence_score": user_analysis.confidence_score,
                        "key_topics": user_analysis.key_topics,
                        "pain_points": user_analysis.pain_points,
                        "opportunities": user_analysis.opportunities,
                        "escalation_needed": response_strategy.get("escalation_needed", False)
                    },
                    "domain_expertise": {
                        "enabled": domain_enabled,
                        "persona": persona_profile.get("name") if isinstance(persona_profile, dict) else None,
                        "persona_type": domain_expertise_type.value if domain_expertise_type else None,
                        "knowledge_documents": domain_document_ids,
                        "web_search_enabled": allow_web_search,
                        "grounding_mode": grounding_mode,
                        "expertise_level": expertise_level
                    }
                }
            }

        except Exception as e:
            # Fallback to basic response if intelligent features fail
            print(f"Error in intelligent RAG: {e}")
            try:
                fallback_response = await gemini_service.generate_response(
                    prompt=query,
                    system_prompt=system_prompt,
                    temperature=agent_config.get("temperature", 0.7),
                    max_tokens=agent_config.get("max_tokens", 500)
                )

                return {
                    "content": fallback_response,
                    "customer_context": {
                        "visitor_id": visitor_id,
                        "personalization_applied": False,
                        "fallback_mode": True
                    },
                    "model": "gemini-2.0-flash-exp",
                    "usage": {
                        "completion_tokens": len(fallback_response.split()),
                        "prompt_tokens": len(query.split()),
                        "total_tokens": len(fallback_response.split()) + len(query.split())
                    }
                }
            except Exception as fallback_error:
                print(f"Fallback also failed: {fallback_error}")
                return {
                    "content": "I apologize, but I'm experiencing technical difficulties. Please try again later.",
                    "customer_context": {
                        "visitor_id": visitor_id,
                        "personalization_applied": False,
                        "error": True
                    },
                    "model": "fallback",
                    "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
                }

    def _synthesize_context(
        self,
        query: str,
        memory_context: Dict[str, Any],
        document_context: List[Dict[str, Any]],
        customer_profile: Any
    ) -> str:
        """Combine all context sources into coherent context"""

        context_parts = []

        # Customer context
        if memory_context.get('customer_profile'):
            profile = memory_context['customer_profile']
            context_parts.append(f"""
CUSTOMER PROFILE:
- Name: {profile['name']}
- Communication Style: {profile['communication_style']}
- Technical Level: {profile['technical_level']}
- Engagement Level: {profile['engagement_level']}
- Returning Customer: {profile['is_returning']}
- Journey Stage: {profile['journey_stage']}
- Primary Interests: {', '.join(profile['primary_interests'])}
- Pain Points: {', '.join(profile['pain_points'])}
""")

        # Memory-based facts
        if memory_context.get('factual_memories'):
            facts = memory_context['factual_memories']
            if facts:
                context_parts.append("KNOWN FACTS ABOUT CUSTOMER:")
                for fact in facts[:5]:  # Top 5 most relevant
                    context_parts.append(f"- {fact['key']}: {fact['value']}")

        # Behavioral insights
        if memory_context.get('behavioral_insights'):
            insights = memory_context['behavioral_insights']
            if insights:
                context_parts.append("\nBEHAVIORAL PATTERNS:")
                for insight in insights[:3]:  # Top 3 patterns
                    context_parts.append(f"- {insight['pattern']}: {insight['description']}")

        # Document context (existing RAG)
        if document_context:
            context_parts.append("\nRELEVANT KNOWLEDGE:")
            for doc in document_context[:3]:  # Top 3 documents
                context_parts.append(f"- {doc['content'][:200]}...")

        # Conversation history
        if memory_context.get('conversation_history'):
            history = memory_context['conversation_history']
            if history:
                context_parts.append("\nRECENT CONVERSATION HISTORY:")
                for conv in history:
                    context_parts.append(f"- {conv['date']}: {conv['summary']}")

        return "\n".join(context_parts)

    def _build_personalized_messages(
        self,
        query: str,
        context: str,
        customer_profile: Any,
        system_prompt: str
    ) -> List[Dict[str, str]]:
        """Build messages with personalization instructions"""

        # Enhance system prompt with personalization
        personalization_instructions = self._get_personalization_instructions(customer_profile)

        enhanced_system_prompt = f"""
{system_prompt}

PERSONALIZATION INSTRUCTIONS:
{personalization_instructions}

CONTEXT:
{context}

Remember to:
1. Adapt your communication style to match the customer's preference
2. Reference relevant past interactions when appropriate
3. Consider their technical level when explaining concepts
4. Be mindful of their journey stage and current interests
5. Build on previously identified pain points or goals
"""

        return [
            {"role": "system", "content": enhanced_system_prompt},
            {"role": "user", "content": query}
        ]

    def _get_personalization_instructions(self, customer_profile: Any) -> str:
        """Generate specific personalization instructions"""

        instructions = []

        # Communication style
        if customer_profile.communication_style == "formal":
            instructions.append("Use professional, courteous language with proper titles")
        elif customer_profile.communication_style == "casual":
            instructions.append("Use friendly, conversational tone with casual language")
        elif customer_profile.communication_style == "technical":
            instructions.append("Use precise technical terminology and detailed explanations")

        # Technical level
        if customer_profile.technical_level == "beginner":
            instructions.append("Explain technical concepts in simple terms with analogies")
        elif customer_profile.technical_level == "expert":
            instructions.append("Use advanced technical language and assume deep knowledge")

        # Response length preference
        if customer_profile.response_length_preference == "brief":
            instructions.append("Keep responses concise and to the point")
        elif customer_profile.response_length_preference == "detailed":
            instructions.append("Provide comprehensive, detailed explanations")

        # Engagement level adaptations
        if customer_profile.engagement_level == "new":
            instructions.append("Focus on onboarding and basic information")
        elif customer_profile.engagement_level == "highly_engaged":
            instructions.append("Provide advanced insights and proactive suggestions")

        return ". ".join(instructions) if instructions else "Use a helpful, professional tone"

    async def _generate_adaptive_response(
        self,
        messages: List[Dict[str, str]],
        customer_profile: Any,
        agent_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate response with adaptive parameters"""

        # Adjust generation parameters based on customer profile
        adaptive_config = agent_config.copy()

        # Adjust temperature based on customer style
        if customer_profile.communication_style == "formal":
            adaptive_config["temperature"] = min(adaptive_config.get("temperature", 0.7), 0.5)
        elif customer_profile.communication_style == "casual":
            adaptive_config["temperature"] = max(adaptive_config.get("temperature", 0.7), 0.8)

        # Adjust response length
        if customer_profile.response_length_preference == "brief":
            adaptive_config["max_tokens"] = min(adaptive_config.get("max_tokens", 500), 300)
        elif customer_profile.response_length_preference == "detailed":
            adaptive_config["max_tokens"] = max(adaptive_config.get("max_tokens", 500), 800)

        # Generate response using Gemini with adaptive config
        response_text = await gemini_service.generate_response(
            prompt=messages[-1]["content"],
            system_prompt=messages[0]["content"] if messages[0]["role"] == "system" else None,
            temperature=adaptive_config.get("temperature", 0.7),
            max_tokens=adaptive_config.get("max_tokens", 500)
        )

        return {
            "content": response_text,
            "model": "gemini-2.0-flash-exp",
            "usage": {
                "completion_tokens": len(response_text.split()),
                "prompt_tokens": len(" ".join([msg["content"] for msg in messages]).split()),
                "total_tokens": len(response_text.split()) + len(" ".join([msg["content"] for msg in messages]).split())
            }
        }

    async def _learn_from_interaction(
        self,
        customer_profile_id: int,
        conversation_id: int,
        query: str,
        response: str,
        context_used: str
    ) -> None:
        """Extract and store learnings from this interaction"""

        # Prepare conversation messages for analysis
        messages = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": response}
        ]

        # Use memory service to learn from conversation
        await self.memory_service.learn_from_conversation(
            customer_profile_id=customer_profile_id,
            conversation_id=conversation_id,
            messages=messages
        )

    async def _update_engagement_metrics(
        self,
        customer_profile_id: int,
        response: Dict[str, Any]
    ) -> None:
        """Update customer engagement metrics"""

        # Calculate estimated session duration (simple heuristic)
        estimated_duration = len(response.get('content', '')) / 200 * 0.5  # 0.5 min per 200 chars

        # Update profile
        await self.memory_service.update_customer_profile(
            customer_profile_id=customer_profile_id,
            updates={
                "last_seen_at": datetime.utcnow()
            }
        )

    async def _get_recent_conversation_history(
        self,
        customer_profile_id: int,
        conversation_id: int = None,
        limit: int = 5
    ) -> List[Dict[str, str]]:
        """Get recent conversation history for context"""
        try:
            # TODO: Implement proper conversation history retrieval
            # For now, return empty list - this would query the messages table
            return []
        except Exception as e:
            print(f"Error getting conversation history: {e}")
            return []

    def _build_intelligent_messages(
        self,
        query: str,
        context: str,
        customer_profile: Any,
        user_analysis: Any,
        response_strategy: Dict[str, Any],
        system_prompt: str
    ) -> List[Dict[str, str]]:
        """Build messages with intelligence-driven personalization"""

        # Enhanced system prompt with intelligence insights
        intelligence_instructions = self._get_intelligence_instructions(user_analysis, response_strategy)
        personalization_instructions = self._get_personalization_instructions(customer_profile)

        enhanced_system_prompt = f"""
{system_prompt}

INTELLIGENT RESPONSE GUIDELINES:
{intelligence_instructions}

PERSONALIZATION INSTRUCTIONS:
{personalization_instructions}

CONTEXT:
{context}

CRITICAL INSTRUCTIONS:
1. Adapt your tone based on the user's emotional state: {user_analysis.emotional_state.value}
2. Handle urgency level: {user_analysis.urgency_level.value}
3. Address user intent: {user_analysis.intent_category.value}
4. Use response strategy: {response_strategy.get('response_tone', 'professional_friendly')}
5. Keep response length: {response_strategy.get('response_length', 'medium')}
6. Apply personalization: {response_strategy.get('personalization_elements', [])}
"""

        if user_analysis.pain_points:
            enhanced_system_prompt += f"\nUser Pain Points to Address: {', '.join(user_analysis.pain_points)}"

        if user_analysis.opportunities:
            enhanced_system_prompt += f"\nBusiness Opportunities: {', '.join(user_analysis.opportunities)}"

        if response_strategy.get("escalation_needed"):
            enhanced_system_prompt += "\nIMPORTANT: This conversation may need escalation to a human agent. Provide excellent service while offering escalation options."

        return [
            {"role": "system", "content": enhanced_system_prompt},
            {"role": "user", "content": query}
        ]

    def _get_intelligence_instructions(self, user_analysis: Any, response_strategy: Dict[str, Any]) -> str:
        """Generate specific instructions based on user intelligence"""

        instructions = []

        # Emotional state instructions
        if user_analysis.emotional_state.value == "frustrated":
            instructions.append("User is frustrated - be extra patient, empathetic, and solution-focused")
        elif user_analysis.emotional_state.value == "excited":
            instructions.append("User is excited - match their enthusiasm while being helpful")
        elif user_analysis.emotional_state.value == "confused":
            instructions.append("User is confused - provide clear, step-by-step explanations")
        elif user_analysis.emotional_state.value == "angry":
            instructions.append("User is angry - be extremely empathetic, apologetic, and offer immediate solutions")

        # Urgency instructions
        if user_analysis.urgency_level.value == "critical":
            instructions.append("URGENT: Provide immediate, actionable solutions. Offer escalation options.")
        elif user_analysis.urgency_level.value == "high":
            instructions.append("High priority - provide quick, efficient responses with clear next steps")

        # Intent-based instructions
        if user_analysis.intent_category.value == "purchase_intent":
            instructions.append("User shows purchase intent - provide detailed product information and offer consultation")
        elif user_analysis.intent_category.value == "cancellation":
            instructions.append("User may want to cancel - understand their reasons and offer retention solutions")
        elif user_analysis.intent_category.value == "complaint":
            instructions.append("User has a complaint - be apologetic, understanding, and focus on resolution")

        return ". ".join(instructions) if instructions else "Provide helpful, professional assistance"

    async def _generate_intelligent_response(
        self,
        messages: List[Dict[str, str]],
        customer_profile: Any,
        user_analysis: Any,
        response_strategy: Dict[str, Any],
        agent_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate response with intelligence-driven adaptations"""

        # Adjust generation parameters based on intelligence
        adaptive_config = agent_config.copy()

        # Adjust temperature based on emotional state and urgency
        if user_analysis.emotional_state.value in ["angry", "frustrated"]:
            adaptive_config["temperature"] = min(adaptive_config.get("temperature", 0.7), 0.3)  # More consistent
        elif user_analysis.urgency_level.value == "critical":
            adaptive_config["temperature"] = min(adaptive_config.get("temperature", 0.7), 0.2)  # Very consistent

        # Adjust response length based on strategy
        length_mapping = {
            "brief_actionable": 300,
            "medium": 500,
            "detailed_with_examples": 800,
            "comprehensive": 1200
        }

    async def _build_domain_document_context(self, document_ids: List[int]) -> List[Dict[str, Any]]:
        documents = await db_service.get_documents_by_ids(document_ids)
        context_entries: List[Dict[str, Any]] = []
        for document in documents[:10]:
            content = (document.content or "").strip()
            if not content:
                continue
            snippet = content[:1200]
            context_entries.append({
                "content": f"[{document.filename}]\n{snippet}",
                "metadata": {"source": document.filename, "document_id": document.id},
                "similarity_score": 0.95
            })
        return context_entries

    async def _should_use_web_search(
        self,
        query: str,
        context_result: Dict[str, Any],
        domain_document_context: List[Dict[str, Any]]
    ) -> bool:
        if context_result.get("chunks_used", 0) == 0 and not domain_document_context:
            return True
        lowered = query.lower()
        if any(trigger in lowered for trigger in ["latest", "news", "recent", "update"]):
            return True
        return False

    def _format_web_results_block(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return ""
        lines = ["Web Search Findings:"]
        for idx, result in enumerate(results[:5], start=1):
            title = result.get("title") or result.get("url") or "Result"
            snippet = result.get("snippet") or ""
            url = result.get("url") or ""
            lines.append(f"[{idx}] {title}\n{snippet}\nSource: {url}")
        return "\n\n".join(lines)

    def _format_persona_tactics(self, tactics: Any) -> str:
        if isinstance(tactics, dict):
            lines = []
            style = tactics.get("style")
            if style:
                lines.append(f"Preferred style: {style}")
            steps = tactics.get("steps")
            if isinstance(steps, list):
                lines.append("Follow these steps:")
                for idx, step in enumerate(steps, start=1):
                    lines.append(f"{idx}. {step}")
            tips = tactics.get("tips")
            if isinstance(tips, list):
                lines.append("Tips:")
                for tip in tips:
                    lines.append(f"- {tip}")
            if lines:
                return "\n".join(lines)
        return ""

    async def get_customer_insights(
        self,
        agent_id: int,
        visitor_id: str = None,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """Get comprehensive customer insights for agent optimization"""

        if visitor_id:
            # Get specific customer insights
            customer_profile = await self.memory_service.get_or_create_customer_profile(
                visitor_id=visitor_id,
                agent_id=agent_id
            )

            memory_context = await self.memory_service.get_contextual_memory(
                customer_profile_id=customer_profile.id,
                query_text=""
            )

            return {
                "customer_profile": memory_context.get('customer_profile', {}),
                "key_memories": memory_context.get('factual_memories', [])[:10],
                "behavioral_insights": memory_context.get('behavioral_insights', [])[:5],
                "conversation_summary": memory_context.get('conversation_history', [])
            }
        else:
            # Get aggregate insights for all customers
            # This would require additional database queries
            return {
                "aggregate_insights": "Feature not yet implemented",
                "top_interests": [],
                "common_pain_points": [],
                "engagement_patterns": {}
            }


# Global instance
intelligent_rag_service = IntelligentRAGService()
