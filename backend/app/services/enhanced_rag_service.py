"""
Enhanced RAG Service with World-Class Concierge Intelligence
Feeds comprehensive user context to the LLM for truly personalized, delightful experiences.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import asdict

from app.services.concierge_intelligence_service import concierge_intelligence, ConciergeCase
from app.services.contextual_case_builder import contextual_case_builder
from app.services.intelligent_rag_service import intelligent_rag_service
from app.services.gemini_service import gemini_service
from app.services.memory_service import memory_service
from app.models.customer_memory import MemoryType, MemoryImportance


class WorldClassRAGService:
    """RAG service enhanced with deep user intelligence for concierge-level interactions"""

    def __init__(self):
        self.context_prompt_templates = {
            "technical_expert": """
You are interacting with a technical expert who prefers:
- Detailed, technical explanations
- Code examples and API documentation
- Architecture diagrams and implementation details
- Direct, efficient communication
""",
            "business_decision_maker": """
You are interacting with a business decision maker who needs:
- ROI and business value focus
- High-level strategic information
- Risk assessment and mitigation
- Executive summary style responses
""",
            "first_time_user": """
You are helping a first-time user who needs:
- Clear, step-by-step guidance
- Friendly, welcoming tone
- Comprehensive explanations
- Proactive suggestions and tips
""",
            "frustrated_user": """
You are helping a frustrated user who needs:
- Immediate acknowledgment of their frustration
- Quick, actionable solutions
- Empathetic and understanding tone
- Clear next steps and escalation paths
""",
            "high_value_customer": """
You are assisting a high-value customer who deserves:
- Premium, white-glove service
- Personalized attention and solutions
- Proactive support and recommendations
- Direct access to specialized resources
"""
        }

    async def generate_world_class_response(
        self,
        message: str,
        customer_profile_id: int,
        session_context: Dict[str, Any],
        agent_id: int,
        conversation_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Generate a world-class concierge response with full context intelligence"""

        # Build comprehensive case context
        case_context = await contextual_case_builder.build_comprehensive_case_context(
            customer_profile_id=customer_profile_id,
            current_message=message,
            session_context=session_context,
            agent_id=agent_id
        )

        # Build concierge case for strategic response
        concierge_case = await concierge_intelligence.build_concierge_case(
            customer_profile_id=customer_profile_id,
            current_message=message,
            session_context=session_context,
            agent_id=agent_id
        )

        # Get relevant knowledge base content
        knowledge_context = await intelligent_rag_service.get_contextual_knowledge(
            query=message,
            customer_profile_id=customer_profile_id,
            conversation_history=conversation_history or []
        )

        # Generate response with full intelligence
        response = await self._generate_intelligent_response(
            message=message,
            case_context=case_context,
            concierge_case=concierge_case,
            knowledge_context=knowledge_context,
            conversation_history=conversation_history or []
        )

        # Save interaction insights to memory
        await self._save_interaction_insights(
            customer_profile_id=customer_profile_id,
            message=message,
            response=response,
            case_context=case_context,
            concierge_case=concierge_case
        )

        return {
            "response": response["content"],
            "response_metadata": {
                "confidence_score": response["confidence"],
                "personalization_applied": response["personalization_elements"],
                "context_used": response["context_sources"],
                "intelligence_insights": {
                    "user_type": concierge_case.relationship_stage,
                    "emotional_state": concierge_case.emotional_state,
                    "urgency_level": concierge_case.urgency_level,
                    "intent": concierge_case.current_intent
                },
                "next_best_actions": concierge_case.next_best_actions,
                "escalation_recommended": any("escalate" in action.lower() for action in concierge_case.next_best_actions)
            },
            "context_insights": {
                "user_story": concierge_case.user_story,
                "relationship_stage": concierge_case.relationship_stage,
                "trust_level": concierge_case.trust_level,
                "satisfaction_trend": concierge_case.satisfaction_trajectory[-1] if concierge_case.satisfaction_trajectory else None
            }
        }

    async def _generate_intelligent_response(
        self,
        message: str,
        case_context: Dict[str, Any],
        concierge_case: ConciergeCase,
        knowledge_context: Dict[str, Any],
        conversation_history: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Generate intelligent response using all available context"""

        # Determine response strategy
        response_strategy = await self._determine_response_strategy(concierge_case)

        # Build comprehensive context for the LLM
        llm_context = await self._build_llm_context(
            message=message,
            case_context=case_context,
            concierge_case=concierge_case,
            knowledge_context=knowledge_context,
            conversation_history=conversation_history,
            strategy=response_strategy
        )

        # Generate response with strategy-specific prompting
        response_prompt = self._build_response_prompt(
            message=message,
            context=llm_context,
            strategy=response_strategy
        )

        try:
            # Generate response using Gemini
            raw_response = await gemini_service.generate_response(
                prompt=response_prompt,
                temperature=self._get_temperature_for_strategy(response_strategy),
                max_tokens=self._get_max_tokens_for_strategy(response_strategy)
            )

            # Post-process response
            processed_response = await self._post_process_response(
                raw_response=raw_response,
                strategy=response_strategy,
                concierge_case=concierge_case
            )

            return processed_response

        except Exception as e:
            # Fallback response with basic context
            return {
                "content": "I apologize, but I'm experiencing technical difficulties. Let me connect you with a human agent who can assist you immediately.",
                "confidence": 0.3,
                "personalization_elements": [],
                "context_sources": ["fallback"],
                "error": str(e)
            }

    async def _determine_response_strategy(self, concierge_case: ConciergeCase) -> Dict[str, Any]:
        """Determine optimal response strategy based on case analysis"""

        strategy = {
            "approach": concierge_case.recommended_approach,
            "tone": self._determine_tone(concierge_case),
            "depth": self._determine_response_depth(concierge_case),
            "personalization_level": self._determine_personalization_level(concierge_case),
            "urgency_handling": self._determine_urgency_handling(concierge_case)
        }

        # Special handling for high-risk situations
        if concierge_case.urgency_level == "critical" or concierge_case.emotional_state == "angry":
            strategy["approach"] = "crisis_management"
            strategy["tone"] = "empathetic_urgent"

        # Special handling for high-value customers
        if concierge_case.potential_value > 10000:
            strategy["personalization_level"] = "premium"
            strategy["approach"] = "white_glove"

        return strategy

    async def _build_llm_context(
        self,
        message: str,
        case_context: Dict[str, Any],
        concierge_case: ConciergeCase,
        knowledge_context: Dict[str, Any],
        conversation_history: List[Dict[str, str]],
        strategy: Dict[str, Any]
    ) -> str:
        """Build comprehensive context for the LLM"""

        context_sections = []

        # User Intelligence Summary
        context_sections.append(f"""
# USER INTELLIGENCE SUMMARY

## Who This User Is
{concierge_case.user_story}

## Current Situation
- **Intent**: {concierge_case.current_intent}
- **Emotional State**: {concierge_case.emotional_state}
- **Urgency Level**: {concierge_case.urgency_level}
- **Relationship Stage**: {concierge_case.relationship_stage}
- **Trust Level**: {concierge_case.trust_level:.1f}/1.0
""")

        # Context Intelligence
        if case_context.get("context_summary"):
            context_sections.append(f"""
## Context Intelligence
{case_context["context_summary"]}

**Confidence Score**: {case_context["confidence_score"]:.1f}/1.0
**Data Completeness**: {case_context["completeness_score"]:.1f}/1.0
""")

        # User Preferences & Patterns
        if concierge_case.preferences:
            context_sections.append(f"""
## User Preferences
{json.dumps(concierge_case.preferences, indent=2)}
""")

        # Success Patterns
        if concierge_case.success_patterns:
            context_sections.append(f"""
## What Works With This User
{chr(10).join([f"- {pattern}" for pattern in concierge_case.success_patterns])}
""")

        # Current Challenges
        if concierge_case.pain_points or concierge_case.risk_factors:
            context_sections.append(f"""
## Current Challenges & Risks
**Pain Points**: {', '.join(concierge_case.pain_points) if concierge_case.pain_points else 'None identified'}
**Risk Factors**: {chr(10).join([f"- {risk}" for risk in concierge_case.risk_factors])}
""")

        # Knowledge Base Context
        if knowledge_context.get("relevant_content"):
            context_sections.append(f"""
## Relevant Knowledge Base Information
{knowledge_context["relevant_content"]}
""")

        # Conversation History
        if conversation_history:
            recent_history = conversation_history[-5:]  # Last 5 exchanges
            context_sections.append(f"""
## Recent Conversation History
{chr(10).join([f"{msg['role']}: {msg['content']}" for msg in recent_history])}
""")

        # Strategy Guidelines
        context_sections.append(f"""
## Response Strategy
- **Approach**: {strategy["approach"]}
- **Tone**: {strategy["tone"]}
- **Depth**: {strategy["depth"]}
- **Personalization**: {strategy["personalization_level"]}

## Key Talking Points
{chr(10).join([f"- {point}" for point in concierge_case.talking_points])}

## Things to Avoid
{chr(10).join([f"- {avoid}" for avoid in concierge_case.things_to_avoid])}

## Recommended Next Actions
{chr(10).join([f"- {action}" for action in concierge_case.next_best_actions])}
""")

        return "\n".join(context_sections)

    def _build_response_prompt(
        self,
        message: str,
        context: str,
        strategy: Dict[str, Any]
    ) -> str:
        """Build the final prompt for response generation"""

        # Get strategy-specific prompt template
        base_template = self.context_prompt_templates.get(
            strategy["approach"],
            self.context_prompt_templates["first_time_user"]
        )

        prompt = f"""
{base_template}

{context}

## USER'S CURRENT MESSAGE
"{message}"

## RESPONSE INSTRUCTIONS

You are a world-class concierge AI assistant. Your goal is to provide an exceptional, personalized experience that delights this specific user.

**Response Guidelines:**
1. **Personalization**: Use the user intelligence to tailor your response perfectly to this individual
2. **Tone**: Adopt the {strategy["tone"]} tone that matches their emotional state and preferences
3. **Depth**: Provide {strategy["depth"]} level of detail based on their expertise and current needs
4. **Context Awareness**: Reference relevant past interactions and show you understand their journey
5. **Proactive Value**: Anticipate their needs and offer additional helpful insights
6. **Trust Building**: Demonstrate competence and reliability to strengthen the relationship

**Your Response Should:**
- Directly address their question with accuracy and completeness
- Show understanding of their specific situation and context
- Provide personalized recommendations based on their profile
- Anticipate and address likely follow-up questions
- Offer next steps that align with their journey stage
- Maintain the appropriate tone for their emotional state

**Remember:** You're not just answering questions - you're providing a concierge-level experience that makes this user feel understood, valued, and expertly assisted.

Generate your response now:
"""

        return prompt

    def _get_temperature_for_strategy(self, strategy: Dict[str, Any]) -> float:
        """Get appropriate temperature based on strategy"""
        if strategy["approach"] == "crisis_management":
            return 0.1  # Very focused and consistent
        elif strategy["approach"] == "technical_expert":
            return 0.2  # Precise and accurate
        elif strategy["personalization_level"] == "premium":
            return 0.4  # More creative and personalized
        else:
            return 0.3  # Balanced

    def _get_max_tokens_for_strategy(self, strategy: Dict[str, Any]) -> int:
        """Get appropriate token limit based on strategy"""
        if strategy["depth"] == "comprehensive":
            return 800
        elif strategy["depth"] == "detailed":
            return 600
        elif strategy["urgency_handling"] == "immediate":
            return 300
        else:
            return 500

    async def _post_process_response(
        self,
        raw_response: str,
        strategy: Dict[str, Any],
        concierge_case: ConciergeCase
    ) -> Dict[str, Any]:
        """Post-process the generated response"""

        # Clean up response
        cleaned_response = raw_response.strip()

        # Add personalization elements based on strategy
        personalization_elements = []
        if strategy["personalization_level"] == "premium":
            personalization_elements.append("premium_service")
        if strategy["approach"] == "technical_expert":
            personalization_elements.append("technical_depth")

        # Calculate confidence based on context richness
        confidence = 0.8
        if concierge_case.trust_level > 0.8:
            confidence += 0.1
        if len(concierge_case.success_patterns) > 2:
            confidence += 0.1

        return {
            "content": cleaned_response,
            "confidence": min(0.95, confidence),
            "personalization_elements": personalization_elements,
            "context_sources": ["user_intelligence", "knowledge_base", "conversation_history"]
        }

    async def _save_interaction_insights(
        self,
        customer_profile_id: int,
        message: str,
        response: Dict[str, Any],
        case_context: Dict[str, Any],
        concierge_case: ConciergeCase
    ):
        """Save insights from this interaction for future use"""

        # Save to memory service
        await memory_service.store_memory(
            customer_profile_id=customer_profile_id,
            memory_type=MemoryType.EPISODIC,
            key=f"interaction_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            value=json.dumps({
                "message": message[:200],  # Truncate for storage
                "response_confidence": response["confidence"],
                "emotional_state": concierge_case.emotional_state,
                "intent": concierge_case.current_intent,
                "satisfaction_predicted": concierge_case.trust_level
            }),
            importance=MemoryImportance.MEDIUM,
            tags=["interaction", "response_quality"]
        )

        # Save successful patterns
        if response["confidence"] > 0.8:
            await memory_service.store_memory(
                customer_profile_id=customer_profile_id,
                memory_type=MemoryType.PROCEDURAL,
                key="successful_response_pattern",
                value=f"High confidence response for {concierge_case.current_intent} intent",
                importance=MemoryImportance.HIGH,
                tags=["success_pattern", "response_strategy"]
            )

    # Helper methods for strategy determination
    def _determine_tone(self, concierge_case: ConciergeCase) -> str:
        """Determine appropriate tone based on case analysis"""
        if concierge_case.emotional_state == "frustrated":
            return "empathetic_understanding"
        elif concierge_case.emotional_state == "excited":
            return "enthusiastic_supportive"
        elif concierge_case.relationship_stage == "first_time":
            return "welcoming_helpful"
        elif concierge_case.potential_value > 10000:
            return "professional_premium"
        else:
            return "friendly_professional"

    def _determine_response_depth(self, concierge_case: ConciergeCase) -> str:
        """Determine appropriate response depth"""
        if concierge_case.urgency_level == "critical":
            return "focused_actionable"
        elif "technical" in concierge_case.user_story.lower():
            return "comprehensive"
        elif concierge_case.relationship_stage == "first_time":
            return "detailed_explanatory"
        else:
            return "balanced"

    def _determine_personalization_level(self, concierge_case: ConciergeCase) -> str:
        """Determine level of personalization to apply"""
        if concierge_case.potential_value > 10000:
            return "premium"
        elif concierge_case.trust_level > 0.8:
            return "high"
        elif len(concierge_case.success_patterns) > 2:
            return "medium"
        else:
            return "standard"

    def _determine_urgency_handling(self, concierge_case: ConciergeCase) -> str:
        """Determine urgency handling approach"""
        if concierge_case.urgency_level == "critical":
            return "immediate"
        elif concierge_case.urgency_level == "high":
            return "priority"
        else:
            return "standard"


# Global instance
world_class_rag = WorldClassRAGService()