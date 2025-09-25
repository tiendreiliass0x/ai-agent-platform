"""
Concierge Intelligence Service - The brain of our world-class concierge system.
Aggregates ALL user touchpoints and builds comprehensive intelligence for truly personalized experiences.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

from app.services.user_intelligence_service import user_intelligence_service, ConversationAnalysis
from app.services.memory_service import memory_service
from app.models.customer_profile import CustomerProfile
from app.models.customer_memory import CustomerMemory, MemoryType, MemoryImportance
from app.services.integrations.crm_service import crm_service


class ProfileCompleteness(str, Enum):
    """Profile completeness levels"""
    MINIMAL = "minimal"          # Basic session data only
    BASIC = "basic"              # Some behavioral data
    ENRICHED = "enriched"        # Good understanding
    COMPREHENSIVE = "comprehensive"  # Deep insights
    WORLD_CLASS = "world_class"  # Perfect concierge knowledge


@dataclass
class UserTouchpoint:
    """Represents any interaction point with the user"""
    touchpoint_type: str  # "message", "page_visit", "document_view", "search", "click"
    timestamp: datetime
    data: Dict[str, Any]
    context: Dict[str, Any]
    insights: Dict[str, Any]
    importance_score: float


@dataclass
class ConciergeCase:
    """Comprehensive case context for the current interaction"""
    case_id: str
    customer_profile_id: int

    # Current Context
    current_intent: str
    urgency_level: str
    emotional_state: str
    session_context: Dict[str, Any]

    # Aggregated Intelligence
    user_story: str  # Natural language summary of who the user is
    interaction_history: List[UserTouchpoint]
    key_insights: List[Dict[str, Any]]
    preferences: Dict[str, Any]
    pain_points: List[str]
    success_patterns: List[str]

    # Relationship Intelligence
    relationship_stage: str  # "first_time", "exploring", "evaluating", "customer", "advocate"
    trust_level: float  # 0-1 score
    satisfaction_trajectory: List[float]  # Historical satisfaction

    # Business Context
    potential_value: float  # Estimated customer lifetime value
    risk_factors: List[str]  # Churn or escalation risks
    opportunities: List[str]  # Upsell/cross-sell opportunities

    # Concierge Strategy
    recommended_approach: str
    talking_points: List[str]
    things_to_avoid: List[str]
    next_best_actions: List[str]


class ConciergeIntelligenceService:
    """The master intelligence service that orchestrates all user understanding"""

    def __init__(self):
        self.touchpoint_weights = {
            "message": 1.0,
            "page_visit": 0.3,
            "document_view": 0.7,
            "search": 0.5,
            "click": 0.2,
            "form_interaction": 0.6,
            "support_ticket": 0.9,
            "purchase": 1.5,
            "cancellation": 1.2
        }

    async def build_concierge_case(
        self,
        customer_profile_id: int,
        current_message: str,
        session_context: Dict[str, Any],
        agent_id: int
    ) -> ConciergeCase:
        """Build comprehensive case context for world-class concierge experience"""

        # Get current conversation analysis
        conversation_analysis = await user_intelligence_service.analyze_user_message(
            message=current_message,
            customer_profile_id=customer_profile_id,
            session_context=session_context
        )

        # Aggregate all intelligence in parallel for speed
        intelligence_tasks = [
            self._get_customer_profile_with_enrichment(customer_profile_id),
            self._aggregate_interaction_history(customer_profile_id, limit=50),
            self._build_user_story(customer_profile_id, current_message),
            self._analyze_relationship_stage(customer_profile_id),
            self._calculate_business_metrics(agent_id, customer_profile_id),
            self._generate_concierge_strategy(customer_profile_id, conversation_analysis),
            self._identify_success_patterns(customer_profile_id),
            self._assess_risk_factors(customer_profile_id, conversation_analysis)
        ]

        results = await asyncio.gather(*intelligence_tasks)

        case_id = f"case_{customer_profile_id}_{int(datetime.now().timestamp())}"

        return ConciergeCase(
            case_id=case_id,
            customer_profile_id=customer_profile_id,
            current_intent=conversation_analysis.intent_category.value,
            urgency_level=conversation_analysis.urgency_level.value,
            emotional_state=conversation_analysis.emotional_state.value,
            session_context=session_context,
            user_story=results[2],
            interaction_history=results[1],
            key_insights=await self._extract_key_insights(results[0], results[1]),
            preferences=results[0].get('preferences', {}),
            pain_points=conversation_analysis.pain_points,
            success_patterns=results[6],
            relationship_stage=results[3]['stage'],
            trust_level=results[3]['trust_level'],
            satisfaction_trajectory=results[3]['satisfaction_history'],
            potential_value=results[4]['estimated_value'],
            risk_factors=results[7],
            opportunities=conversation_analysis.opportunities,
            recommended_approach=results[5]['approach'],
            talking_points=results[5]['talking_points'],
            things_to_avoid=results[5]['avoid_topics'],
            next_best_actions=results[5]['next_actions']
        )

    async def _get_customer_profile_with_enrichment(self, customer_profile_id: int) -> Dict[str, Any]:
        """Get enriched customer profile with all computed insights"""

        # This would integrate with your database layer
        # For now, showing the structure of what we'd return
        profile_data = {
            "basic_info": {},
            "behavioral_patterns": {},
            "preferences": {},
            "engagement_metrics": {},
            "personality_insights": {},
            "technical_profile": {},
            "business_context": {}
        }

        # Enrich with real-time calculations
        profile_data["completeness_score"] = await self._calculate_profile_completeness(customer_profile_id)
        profile_data["confidence_score"] = await self._calculate_confidence_score(customer_profile_id)

        return profile_data

    async def _aggregate_interaction_history(
        self,
        customer_profile_id: int,
        limit: int = 50
    ) -> List[UserTouchpoint]:
        """Aggregate ALL touchpoints across all channels"""

        touchpoints = []

        # Aggregate from multiple sources
        touchpoint_sources = [
            self._get_conversation_touchpoints(customer_profile_id),
            self._get_web_behavior_touchpoints(customer_profile_id),
            self._get_support_touchpoints(customer_profile_id),
            self._get_business_touchpoints(customer_profile_id)
        ]

        all_touchpoints = []
        for source_touchpoints in await asyncio.gather(*touchpoint_sources):
            all_touchpoints.extend(source_touchpoints)

        # Sort by importance and recency
        all_touchpoints.sort(
            key=lambda tp: (tp.importance_score, tp.timestamp.timestamp()),
            reverse=True
        )

        return all_touchpoints[:limit]

    async def _build_user_story(
        self,
        customer_profile_id: int,
        current_message: str
    ) -> str:
        """Generate a natural language story of who this user is"""

        # Get profile and interaction data
        profile = await self._get_customer_profile_with_enrichment(customer_profile_id)
        recent_interactions = await self._aggregate_interaction_history(customer_profile_id, 10)

        # Build narrative components
        story_elements = []

        # Who they are
        engagement_level = profile.get("engagement_metrics", {}).get("level", "new")
        if engagement_level == "new":
            story_elements.append("This is a first-time visitor exploring our platform.")
        elif engagement_level == "exploring":
            story_elements.append("This user is actively exploring our features and evaluating fit.")
        elif engagement_level == "engaged":
            story_elements.append("This is an engaged user who regularly interacts with us.")
        else:
            story_elements.append("This is a highly engaged power user.")

        # What they care about
        interests = profile.get("preferences", {}).get("primary_interests", [])
        if interests:
            story_elements.append(f"They're particularly interested in: {', '.join(interests[:3])}.")

        # How they communicate
        comm_style = profile.get("behavioral_patterns", {}).get("communication_style", "neutral")
        if comm_style == "technical":
            story_elements.append("They prefer technical details and precise information.")
        elif comm_style == "casual":
            story_elements.append("They prefer friendly, conversational interactions.")
        elif comm_style == "formal":
            story_elements.append("They prefer professional, structured communication.")

        # Current context
        story_elements.append(f"Right now, they're asking: '{current_message[:100]}...'")

        return " ".join(story_elements)

    async def _analyze_relationship_stage(self, customer_profile_id: int) -> Dict[str, Any]:
        """Analyze the relationship stage and trust level"""
        # Use CRM integration (or heuristic) to determine stage and trust.
        # Note: agent_id not available here, so we use a heuristic fallback.
        # Callers that have agent_id should prefer crm_service directly.
        # For now, derive based on profile engagement via memory service.
        try:
            # Heuristic fallback
            profile = await memory_service._get_customer_profile(customer_profile_id)
            if not profile:
                return {
                    "stage": "first_time",
                    "trust_level": 0.3,
                    "satisfaction_history": [4.0],
                    "relationship_trend": "neutral"
                }

            if profile.total_conversations == 0:
                stage = "first_time"
                trust = 0.3
            elif profile.total_conversations < 3:
                stage = "exploring"
                trust = 0.5
            elif profile.total_conversations < 10:
                stage = "evaluating"
                trust = 0.65
            else:
                stage = "established_customer"
                trust = 0.8

            return {
                "stage": stage,
                "trust_level": trust,
                "satisfaction_history": [profile.satisfaction_score] if profile.satisfaction_score is not None else [4.0],
                "relationship_trend": "positive" if trust >= 0.6 else "neutral"
            }
        except Exception:
            return {
                "stage": "evaluating",
                "trust_level": 0.6,
                "satisfaction_history": [4.0],
                "relationship_trend": "neutral"
            }

    async def _calculate_business_metrics(self, agent_id: int, customer_profile_id: int) -> Dict[str, Any]:
        """Calculate business-relevant metrics"""
        # In production, pull from CRM/billing. Here, return stub/heuristic.
        return await crm_service.get_business_metrics(agent_id=agent_id, customer_profile_id=customer_profile_id)

    async def _generate_concierge_strategy(
        self,
        customer_profile_id: int,
        conversation_analysis: ConversationAnalysis
    ) -> Dict[str, Any]:
        """Generate optimal concierge approach strategy"""

        strategy = {
            "approach": "consultative_expert",
            "talking_points": [
                "Acknowledge their technical expertise",
                "Focus on advanced features and customization",
                "Provide detailed technical documentation",
                "Offer architecture consultation"
            ],
            "avoid_topics": [
                "Basic feature explanations",
                "Overly simplified language",
                "Sales pressure"
            ],
            "next_actions": [
                "Provide comprehensive technical answer",
                "Offer deep-dive demo",
                "Connect with solutions architect"
            ]
        }

        # Adjust based on emotional state
        if conversation_analysis.emotional_state.value == "frustrated":
            strategy["approach"] = "empathetic_solution_focused"
            strategy["talking_points"].insert(0, "Acknowledge frustration and apologize")

        return strategy

    async def _extract_key_insights(
        self,
        profile: Dict[str, Any],
        interactions: List[UserTouchpoint]
    ) -> List[Dict[str, Any]]:
        """Extract the most important insights about this user"""

        insights = []

        # Behavioral insights
        if len(interactions) > 5:
            insights.append({
                "type": "behavioral",
                "insight": "Highly engaged user with consistent interaction patterns",
                "confidence": 0.9,
                "evidence": f"Has {len(interactions)} touchpoints in recent history"
            })

        # Technical level insights
        technical_signals = sum(1 for tp in interactions if "api" in str(tp.data).lower())
        if technical_signals > 2:
            insights.append({
                "type": "technical",
                "insight": "Technical user interested in integrations and APIs",
                "confidence": 0.8,
                "evidence": f"{technical_signals} technical touchpoints detected"
            })

        return insights

    async def _identify_success_patterns(self, customer_profile_id: int) -> List[str]:
        """Identify what has worked well with this user in the past"""

        return [
            "Responds well to detailed technical explanations",
            "Prefers self-service resources over calls",
            "Values quick response times"
        ]

    async def _assess_risk_factors(
        self,
        customer_profile_id: int,
        conversation_analysis: ConversationAnalysis
    ) -> List[str]:
        """Identify potential risks or concerns"""

        risks = []

        if conversation_analysis.urgency_level.value == "critical":
            risks.append("High urgency - potential escalation risk if not handled quickly")

        if conversation_analysis.emotional_state.value in ["frustrated", "angry"]:
            risks.append("Negative emotional state - risk of dissatisfaction")

        if conversation_analysis.intent_category.value == "cancellation":
            risks.append("Cancellation intent - high churn risk")

        return risks

    async def generate_llm_context(self, case: ConciergeCase) -> str:
        """Generate rich context to feed the LLM for world-class responses"""

        context = f"""
# CONCIERGE CASE CONTEXT

## Customer Profile Summary
{case.user_story}

## Current Situation
- **Intent**: {case.current_intent}
- **Emotional State**: {case.emotional_state}
- **Urgency**: {case.urgency_level}
- **Relationship Stage**: {case.relationship_stage}
- **Trust Level**: {case.trust_level:.1f}/1.0

## Key Insights
"""

        for insight in case.key_insights[:3]:
            context += f"- {insight['insight']} (confidence: {insight['confidence']:.1f})\n"

        context += f"""

## What Works With This User
"""
        for pattern in case.success_patterns:
            context += f"- {pattern}\n"

        context += f"""

## Current Challenges & Risks
"""
        for risk in case.risk_factors:
            context += f"- {risk}\n"

        context += f"""

## Recommended Approach
**Strategy**: {case.recommended_approach}

**Key Talking Points**:
"""
        for point in case.talking_points:
            context += f"- {point}\n"

        context += f"""

**Things to Avoid**:
"""
        for avoid in case.things_to_avoid:
            context += f"- {avoid}\n"

        context += f"""

## Business Context
- **Estimated Value**: ${case.potential_value:,.2f}
- **Relationship Stage**: {case.relationship_stage}

## Next Best Actions
"""
        for action in case.next_best_actions:
            context += f"- {action}\n"

        return context

    # Helper methods for touchpoint aggregation
    async def _get_conversation_touchpoints(self, customer_profile_id: int) -> List[UserTouchpoint]:
        """Get conversation-based touchpoints"""
        # Would query conversation history
        return []

    async def _get_web_behavior_touchpoints(self, customer_profile_id: int) -> List[UserTouchpoint]:
        """Get web behavior touchpoints (page visits, clicks, etc.)"""
        # Would integrate with web analytics
        return []

    async def _get_support_touchpoints(self, customer_profile_id: int) -> List[UserTouchpoint]:
        """Get support interaction touchpoints"""
        # Would integrate with support system
        return []

    async def _get_business_touchpoints(self, customer_profile_id: int) -> List[UserTouchpoint]:
        """Get business touchpoints (purchases, subscriptions, etc.)"""
        # Would integrate with billing/CRM systems
        return []

    async def _calculate_profile_completeness(self, customer_profile_id: int) -> float:
        """Calculate how complete our understanding of this user is"""
        # Would analyze available data points
        return 0.75

    async def _calculate_confidence_score(self, customer_profile_id: int) -> float:
        """Calculate confidence in our insights about this user"""
        # Would analyze data quality and consistency
        return 0.85

# Global instance
concierge_intelligence = ConciergeIntelligenceService()
