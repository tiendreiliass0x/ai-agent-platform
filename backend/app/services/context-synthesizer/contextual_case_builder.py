"""
Contextual Case Builder - Builds comprehensive case context by aggregating ALL available data points
for world-class concierge understanding.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import hashlib

from app.services.concierge_intelligence_service import ConciergeCase, UserTouchpoint
from app.models.customer_profile import CustomerProfile
from app.models.customer_memory import CustomerMemory, MemoryType
from app.services.gemini_service import gemini_service


@dataclass
class DataSource:
    """Represents a source of user data"""
    source_id: str
    source_type: str
    priority: int
    last_updated: datetime
    data: Dict[str, Any]
    reliability_score: float


@dataclass
class ContextLayer:
    """Different layers of context we build"""
    layer_name: str
    context_data: Dict[str, Any]
    confidence_score: float
    source_count: int
    last_updated: datetime


@dataclass
class CaseEvidence:
    """Evidence supporting insights about the user"""
    evidence_type: str  # "behavioral", "explicit", "inferred", "contextual"
    claim: str
    supporting_data: List[str]
    confidence: float
    timestamp: datetime


class ContextualCaseBuilder:
    """Builds rich, evidence-based context for each user interaction"""

    def __init__(self):
        self.context_layers = [
            "identity",           # Who they are
            "behavioral",         # How they behave
            "contextual",         # Current situation
            "historical",         # Past interactions
            "predictive",         # Future intentions
            "emotional",          # Emotional intelligence
            "business",           # Business context
            "technical",          # Technical preferences
            "relational"          # Relationship dynamics
        ]

    async def build_comprehensive_case_context(
        self,
        customer_profile_id: int,
        current_message: str,
        session_context: Dict[str, Any],
        agent_id: int
    ) -> Dict[str, Any]:
        """Build the most comprehensive context possible"""

        # Gather all available data sources
        data_sources = await self._gather_all_data_sources(
            customer_profile_id,
            session_context,
            agent_id
        )

        # Build context layers in parallel
        layer_tasks = [
            self._build_identity_context(customer_profile_id, data_sources),
            self._build_behavioral_context(customer_profile_id, data_sources),
            self._build_contextual_situation(current_message, session_context, data_sources),
            self._build_historical_context(customer_profile_id, data_sources),
            self._build_predictive_context(customer_profile_id, current_message, data_sources),
            self._build_emotional_intelligence(customer_profile_id, current_message, data_sources),
            self._build_business_context(customer_profile_id, data_sources),
            self._build_technical_context(customer_profile_id, data_sources),
            self._build_relational_context(customer_profile_id, data_sources)
        ]

        context_layers = await asyncio.gather(*layer_tasks)

        # Build evidence-based insights
        evidence = await self._gather_case_evidence(customer_profile_id, data_sources)

        # Generate smart hypotheses about the user
        hypotheses = await self._generate_user_hypotheses(
            customer_profile_id,
            current_message,
            context_layers,
            evidence
        )

        # Create the comprehensive case context
        case_context = {
            "case_id": f"ctx_{customer_profile_id}_{int(datetime.now().timestamp())}",
            "customer_profile_id": customer_profile_id,
            "build_timestamp": datetime.now().isoformat(),
            "data_sources": [asdict(ds) for ds in data_sources],
            "context_layers": {
                layer_name: asdict(layer_data)
                for layer_name, layer_data in zip(self.context_layers, context_layers)
            },
            "evidence": [asdict(e) for e in evidence],
            "user_hypotheses": hypotheses,
            "confidence_score": self._calculate_overall_confidence(context_layers, evidence),
            "completeness_score": self._calculate_completeness_score(context_layers),
            "context_summary": await self._generate_context_summary(context_layers, evidence)
        }

        return case_context

    async def _gather_all_data_sources(
        self,
        customer_profile_id: int,
        session_context: Dict[str, Any],
        agent_id: int
    ) -> List[DataSource]:
        """Gather data from all available sources"""

        data_sources = []

        # Customer Profile Database
        data_sources.append(DataSource(
            source_id="customer_profile",
            source_type="database",
            priority=10,
            last_updated=datetime.now(),
            data=await self._get_customer_profile_data(customer_profile_id),
            reliability_score=0.95
        ))

        # Memory System
        data_sources.append(DataSource(
            source_id="customer_memory",
            source_type="memory",
            priority=9,
            last_updated=datetime.now(),
            data=await self._get_memory_data(customer_profile_id),
            reliability_score=0.90
        ))

        # Conversation History
        data_sources.append(DataSource(
            source_id="conversation_history",
            source_type="interactions",
            priority=8,
            last_updated=datetime.now(),
            data=await self._get_conversation_data(customer_profile_id),
            reliability_score=0.95
        ))

        # Session Context
        data_sources.append(DataSource(
            source_id="session_context",
            source_type="realtime",
            priority=7,
            last_updated=datetime.now(),
            data=session_context,
            reliability_score=0.85
        ))

        # Web Analytics (if available)
        data_sources.append(DataSource(
            source_id="web_analytics",
            source_type="behavioral",
            priority=6,
            last_updated=datetime.now(),
            data=await self._get_web_analytics_data(customer_profile_id),
            reliability_score=0.80
        ))

        # Business Systems (CRM, Billing, Support)
        data_sources.append(DataSource(
            source_id="business_systems",
            source_type="business",
            priority=5,
            last_updated=datetime.now(),
            data=await self._get_business_systems_data(customer_profile_id),
            reliability_score=0.90
        ))

        return data_sources

    async def _build_identity_context(
        self,
        customer_profile_id: int,
        data_sources: List[DataSource]
    ) -> ContextLayer:
        """Build identity context layer"""

        identity_data = {}

        # Aggregate identity information from all sources
        for source in data_sources:
            if "name" in source.data:
                identity_data["name"] = source.data["name"]
            if "email" in source.data:
                identity_data["email"] = source.data["email"]
            if "company" in source.data:
                identity_data["company"] = source.data["company"]
            if "role" in source.data:
                identity_data["role"] = source.data["role"]

        # Infer additional identity characteristics
        identity_data["user_type"] = await self._infer_user_type(data_sources)
        identity_data["experience_level"] = await self._infer_experience_level(data_sources)
        identity_data["authority_level"] = await self._infer_authority_level(data_sources)

        return ContextLayer(
            layer_name="identity",
            context_data=identity_data,
            confidence_score=self._calculate_layer_confidence(identity_data, data_sources),
            source_count=len([s for s in data_sources if any(k in s.data for k in identity_data.keys())]),
            last_updated=datetime.now()
        )

    async def _build_behavioral_context(
        self,
        customer_profile_id: int,
        data_sources: List[DataSource]
    ) -> ContextLayer:
        """Build behavioral context layer"""

        behavioral_patterns = {}

        # Communication patterns
        behavioral_patterns["communication_style"] = await self._analyze_communication_style(data_sources)
        behavioral_patterns["response_patterns"] = await self._analyze_response_patterns(data_sources)
        behavioral_patterns["engagement_patterns"] = await self._analyze_engagement_patterns(data_sources)
        behavioral_patterns["decision_making_style"] = await self._infer_decision_making_style(data_sources)

        # Usage patterns
        behavioral_patterns["usage_frequency"] = await self._calculate_usage_frequency(data_sources)
        behavioral_patterns["preferred_channels"] = await self._identify_preferred_channels(data_sources)
        behavioral_patterns["time_preferences"] = await self._analyze_time_preferences(data_sources)

        # Problem-solving patterns
        behavioral_patterns["help_seeking_behavior"] = await self._analyze_help_seeking(data_sources)
        behavioral_patterns["technical_comfort"] = await self._assess_technical_comfort(data_sources)

        return ContextLayer(
            layer_name="behavioral",
            context_data=behavioral_patterns,
            confidence_score=self._calculate_layer_confidence(behavioral_patterns, data_sources),
            source_count=len(data_sources),
            last_updated=datetime.now()
        )

    async def _build_contextual_situation(
        self,
        current_message: str,
        session_context: Dict[str, Any],
        data_sources: List[DataSource]
    ) -> ContextLayer:
        """Build current situational context"""

        situation_data = {
            "current_message": current_message,
            "message_intent": await self._analyze_message_intent(current_message),
            "session_context": session_context,
            "current_page": session_context.get("current_page", "unknown"),
            "referrer": session_context.get("referrer", "unknown"),
            "time_of_day": datetime.now().hour,
            "day_of_week": datetime.now().weekday(),
            "session_duration": session_context.get("session_duration", 0),
            "pages_visited": session_context.get("pages_visited", []),
            "previous_searches": session_context.get("searches", [])
        }

        # Contextual urgency assessment
        situation_data["urgency_indicators"] = await self._assess_urgency_indicators(
            current_message, session_context, data_sources
        )

        return ContextLayer(
            layer_name="contextual",
            context_data=situation_data,
            confidence_score=0.90,  # High confidence in current data
            source_count=1,
            last_updated=datetime.now()
        )

    async def _build_predictive_context(
        self,
        customer_profile_id: int,
        current_message: str,
        data_sources: List[DataSource]
    ) -> ContextLayer:
        """Build predictive context about user intentions and needs"""

        predictions = {}

        # Intent prediction
        predictions["likely_next_actions"] = await self._predict_next_actions(current_message, data_sources)
        predictions["conversion_probability"] = await self._predict_conversion(data_sources)
        predictions["churn_risk"] = await self._assess_churn_risk(data_sources)
        predictions["escalation_likelihood"] = await self._predict_escalation_need(current_message, data_sources)

        # Content predictions
        predictions["information_needs"] = await self._predict_information_needs(current_message, data_sources)
        predictions["question_categories"] = await self._predict_follow_up_questions(current_message, data_sources)

        return ContextLayer(
            layer_name="predictive",
            context_data=predictions,
            confidence_score=0.70,  # Lower confidence for predictions
            source_count=len(data_sources),
            last_updated=datetime.now()
        )

    async def _generate_user_hypotheses(
        self,
        customer_profile_id: int,
        current_message: str,
        context_layers: List[ContextLayer],
        evidence: List[CaseEvidence]
    ) -> List[Dict[str, Any]]:
        """Generate smart hypotheses about the user based on all available data"""

        hypotheses = []

        # Analyze patterns to generate hypotheses
        strong_evidence = [e for e in evidence if e.confidence > 0.8]

        # Generate hypotheses using AI
        hypothesis_prompt = f"""
Based on this user data, generate 3-5 smart hypotheses about who this user is and what they need.

Current message: "{current_message}"

Key evidence:
{chr(10).join([f"- {e.claim} (confidence: {e.confidence:.2f})" for e in strong_evidence[:5]])}

Context summary:
- Identity: {context_layers[0].context_data if len(context_layers) > 0 else 'Limited data'}
- Behavior: {context_layers[1].context_data if len(context_layers) > 1 else 'Limited data'}

Generate hypotheses in this format:
1. [Hypothesis about user type/role]
2. [Hypothesis about current need/goal]
3. [Hypothesis about preferred interaction style]
4. [Hypothesis about technical level]
5. [Hypothesis about decision-making authority]

Each hypothesis should be specific and actionable for a concierge.
"""

        try:
            hypotheses_text = await gemini_service.generate_response(
                prompt=hypothesis_prompt,
                temperature=0.3,
                max_tokens=300
            )

            # Parse hypotheses
            for i, line in enumerate(hypotheses_text.split('\n'), 1):
                if line.strip() and (line.strip().startswith(f"{i}.") or line.strip().startswith("-")):
                    hypotheses.append({
                        "id": f"hyp_{i}",
                        "hypothesis": line.strip().split(".", 1)[1].strip() if "." in line else line.strip(),
                        "confidence": 0.6 + (len(strong_evidence) * 0.1),  # Confidence based on evidence
                        "supporting_evidence": [e.claim for e in strong_evidence[:2]],
                        "generated_at": datetime.now().isoformat()
                    })

        except Exception as e:
            # Fallback hypotheses based on patterns
            hypotheses = [
                {
                    "id": "hyp_fallback",
                    "hypothesis": "User seeking information or problem resolution",
                    "confidence": 0.5,
                    "supporting_evidence": ["Current message indicates help-seeking behavior"],
                    "generated_at": datetime.now().isoformat()
                }
            ]

        return hypotheses

    async def _generate_context_summary(
        self,
        context_layers: List[ContextLayer],
        evidence: List[CaseEvidence]
    ) -> str:
        """Generate a concise summary of all context"""

        summary_parts = []

        # Summarize each layer
        for i, layer in enumerate(context_layers):
            layer_name = self.context_layers[i] if i < len(self.context_layers) else f"layer_{i}"

            if layer.context_data:
                key_points = list(layer.context_data.keys())[:3]
                summary_parts.append(f"{layer_name.title()}: {', '.join(key_points)}")

        # Add strongest evidence
        strong_evidence = sorted(evidence, key=lambda x: x.confidence, reverse=True)[:3]
        if strong_evidence:
            summary_parts.append(f"Key insights: {'; '.join([e.claim for e in strong_evidence])}")

        return " | ".join(summary_parts)

    # Helper methods for data gathering
    async def _get_customer_profile_data(self, customer_profile_id: int) -> Dict[str, Any]:
        """Get customer profile from database"""
        # Would integrate with your database
        return {"profile": "data"}

    async def _get_memory_data(self, customer_profile_id: int) -> Dict[str, Any]:
        """Get memory data"""
        # Would integrate with memory service
        return {"memories": []}

    async def _get_conversation_data(self, customer_profile_id: int) -> Dict[str, Any]:
        """Get conversation history"""
        return {"conversations": []}

    async def _get_web_analytics_data(self, customer_profile_id: int) -> Dict[str, Any]:
        """Get web analytics data"""
        return {"page_views": [], "sessions": []}

    async def _get_business_systems_data(self, customer_profile_id: int) -> Dict[str, Any]:
        """Get business system data"""
        return {"purchases": [], "support_tickets": []}

    # Analysis helper methods
    async def _analyze_message_intent(self, message: str) -> str:
        """Analyze the intent of current message"""
        return "information_seeking"

    async def _infer_user_type(self, data_sources: List[DataSource]) -> str:
        """Infer user type from data"""
        return "technical_evaluator"

    # Additional analysis methods would be implemented...
    async def _infer_experience_level(self, data_sources: List[DataSource]) -> str:
        return "intermediate"

    async def _infer_authority_level(self, data_sources: List[DataSource]) -> str:
        return "decision_maker"

    async def _analyze_communication_style(self, data_sources: List[DataSource]) -> str:
        return "professional"

    def _calculate_layer_confidence(self, layer_data: Dict, data_sources: List[DataSource]) -> float:
        """Calculate confidence score for a context layer"""
        if not layer_data:
            return 0.0

        # Base confidence on data richness and source reliability
        data_richness = len(layer_data) / 10  # Normalize to 0-1
        avg_source_reliability = sum(s.reliability_score for s in data_sources) / len(data_sources)

        return min(0.95, (data_richness * 0.5) + (avg_source_reliability * 0.5))

    def _calculate_overall_confidence(self, context_layers: List[ContextLayer], evidence: List[CaseEvidence]) -> float:
        """Calculate overall confidence in our understanding"""
        if not context_layers:
            return 0.0

        layer_confidences = [layer.confidence_score for layer in context_layers]
        evidence_strength = sum(e.confidence for e in evidence) / len(evidence) if evidence else 0.5

        return (sum(layer_confidences) / len(layer_confidences)) * 0.7 + evidence_strength * 0.3

    def _calculate_completeness_score(self, context_layers: List[ContextLayer]) -> float:
        """Calculate how complete our context understanding is"""
        total_possible = len(self.context_layers)
        layers_with_data = sum(1 for layer in context_layers if layer.context_data)

        return layers_with_data / total_possible

    # Placeholder methods for various analysis functions
    async def _analyze_response_patterns(self, data_sources: List[DataSource]) -> Dict:
        return {"average_response_time": "2_minutes", "preferred_length": "medium"}

    async def _analyze_engagement_patterns(self, data_sources: List[DataSource]) -> Dict:
        return {"frequency": "weekly", "depth": "thorough"}

    async def _infer_decision_making_style(self, data_sources: List[DataSource]) -> str:
        return "analytical"

    async def _calculate_usage_frequency(self, data_sources: List[DataSource]) -> str:
        return "regular"

    async def _identify_preferred_channels(self, data_sources: List[DataSource]) -> List[str]:
        return ["chat", "email"]

    async def _analyze_time_preferences(self, data_sources: List[DataSource]) -> Dict:
        return {"preferred_hours": "business_hours", "timezone": "UTC-8"}

    async def _analyze_help_seeking(self, data_sources: List[DataSource]) -> str:
        return "self_service_first"

    async def _assess_technical_comfort(self, data_sources: List[DataSource]) -> str:
        return "high"

    async def _assess_urgency_indicators(self, message: str, session_context: Dict, data_sources: List[DataSource]) -> List[str]:
        return []

    async def _predict_next_actions(self, message: str, data_sources: List[DataSource]) -> List[str]:
        return ["ask_follow_up", "request_demo"]

    async def _predict_conversion(self, data_sources: List[DataSource]) -> float:
        return 0.65

    async def _assess_churn_risk(self, data_sources: List[DataSource]) -> float:
        return 0.15

    async def _predict_escalation_need(self, message: str, data_sources: List[DataSource]) -> float:
        return 0.1

    async def _predict_information_needs(self, message: str, data_sources: List[DataSource]) -> List[str]:
        return ["pricing", "features", "integration"]

    async def _predict_follow_up_questions(self, message: str, data_sources: List[DataSource]) -> List[str]:
        return ["how_to_implement", "cost_details"]

    async def _gather_case_evidence(self, customer_profile_id: int, data_sources: List[DataSource]) -> List[CaseEvidence]:
        """Gather evidence to support insights"""
        evidence = []

        # Example evidence gathering
        for source in data_sources:
            if source.reliability_score > 0.8:
                evidence.append(CaseEvidence(
                    evidence_type="behavioral",
                    claim=f"High-reliability data from {source.source_type}",
                    supporting_data=[str(source.data)[:100]],
                    confidence=source.reliability_score,
                    timestamp=datetime.now()
                ))

        return evidence


    # Additional context layer builders
    async def _build_historical_context(self, customer_profile_id: int, data_sources: List[DataSource]) -> ContextLayer:
        """Build historical context layer"""
        historical_data = {"interaction_count": 0, "first_seen": "unknown", "key_milestones": []}
        return ContextLayer(
            layer_name="historical",
            context_data=historical_data,
            confidence_score=0.7,
            source_count=1,
            last_updated=datetime.now()
        )

    async def _build_emotional_intelligence(self, customer_profile_id: int, message: str, data_sources: List[DataSource]) -> ContextLayer:
        """Build emotional intelligence layer"""
        emotional_data = {"current_sentiment": "neutral", "emotional_history": [], "triggers": []}
        return ContextLayer(
            layer_name="emotional",
            context_data=emotional_data,
            confidence_score=0.6,
            source_count=1,
            last_updated=datetime.now()
        )

    async def _build_business_context(self, customer_profile_id: int, data_sources: List[DataSource]) -> ContextLayer:
        """Build business context layer"""
        business_data = {"company_size": "unknown", "industry": "unknown", "budget_range": "unknown"}
        return ContextLayer(
            layer_name="business",
            context_data=business_data,
            confidence_score=0.5,
            source_count=1,
            last_updated=datetime.now()
        )

    async def _build_technical_context(self, customer_profile_id: int, data_sources: List[DataSource]) -> ContextLayer:
        """Build technical context layer"""
        technical_data = {"expertise_level": "intermediate", "tech_stack": [], "integration_needs": []}
        return ContextLayer(
            layer_name="technical",
            context_data=technical_data,
            confidence_score=0.6,
            source_count=1,
            last_updated=datetime.now()
        )

    async def _build_relational_context(self, customer_profile_id: int, data_sources: List[DataSource]) -> ContextLayer:
        """Build relational context layer"""
        relational_data = {"relationship_stage": "new", "trust_level": 0.5, "satisfaction_trend": "neutral"}
        return ContextLayer(
            layer_name="relational",
            context_data=relational_data,
            confidence_score=0.7,
            source_count=1,
            last_updated=datetime.now()
        )


# Global instance
contextual_case_builder = ContextualCaseBuilder()