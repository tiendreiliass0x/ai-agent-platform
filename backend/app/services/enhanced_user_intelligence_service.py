"""
Enhanced User Intelligence Service with Governance and Confidence

Advanced user behavior analysis with privacy controls, confidence scoring,
and safety detection for enterprise-grade customer intelligence.
"""

import asyncio
import json
import re
import math
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from app.services.gemini_service import gemini_service
from app.services.memory_service import memory_service
from ..core.governance import (
    ConsentContext, ConsentScope, PredictionWithConfidence, EvidenceItem, SafetyFlags,
    InferenceCategory, governance_engine
)
from .enhanced_concierge_case import (
    EnhancedConciergeCase, CustomerEntitlements, CustomerTier, TouchpointWeight,
    BusinessMetrics, NextBestAction, ProvenanceInfo
)


class UrgencyLevel(str, Enum):
    """User urgency detection - simplified as per GPT-5-Pro recommendations"""
    CRITICAL = "critical"      # Service down, time-bound issues
    HIGH = "high"             # Priority response required
    NORMAL = "normal"         # Standard response time
    LOW = "low"              # Can be queued


class EmotionalState(str, Enum):
    """Simplified emotional states with higher accuracy"""
    CALM = "calm"
    FRUSTRATED = "frustrated"
    CONFUSED = "confused"
    URGENT = "urgent"
    NEUTRAL = "neutral"


class IntentCategory(str, Enum):
    """Refined intent categories for better classification"""
    INFO = "info"                     # Looking for information
    TROUBLESHOOT = "troubleshoot"     # Solving technical issues
    PURCHASE = "purchase"             # Buying or upgrading
    SUPPORT = "support"               # Help with existing service
    COMPLAINT = "complaint"           # Expressing dissatisfaction
    COMPARE = "compare"               # Comparing options
    BILLING = "billing"               # Billing and payment issues
    ACCOUNT = "account"               # Account management
    POLICY = "policy"                 # Policy questions
    OTHER = "other"                   # Unclear or mixed intent


class RiskCategory(str, Enum):
    """Risk assessment categories"""
    FRAUD_SUSPECTED = "fraud_suspected"
    CHURN_RISK = "churn_risk"
    ESCALATION_RISK = "escalation_risk"
    NONE = "none"


@dataclass
class IntelligenceAnalysis:
    """Complete intelligence analysis with confidence and governance"""
    # Core predictions
    intent: PredictionWithConfidence
    urgency: PredictionWithConfidence
    emotion: PredictionWithConfidence
    risk: PredictionWithConfidence

    # Extracted insights
    key_topics: List[str]
    pain_points: List[str]
    opportunities: List[str]
    personalization_cues: Dict[str, Any]

    # Context and evidence
    evidence_items: List[EvidenceItem]
    confidence_score: float
    processing_time_ms: float

    # Safety and governance
    safety_flags: SafetyFlags
    consent_compliant: bool


class EnhancedUserIntelligenceService:
    """Enhanced user intelligence with governance, confidence, and safety"""

    def __init__(self):
        # Improved pattern matching for higher accuracy
        self.emotional_indicators = {
            EmotionalState.FRUSTRATED: [
                "frustrated", "annoying", "ridiculous", "terrible", "awful",
                "hate", "stupid", "useless", "broken", "not working", "failed"
            ],
            EmotionalState.CONFUSED: [
                "confused", "don't understand", "unclear", "how do", "what does",
                "help me understand", "explain", "lost", "stuck"
            ],
            EmotionalState.URGENT: [
                "urgent", "asap", "immediately", "right now", "emergency",
                "critical", "deadline", "time sensitive"
            ],
            EmotionalState.CALM: [
                "thank you", "appreciate", "helpful", "great", "perfect",
                "exactly what", "solved"
            ]
        }

        self.urgency_indicators = {
            UrgencyLevel.CRITICAL: [
                "down", "not working", "emergency", "urgent", "critical",
                "production", "outage", "system failure"
            ],
            UrgencyLevel.HIGH: [
                "asap", "soon", "priority", "important", "deadline",
                "time sensitive", "need help"
            ],
            UrgencyLevel.LOW: [
                "when you can", "no rush", "eventually", "curious",
                "wondering", "maybe"
            ]
        }

        self.intent_patterns = {
            IntentCategory.TROUBLESHOOT: [
                r"\bnot working\b", r"\berror\b", r"\bproblem\b", r"\bissue\b",
                r"\bbroken\b", r"\bfailed\b", r"\bfix\b", r"\btrouble\b"
            ],
            IntentCategory.PURCHASE: [
                r"\bbuy\b", r"\bpurchase\b", r"\bget\b", r"\border\b",
                r"\bprice\b", r"\bcost\b", r"\bupgrade\b", r"\bplan\b"
            ],
            IntentCategory.BILLING: [
                r"\bbill\b", r"\bcharge\b", r"\bpayment\b", r"\binvoice\b",
                r"\brefund\b", r"\bcredit\b"
            ],
            IntentCategory.COMPLAINT: [
                r"\bcomplaint\b", r"\bunhappy\b", r"\bdissatisfied\b",
                r"\bterrible\b", r"\bawful\b"
            ]
        }

        # Fraud and abuse patterns
        self.risk_patterns = {
            "fraud": [
                r"free\s+money", r"get\s+rich", r"click\s+here",
                r"urgent\s+action", r"verify\s+account"
            ],
            "churn": [
                r"cancel", r"close\s+account", r"switching\s+to",
                r"competitor", r"dissatisfied"
            ]
        }

    async def analyze_user_message(
        self,
        message: str,
        customer_profile_id: Optional[int] = None,
        conversation_history: List[Dict[str, str]] = None,
        session_context: Dict[str, Any] = None,
        consent_context: Optional[ConsentContext] = None
    ) -> IntelligenceAnalysis:
        """
        Comprehensive user intelligence analysis with governance controls
        """
        start_time = datetime.now()

        # Safety check first
        safety_flags = governance_engine.detect_safety_issues(message)
        if safety_flags.requires_human_review:
            return self._create_safety_fallback_analysis(message, safety_flags)

        # Validate consent for different types of analysis
        if consent_context is None:
            consent_context = ConsentContext()  # Default minimal consent

        consent_compliant = self._validate_analysis_consent(consent_context)

        # Run parallel analysis tasks
        analysis_tasks = []

        # Core predictions (always allowed)
        analysis_tasks.extend([
            self._detect_emotional_state_with_confidence(message, conversation_history or []),
            self._detect_urgency_level_with_confidence(message, session_context or {}),
            self._predict_intent_with_confidence(message, conversation_history or []),
            self._assess_risk_with_confidence(message, conversation_history or [])
        ])

        # Additional analysis (consent-dependent)
        if consent_compliant:
            analysis_tasks.extend([
                self._extract_key_topics(message),
                self._identify_pain_points(message),
                self._spot_opportunities(message, conversation_history or []),
                self._extract_personalization_cues(message, session_context or {})
            ])

        # Execute all tasks in parallel
        results = await asyncio.gather(*analysis_tasks, return_exceptions=True)

        # Parse results (handle any exceptions)
        emotion_pred = results[0] if not isinstance(results[0], Exception) else self._default_emotion_prediction()
        urgency_pred = results[1] if not isinstance(results[1], Exception) else self._default_urgency_prediction()
        intent_pred = results[2] if not isinstance(results[2], Exception) else self._default_intent_prediction()
        risk_pred = results[3] if not isinstance(results[3], Exception) else self._default_risk_prediction()

        # Optional results (only if consent given)
        if consent_compliant and len(results) > 4:
            key_topics = results[4] if not isinstance(results[4], Exception) else []
            pain_points = results[5] if not isinstance(results[5], Exception) else []
            opportunities = results[6] if not isinstance(results[6], Exception) else []
            personalization_cues = results[7] if not isinstance(results[7], Exception) else {}
        else:
            key_topics, pain_points, opportunities, personalization_cues = [], [], [], {}

        # Calculate overall confidence
        confidence_score = self._calculate_overall_confidence([
            emotion_pred, urgency_pred, intent_pred, risk_pred
        ])

        # Create evidence items
        evidence_items = self._build_evidence_items(
            message, emotion_pred, urgency_pred, intent_pred, consent_context
        )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return IntelligenceAnalysis(
            intent=intent_pred,
            urgency=urgency_pred,
            emotion=emotion_pred,
            risk=risk_pred,
            key_topics=key_topics,
            pain_points=pain_points,
            opportunities=opportunities,
            personalization_cues=personalization_cues,
            evidence_items=evidence_items,
            confidence_score=confidence_score,
            processing_time_ms=processing_time,
            safety_flags=safety_flags,
            consent_compliant=consent_compliant
        )

    async def _detect_emotional_state_with_confidence(
        self,
        message: str,
        conversation_history: List[Dict[str, str]]
    ) -> PredictionWithConfidence:
        """Detect emotional state with confidence scoring"""

        message_lower = message.lower()
        evidence_count = 0

        # Fast keyword matching first
        for emotion, keywords in self.emotional_indicators.items():
            matches = sum(1 for keyword in keywords if keyword in message_lower)
            if matches > 0:
                confidence = min(0.6 + (matches * 0.1), 0.9)  # Base confidence from keywords
                evidence_strength = min(matches / 3.0, 1.0)    # Evidence strength
                return PredictionWithConfidence(
                    label=emotion.value,
                    probability=confidence,
                    confidence=confidence,
                    evidence_strength=evidence_strength,
                    evidence_count=matches,
                    model_version="keyword_v1.0"
                )

        # AI analysis for nuanced detection
        try:
            emotion_prompt = f"""
Analyze the emotional state in this message. Respond with only one word from: calm, frustrated, confused, urgent, neutral

Message: "{message}"

Previous context: {conversation_history[-2:] if conversation_history else "None"}

Emotion:"""

            emotion_result = await gemini_service.generate_response(
                prompt=emotion_prompt,
                temperature=0.1,  # Low temperature for consistent classification
                max_tokens=10
            )

            emotion_label = emotion_result.strip().lower()

            # Validate the response
            valid_emotions = [e.value for e in EmotionalState]
            if emotion_label not in valid_emotions:
                emotion_label = EmotionalState.NEUTRAL.value

            # AI predictions have medium confidence
            return PredictionWithConfidence(
                label=emotion_label,
                probability=0.7,
                confidence=0.6,  # Medium confidence for AI predictions
                evidence_strength=0.5,
                evidence_count=1,
                model_version="gemini_emotion_v1.0"
            )

        except Exception as e:
            # Fallback to neutral with low confidence
            return PredictionWithConfidence(
                label=EmotionalState.NEUTRAL.value,
                probability=0.5,
                confidence=0.3,
                evidence_strength=0.2,
                evidence_count=0,
                model_version="fallback_v1.0"
            )

    async def _detect_urgency_level_with_confidence(
        self,
        message: str,
        session_context: Dict[str, Any]
    ) -> PredictionWithConfidence:
        """Detect urgency with multi-factor analysis"""

        message_lower = message.lower()
        urgency_score = 0.0
        evidence_factors = []

        # Factor 1: Keyword-based urgency
        for urgency, keywords in self.urgency_indicators.items():
            matches = sum(1 for keyword in keywords if keyword in message_lower)
            if matches > 0:
                if urgency == UrgencyLevel.CRITICAL:
                    urgency_score += matches * 0.4
                elif urgency == UrgencyLevel.HIGH:
                    urgency_score += matches * 0.3
                elif urgency == UrgencyLevel.LOW:
                    urgency_score -= matches * 0.2
                evidence_factors.append(f"keywords_{urgency.value}")

        # Factor 2: Page context
        current_page = session_context.get("current_page", "")
        if "/support" in current_page or "/help" in current_page:
            urgency_score += 0.2
            evidence_factors.append("support_page")

        # Factor 3: Time patterns (off-hours = higher urgency)
        current_hour = datetime.now().hour
        if current_hour < 8 or current_hour > 18:  # Outside business hours
            urgency_score += 0.1
            evidence_factors.append("off_hours")

        # Factor 4: Message patterns
        if "!!!" in message or message.isupper():
            urgency_score += 0.3
            evidence_factors.append("emphasis_markers")

        # Determine urgency level
        if urgency_score >= 0.7:
            label = UrgencyLevel.CRITICAL
        elif urgency_score >= 0.4:
            label = UrgencyLevel.HIGH
        elif urgency_score >= 0.1:
            label = UrgencyLevel.NORMAL
        else:
            label = UrgencyLevel.LOW

        confidence = min(0.5 + (len(evidence_factors) * 0.1), 0.9)
        evidence_strength = min(urgency_score, 1.0)

        return PredictionWithConfidence(
            label=label.value,
            probability=confidence,
            confidence=confidence,
            evidence_strength=evidence_strength,
            evidence_count=len(evidence_factors),
            model_version="urgency_multiactor_v1.0"
        )

    async def _predict_intent_with_confidence(
        self,
        message: str,
        conversation_history: List[Dict[str, str]]
    ) -> PredictionWithConfidence:
        """Predict intent with pattern matching and AI backup"""

        message_lower = message.lower()

        # Pattern-based intent detection
        for intent, patterns in self.intent_patterns.items():
            matches = sum(1 for pattern in patterns if re.search(pattern, message_lower))
            if matches > 0:
                confidence = min(0.6 + (matches * 0.1), 0.8)
                evidence_strength = min(matches / 2.0, 1.0)
                return PredictionWithConfidence(
                    label=intent.value,
                    probability=confidence,
                    confidence=confidence,
                    evidence_strength=evidence_strength,
                    evidence_count=matches,
                    model_version="intent_patterns_v1.0"
                )

        # AI-based intent prediction for complex cases
        try:
            intent_prompt = f"""
Classify the user's intent. Choose exactly one from: info, troubleshoot, purchase, support, complaint, compare, billing, account, policy, other

Message: "{message}"
Context: {conversation_history[-1:] if conversation_history else "None"}

Intent:"""

            intent_result = await gemini_service.generate_response(
                prompt=intent_prompt,
                temperature=0.1,
                max_tokens=10
            )

            intent_label = intent_result.strip().lower()

            # Validate response
            valid_intents = [i.value for i in IntentCategory]
            if intent_label not in valid_intents:
                intent_label = IntentCategory.OTHER.value

            return PredictionWithConfidence(
                label=intent_label,
                probability=0.6,
                confidence=0.5,  # Medium confidence for AI
                evidence_strength=0.4,
                evidence_count=1,
                model_version="gemini_intent_v1.0"
            )

        except Exception:
            return PredictionWithConfidence(
                label=IntentCategory.OTHER.value,
                probability=0.4,
                confidence=0.3,
                evidence_strength=0.2,
                evidence_count=0,
                model_version="fallback_v1.0"
            )

    async def _assess_risk_with_confidence(
        self,
        message: str,
        conversation_history: List[Dict[str, str]]
    ) -> PredictionWithConfidence:
        """Assess various risk categories"""

        message_lower = message.lower()
        risk_score = 0.0
        detected_risks = []

        # Check for fraud patterns
        for pattern in self.risk_patterns["fraud"]:
            if re.search(pattern, message_lower):
                risk_score += 0.4
                detected_risks.append("fraud_pattern")

        # Check for churn indicators
        for pattern in self.risk_patterns["churn"]:
            if re.search(pattern, message_lower):
                risk_score += 0.3
                detected_risks.append("churn_pattern")

        # Determine risk level
        if risk_score >= 0.5:
            if "fraud" in str(detected_risks):
                label = RiskCategory.FRAUD_SUSPECTED
            else:
                label = RiskCategory.CHURN_RISK
        elif risk_score >= 0.2:
            label = RiskCategory.ESCALATION_RISK
        else:
            label = RiskCategory.NONE

        confidence = min(0.4 + (len(detected_risks) * 0.2), 0.8)
        evidence_strength = min(risk_score, 1.0)

        return PredictionWithConfidence(
            label=label.value,
            probability=confidence,
            confidence=confidence,
            evidence_strength=evidence_strength,
            evidence_count=len(detected_risks),
            model_version="risk_patterns_v1.0"
        )

    # Helper methods for default predictions
    def _default_emotion_prediction(self) -> PredictionWithConfidence:
        return PredictionWithConfidence(
            label=EmotionalState.NEUTRAL.value,
            probability=0.5, confidence=0.3, evidence_strength=0.1,
            evidence_count=0, model_version="default_v1.0"
        )

    def _default_urgency_prediction(self) -> PredictionWithConfidence:
        return PredictionWithConfidence(
            label=UrgencyLevel.NORMAL.value,
            probability=0.5, confidence=0.3, evidence_strength=0.1,
            evidence_count=0, model_version="default_v1.0"
        )

    def _default_intent_prediction(self) -> PredictionWithConfidence:
        return PredictionWithConfidence(
            label=IntentCategory.OTHER.value,
            probability=0.4, confidence=0.3, evidence_strength=0.1,
            evidence_count=0, model_version="default_v1.0"
        )

    def _default_risk_prediction(self) -> PredictionWithConfidence:
        return PredictionWithConfidence(
            label=RiskCategory.NONE.value,
            probability=0.8, confidence=0.7, evidence_strength=0.8,
            evidence_count=0, model_version="default_v1.0"
        )

    async def _extract_key_topics(self, message: str) -> List[str]:
        """Extract key topics and entities"""
        # Simple keyword extraction - could be enhanced with NER
        topics = []

        # Technical terms
        tech_terms = ["api", "integration", "sdk", "webhook", "database", "ssl", "authentication"]
        for term in tech_terms:
            if term in message.lower():
                topics.append(term)

        # Business terms
        business_terms = ["pricing", "plan", "billing", "account", "subscription", "feature"]
        for term in business_terms:
            if term in message.lower():
                topics.append(term)

        return topics[:5]  # Limit to top 5

    async def _identify_pain_points(self, message: str) -> List[str]:
        """Identify customer pain points"""
        pain_indicators = [
            ("confusing", "unclear documentation"),
            ("slow", "performance issues"),
            ("expensive", "cost concerns"),
            ("difficult", "usability problems"),
            ("broken", "functionality issues")
        ]

        pain_points = []
        message_lower = message.lower()

        for indicator, pain_point in pain_indicators:
            if indicator in message_lower:
                pain_points.append(pain_point)

        return pain_points

    async def _spot_opportunities(self, message: str, conversation_history: List[Dict[str, str]]) -> List[str]:
        """Identify business opportunities"""
        opportunities = []
        message_lower = message.lower()

        if any(word in message_lower for word in ["demo", "trial", "test"]):
            opportunities.append("demo_request")

        if any(word in message_lower for word in ["upgrade", "more features", "advanced"]):
            opportunities.append("upgrade_potential")

        if any(word in message_lower for word in ["team", "company", "enterprise"]):
            opportunities.append("enterprise_sale")

        return opportunities

    async def _extract_personalization_cues(self, message: str, session_context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract personalization cues"""
        cues = {}
        message_lower = message.lower()

        # Technical level detection
        if any(term in message_lower for term in ['api', 'code', 'integration', 'webhook']):
            cues['technical_level'] = 'expert'
        elif any(term in message_lower for term in ['simple', 'easy', 'beginner', 'how to']):
            cues['technical_level'] = 'beginner'

        # Communication style
        if any(term in message_lower for term in ['quick', 'brief', 'summary']):
            cues['response_style'] = 'concise'
        elif any(term in message_lower for term in ['detail', 'explain', 'walkthrough']):
            cues['response_style'] = 'detailed'

        # Device context
        user_agent = session_context.get("user_agent", "")
        if "mobile" in user_agent.lower():
            cues['device'] = 'mobile'
            cues['response_length'] = 'brief'

        return cues

    def _validate_analysis_consent(self, consent_context: ConsentContext) -> bool:
        """Validate consent for behavioral analysis"""
        return governance_engine.validate_consent(
            consent_context,
            ConsentScope.ANALYZE_BEHAVIOR
        )

    def _calculate_overall_confidence(self, predictions: List[PredictionWithConfidence]) -> float:
        """Calculate overall confidence score"""
        if not predictions:
            return 0.0

        confidences = [p.confidence for p in predictions if p.confidence > 0]
        if not confidences:
            return 0.0

        # Weighted average with emphasis on evidence strength
        weighted_sum = sum(p.confidence * p.evidence_strength for p in predictions)
        weight_sum = sum(p.evidence_strength for p in predictions)

        return weighted_sum / weight_sum if weight_sum > 0 else sum(confidences) / len(confidences)

    def _build_evidence_items(
        self,
        message: str,
        emotion_pred: PredictionWithConfidence,
        urgency_pred: PredictionWithConfidence,
        intent_pred: PredictionWithConfidence,
        consent_context: ConsentContext
    ) -> List[EvidenceItem]:
        """Build evidence items for audit trail"""
        evidence_items = []

        # Redact message if needed
        clean_message = governance_engine.redact_pii(message) if consent_context.pii_redaction_enabled else message

        evidence_items.append(EvidenceItem(
            evidence_type="explicit",
            source_ref="user_message",
            summary=f"User message: {clean_message[:100]}...",
            confidence=1.0
        ))

        if emotion_pred.evidence_count > 0:
            evidence_items.append(EvidenceItem(
                evidence_type="behavioral",
                source_ref="emotion_analysis",
                summary=f"Emotion '{emotion_pred.label}' detected with {emotion_pred.evidence_count} indicators",
                confidence=emotion_pred.confidence
            ))

        return evidence_items

    def _create_safety_fallback_analysis(self, message: str, safety_flags: SafetyFlags) -> IntelligenceAnalysis:
        """Create safe fallback analysis when safety issues detected"""
        return IntelligenceAnalysis(
            intent=self._default_intent_prediction(),
            urgency=PredictionWithConfidence(
                label=UrgencyLevel.HIGH.value,  # Safety issues are high priority
                probability=0.9, confidence=0.9, evidence_strength=0.9,
                evidence_count=1, model_version="safety_override_v1.0"
            ),
            emotion=self._default_emotion_prediction(),
            risk=PredictionWithConfidence(
                label=RiskCategory.ESCALATION_RISK.value,
                probability=0.9, confidence=0.9, evidence_strength=0.9,
                evidence_count=1, model_version="safety_override_v1.0"
            ),
            key_topics=["safety_issue"],
            pain_points=["potential_abuse"],
            opportunities=[],
            personalization_cues={},
            evidence_items=[],
            confidence_score=0.9,  # High confidence in safety detection
            processing_time_ms=0.0,
            safety_flags=safety_flags,
            consent_compliant=False  # Don't do behavioral analysis on safety issues
        )


# Global service instance
enhanced_user_intelligence_service = EnhancedUserIntelligenceService()