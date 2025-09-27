"""
Enhanced ConciergeCase - Comprehensive customer context with governance

This module provides the core data structure that aggregates all customer context
while respecting privacy, consent, and governance requirements.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
from enum import Enum

from ..core.governance import (
    ConsentContext, PredictionWithConfidence, EvidenceItem, SafetyFlags,
    InferenceCategory, DataRetentionPolicy, governance_engine
)


class CustomerTier(str, Enum):
    """Customer tier for entitlements"""
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    VIP = "vip"


@dataclass
class CustomerEntitlements:
    """Customer entitlements and permissions based on their plan"""
    # Plan information
    tier: CustomerTier = CustomerTier.BASIC
    plan_name: str = "Basic Plan"

    # Financial limits
    refund_limit_usd: float = 50.0
    discount_limit_percent: float = 10.0

    # Service levels
    priority_support: bool = False
    sla_hours: int = 48
    dedicated_success_manager: bool = False

    # Feature access
    feature_access: Set[str] = field(default_factory=lambda: {"basic_chat", "knowledge_base"})
    api_rate_limit: int = 100  # requests per hour

    # Geographic and legal
    region: str = "US"
    data_residency_required: bool = False
    compliance_requirements: Set[str] = field(default_factory=set)  # GDPR, CCPA, etc.


@dataclass
class TouchpointWeight:
    """Weighted touchpoint with time decay"""
    touchpoint_id: str
    source: str                    # "support_ticket", "page_view", "purchase", etc.
    base_weight: float
    timestamp: datetime
    decay_rate: float = 0.3        # λ for exponential decay

    @property
    def current_weight(self) -> float:
        """Calculate current weight with time decay"""
        age_days = (datetime.now() - self.timestamp).days
        decayed_weight = self.base_weight * math.exp(-self.decay_rate * age_days)
        return max(decayed_weight, 0.01)  # Minimum threshold

    @property
    def age_days(self) -> int:
        return (datetime.now() - self.timestamp).days


@dataclass
class BusinessMetrics:
    """Business intelligence and metrics"""
    # Customer value
    clv_estimate: float = 0.0
    total_spent: float = 0.0
    avg_order_value: float = 0.0

    # Risk assessment
    churn_probability: float = 0.0
    escalation_risk: float = 0.0
    satisfaction_trend: str = "stable"  # "improving", "declining", "stable"

    # Opportunities
    upsell_opportunities: List[str] = field(default_factory=list)
    cross_sell_potential: List[str] = field(default_factory=list)
    renewal_likelihood: float = 0.5


@dataclass
class NextBestAction:
    """Recommended action with business context"""
    action_id: str
    action_type: str               # "guided_troubleshooting", "offer_discount", "escalate", etc.
    description: str
    expected_value: float          # Business value (revenue, satisfaction points, etc.)
    confidence: float              # How confident we are this is the right action
    preconditions: List[str] = field(default_factory=list)
    estimated_effort: str = "medium"  # "low", "medium", "high"
    requires_approval: bool = False


@dataclass
class ProvenanceInfo:
    """Provenance and model information"""
    model_versions: Dict[str, str] = field(default_factory=dict)
    data_sources: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)
    processing_time_ms: float = 0.0
    feature_importance: Dict[str, float] = field(default_factory=dict)


@dataclass
class EnhancedConciergeCase:
    """
    Comprehensive customer context with governance controls

    This is the central data structure that aggregates all customer intelligence
    while respecting privacy, consent, and business requirements.
    """

    # Core Identity
    case_id: str
    customer_ref: str
    session_id: str

    # Governance and Consent
    consent_context: ConsentContext
    safety_flags: SafetyFlags

    # Predictions with Confidence
    intent_prediction: PredictionWithConfidence
    urgency_prediction: PredictionWithConfidence
    emotion_prediction: PredictionWithConfidence

    # Business Context
    entitlements: CustomerEntitlements
    business_metrics: BusinessMetrics

    # Evidence and Touchpoints
    evidence_items: List[EvidenceItem] = field(default_factory=list)
    touchpoint_weights: List[TouchpointWeight] = field(default_factory=list)

    # Session Context
    session_context: Dict[str, Any] = field(default_factory=dict)
    conversation_history_summary: str = ""

    # Recommendations
    next_best_actions: List[NextBestAction] = field(default_factory=list)
    recommended_strategy: str = "guided_resolution"

    # System Metadata
    provenance: ProvenanceInfo = field(default_factory=ProvenanceInfo)
    ttl_hours: int = 24
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def expires_at(self) -> datetime:
        """When this case expires"""
        return self.created_at + timedelta(hours=self.ttl_hours)

    @property
    def is_expired(self) -> bool:
        """Check if case has expired"""
        return datetime.now() > self.expires_at

    @property
    def overall_confidence(self) -> float:
        """Overall confidence across all predictions"""
        predictions = [self.intent_prediction, self.urgency_prediction, self.emotion_prediction]
        confidences = [p.confidence for p in predictions if p.confidence > 0]
        return sum(confidences) / len(confidences) if confidences else 0.0

    @property
    def requires_clarification(self) -> bool:
        """Whether we should ask clarifying questions"""
        return (self.overall_confidence < 0.65 or
                self.intent_prediction.requires_clarification)

    @property
    def requires_human_escalation(self) -> bool:
        """Whether this case requires human intervention"""
        return (self.safety_flags.requires_human_review or
                self.emotion_prediction.label in ["frustrated", "angry"] and self.emotion_prediction.confidence > 0.8 or
                self.entitlements.tier == CustomerTier.VIP and self.urgency_prediction.label == "critical")

    @property
    def top_touchpoints(self, limit: int = 5) -> List[TouchpointWeight]:
        """Get top touchpoints by current weight"""
        return sorted(self.touchpoint_weights,
                     key=lambda t: t.current_weight,
                     reverse=True)[:limit]

    def add_evidence(self, evidence_type: str, source_ref: str, summary: str,
                    confidence: float = 1.0) -> None:
        """Add evidence item with governance checks"""

        # Check if we can store this type of evidence
        if not governance_engine.should_persist_data(evidence_type, self.consent_context):
            # Store as session-only
            retention_policy = DataRetentionPolicy.SESSION_ONLY
        else:
            retention_policy = self.consent_context.data_retention_policy

        evidence = EvidenceItem(
            evidence_type=evidence_type,
            source_ref=source_ref,
            summary=governance_engine.redact_pii(summary) if self.consent_context.pii_redaction_enabled else summary,
            confidence=confidence,
            retention_policy=retention_policy
        )

        self.evidence_items.append(evidence)

    def add_touchpoint(self, touchpoint_id: str, source: str, base_weight: float,
                      timestamp: Optional[datetime] = None) -> None:
        """Add touchpoint with time decay calculation"""

        if timestamp is None:
            timestamp = datetime.now()

        # Get decay rate based on source type
        decay_rates = {
            "support_ticket": 0.1,      # Slow decay - important for longer
            "page_view": 0.5,           # Fast decay - less important over time
            "purchase": 0.05,           # Very slow decay - always relevant
            "complaint": 0.2,           # Medium decay
            "satisfaction_survey": 0.15  # Slow-medium decay
        }

        decay_rate = decay_rates.get(source, 0.3)  # Default decay rate

        touchpoint = TouchpointWeight(
            touchpoint_id=touchpoint_id,
            source=source,
            base_weight=base_weight,
            timestamp=timestamp,
            decay_rate=decay_rate
        )

        self.touchpoint_weights.append(touchpoint)

    def get_evidence_by_type(self, evidence_type: str) -> List[EvidenceItem]:
        """Get evidence items of a specific type"""
        return [e for e in self.evidence_items if e.evidence_type == evidence_type]

    def calculate_context_quality_score(self) -> float:
        """Calculate overall context quality score (0-1)"""
        score = 0.0

        # Prediction confidence (40% of score)
        score += self.overall_confidence * 0.4

        # Evidence strength (30% of score)
        if self.evidence_items:
            evidence_score = sum(e.confidence for e in self.evidence_items) / len(self.evidence_items)
            score += evidence_score * 0.3

        # Touchpoint recency and relevance (20% of score)
        if self.touchpoint_weights:
            touchpoint_score = sum(t.current_weight for t in self.top_touchpoints(3)) / 3
            score += touchpoint_score * 0.2

        # Consent completeness (10% of score)
        consent_score = len(self.consent_context.consents) / len(list(self.consent_context.allowed_inferences))
        score += consent_score * 0.1

        return min(score, 1.0)

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to summary dictionary for API responses"""
        return {
            "case_id": self.case_id,
            "customer_ref": self.customer_ref,
            "overall_confidence": self.overall_confidence,
            "context_quality": self.calculate_context_quality_score(),

            "predictions": {
                "intent": {
                    "label": self.intent_prediction.label,
                    "confidence": self.intent_prediction.confidence
                },
                "urgency": {
                    "label": self.urgency_prediction.label,
                    "confidence": self.urgency_prediction.confidence
                },
                "emotion": {
                    "label": self.emotion_prediction.label,
                    "confidence": self.emotion_prediction.confidence
                }
            },

            "recommendations": {
                "strategy": self.recommended_strategy,
                "requires_clarification": self.requires_clarification,
                "requires_escalation": self.requires_human_escalation,
                "next_actions": [
                    {
                        "action": action.action_type,
                        "description": action.description,
                        "confidence": action.confidence
                    } for action in self.next_best_actions[:3]
                ]
            },

            "business_context": {
                "tier": self.entitlements.tier.value,
                "priority_support": self.entitlements.priority_support,
                "clv_estimate": self.business_metrics.clv_estimate,
                "churn_risk": self.business_metrics.churn_probability
            },

            "governance": {
                "consent_version": self.consent_context.consent_version,
                "data_retention": self.consent_context.data_retention_policy.value,
                "pii_redacted": self.consent_context.pii_redaction_enabled,
                "expires_at": self.expires_at.isoformat()
            }
        }

    def to_audit_log(self) -> Dict[str, Any]:
        """Create audit log entry for this case"""
        return governance_engine.create_audit_log(
            action="concierge_case_created",
            customer_id=self.customer_ref,
            data_access={
                "predictions": ["intent", "urgency", "emotion"],
                "touchpoints": [t.source for t in self.touchpoint_weights],
                "evidence_types": list(set(e.evidence_type for e in self.evidence_items))
            },
            consent_context=self.consent_context
        )


class ConciergeStrategy(str, Enum):
    """Available concierge strategies"""
    GENERIC_HELPFUL = "generic_helpful"           # No personalization
    CLARIFY_FIRST = "clarify_first"              # Ask questions before acting
    HELPFUL_IMMEDIATE = "helpful_immediate"       # Be immediately helpful
    CLARIFY_AND_SIMPLIFY = "clarify_and_simplify"  # For confused customers
    PROACTIVE_ACTION = "proactive_action"        # Take confident action
    GUIDED_RESOLUTION = "guided_resolution"      # Standard guided approach


def determine_concierge_strategy(case: EnhancedConciergeCase) -> ConciergeStrategy:
    """Determine the best concierge strategy for this case"""

    # Safety and consent checks first
    if case.safety_flags.requires_human_review:
        return ConciergeStrategy.GENERIC_HELPFUL

    # No consent for personalization
    from ..core.governance import ConsentScope
    if not governance_engine.validate_consent(case.consent_context,
                                            ConsentScope.PERSONALIZE_RESPONSES):
        return ConciergeStrategy.GENERIC_HELPFUL

    # High-priority customers with clear intent
    if (case.entitlements.priority_support and
        case.intent_prediction.is_high_confidence and
        case.emotion_prediction.label != "frustrated"):
        return ConciergeStrategy.PROACTIVE_ACTION

    # Frustrated customers need immediate help
    if (case.emotion_prediction.label in ["frustrated", "angry"] and
        case.emotion_prediction.confidence > 0.7):
        return ConciergeStrategy.HELPFUL_IMMEDIATE

    # Confused customers need simplification
    if (case.emotion_prediction.label == "confused" and
        case.emotion_prediction.confidence > 0.6):
        return ConciergeStrategy.CLARIFY_AND_SIMPLIFY

    # Low confidence requires clarification
    if case.requires_clarification:
        return ConciergeStrategy.CLARIFY_FIRST

    # High confidence enables proactive action
    if case.overall_confidence > 0.8:
        return ConciergeStrategy.PROACTIVE_ACTION

    # Default to guided resolution
    return ConciergeStrategy.GUIDED_RESOLUTION