"""
Governance and Consent Management for AI Agent Platform

This module handles data governance, consent management, and privacy compliance
for customer intelligence and personalization features.
"""

import re
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
from enum import Enum
import json


class ConsentScope(str, Enum):
    """Specific consent scopes for different data uses"""
    STORE_PREFERENCES = "store_preferences"
    USE_CONVERSATION_HISTORY = "use_conversation_history"
    PERSIST_SATISFACTION = "persist_satisfaction"
    PERSONALIZE_RESPONSES = "personalize_responses"
    ANALYZE_BEHAVIOR = "analyze_behavior"
    CROSS_SESSION_MEMORY = "cross_session_memory"


class DataRetentionPolicy(str, Enum):
    """Data retention policies with different TTLs"""
    SESSION_ONLY = "session_only"          # 24 hours
    SHORT_TERM = "short_term"              # 30 days
    MEDIUM_TERM = "medium_term"            # 90 days
    LONG_TERM = "long_term"                # 365 days
    PREFERENCES = "preferences"            # 12 months
    COMPLIANCE = "compliance"              # 7 years (legal)


class InferenceCategory(str, Enum):
    """Categories of inferences we can make about customers"""
    # Allowed inferences
    INTENT = "intent"                      # What they want to do
    URGENCY = "urgency"                    # How urgent their need is
    TECHNICAL_LEVEL = "technical_level"    # Their technical sophistication
    COMMUNICATION_STYLE = "communication_style"  # How they prefer to communicate
    SATISFACTION = "satisfaction"          # Their satisfaction level

    # Blocked inferences (protected attributes)
    HEALTH = "health"                      # Health conditions
    RELIGION = "religion"                  # Religious beliefs
    POLITICS = "politics"                  # Political views
    FINANCIAL_STATUS = "financial_status"  # Detailed financial situation
    PERSONAL_RELATIONSHIPS = "personal_relationships"  # Family/relationship status


@dataclass
class ConsentContext:
    """Customer consent and data governance context"""
    # Core consent flags
    consents: Set[ConsentScope] = field(default_factory=set)
    consent_timestamp: Optional[datetime] = None
    consent_version: str = "1.0"

    # Data handling preferences
    data_retention_policy: DataRetentionPolicy = DataRetentionPolicy.SHORT_TERM
    pii_redaction_enabled: bool = True
    hash_identifiers: bool = True

    # Inference controls
    allowed_inferences: Set[InferenceCategory] = field(default_factory=lambda: {
        InferenceCategory.INTENT,
        InferenceCategory.URGENCY,
        InferenceCategory.TECHNICAL_LEVEL,
        InferenceCategory.COMMUNICATION_STYLE,
        InferenceCategory.SATISFACTION
    })

    # Geographic and legal context
    region: Optional[str] = None
    gdpr_subject: bool = False
    ccpa_subject: bool = False
    data_residency_required: bool = False

    # Audit trail
    consent_source: str = "implicit"  # "explicit", "implicit", "inferred"
    last_updated: Optional[datetime] = None


@dataclass
class PredictionWithConfidence:
    """A prediction with confidence metrics and evidence"""
    label: str
    probability: float
    confidence: float                      # How certain we are (0-1)
    evidence_strength: float               # How much evidence supports this (0-1)
    evidence_count: int = 0                # Number of evidence points
    model_version: str = "unknown"
    generated_at: datetime = field(default_factory=datetime.now)

    @property
    def is_high_confidence(self) -> bool:
        """High confidence threshold for automated actions"""
        return self.confidence >= 0.8 and self.evidence_strength >= 0.7

    @property
    def requires_clarification(self) -> bool:
        """Low confidence threshold requiring clarification"""
        return self.confidence < 0.65 or self.evidence_strength < 0.5


@dataclass
class EvidenceItem:
    """Evidence supporting a prediction or insight"""
    evidence_type: str                     # "explicit", "behavioral", "inferred", "contextual"
    source_ref: str                        # Reference to source (message_id, event_id, etc.)
    summary: str                           # Human-readable summary
    confidence: float                      # Confidence in this evidence (0-1)
    timestamp: datetime = field(default_factory=datetime.now)

    # Governance
    pii_redacted: bool = False
    retention_policy: DataRetentionPolicy = DataRetentionPolicy.SHORT_TERM


@dataclass
class SafetyFlags:
    """Safety and abuse detection flags"""
    abuse_detected: bool = False
    fraud_suspected: bool = False
    prompt_injection: bool = False
    pii_exposed: bool = False
    policy_violation: bool = False

    # Details
    detected_patterns: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    requires_human_review: bool = False


class GovernanceEngine:
    """Core governance engine for consent and data handling"""

    def __init__(self):
        self.retention_ttls = {
            DataRetentionPolicy.SESSION_ONLY: timedelta(hours=24),
            DataRetentionPolicy.SHORT_TERM: timedelta(days=30),
            DataRetentionPolicy.PREFERENCES: timedelta(days=365),
            DataRetentionPolicy.COMPLIANCE: timedelta(days=2555)  # 7 years
        }

        # PII patterns for redaction
        self.pii_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            "address": r'\b\d+\s+[A-Za-z0-9\s,.-]+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Way|Place|Pl)\b'
        }

        # Abuse detection patterns
        self.abuse_patterns = [
            r"ignore\s+previous\s+instructions",
            r"jailbreak",
            r"pretend\s+you\s+are",
            r"act\s+as\s+(?!a\s+helpful|an?\s+assistant)",
            r"system\s+prompt",
            r"forget\s+everything",
            r"new\s+instructions"
        ]

    def validate_consent(self, consent_context: ConsentContext, required_scope: ConsentScope) -> bool:
        """Validate that customer has given required consent"""
        if required_scope not in consent_context.consents:
            return False

        # Check if consent is still valid (not expired)
        if consent_context.consent_timestamp:
            max_age = timedelta(days=365)  # Consent expires after 1 year
            if datetime.now() - consent_context.consent_timestamp > max_age:
                return False

        return True

    def can_make_inference(self, consent_context: ConsentContext, inference_type: InferenceCategory) -> bool:
        """Check if we're allowed to make a specific type of inference"""
        # Block protected attributes regardless of consent
        if inference_type in {InferenceCategory.HEALTH, InferenceCategory.RELIGION,
                             InferenceCategory.POLITICS, InferenceCategory.FINANCIAL_STATUS,
                             InferenceCategory.PERSONAL_RELATIONSHIPS}:
            return False

        # Check explicit consent for behavioral analysis
        if inference_type in {InferenceCategory.SATISFACTION, InferenceCategory.COMMUNICATION_STYLE}:
            return self.validate_consent(consent_context, ConsentScope.ANALYZE_BEHAVIOR)

        # Default allowed inferences
        return inference_type in consent_context.allowed_inferences

    def redact_pii(self, text: str, hash_identifiers: bool = True) -> str:
        """Redact or hash PII from text"""
        redacted_text = text

        for pii_type, pattern in self.pii_patterns.items():
            matches = re.finditer(pattern, redacted_text, re.IGNORECASE)
            for match in reversed(list(matches)):  # Reverse to maintain indices
                pii_value = match.group()

                if hash_identifiers:
                    # Hash the PII value for consistent anonymization
                    hashed = hashlib.sha256(pii_value.encode()).hexdigest()[:8]
                    replacement = f"[{pii_type.upper()}_{hashed}]"
                else:
                    replacement = f"[{pii_type.upper()}_REDACTED]"

                redacted_text = redacted_text[:match.start()] + replacement + redacted_text[match.end():]

        return redacted_text

    def detect_safety_issues(self, message: str) -> SafetyFlags:
        """Detect safety and abuse issues in user message"""
        flags = SafetyFlags()
        message_lower = message.lower()

        # Check for abuse patterns
        detected_patterns = []
        for pattern in self.abuse_patterns:
            if re.search(pattern, message_lower):
                detected_patterns.append(pattern)
                flags.abuse_detected = True

        # Check for PII exposure
        for pii_type, pattern in self.pii_patterns.items():
            if re.search(pattern, message):
                flags.pii_exposed = True
                detected_patterns.append(f"pii_{pii_type}")

        # Check for prompt injection
        injection_keywords = ["system:", "assistant:", "user:", "###", "```"]
        if any(keyword in message_lower for keyword in injection_keywords):
            flags.prompt_injection = True
            detected_patterns.append("prompt_injection")

        flags.detected_patterns = detected_patterns
        flags.risk_score = min(len(detected_patterns) * 0.3, 1.0)
        flags.requires_human_review = flags.risk_score > 0.7

        return flags

    def should_persist_data(self, data_type: str, consent_context: ConsentContext) -> bool:
        """Determine if data should be persisted based on governance rules"""

        # Session-only data types
        if data_type in ["emotion", "temporary_state", "one_off_request"]:
            return False

        # Check consent for persistent data
        if data_type in ["preferences", "communication_style"]:
            return self.validate_consent(consent_context, ConsentScope.STORE_PREFERENCES)

        if data_type in ["conversation_history", "satisfaction_history"]:
            return self.validate_consent(consent_context, ConsentScope.USE_CONVERSATION_HISTORY)

        # Default to not persisting without explicit consent
        return False

    def calculate_data_ttl(self, data_type: str, retention_policy: DataRetentionPolicy) -> datetime:
        """Calculate when data should expire"""
        ttl = self.retention_ttls.get(retention_policy, timedelta(days=30))
        return datetime.now() + ttl

    def create_audit_log(self, action: str, customer_id: str, data_access: Dict[str, Any],
                        consent_context: ConsentContext) -> Dict[str, Any]:
        """Create audit log entry for data access"""
        return {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "customer_id": customer_id,
            "consent_version": consent_context.consent_version,
            "data_accessed": list(data_access.keys()),
            "retention_policy": consent_context.data_retention_policy.value,
            "gdpr_subject": consent_context.gdpr_subject,
            "pii_redacted": consent_context.pii_redaction_enabled
        }


# Global governance instance
governance_engine = GovernanceEngine()


# Decision thresholds based on GPT-5-Pro recommendations
class DecisionThresholds:
    """Decision thresholds for ask vs act logic"""
    ASK_THRESHOLD = 0.65          # Below this confidence, ask clarifying questions
    ACT_THRESHOLD = 0.80          # Above this confidence, take action
    ESCALATE_THRESHOLD = 0.95     # Critical decisions need human review

    # Emotion-based overrides
    FRUSTRATED_OVERRIDE = 0.70    # Lower threshold for frustrated customers
    CONFUSED_OVERRIDE = 0.60      # Even lower for confused customers


def determine_interaction_strategy(prediction: PredictionWithConfidence,
                                 emotion_prediction: PredictionWithConfidence,
                                 consent_context: ConsentContext) -> str:
    """Determine how to interact based on confidence and consent"""

    # Safety first - if no consent for personalization, use generic approach
    if not governance_engine.validate_consent(consent_context, ConsentScope.PERSONALIZE_RESPONSES):
        return "generic_helpful"

    # High-confidence frustrated customer - be immediately helpful
    if (emotion_prediction.label == "frustrated" and
        emotion_prediction.confidence > DecisionThresholds.FRUSTRATED_OVERRIDE):
        return "helpful_immediate"

    # Confused customer - simplify and clarify
    if (emotion_prediction.label == "confused" and
        emotion_prediction.confidence > DecisionThresholds.CONFUSED_OVERRIDE):
        return "clarify_and_simplify"

    # Low confidence on intent - ask clarifying questions
    if prediction.confidence < DecisionThresholds.ASK_THRESHOLD:
        return "clarify_first"

    # High confidence - take proactive action
    if prediction.confidence > DecisionThresholds.ACT_THRESHOLD:
        return "proactive_action"

    # Medium confidence - guided resolution
    return "guided_resolution"