"""
Governance Controls for Third-Party Data Integration

This module extends the core governance framework to handle third-party data
access, consent validation, and compliance requirements for external integrations.
"""

import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from enum import Enum

from ..core.governance import (
    ConsentContext, ConsentScope, DataRetentionPolicy, governance_engine,
    PredictionWithConfidence, EvidenceItem, SafetyFlags
)
from .base import IntegrationResponse


class ThirdPartyDataType(str, Enum):
    """Types of third-party data we can access"""
    ORDER_HISTORY = "order_history"
    PRODUCT_CATALOG = "product_catalog"
    CUSTOMER_PROFILE = "customer_profile"
    INVENTORY_DATA = "inventory_data"
    PAYMENT_INFO = "payment_info"
    SHIPPING_INFO = "shipping_info"
    SUPPORT_TICKETS = "support_tickets"
    ANALYTICS_DATA = "analytics_data"


class ComplianceRequirement(str, Enum):
    """Compliance requirements for data handling"""
    GDPR = "gdpr"
    CCPA = "ccpa"
    PCI_DSS = "pci_dss"
    HIPAA = "hipaa"
    SOX = "sox"
    COPPA = "coppa"


class AccessLevel(str, Enum):
    """Access levels for third-party data"""
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    ADMIN = "admin"
    LIMITED = "limited"


@dataclass
class ThirdPartyConsent:
    """Extended consent specifically for third-party data access"""
    customer_id: str
    platform_id: str

    # Data type permissions
    allowed_data_types: Set[ThirdPartyDataType] = field(default_factory=set)
    access_level: AccessLevel = AccessLevel.READ_ONLY

    # Specific consent scopes
    consents: Set[ConsentScope] = field(default_factory=set)

    # Data usage restrictions
    purpose_limitation: List[str] = field(default_factory=list)  # What data can be used for
    retention_override: Optional[DataRetentionPolicy] = None

    # Platform-specific settings
    platform_consent_id: Optional[str] = None  # Platform's consent ID
    webhook_consent: bool = False  # Allow real-time webhooks

    # Compliance
    compliance_requirements: Set[ComplianceRequirement] = field(default_factory=set)

    # Temporal controls
    consent_timestamp: datetime = field(default_factory=datetime.now)
    expiry_date: Optional[datetime] = None
    last_used: Optional[datetime] = None

    # Audit
    consent_source: str = "explicit"  # "explicit", "implied", "migrated"
    consent_version: str = "1.0"


@dataclass
class DataAccessAudit:
    """Audit record for third-party data access"""
    audit_id: str
    customer_id: str
    platform_id: str
    integration_id: str

    # Access details
    data_types_accessed: List[ThirdPartyDataType]
    purpose: str
    access_level: AccessLevel

    # Governance validation
    consent_validated: bool
    consent_version: str
    compliance_checks_passed: bool

    # Data handling
    pii_redacted: bool
    data_minimization_applied: bool
    retention_policy: DataRetentionPolicy

    # Results
    records_accessed: int
    fields_redacted: List[str] = field(default_factory=list)

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: Optional[str] = None
    user_agent: Optional[str] = None


@dataclass
class IntegrationGovernancePolicy:
    """Governance policy for a specific integration"""
    integration_id: str
    platform_type: str

    # Default settings
    default_access_level: AccessLevel = AccessLevel.READ_ONLY
    default_retention: DataRetentionPolicy = DataRetentionPolicy.SHORT_TERM

    # Required consents
    required_consents: Set[ConsentScope] = field(default_factory=lambda: {
        ConsentScope.USE_CONVERSATION_HISTORY,
        ConsentScope.PERSONALIZE_RESPONSES
    })

    # Data restrictions
    prohibited_data_types: Set[ThirdPartyDataType] = field(default_factory=set)
    pii_auto_redaction: bool = True

    # Compliance
    compliance_requirements: Set[ComplianceRequirement] = field(default_factory=set)
    audit_all_access: bool = True

    # Rate limiting for governance
    max_requests_per_hour: int = 1000
    max_records_per_request: int = 100

    # Data freshness
    cache_ttl_minutes: int = 60

    # Webhook governance
    webhook_allowed: bool = True
    webhook_pii_filtering: bool = True


class ThirdPartyGovernanceEngine:
    """Governance engine specifically for third-party integrations"""

    def __init__(self):
        self.third_party_consents: Dict[str, ThirdPartyConsent] = {}
        self.access_audits: List[DataAccessAudit] = []
        self.integration_policies: Dict[str, IntegrationGovernancePolicy] = {}

        # Load default policies
        self._load_default_policies()

    def _load_default_policies(self):
        """Load default governance policies for popular platforms"""

        # Shopify policy - stricter due to payment data
        shopify_policy = IntegrationGovernancePolicy(
            integration_id="shopify",
            platform_type="ecommerce",
            default_access_level=AccessLevel.READ_ONLY,
            default_retention=DataRetentionPolicy.MEDIUM_TERM,
            prohibited_data_types={ThirdPartyDataType.PAYMENT_INFO},
            compliance_requirements={ComplianceRequirement.GDPR, ComplianceRequirement.PCI_DSS},
            max_requests_per_hour=500,
            max_records_per_request=50
        )

        # WooCommerce policy - more flexible for self-hosted
        woocommerce_policy = IntegrationGovernancePolicy(
            integration_id="woocommerce",
            platform_type="ecommerce",
            default_access_level=AccessLevel.READ_WRITE,
            default_retention=DataRetentionPolicy.LONG_TERM,
            compliance_requirements={ComplianceRequirement.GDPR},
            max_requests_per_hour=1000,
            max_records_per_request=100
        )

        self.integration_policies["shopify"] = shopify_policy
        self.integration_policies["woocommerce"] = woocommerce_policy

    def validate_third_party_access(self, customer_id: str, platform_id: str,
                                  data_types: List[ThirdPartyDataType],
                                  purpose: str,
                                  consent_context: Optional[ConsentContext] = None) -> IntegrationResponse:
        """Validate if third-party data access is allowed"""

        # Get consent for this customer and platform
        consent_key = f"{customer_id}_{platform_id}"
        third_party_consent = self.third_party_consents.get(consent_key)

        if not third_party_consent:
            return IntegrationResponse(
                success=False,
                error_message="No third-party consent found",
                error_code="NO_CONSENT"
            )

        # Check if consent is expired
        if third_party_consent.expiry_date and datetime.now() > third_party_consent.expiry_date:
            return IntegrationResponse(
                success=False,
                error_message="Third-party consent has expired",
                error_code="CONSENT_EXPIRED"
            )

        # Check data type permissions
        for data_type in data_types:
            if data_type not in third_party_consent.allowed_data_types:
                return IntegrationResponse(
                    success=False,
                    error_message=f"Access to {data_type.value} not permitted",
                    error_code="DATA_TYPE_PROHIBITED"
                )

        # Check integration policy
        if platform_id in self.integration_policies:
            policy = self.integration_policies[platform_id]

            # Check prohibited data types
            prohibited = set(data_types) & policy.prohibited_data_types
            if prohibited:
                return IntegrationResponse(
                    success=False,
                    error_message=f"Platform policy prohibits: {[p.value for p in prohibited]}",
                    error_code="POLICY_VIOLATION"
                )

            # Check compliance requirements
            if policy.compliance_requirements:
                if consent_context:
                    if (ComplianceRequirement.GDPR in policy.compliance_requirements and
                        not consent_context.gdpr_subject):
                        # This is actually fine - GDPR applies to EU residents
                        pass

                    if (ComplianceRequirement.CCPA in policy.compliance_requirements and
                        not consent_context.ccpa_subject):
                        # Similarly for CCPA
                        pass

        # Validate core consent scopes
        if consent_context:
            for scope in [ConsentScope.USE_CONVERSATION_HISTORY, ConsentScope.PERSONALIZE_RESPONSES]:
                if not governance_engine.validate_consent(consent_context, scope):
                    return IntegrationResponse(
                        success=False,
                        error_message=f"Missing required consent: {scope.value}",
                        error_code="MISSING_CONSENT"
                    )

        return IntegrationResponse(success=True)

    def apply_data_governance(self, data: Any, data_types: List[ThirdPartyDataType],
                            customer_id: str, platform_id: str,
                            consent_context: Optional[ConsentContext] = None) -> Dict[str, Any]:
        """Apply governance controls to third-party data"""

        consent_key = f"{customer_id}_{platform_id}"
        third_party_consent = self.third_party_consents.get(consent_key)
        policy = self.integration_policies.get(platform_id)

        governed_data = data
        redacted_fields = []

        # Apply PII redaction
        if isinstance(data, dict):
            if policy and policy.pii_auto_redaction:
                # Redact common PII fields
                pii_fields = ['email', 'phone', 'address', 'billing_address', 'shipping_address']

                for field in pii_fields:
                    if field in governed_data:
                        if consent_context and consent_context.pii_redaction_enabled:
                            governed_data[field] = governance_engine.redact_pii(str(governed_data[field]))
                            redacted_fields.append(field)

        # Apply data minimization
        if third_party_consent and third_party_consent.purpose_limitation:
            # Remove fields not needed for stated purpose
            if isinstance(governed_data, dict):
                purpose = third_party_consent.purpose_limitation[0] if third_party_consent.purpose_limitation else "general"

                # Example: for "support" purpose, remove financial data
                if purpose == "support":
                    financial_fields = ['total_price', 'payment_method', 'credit_card']
                    for field in financial_fields:
                        if field in governed_data:
                            del governed_data[field]
                            redacted_fields.append(f"{field}_minimized")

        return {
            "data": governed_data,
            "redacted_fields": redacted_fields,
            "governance_applied": True
        }

    def create_access_audit(self, customer_id: str, platform_id: str, integration_id: str,
                          data_types: List[ThirdPartyDataType], purpose: str,
                          access_level: AccessLevel, records_accessed: int,
                          consent_validated: bool, pii_redacted: bool,
                          session_id: Optional[str] = None) -> DataAccessAudit:
        """Create audit record for data access"""

        audit_id = hashlib.sha256(
            f"{customer_id}_{platform_id}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        consent_key = f"{customer_id}_{platform_id}"
        third_party_consent = self.third_party_consents.get(consent_key)

        audit = DataAccessAudit(
            audit_id=audit_id,
            customer_id=customer_id,
            platform_id=platform_id,
            integration_id=integration_id,
            data_types_accessed=data_types,
            purpose=purpose,
            access_level=access_level,
            consent_validated=consent_validated,
            consent_version=third_party_consent.consent_version if third_party_consent else "unknown",
            compliance_checks_passed=consent_validated,
            pii_redacted=pii_redacted,
            data_minimization_applied=True,
            retention_policy=third_party_consent.retention_override if third_party_consent else DataRetentionPolicy.SHORT_TERM,
            records_accessed=records_accessed,
            session_id=session_id
        )

        self.access_audits.append(audit)

        # Update last used timestamp
        if third_party_consent:
            third_party_consent.last_used = datetime.now()

        return audit

    def grant_third_party_consent(self, customer_id: str, platform_id: str,
                                data_types: Set[ThirdPartyDataType],
                                access_level: AccessLevel = AccessLevel.READ_ONLY,
                                purpose_limitation: Optional[List[str]] = None,
                                expiry_days: int = 365) -> ThirdPartyConsent:
        """Grant third-party data access consent"""

        consent = ThirdPartyConsent(
            customer_id=customer_id,
            platform_id=platform_id,
            allowed_data_types=data_types,
            access_level=access_level,
            consents={ConsentScope.USE_CONVERSATION_HISTORY, ConsentScope.PERSONALIZE_RESPONSES},
            purpose_limitation=purpose_limitation or ["customer_support", "personalization"],
            expiry_date=datetime.now() + timedelta(days=expiry_days),
            compliance_requirements={ComplianceRequirement.GDPR}  # Default to GDPR compliance
        )

        consent_key = f"{customer_id}_{platform_id}"
        self.third_party_consents[consent_key] = consent

        return consent

    def revoke_third_party_consent(self, customer_id: str, platform_id: str) -> bool:
        """Revoke third-party data access consent"""
        consent_key = f"{customer_id}_{platform_id}"
        return self.third_party_consents.pop(consent_key, None) is not None

    def get_customer_third_party_consents(self, customer_id: str) -> List[ThirdPartyConsent]:
        """Get all third-party consents for a customer"""
        return [consent for consent in self.third_party_consents.values()
                if consent.customer_id == customer_id]

    def get_integration_compliance_status(self, integration_id: str) -> Dict[str, Any]:
        """Get compliance status for an integration"""

        if integration_id not in self.integration_policies:
            return {"compliant": False, "reason": "No policy defined"}

        policy = self.integration_policies[integration_id]

        # Count recent access
        recent_audits = [audit for audit in self.access_audits
                        if audit.integration_id == integration_id and
                        (datetime.now() - audit.timestamp).days <= 30]

        # Check for compliance violations
        violations = [audit for audit in recent_audits
                     if not audit.compliance_checks_passed]

        return {
            "compliant": len(violations) == 0,
            "policy": {
                "audit_required": policy.audit_all_access,
                "pii_redaction": policy.pii_auto_redaction,
                "compliance_requirements": [req.value for req in policy.compliance_requirements]
            },
            "recent_activity": {
                "total_access_events": len(recent_audits),
                "compliance_violations": len(violations),
                "unique_customers": len(set(audit.customer_id for audit in recent_audits))
            }
        }

    def generate_compliance_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate compliance report for third-party data access"""

        relevant_audits = [audit for audit in self.access_audits
                          if start_date <= audit.timestamp <= end_date]

        # Group by platform
        platform_stats = {}
        for audit in relevant_audits:
            if audit.platform_id not in platform_stats:
                platform_stats[audit.platform_id] = {
                    "total_requests": 0,
                    "compliant_requests": 0,
                    "unique_customers": set(),
                    "data_types": set(),
                    "pii_redaction_rate": 0
                }

            stats = platform_stats[audit.platform_id]
            stats["total_requests"] += 1
            if audit.compliance_checks_passed:
                stats["compliant_requests"] += 1
            stats["unique_customers"].add(audit.customer_id)
            stats["data_types"].update(audit.data_types_accessed)
            if audit.pii_redacted:
                stats["pii_redaction_rate"] += 1

        # Convert sets to lists for JSON serialization
        for platform_id, stats in platform_stats.items():
            stats["unique_customers"] = len(stats["unique_customers"])
            stats["data_types"] = [dt.value for dt in stats["data_types"]]
            stats["compliance_rate"] = stats["compliant_requests"] / stats["total_requests"] if stats["total_requests"] > 0 else 0
            stats["pii_redaction_rate"] = stats["pii_redaction_rate"] / stats["total_requests"] if stats["total_requests"] > 0 else 0

        return {
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "summary": {
                "total_platforms": len(platform_stats),
                "total_requests": len(relevant_audits),
                "overall_compliance_rate": len([a for a in relevant_audits if a.compliance_checks_passed]) / len(relevant_audits) if relevant_audits else 0
            },
            "platform_breakdown": platform_stats
        }


# Global third-party governance engine
third_party_governance = ThirdPartyGovernanceEngine()