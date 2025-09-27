"""
Concierge Orchestrator - Main service that builds EnhancedConciergeCase

This service orchestrates all the intelligence gathering, governance validation,
and context building to create comprehensive customer concierge experiences.
"""

import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..core.governance import ConsentContext, ConsentScope, DataRetentionPolicy, governance_engine
from .enhanced_concierge_case import (
    EnhancedConciergeCase, CustomerEntitlements, CustomerTier, TouchpointWeight,
    BusinessMetrics, NextBestAction, ProvenanceInfo, determine_concierge_strategy
)
from .enhanced_user_intelligence_service import enhanced_user_intelligence_service
from .customer_data_service import customer_data_service
from ..models.customer_profile import CustomerProfile
from ..models.organization import Organization


class EntitlementsService:
    """Service for determining customer entitlements and permissions"""

    def __init__(self):
        self.tier_configs = {
            CustomerTier.BASIC: {
                "refund_limit_usd": 50.0,
                "discount_limit_percent": 5.0,
                "priority_support": False,
                "sla_hours": 48,
                "feature_access": {"basic_chat", "knowledge_base"},
                "api_rate_limit": 100
            },
            CustomerTier.PROFESSIONAL: {
                "refund_limit_usd": 200.0,
                "discount_limit_percent": 15.0,
                "priority_support": True,
                "sla_hours": 24,
                "feature_access": {"basic_chat", "knowledge_base", "priority_support", "analytics"},
                "api_rate_limit": 500
            },
            CustomerTier.ENTERPRISE: {
                "refund_limit_usd": 1000.0,
                "discount_limit_percent": 25.0,
                "priority_support": True,
                "sla_hours": 4,
                "feature_access": {"basic_chat", "knowledge_base", "priority_support", "analytics", "custom_integrations"},
                "api_rate_limit": 2000
            },
            CustomerTier.VIP: {
                "refund_limit_usd": 5000.0,
                "discount_limit_percent": 40.0,
                "priority_support": True,
                "sla_hours": 1,
                "feature_access": {"all_features"},
                "api_rate_limit": 10000
            }
        }

    async def get_customer_entitlements(
        self,
        customer_profile: Optional[CustomerProfile],
        organization: Organization,
        db: AsyncSession
    ) -> CustomerEntitlements:
        """Get customer entitlements based on their profile and plan"""

        # Default to basic tier for unknown customers
        tier = CustomerTier.BASIC
        plan_name = "Basic Plan"

        if customer_profile:
            # Determine tier from customer profile
            if customer_profile.total_spent > 10000:
                tier = CustomerTier.VIP
                plan_name = "VIP Customer"
            elif customer_profile.total_spent > 5000:
                tier = CustomerTier.ENTERPRISE
                plan_name = "Enterprise Customer"
            elif customer_profile.total_spent > 1000:
                tier = CustomerTier.PROFESSIONAL
                plan_name = "Professional Customer"

            # Check if they have explicit subscription
            if hasattr(customer_profile, 'subscription_tier'):
                tier = CustomerTier(customer_profile.subscription_tier)

        # Get tier configuration
        config = self.tier_configs.get(tier, self.tier_configs[CustomerTier.BASIC])

        return CustomerEntitlements(
            tier=tier,
            plan_name=plan_name,
            refund_limit_usd=config["refund_limit_usd"],
            discount_limit_percent=config["discount_limit_percent"],
            priority_support=config["priority_support"],
            sla_hours=config["sla_hours"],
            feature_access=config["feature_access"],
            api_rate_limit=config["api_rate_limit"],
            region=organization.settings.get("region", "US") if organization.settings else "US",
            data_residency_required=organization.settings.get("data_residency_required", False) if organization.settings else False
        )

    def can_perform_action(self, entitlements: CustomerEntitlements, action: str, value: float = 0) -> bool:
        """Check if customer can perform a specific action"""

        if action == "refund" and value > entitlements.refund_limit_usd:
            return False

        if action == "discount" and value > entitlements.discount_limit_percent:
            return False

        if action == "priority_support" and not entitlements.priority_support:
            return False

        return True


class TouchpointAnalyzer:
    """Analyzes and weights customer touchpoints with time decay"""

    def __init__(self):
        self.source_weights = {
            "support_ticket": 0.9,
            "purchase": 0.8,
            "complaint": 0.8,
            "satisfaction_survey": 0.7,
            "feature_request": 0.6,
            "page_view": 0.3,
            "email_open": 0.2,
            "login": 0.1
        }

        self.decay_rates = {
            "support_ticket": 0.1,    # Slow decay - important longer
            "purchase": 0.05,         # Very slow decay - always relevant
            "complaint": 0.2,         # Medium decay
            "satisfaction_survey": 0.15,  # Slow-medium decay
            "page_view": 0.5,         # Fast decay
            "email_open": 0.4,        # Fast decay
            "login": 0.6,             # Very fast decay
            "feature_request": 0.3    # Medium-slow decay
        }

    async def get_customer_touchpoints(
        self,
        customer_profile: Optional[CustomerProfile],
        session_context: Dict[str, Any],
        db: AsyncSession
    ) -> List[TouchpointWeight]:
        """Get and analyze customer touchpoints with time decay"""

        touchpoints = []

        # Add session-based touchpoints
        current_page = session_context.get("current_page", "")
        if current_page:
            touchpoints.append(TouchpointWeight(
                touchpoint_id=f"session_{uuid.uuid4()}",
                source="page_view",
                base_weight=self.source_weights.get("page_view", 0.3),
                timestamp=datetime.now()
            ))

        if not customer_profile:
            return touchpoints

        # Add historical touchpoints (mock data - would integrate with real data)
        # Support tickets
        if hasattr(customer_profile, 'support_tickets_count') and customer_profile.support_tickets_count > 0:
            touchpoints.append(TouchpointWeight(
                touchpoint_id=f"support_{customer_profile.id}",
                source="support_ticket",
                base_weight=self.source_weights.get("support_ticket", 0.9),
                timestamp=customer_profile.last_seen_at or datetime.now() - timedelta(days=7)
            ))

        # Recent purchases
        if customer_profile.total_spent > 0:
            touchpoints.append(TouchpointWeight(
                touchpoint_id=f"purchase_{customer_profile.id}",
                source="purchase",
                base_weight=self.source_weights.get("purchase", 0.8),
                timestamp=customer_profile.last_seen_at or datetime.now() - timedelta(days=30)
            ))

        # Satisfaction surveys
        if customer_profile.satisfaction_score and customer_profile.satisfaction_score > 0:
            touchpoints.append(TouchpointWeight(
                touchpoint_id=f"satisfaction_{customer_profile.id}",
                source="satisfaction_survey",
                base_weight=self.source_weights.get("satisfaction_survey", 0.7),
                timestamp=customer_profile.last_seen_at or datetime.now() - timedelta(days=14)
            ))

        return touchpoints


class BusinessMetricsCalculator:
    """Calculates business metrics and opportunities"""

    async def calculate_business_metrics(
        self,
        customer_profile: Optional[CustomerProfile],
        intelligence_analysis,
        entitlements: CustomerEntitlements
    ) -> BusinessMetrics:
        """Calculate comprehensive business metrics"""

        if not customer_profile:
            return BusinessMetrics()

        # Calculate CLV estimate (simplified)
        clv_estimate = customer_profile.total_spent * 2.5  # Simple multiplier

        # Calculate churn probability based on various factors
        churn_probability = 0.1  # Base rate
        if customer_profile.satisfaction_score and customer_profile.satisfaction_score < 3:
            churn_probability += 0.3
        if intelligence_analysis.emotion.label in ["frustrated", "angry"]:
            churn_probability += 0.2
        if "cancel" in intelligence_analysis.key_topics:
            churn_probability += 0.4

        # Calculate escalation risk
        escalation_risk = 0.1  # Base rate
        if intelligence_analysis.urgency.label == "critical":
            escalation_risk += 0.4
        if intelligence_analysis.emotion.label in ["frustrated", "angry"]:
            escalation_risk += 0.3
        if entitlements.tier in [CustomerTier.ENTERPRISE, CustomerTier.VIP]:
            escalation_risk += 0.2  # High-value customers escalate more

        # Identify opportunities
        opportunities = []
        if "upgrade" in intelligence_analysis.key_topics:
            opportunities.append("upgrade_opportunity")
        if "demo" in intelligence_analysis.opportunities:
            opportunities.append("demo_request")
        if entitlements.tier == CustomerTier.BASIC and customer_profile.total_spent > 500:
            opportunities.append("tier_upgrade")

        return BusinessMetrics(
            clv_estimate=clv_estimate,
            total_spent=customer_profile.total_spent,
            avg_order_value=customer_profile.total_spent / max(customer_profile.total_conversations, 1),
            churn_probability=min(churn_probability, 0.9),
            escalation_risk=min(escalation_risk, 0.9),
            satisfaction_trend="stable",  # Would calculate from trend data
            upsell_opportunities=opportunities
        )


class NextBestActionGenerator:
    """Generates contextual next best actions"""

    def generate_next_best_actions(
        self,
        intelligence_analysis,
        entitlements: CustomerEntitlements,
        business_metrics: BusinessMetrics
    ) -> List[NextBestAction]:
        """Generate prioritized next best actions"""

        actions = []

        # Safety and escalation actions first
        if intelligence_analysis.safety_flags.requires_human_review:
            actions.append(NextBestAction(
                action_id="escalate_safety",
                action_type="escalate_to_human",
                description="Escalate to human agent due to safety concerns",
                expected_value=100.0,  # High value for safety
                confidence=0.95,
                requires_approval=False
            ))

        # High-priority customer actions
        if entitlements.tier in [CustomerTier.ENTERPRISE, CustomerTier.VIP]:
            if intelligence_analysis.urgency.label == "critical":
                actions.append(NextBestAction(
                    action_id="priority_escalation",
                    action_type="priority_escalation",
                    description="Immediate escalation for high-tier customer",
                    expected_value=business_metrics.clv_estimate * 0.1,
                    confidence=0.9
                ))

        # Emotion-based actions
        if intelligence_analysis.emotion.label == "frustrated" and intelligence_analysis.emotion.confidence > 0.7:
            if entitlements.can_perform_action(entitlements, "discount", 10):
                actions.append(NextBestAction(
                    action_id="offer_discount",
                    action_type="offer_discount",
                    description="Offer discount to frustrated customer",
                    expected_value=business_metrics.clv_estimate * 0.05,
                    confidence=0.8,
                    preconditions=["frustration_confirmed"]
                ))

        # Intent-based actions
        if intelligence_analysis.intent.label == "purchase":
            actions.append(NextBestAction(
                action_id="guided_purchase",
                action_type="guided_purchase",
                description="Provide guided purchase assistance",
                expected_value=business_metrics.avg_order_value,
                confidence=intelligence_analysis.intent.confidence
            ))

        elif intelligence_analysis.intent.label == "troubleshoot":
            actions.append(NextBestAction(
                action_id="technical_support",
                action_type="technical_troubleshooting",
                description="Provide technical troubleshooting guidance",
                expected_value=50.0,  # Satisfaction value
                confidence=intelligence_analysis.intent.confidence
            ))

        # Business opportunity actions
        for opportunity in intelligence_analysis.opportunities:
            if opportunity == "demo_request":
                actions.append(NextBestAction(
                    action_id="schedule_demo",
                    action_type="schedule_demo",
                    description="Schedule product demo",
                    expected_value=business_metrics.avg_order_value * 0.3,
                    confidence=0.7
                ))

        # Default helpful action
        if not actions:
            actions.append(NextBestAction(
                action_id="general_assistance",
                action_type="general_assistance",
                description="Provide general helpful assistance",
                expected_value=25.0,
                confidence=0.6
            ))

        # Sort by expected value and return top 5
        actions.sort(key=lambda x: x.expected_value * x.confidence, reverse=True)
        return actions[:5]


class ConciergeOrchestrator:
    """Main orchestrator for creating EnhancedConciergeCase"""

    def __init__(self):
        self.entitlements_service = EntitlementsService()
        self.touchpoint_analyzer = TouchpointAnalyzer()
        self.business_calculator = BusinessMetricsCalculator()
        self.action_generator = NextBestActionGenerator()

    async def create_concierge_case(
        self,
        message: str,
        customer_id: Optional[str],
        session_context: Dict[str, Any],
        agent_id: int,
        db: AsyncSession,
        consent_context: Optional[ConsentContext] = None
    ) -> EnhancedConciergeCase:
        """
        Create comprehensive concierge case with full context analysis
        """

        start_time = datetime.now()

        # Create default consent if not provided
        if consent_context is None:
            consent_context = ConsentContext(
                consents={ConsentScope.PERSONALIZE_RESPONSES},
                data_retention_policy=DataRetentionPolicy.SHORT_TERM,
                pii_redaction_enabled=True
            )

        # Safety check first
        safety_flags = governance_engine.detect_safety_issues(message)

        # Get customer profile and organization
        customer_profile = None
        if customer_id:
            result = await db.execute(select(CustomerProfile).where(CustomerProfile.visitor_id == customer_id))
            customer_profile = result.scalar_one_or_none()

        # Get organization for entitlements
        from ..models.agent import Agent
        agent_result = await db.execute(select(Agent).where(Agent.id == agent_id))
        agent = agent_result.scalar_one_or_none()

        organization_result = await db.execute(select(Organization).where(Organization.id == agent.organization_id))
        organization = organization_result.scalar_one_or_none()

        # Parallel analysis tasks
        analysis_tasks = [
            enhanced_user_intelligence_service.analyze_user_message(
                message=message,
                customer_profile_id=customer_profile.id if customer_profile else None,
                conversation_history=session_context.get("conversation_history", []),
                session_context=session_context,
                consent_context=consent_context
            ),
            self.entitlements_service.get_customer_entitlements(customer_profile, organization, db),
            self.touchpoint_analyzer.get_customer_touchpoints(customer_profile, session_context, db)
        ]

        # Execute analysis in parallel
        intelligence_analysis, entitlements, touchpoints = await asyncio.gather(*analysis_tasks)

        # Calculate business metrics
        business_metrics = await self.business_calculator.calculate_business_metrics(
            customer_profile, intelligence_analysis, entitlements
        )

        # Generate next best actions
        next_best_actions = self.action_generator.generate_next_best_actions(
            intelligence_analysis, entitlements, business_metrics
        )

        # Create case ID and provenance
        case_id = f"case_{uuid.uuid4().hex[:8]}"
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        provenance = ProvenanceInfo(
            model_versions={
                "intelligence": "enhanced_v1.0",
                "governance": "v1.0",
                "entitlements": "v1.0"
            },
            data_sources=["user_message", "customer_profile", "session_context"],
            processing_time_ms=processing_time
        )

        # Create the enhanced concierge case
        case = EnhancedConciergeCase(
            case_id=case_id,
            customer_ref=customer_id or f"anon_{uuid.uuid4().hex[:8]}",
            session_id=session_context.get("session_id", f"session_{uuid.uuid4().hex[:8]}"),
            consent_context=consent_context,
            safety_flags=intelligence_analysis.safety_flags,
            intent_prediction=intelligence_analysis.intent,
            urgency_prediction=intelligence_analysis.urgency,
            emotion_prediction=intelligence_analysis.emotion,
            entitlements=entitlements,
            business_metrics=business_metrics,
            evidence_items=intelligence_analysis.evidence_items,
            touchpoint_weights=touchpoints,
            session_context=session_context,
            next_best_actions=next_best_actions,
            provenance=provenance
        )

        # Add evidence from the message
        case.add_evidence(
            evidence_type="explicit",
            source_ref="user_message",
            summary=f"Customer message with intent '{intelligence_analysis.intent.label}'",
            confidence=1.0
        )

        # Determine strategy
        case.recommended_strategy = determine_concierge_strategy(case).value

        return case


# Global orchestrator instance
concierge_orchestrator = ConciergeOrchestrator()