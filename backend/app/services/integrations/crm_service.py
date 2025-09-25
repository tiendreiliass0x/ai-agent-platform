"""
CRM Integration Service (Stub)

Provides a seam for pulling relationship stage and business metrics from a CRM.
In this environment, it derives values heuristically from local data.
"""

from typing import Dict, Any, Optional
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from ...models.agent import Agent
from ...models.customer_profile import CustomerProfile
from ...models.organization import Organization


class CRMService:
    def __init__(self):
        pass

    async def get_session(self) -> AsyncSession:
        from ...core.database import get_db_session
        return await get_db_session()

    async def get_relationship_info(
        self,
        agent_id: int,
        customer_profile_id: int
    ) -> Dict[str, Any]:
        """Return relationship stage, trust level, and optional business metrics.

        In production, connect to CRM (e.g., Salesforce/HubSpot) using Organization.settings.
        Here, derive stage heuristically from engagement and profile fields.
        """
        async with await self.get_session() as db:
            # Load agent and profile
            profile_result = await db.execute(select(CustomerProfile).where(CustomerProfile.id == customer_profile_id))
            profile = profile_result.scalar_one_or_none()

            agent_result = await db.execute(select(Agent).where(Agent.id == agent_id))
            agent = agent_result.scalar_one_or_none()

            # Defaults
            stage = "first_time"
            trust = 0.3
            satisfaction_history = []

            if profile:
                # Simple heuristic mapping
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

                if profile.satisfaction_score is not None:
                    satisfaction_history = [profile.satisfaction_score]

            # If Organization settings provided CRM configs, we could alter logic here
            # Example: org.settings.get('crm', { 'provider': 'hubspot', ... })

            return {
                "stage": stage,
                "trust_level": trust,
                "satisfaction_history": satisfaction_history or [4.0],
                "relationship_trend": "positive" if trust >= 0.6 else "neutral"
            }

    async def get_business_metrics(
        self,
        agent_id: int,
        customer_profile_id: int
    ) -> Dict[str, Any]:
        """Return business metrics (CLV, churn risk, conversion probability)."""
        # Heuristic placeholder; replace with CRM/billing lookups.
        return {
            "estimated_value": 5000.0,
            "conversion_probability": 0.6,
            "churn_risk": 0.2,
            "upsell_potential": 0.7
        }

    # Configuration helpers
    async def get_org_crm_config(self, organization_id: int) -> Dict[str, Any]:
        from ...services.database_service import db_service
        org = await db_service.get_organization_by_id(organization_id)
        settings = (org.settings or {}) if org else {}
        return settings.get("crm", {})

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        provider = (config or {}).get("provider")
        enabled = bool((config or {}).get("enabled", False))
        if not provider:
            return {"ok": False, "message": "Missing provider"}
        if provider not in ("hubspot", "salesforce", "custom"):
            return {"ok": False, "message": f"Unsupported provider: {provider}"}
        # In real implementation, verify auth validity with provider API
        return {"ok": True, "message": "Configuration looks valid", "enabled": enabled}

    async def test_connection(self, config: Dict[str, Any]) -> Dict[str, Any]:
        validation = self.validate_config(config)
        if not validation.get("ok"):
            return {"success": False, "message": validation.get("message", "Invalid configuration")}
        # No external calls here; assume success if provider present
        return {"success": True, "message": f"Connected to {config.get('provider')} (mock)", "provider": config.get("provider")}

    async def sync(self, organization_id: int) -> Dict[str, Any]:
        # Placeholder: In production, fetch entities and update local profiles
        now = datetime.utcnow().isoformat()
        return {"success": True, "synced": 0, "last_sync_at": now}


# Global instance
crm_service = CRMService()
