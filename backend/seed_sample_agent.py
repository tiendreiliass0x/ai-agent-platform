"""Seed script: create sample organization, user, agent, and document."""

import asyncio
from pathlib import Path
from datetime import datetime

from sqlalchemy import insert, select, text

from app.core.database import get_db_session
from app.core.auth import get_password_hash
from app.models.organization import Organization
from app.models.user import User
from app.models.agent import Agent, AgentTier
from app.models.document import Document

SAMPLE_ORG = {
    "name": "Demo Org",
    "slug": "demo-org",
    "plan": "professional",
    "subscription_status": "active",
    "max_agents": 10,
    "max_users": 5,
    "max_documents_per_agent": 100,
    "settings": {},
}

SAMPLE_USER_PASSWORD = "Dem0Admin!"
SAMPLE_USER = {
    "email": "demo@demo-org.example",
    "password_hash": get_password_hash(SAMPLE_USER_PASSWORD),
    "name": "Demo Admin",
    "plan": "professional",
    "is_active": True,
    "is_superuser": False,
}

SAMPLE_AGENT = {
    "name": "Shopify Concierge",
    "description": "Demo agent for Shopify order management",
    "system_prompt": "You are a helpful assistant focused on Shopify + Katana workflows.",
    "tier": AgentTier.professional,
    "config": {},
    "widget_config": {},
}

SAMPLE_DOC = {
    "filename": "shopify-order-management.html",
    "content_type": "text/html",
    "status": "completed",
    "doc_metadata": {"source_url": "https://katanamrp.com/integrations/shopify/shopify-order-management/"},
}

async def seed():
    async with (await get_db_session()) as session:
        # Cleanup previous seed data for idempotency
        await session.execute(
            text(
                "DELETE FROM user_organizations "
                "WHERE organization_id IN (SELECT id FROM organizations WHERE slug = :slug)"
            ),
            {"slug": SAMPLE_ORG["slug"]}
        )
        await session.execute(
            text("DELETE FROM agents WHERE name = :agent_name"),
            {"agent_name": SAMPLE_AGENT["name"]}
        )
        await session.execute(
            text("DELETE FROM users WHERE email = :email"),
            {"email": SAMPLE_USER["email"]}
        )
        await session.execute(
            text("DELETE FROM organizations WHERE slug = :slug"),
            {"slug": SAMPLE_ORG["slug"]}
        )
        await session.commit()

        # Reset sequences to avoid primary key collisions
        await session.execute(text("SELECT setval('organizations_id_seq', COALESCE((SELECT MAX(id) FROM organizations), 0) + 1, false)"))
        await session.execute(text("SELECT setval('users_id_seq', COALESCE((SELECT MAX(id) FROM users), 0) + 1, false)"))
        await session.execute(text("SELECT setval('agents_id_seq', COALESCE((SELECT MAX(id) FROM agents), 0) + 1, false)"))
        await session.commit()

        result = await session.execute(
            insert(Organization).values(SAMPLE_ORG).returning(Organization.id)
        )
        organization_id = result.scalar_one()

        user_values = {**SAMPLE_USER}
        result = await session.execute(
            insert(User).values(user_values).returning(User.id)
        )
        user_id = result.scalar_one()

        from app.models.user_organization import UserOrganization, OrganizationRole

        await session.execute(
            insert(UserOrganization).values(
                user_id=user_id,
                organization_id=organization_id,
                role=OrganizationRole.OWNER.value,
                is_active=True
            )
        )

        existing_agent = await session.execute(
            select(Agent).where(Agent.name == SAMPLE_AGENT["name"], Agent.organization_id == organization_id)
        )
        agent = existing_agent.scalar_one_or_none()
        if agent:
            agent_id = agent.id
        else:
            agent_values = {
                **SAMPLE_AGENT,
                "organization_id": organization_id,
                "user_id": user_id,
                "created_at": datetime.utcnow(),
            }
            result = await session.execute(
                insert(Agent).values(agent_values).returning(Agent.id)
            )
            agent_id = result.scalar_one()

        content_path = Path(__file__).parent / "sample_data" / SAMPLE_DOC["filename"]
        content = content_path.read_text(encoding="utf-8") if content_path.exists() else ""

        doc_values = {
            **SAMPLE_DOC,
            "agent_id": agent_id,
            "content": content,
            "size": len(content.encode()),
        }
        await session.execute(insert(Document).values(doc_values))

        await session.commit()

        print("Seeded sample data. Agent ID:", agent_id)

if __name__ == "__main__":
    asyncio.run(seed())
