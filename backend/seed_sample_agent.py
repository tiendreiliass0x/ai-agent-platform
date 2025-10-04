"""
Seed script: create sample organization, user, agent, and document.

This script creates a complete demo environment with:
- Demo organization
- Demo user with OWNER role
- Demo agent with professional tier
- Sample document with embeddings

Safety features:
- Environment validation (blocks production)
- Comprehensive error handling
- Transaction rollback on failure
- Database-agnostic implementation
- Proper logging
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

from app.core.config import settings
from app.core.database import get_db_session
from app.core.auth import get_password_hash
from app.models.organization import Organization
from app.models.user import User
from app.models.agent import Agent, AgentTier
from app.models.document import Document
from app.models.user_organization import UserOrganization, OrganizationRole

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sample data configuration
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

SAMPLE_USER = {
    "email": "demo@demo-org.example",
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
    "domain_expertise_enabled": False,
    "config": {},
    "widget_config": {},
}

SAMPLE_DOC = {
    "filename": "shopify-order-management.html",
    "content_type": "text/html",
    "status": "completed",
    "doc_metadata": {"source_url": "https://katanamrp.com/integrations/shopify/shopify-order-management/"},
}


def get_password_from_env() -> str:
    """Get demo password from environment variable or use default."""
    password = os.getenv("DEMO_USER_PASSWORD", "Dem0Admin!")

    # Validate password strength
    if len(password) < 8:
        raise ValueError("Demo password must be at least 8 characters long")

    return password


def validate_environment() -> None:
    """Validate that seeding is safe in current environment."""
    env = (settings.ENVIRONMENT or "development").lower()

    if env == "production":
        logger.error("❌ Cannot seed sample data in PRODUCTION environment!")
        logger.error("This would create demo credentials in a production database.")
        raise EnvironmentError(
            "Seeding is blocked in production for security reasons. "
            "Set ENVIRONMENT to 'development' or 'staging' to seed data."
        )

    logger.info(f"✓ Environment validated: {env}")


def validate_sample_data_files() -> Path:
    """Validate that sample data directory and files exist."""
    # Try multiple possible locations
    possible_paths = [
        Path(__file__).parent / "sample_data",
        Path(__file__).parent / "scripts" / "sample_data",
    ]

    sample_data_dir = None
    for path in possible_paths:
        if path.exists() and path.is_dir():
            sample_data_dir = path
            break

    if not sample_data_dir:
        raise FileNotFoundError(
            f"Sample data directory not found. Tried: {', '.join(str(p) for p in possible_paths)}"
        )

    # Validate document file exists
    doc_path = sample_data_dir / SAMPLE_DOC["filename"]
    if not doc_path.exists():
        raise FileNotFoundError(
            f"Sample document not found: {doc_path}\n"
            f"Please ensure '{SAMPLE_DOC['filename']}' exists in the sample_data directory."
        )

    logger.info(f"✓ Sample data directory validated: {sample_data_dir}")
    return sample_data_dir


async def cleanup_existing_data(session) -> None:
    """
    Clean up existing seed data in correct order to respect foreign key constraints.

    Order:
    1. Documents (references agents)
    2. User-Organization links (references users & organizations)
    3. Agents (references users & organizations)
    4. Users
    5. Organizations
    """
    try:
        # Find organization by slug
        result = await session.execute(
            select(Organization.id, Organization.name).where(Organization.slug == SAMPLE_ORG["slug"])
        )
        org_data = result.first()

        if org_data:
            org_id, org_name = org_data
            logger.info(f"Found existing organization: {org_name} (ID: {org_id})")

            # Find and delete agents by ID to avoid column issues
            result = await session.execute(
                select(Agent.id).where(Agent.organization_id == org_id)
            )
            agent_ids = [row[0] for row in result.all()]

            if agent_ids:
                logger.info(f"Deleting {len(agent_ids)} existing agents...")
                # Delete documents first
                for agent_id in agent_ids:
                    await session.execute(
                        delete(Document).where(Document.agent_id == agent_id)
                    )

                # Delete agents
                await session.execute(
                    delete(Agent).where(Agent.organization_id == org_id)
                )

            # Delete user-organization links
            await session.execute(
                delete(UserOrganization).where(UserOrganization.organization_id == org_id)
            )

            # Delete user
            result = await session.execute(
                select(User.id, User.email).where(User.email == SAMPLE_USER["email"])
            )
            user_data = result.first()
            if user_data:
                user_id, user_email = user_data
                logger.info(f"Deleting existing user: {user_email}")
                await session.execute(delete(User).where(User.id == user_id))

            # Delete organization
            logger.info(f"Deleting existing organization: {org_name}")
            await session.execute(delete(Organization).where(Organization.id == org_id))

            await session.flush()
            logger.info("✓ Cleanup completed successfully")
        else:
            logger.info("No existing seed data found - starting fresh")

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise


async def create_organization(session) -> int:
    """Create demo organization."""
    result = await session.execute(
        select(Organization.id, Organization.name).where(Organization.slug == SAMPLE_ORG["slug"])
    )
    existing = result.first()

    if existing:
        org_id, org_name = existing
        logger.info(f"Organization already exists: {org_name} (ID: {org_id})")
        return org_id

    org = Organization(**SAMPLE_ORG)
    session.add(org)
    await session.flush()

    logger.info(f"✓ Created organization: {org.name} (ID: {org.id})")
    return org.id


async def create_user(session, password: str) -> int:
    """Create demo user with hashed password."""
    result = await session.execute(
        select(User.id, User.email).where(User.email == SAMPLE_USER["email"])
    )
    existing = result.first()

    if existing:
        user_id, user_email = existing
        logger.info(f"User already exists: {user_email} (ID: {user_id})")
        return user_id

    user_data = {
        **SAMPLE_USER,
        "password_hash": get_password_hash(password)
    }
    user = User(**user_data)
    session.add(user)
    await session.flush()

    logger.info(f"✓ Created user: {user.email} (ID: {user.id})")
    return user.id


async def create_user_organization_link(session, user_id: int, organization_id: int) -> None:
    """Link user to organization with OWNER role."""
    result = await session.execute(
        select(UserOrganization).where(
            UserOrganization.user_id == user_id,
            UserOrganization.organization_id == organization_id
        )
    )
    existing = result.scalar_one_or_none()

    if existing:
        logger.info(f"User-Organization link already exists (ID: {existing.id})")
        return

    user_org = UserOrganization(
        user_id=user_id,
        organization_id=organization_id,
        role=OrganizationRole.OWNER,
        is_active=True
    )
    session.add(user_org)
    await session.flush()

    logger.info(f"✓ Created user-organization link with OWNER role (ID: {user_org.id})")


async def create_agent(session, user_id: int, organization_id: int) -> int:
    """Create demo agent."""
    result = await session.execute(
        select(Agent.id, Agent.name, Agent.public_id).where(
            Agent.name == SAMPLE_AGENT["name"],
            Agent.organization_id == organization_id
        )
    )
    existing = result.first()

    if existing:
        agent_id, agent_name, public_id = existing
        logger.info(f"Agent already exists: {agent_name} (ID: {agent_id}, public_id: {public_id})")
        return agent_id

    agent_data = {
        **SAMPLE_AGENT,
        "user_id": user_id,
        "organization_id": organization_id,
    }
    agent = Agent(**agent_data)
    session.add(agent)
    await session.flush()

    logger.info(f"✓ Created agent: {agent.name} (ID: {agent.id}, public_id: {agent.public_id})")
    return agent.id


async def create_document(session, agent_id: int, sample_data_dir: Path) -> Optional[int]:
    """Create and process sample document."""
    # Check if document already exists
    result = await session.execute(
        select(Document.id, Document.filename).where(
            Document.agent_id == agent_id,
            Document.filename == SAMPLE_DOC["filename"]
        )
    )
    existing = result.first()

    if existing:
        doc_id, filename = existing
        logger.info(f"Document already exists: {filename} (ID: {doc_id})")
        return doc_id

    # Load document content
    doc_path = sample_data_dir / SAMPLE_DOC["filename"]
    try:
        content = doc_path.read_text(encoding="utf-8")
        logger.info(f"Loaded document content: {len(content)} characters")
    except Exception as e:
        logger.error(f"Failed to read document file: {e}")
        return None

    # Create document
    doc_data = {
        **SAMPLE_DOC,
        "agent_id": agent_id,
        "content": content,
        "size": len(content.encode('utf-8')),
    }
    doc = Document(**doc_data)
    session.add(doc)
    await session.flush()

    logger.info(f"✓ Created document: {doc.filename} (ID: {doc.id}, size: {doc.size} bytes)")
    return doc.id


async def process_document_embeddings(agent_id: int, document_id: int) -> None:
    """Process document to generate embeddings for RAG."""
    try:
        from app.services.document_processor import document_processor

        logger.info(f"Processing document embeddings for document ID: {document_id}")

        # Process document to generate chunks and embeddings
        await document_processor.process_document_content(
            agent_id=agent_id,
            document_id=document_id
        )

        logger.info("✓ Document embeddings processed successfully")
    except ImportError as e:
        logger.warning(f"Document processor not available: {e}")
        logger.warning("Skipping embedding generation - document created but not indexed for RAG")
    except Exception as e:
        logger.error(f"Failed to process document embeddings: {e}")
        logger.warning("Document created but embeddings failed - RAG may not work properly")


async def seed() -> None:
    """
    Main seed function - creates complete demo environment.

    Raises:
        EnvironmentError: If running in production
        FileNotFoundError: If sample data files are missing
        IntegrityError: If database constraints are violated
    """
    logger.info("=" * 80)
    logger.info("Starting sample data seeding process...")
    logger.info("=" * 80)

    # Step 1: Validate environment
    validate_environment()

    # Step 2: Validate files
    sample_data_dir = validate_sample_data_files()

    # Step 3: Get password
    password = get_password_from_env()

    # Step 4: Seed database
    async with (await get_db_session()) as session:
        try:
            # Clean up existing data
            logger.info("\n[1/7] Cleaning up existing seed data...")
            await cleanup_existing_data(session)

            # Create organization
            logger.info("\n[2/7] Creating organization...")
            organization_id = await create_organization(session)

            # Create user
            logger.info("\n[3/7] Creating user...")
            user_id = await create_user(session, password)

            # Link user to organization
            logger.info("\n[4/7] Linking user to organization...")
            await create_user_organization_link(session, user_id, organization_id)

            # Create agent
            logger.info("\n[5/7] Creating agent...")
            agent_id = await create_agent(session, user_id, organization_id)

            # Create document
            logger.info("\n[6/7] Creating document...")
            document_id = await create_document(session, agent_id, sample_data_dir)

            # Commit all changes
            await session.commit()
            logger.info("\n✓ Database changes committed successfully")

            # Process embeddings (after commit)
            if document_id:
                logger.info("\n[7/7] Processing document embeddings...")
                await process_document_embeddings(agent_id, document_id)

            # Success summary
            logger.info("\n" + "=" * 80)
            logger.info("✅ Sample data seeded successfully!")
            logger.info("=" * 80)
            logger.info(f"\nDemo Environment Details:")
            logger.info(f"  Organization: {SAMPLE_ORG['name']} (slug: {SAMPLE_ORG['slug']})")
            logger.info(f"  User Email:   {SAMPLE_USER['email']}")
            logger.info(f"  Password:     {password}")
            logger.info(f"  Agent Name:   {SAMPLE_AGENT['name']}")
            logger.info(f"  Agent ID:     {agent_id}")
            if document_id:
                logger.info(f"  Document:     {SAMPLE_DOC['filename']} (ID: {document_id})")
            logger.info("\n⚠️  IMPORTANT: Change the demo password before deploying to any environment!")
            logger.info("=" * 80 + "\n")

        except IntegrityError as e:
            await session.rollback()
            logger.error(f"\n❌ Database integrity error: {e}")
            logger.error("This usually indicates duplicate data or constraint violations.")
            raise

        except Exception as e:
            await session.rollback()
            logger.error(f"\n❌ Seeding failed: {e}")
            logger.error("All changes have been rolled back.")
            raise


if __name__ == "__main__":
    try:
        asyncio.run(seed())
        sys.exit(0)
    except EnvironmentError as e:
        logger.error(f"\n{e}")
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error(f"\n{e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nUnexpected error: {e}")
        sys.exit(1)
