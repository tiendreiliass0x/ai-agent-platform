"""
Database integration tests for schema validation and data integrity.
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from app.models.user import User
from app.models.organization import Organization
from app.models.agent import Agent
from app.models.user_organization import UserOrganization


@pytest.mark.database
@pytest.mark.asyncio
async def test_agent_public_id_column_exists(db_session: AsyncSession, test_organization, test_user):
    """Test that the public_id column exists on agents table."""
    # This is a regression test for the original schema mismatch error
    agent = Agent(
        name="Test Agent",
        description="Test agent for public_id validation",
        user_id=test_user.id,
        organization_id=test_organization.id,
        is_active=True,
        public_id="test-public-id-123"
    )

    db_session.add(agent)
    await db_session.commit()
    await db_session.refresh(agent)

    assert agent.public_id == "test-public-id-123"


@pytest.mark.database
@pytest.mark.asyncio
async def test_agent_public_id_unique_constraint(db_session: AsyncSession, test_organization, test_user):
    """Test that public_id has unique constraint."""
    # Create first agent
    agent1 = Agent(
        name="Agent 1",
        description="First agent",
        user_id=test_user.id,
        organization_id=test_organization.id,
        is_active=True,
        public_id="unique-id-123"
    )

    db_session.add(agent1)
    await db_session.commit()

    # Try to create second agent with same public_id
    agent2 = Agent(
        name="Agent 2",
        description="Second agent",
        user_id=test_user.id,
        organization_id=test_organization.id,
        is_active=True,
        public_id="unique-id-123"  # Same as agent1
    )

    db_session.add(agent2)

    # Should raise integrity error due to unique constraint
    with pytest.raises(IntegrityError):
        await db_session.commit()


@pytest.mark.database
@pytest.mark.asyncio
async def test_agent_public_id_not_null(db_session: AsyncSession, test_organization, test_user):
    """Test that public_id cannot be null."""
    agent = Agent(
        name="Test Agent",
        description="Test agent",
        user_id=test_user.id,
        organization_id=test_organization.id,
        is_active=True,
        public_id=None  # This should fail
    )

    db_session.add(agent)

    # Should raise integrity error due to NOT NULL constraint
    with pytest.raises(IntegrityError):
        await db_session.commit()


@pytest.mark.database
@pytest.mark.asyncio
async def test_organization_agents_relationship(db_session: AsyncSession):
    """Test that organization-agent relationship works correctly."""
    # Create organization
    org = Organization(
        name="Test Corp",
        slug="test-corp",
        plan="pro",
        max_agents=10,
        max_users=5,
        max_documents_per_agent=100,
        is_active=True
    )
    db_session.add(org)
    await db_session.flush()

    # Create user for the agents
    user = User(
        email="agent_owner@example.com",
        name="Agent Owner",
        password_hash="password_hash",
        is_active=True
    )
    db_session.add(user)
    await db_session.flush()

    # Create multiple agents
    agents = []
    for i in range(3):
        agent = Agent(
            name=f"Agent {i+1}",
            description=f"Test agent {i+1}",
            user_id=user.id,
            organization_id=org.id,
            is_active=True,
            public_id=f"agent-{i+1}-test"
        )
        db_session.add(agent)
        agents.append(agent)

    await db_session.commit()

    # Test relationship
    await db_session.refresh(org)
    org_agents = await db_session.execute(
        select(Agent).where(Agent.organization_id == org.id)
    )
    fetched_agents = org_agents.scalars().all()

    assert len(fetched_agents) == 3
    assert all(agent.organization_id == org.id for agent in fetched_agents)


@pytest.mark.database
@pytest.mark.asyncio
async def test_user_organization_relationship(db_session: AsyncSession):
    """Test user-organization many-to-many relationship."""
    # Create user
    user = User(
        email="test@example.com",
        name="Test User",
        password_hash="password_hash",
        is_active=True
    )
    db_session.add(user)

    # Create organization
    org = Organization(
        name="Test Organization",
        slug="test-org",
        plan="enterprise",
        max_agents=100,
        max_users=50,
        max_documents_per_agent=1000,
        is_active=True
    )
    db_session.add(org)
    await db_session.flush()

    # Create user-organization relationship
    user_org = UserOrganization(
        user_id=user.id,
        organization_id=org.id,
        role="admin",
        is_active=True,
        can_manage_users=True,
        can_manage_agents=True,
        can_view_agents=True,
        can_manage_billing=True,
        can_delete_organization=True
    )
    db_session.add(user_org)
    await db_session.commit()

    # Test relationship query
    result = await db_session.execute(
        select(UserOrganization)
        .where(UserOrganization.user_id == user.id)
        .where(UserOrganization.organization_id == org.id)
    )
    fetched_user_org = result.scalar_one()

    assert fetched_user_org.role == "admin"
    assert fetched_user_org.can_manage_users is True


@pytest.mark.database
@pytest.mark.asyncio
async def test_database_schema_integrity(db_session: AsyncSession):
    """Test overall database schema integrity."""
    # Test that all expected tables exist
    tables_query = """
        SELECT name FROM sqlite_master
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
    """

    result = await db_session.execute(text(tables_query))
    tables = [row[0] for row in result.fetchall()]

    # Expected core tables (may vary based on migrations)
    expected_tables = ["users", "organizations", "agents", "user_organizations"]

    for table in expected_tables:
        assert table in tables, f"Expected table '{table}' not found in database"


@pytest.mark.database
@pytest.mark.asyncio
async def test_agent_public_id_index_exists(db_session: AsyncSession):
    """Test that public_id index exists for performance."""
    # Check that the index exists
    index_query = """
        SELECT name FROM sqlite_master
        WHERE type='index' AND name='ix_agents_public_id'
    """

    result = await db_session.execute(text(index_query))
    index_row = result.fetchone()

    assert index_row is not None, "Index 'ix_agents_public_id' not found"


@pytest.mark.database
@pytest.mark.asyncio
async def test_organization_limits_validation(db_session: AsyncSession):
    """Test organization limits and validation."""
    org = Organization(
        name="Limit Test Corp",
        slug="limit-test",
        plan="starter",
        max_agents=5,
        max_users=3,
        max_documents_per_agent=50,
        is_active=True
    )
    db_session.add(org)
    await db_session.commit()

    # Test that organization was created with correct limits
    await db_session.refresh(org)
    assert org.max_agents == 5
    assert org.max_users == 3
    assert org.max_documents_per_agent == 50


@pytest.mark.database
@pytest.mark.asyncio
async def test_cascade_deletion_behavior(db_session: AsyncSession):
    """Test cascade deletion and referential integrity."""
    # Create organization with agents
    org = Organization(
        name="Delete Test Corp",
        slug="delete-test",
        plan="pro",
        max_agents=10,
        max_users=5,
        max_documents_per_agent=100,
        is_active=True
    )
    db_session.add(org)
    await db_session.flush()

    # Create user for the agent
    user = User(
        email="delete_test@example.com",
        name="Delete Test User",
        password_hash="password_hash",
        is_active=True
    )
    db_session.add(user)
    await db_session.flush()

    # Add agent
    agent = Agent(
        name="Test Agent",
        description="Agent for deletion test",
        user_id=user.id,
        organization_id=org.id,
        is_active=True,
        public_id="delete-test-agent"
    )
    db_session.add(agent)
    await db_session.commit()

    # Verify agent exists
    agent_count = await db_session.execute(
        select(Agent).where(Agent.organization_id == org.id)
    )
    assert len(agent_count.scalars().all()) == 1

    # Delete organization
    await db_session.delete(org)
    await db_session.commit()

    # Verify agent still exists but organization is gone
    # (This tests that we don't have cascade delete setup)
    remaining_agents = await db_session.execute(
        select(Agent).where(Agent.public_id == "delete-test-agent")
    )
    remaining_agent = remaining_agents.scalar_one_or_none()

    # Agent should still exist with dangling organization_id
    # This behavior may need to be adjusted based on business requirements
    assert remaining_agent is not None


@pytest.mark.database
@pytest.mark.asyncio
async def test_data_consistency_after_migration(db_session: AsyncSession):
    """Test data consistency after the public_id migration."""
    # Create an agent and verify all fields work together
    org = Organization(
        name="Migration Test Corp",
        slug="migration-test",
        plan="enterprise",
        max_agents=100,
        max_users=50,
        max_documents_per_agent=1000,
        is_active=True
    )
    db_session.add(org)
    await db_session.flush()

    # Create user for the agent
    user = User(
        email="migration_test@example.com",
        name="Migration Test User",
        password_hash="password_hash",
        is_active=True
    )
    db_session.add(user)
    await db_session.flush()

    agent = Agent(
        name="Migration Test Agent",
        description="Agent to test post-migration consistency",
        user_id=user.id,
        organization_id=org.id,
        is_active=True,
        public_id="migration-test-agent-id"
    )
    db_session.add(agent)
    await db_session.commit()

    # Retrieve and verify all fields
    retrieved_agent = await db_session.execute(
        select(Agent).where(Agent.public_id == "migration-test-agent-id")
    )
    agent_data = retrieved_agent.scalar_one()

    assert agent_data.name == "Migration Test Agent"
    assert agent_data.description == "Agent to test post-migration consistency"
    assert agent_data.user_id == user.id
    assert agent_data.organization_id == org.id
    assert agent_data.is_active is True
    assert agent_data.public_id == "migration-test-agent-id"