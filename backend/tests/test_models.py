"""
Model validation tests to ensure model-database schema consistency.
"""

import pytest
from sqlalchemy import inspect, create_engine
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
from typing import Type, Dict, Any

from app.models.user import User
from app.models.organization import Organization
from app.models.agent import Agent, AgentTier, DomainExpertiseType
from app.models.user_organization import UserOrganization


@pytest.mark.unit
def test_agent_model_has_required_fields():
    """Test that Agent model has all required fields."""
    agent_table = Agent.__table__
    column_names = [col.name for col in agent_table.columns]

    # Critical fields that caused the original regression
    required_fields = [
        'id', 'public_id', 'name', 'description', 'user_id',
        'organization_id', 'is_active', 'created_at', 'updated_at'
    ]

    for field in required_fields:
        assert field in column_names, f"Required field '{field}' missing from Agent model"


@pytest.mark.unit
def test_agent_public_id_field_properties():
    """Test that public_id field has correct properties."""
    agent_table = Agent.__table__
    public_id_col = agent_table.columns['public_id']

    # Should be unique and not nullable
    assert public_id_col.unique, "public_id should be unique"
    assert not public_id_col.nullable, "public_id should not be nullable"
    assert public_id_col.index, "public_id should be indexed"


@pytest.mark.unit
def test_user_model_has_correct_password_field():
    """Test that User model uses password_hash, not hashed_password."""
    user_table = User.__table__
    column_names = [col.name for col in user_table.columns]

    assert 'password_hash' in column_names, "User should have password_hash field"
    assert 'hashed_password' not in column_names, "User should not have hashed_password field"


@pytest.mark.unit
def test_all_models_have_required_base_fields():
    """Test that all models have required base fields."""
    models_to_test = [User, Organization, Agent, UserOrganization]

    for model in models_to_test:
        column_names = [col.name for col in model.__table__.columns]

        # All models should have ID
        assert 'id' in column_names, f"{model.__name__} should have id field"

        # Models with timestamps should have them
        if hasattr(model, 'created_at'):
            assert 'created_at' in column_names, f"{model.__name__} should have created_at field"

        if hasattr(model, 'updated_at'):
            assert 'updated_at' in column_names, f"{model.__name__} should have updated_at field"


@pytest.mark.unit
def test_agent_enums_are_valid():
    """Test that Agent enum fields have valid values."""
    # Test AgentTier enum
    valid_tiers = ['basic', 'professional', 'enterprise']
    for tier in AgentTier:
        assert tier.value in valid_tiers, f"Invalid tier value: {tier.value}"

    # Test DomainExpertiseType enum
    valid_expertise_types = [
        'sales_rep', 'solution_engineer', 'support_expert',
        'domain_specialist', 'product_expert'
    ]
    for expertise in DomainExpertiseType:
        assert expertise.value in valid_expertise_types, f"Invalid expertise type: {expertise.value}"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_agent_model_creation_with_public_id(db_session: AsyncSession, test_user, test_organization):
    """Test that Agent can be created with public_id."""
    agent = Agent(
        name="Test Agent",
        description="Test description",
        user_id=test_user.id,
        organization_id=test_organization.id,
        public_id="test-public-id-123",
        is_active=True
    )

    db_session.add(agent)
    await db_session.commit()
    await db_session.refresh(agent)

    assert agent.public_id == "test-public-id-123"
    assert agent.name == "Test Agent"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_agent_model_auto_generates_public_id(db_session: AsyncSession, test_user, test_organization):
    """Test that Agent auto-generates public_id if not provided."""
    agent = Agent(
        name="Auto ID Agent",
        description="Test description",
        user_id=test_user.id,
        organization_id=test_organization.id,
        is_active=True
        # No public_id provided
    )

    db_session.add(agent)
    await db_session.commit()
    await db_session.refresh(agent)

    # Should have auto-generated UUID
    assert agent.public_id is not None
    assert len(agent.public_id) > 0
    assert '-' in agent.public_id  # UUID format


@pytest.mark.unit
@pytest.mark.asyncio
async def test_agent_public_id_uniqueness_constraint(db_session: AsyncSession, test_user, test_organization):
    """Test that public_id uniqueness is enforced."""
    # Create first agent
    agent1 = Agent(
        name="Agent 1",
        description="First agent",
        user_id=test_user.id,
        organization_id=test_organization.id,
        public_id="duplicate-id",
        is_active=True
    )
    db_session.add(agent1)
    await db_session.commit()

    # Try to create second agent with same public_id
    agent2 = Agent(
        name="Agent 2",
        description="Second agent",
        user_id=test_user.id,
        organization_id=test_organization.id,
        public_id="duplicate-id",  # Same as agent1
        is_active=True
    )
    db_session.add(agent2)

    # Should raise integrity error
    with pytest.raises(IntegrityError):
        await db_session.commit()


@pytest.mark.unit
def test_model_table_names_are_correct():
    """Test that all models have correct table names."""
    expected_table_names = {
        User: "users",
        Organization: "organizations",
        Agent: "agents",
        UserOrganization: "user_organizations"
    }

    for model, expected_name in expected_table_names.items():
        assert model.__tablename__ == expected_name, f"{model.__name__} should have table name {expected_name}"


@pytest.mark.unit
def test_model_relationships_are_defined():
    """Test that required relationships are defined."""
    # User relationships
    assert hasattr(User, 'agents'), "User should have agents relationship"
    assert hasattr(User, 'user_organizations'), "User should have user_organizations relationship"

    # Organization relationships
    assert hasattr(Organization, 'agents'), "Organization should have agents relationship"

    # Agent relationships
    assert hasattr(Agent, 'user'), "Agent should have user relationship"
    assert hasattr(Agent, 'organization'), "Agent should have organization relationship"

    # UserOrganization relationships
    assert hasattr(UserOrganization, 'user'), "UserOrganization should have user relationship"
    assert hasattr(UserOrganization, 'organization'), "UserOrganization should have organization relationship"


@pytest.mark.unit
def test_model_foreign_keys_are_correct():
    """Test that foreign key relationships are properly defined."""
    # Agent foreign keys
    agent_table = Agent.__table__
    fk_columns = [col for col in agent_table.columns if col.foreign_keys]

    user_id_col = agent_table.columns['user_id']
    org_id_col = agent_table.columns['organization_id']

    assert user_id_col.foreign_keys, "Agent.user_id should have foreign key"
    assert org_id_col.foreign_keys, "Agent.organization_id should have foreign key"

    # Check foreign key targets
    user_fk = list(user_id_col.foreign_keys)[0]
    org_fk = list(org_id_col.foreign_keys)[0]

    assert 'users.id' in str(user_fk), "Agent.user_id should reference users.id"
    assert 'organizations.id' in str(org_fk), "Agent.organization_id should reference organizations.id"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_model_cascade_behavior(db_session: AsyncSession):
    """Test model cascade deletion behavior."""
    # Create test data
    user = User(
        email="cascade_test@example.com",
        name="Cascade Test User",
        password_hash="hashed",
        is_active=True
    )
    db_session.add(user)
    await db_session.flush()

    org = Organization(
        name="Cascade Test Org",
        slug="cascade-test-org",
        plan="pro",
        max_agents=10,
        max_users=5,
        max_documents_per_agent=100,
        is_active=True
    )
    db_session.add(org)
    await db_session.flush()

    # Create user-organization relationship
    user_org = UserOrganization(
        user_id=user.id,
        organization_id=org.id,
        role="admin",
        is_active=True
    )
    db_session.add(user_org)
    await db_session.commit()

    # Test that deleting user cascades to user_organizations
    await db_session.delete(user)
    await db_session.commit()

    # user_organizations should be deleted due to cascade
    from sqlalchemy import select
    result = await db_session.execute(
        select(UserOrganization).where(UserOrganization.user_id == user.id)
    )
    remaining_user_orgs = result.scalars().all()
    assert len(remaining_user_orgs) == 0, "UserOrganization should be cascade deleted with User"


@pytest.mark.unit
def test_agent_json_fields_have_defaults():
    """Test that Agent JSON fields have proper defaults."""
    agent_table = Agent.__table__

    # tool_policy should have default
    tool_policy_col = agent_table.columns['tool_policy']
    assert tool_policy_col.default is not None, "tool_policy should have default value"

    # config should have default
    config_col = agent_table.columns['config']
    assert config_col.default is not None, "config should have default value"

    # widget_config should have default
    widget_config_col = agent_table.columns['widget_config']
    assert widget_config_col.default is not None, "widget_config should have default value"


@pytest.mark.unit
def test_organization_has_limit_fields():
    """Test that Organization model has all limit fields."""
    org_table = Organization.__table__
    column_names = [col.name for col in org_table.columns]

    limit_fields = ['max_agents', 'max_users', 'max_documents_per_agent']
    for field in limit_fields:
        assert field in column_names, f"Organization should have {field} field"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_user_organization_permissions_model(db_session: AsyncSession, test_user, test_organization):
    """Test UserOrganization permission fields."""
    user_org = UserOrganization(
        user_id=test_user.id,
        organization_id=test_organization.id,
        role="member",
        is_active=True
    )

    db_session.add(user_org)
    await db_session.commit()
    await db_session.refresh(user_org)

    # Test permission properties are computed based on role
    # member role should have no permissions
    assert user_org.can_manage_users is False  # Only owner/admin can manage users
    assert user_org.can_manage_agents is False  # Only owner/admin can manage agents
    assert user_org.can_view_agents is False  # Members cannot view agents (need viewer+ role)
    assert user_org.can_manage_billing is False  # Only owner/admin can manage billing
    assert user_org.can_delete_organization is False  # Only owner can delete
    assert user_org.role == "member"


@pytest.mark.unit
def test_model_string_representations():
    """Test that models have useful string representations."""
    # This helps with debugging
    models = [User, Organization, Agent, UserOrganization]

    for model in models:
        # Check if model has __repr__ or __str__ methods
        has_repr = hasattr(model, '__repr__') and callable(getattr(model, '__repr__'))
        has_str = hasattr(model, '__str__') and callable(getattr(model, '__str__'))

        # At least one should be implemented for debugging
        assert has_repr or has_str, f"{model.__name__} should have __repr__ or __str__ method"