"""
API endpoint tests for critical user and organization functionality.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from app.models.user import User
from app.models.organization import Organization
from app.models.agent import Agent
from app.models.user_organization import UserOrganization


@pytest.mark.api
def test_user_context_endpoint_success(client: TestClient, authenticated_client: TestClient, mock_db_session):
    """Test the user context endpoint that was causing the original public_id error."""
    with patch('app.services.database_service.db_service') as mock_db_service:
        # Mock the database service responses
        mock_user_organizations = [
            AsyncMock(
                is_active=True,
                organization=AsyncMock(
                    id=1,
                    name="Test Organization",
                    slug="test-org",
                    plan="pro",
                    max_agents=10,
                    max_users=5,
                    max_documents_per_agent=100,
                    is_active=True
                ),
                can_manage_users=True,
                can_manage_agents=True,
                can_view_agents=True,
                can_manage_billing=False,
                can_delete_organization=False,
                role="admin"
            )
        ]

        mock_db_service.get_user_organizations.return_value = mock_user_organizations
        mock_db_service.count_organization_agents.return_value = 3
        mock_db_service.count_organization_users.return_value = 2

        # Make the API call
        response = authenticated_client.get("/api/v1/users/context")

        # Assert successful response
        assert response.status_code == 200

        data = response.json()
        assert "organizations" in data
        assert "default_organization_id" in data
        assert len(data["organizations"]) == 1

        org = data["organizations"][0]
        assert org["name"] == "Test Organization"
        assert org["agents_count"] == 3
        assert org["active_users_count"] == 2
        assert org["can_add_agent"] is True  # 3 < 10
        assert org["can_add_user"] is True   # 2 < 5


@pytest.mark.api
def test_user_context_endpoint_with_agents_having_public_id(client: TestClient, authenticated_client: TestClient, mock_db_session):
    """Test that the endpoint works correctly when agents have public_id fields (regression test)."""
    with patch('app.services.database_service.db_service') as mock_db_service:
        # Create mock organization with agents that have public_id
        mock_organization = AsyncMock(
            id=1,
            name="Test Organization",
            slug="test-org",
            plan="enterprise",
            max_agents=100,
            max_users=50,
            max_documents_per_agent=1000,
            is_active=True
        )

        mock_user_organization = AsyncMock(
            is_active=True,
            organization=mock_organization,
            can_manage_users=True,
            can_manage_agents=True,
            can_view_agents=True,
            can_manage_billing=True,
            can_delete_organization=True,
            role="owner"
        )

        # Mock agents with public_id fields
        mock_agents = [
            AsyncMock(
                id=1,
                name="Agent 1",
                public_id="agent-1-public-id",
                organization_id=1,
                is_active=True
            ),
            AsyncMock(
                id=2,
                name="Agent 2",
                public_id="agent-2-public-id",
                organization_id=1,
                is_active=True
            )
        ]

        mock_db_service.get_user_organizations.return_value = [mock_user_organization]
        mock_db_service.count_organization_agents.return_value = len(mock_agents)
        mock_db_service.count_organization_users.return_value = 5

        # Make the API call - this should not fail due to public_id column issues
        response = authenticated_client.get("/api/v1/users/context")

        # Assert successful response
        assert response.status_code == 200

        data = response.json()
        assert data["organizations"][0]["agents_count"] == 2
        assert data["organizations"][0]["user_role"] == "owner"


@pytest.mark.api
def test_user_organizations_endpoint(client: TestClient, authenticated_client: TestClient, mock_db_session):
    """Test the user organizations endpoint."""
    with patch('app.api.v1.users.select') as mock_select, \
         patch('app.api.v1.users.selectinload') as mock_selectinload:

        # Mock the SQLAlchemy query chain
        mock_query = AsyncMock()
        mock_select.return_value = mock_query
        mock_query.options.return_value = mock_query
        mock_query.where.return_value = mock_query

        # Mock the database execution
        mock_result = AsyncMock()
        mock_db_session.execute.return_value = mock_result

        # Mock the user with organization relationships
        mock_organization = AsyncMock(
            id=1,
            name="Test Organization",
            slug="test-org",
            plan="pro",
            max_agents=10,
            max_users=5,
            max_documents_per_agent=100,
            agents_count=2,
            active_users_count=3,
            is_active=True
        )
        mock_organization.can_add_agent.return_value = True
        mock_organization.can_add_user.return_value = True

        mock_user_org = AsyncMock(
            is_active=True,
            organization=mock_organization,
            can_manage_users=False,
            can_manage_agents=True,
            can_view_agents=True,
            can_manage_billing=False,
            can_delete_organization=False,
            role="member"
        )

        mock_user = AsyncMock()
        mock_user.user_organizations = [mock_user_org]
        mock_result.scalar_one_or_none.return_value = mock_user

        # Make the API call
        response = authenticated_client.get("/api/v1/users/organizations")

        # Assert successful response
        assert response.status_code == 200

        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "Test Organization"
        assert data[0]["user_role"] == "member"
        assert data[0]["user_permissions"]["manage_agents"] is True
        assert data[0]["user_permissions"]["manage_billing"] is False


@pytest.mark.api
def test_organization_context_endpoint(client: TestClient, authenticated_client: TestClient, mock_db_session):
    """Test the organization context endpoint."""
    organization_id = 1

    with patch('app.api.v1.users.select') as mock_select, \
         patch('app.api.v1.users.selectinload') as mock_selectinload:

        # Mock the SQLAlchemy query chain
        mock_query = AsyncMock()
        mock_select.return_value = mock_query
        mock_query.options.return_value = mock_query
        mock_query.where.return_value = mock_query

        # Mock the database execution
        mock_result = AsyncMock()
        mock_db_session.execute.return_value = mock_result

        # Mock the organization with proper data
        mock_organization = AsyncMock(
            id=organization_id,
            name="Specific Organization",
            slug="specific-org",
            plan="enterprise",
            max_agents=100,
            max_users=25,
            max_documents_per_agent=500,
            agents_count=15,
            active_users_count=12,
            is_active=True
        )
        mock_organization.can_add_agent.return_value = True
        mock_organization.can_add_user.return_value = True

        mock_user_org = AsyncMock(
            organization_id=organization_id,
            is_active=True,
            organization=mock_organization,
            can_manage_users=True,
            can_manage_agents=True,
            can_view_agents=True,
            can_manage_billing=True,
            can_delete_organization=False,
            role="admin"
        )

        mock_user = AsyncMock()
        mock_user.user_organizations = [mock_user_org]
        mock_result.scalar_one_or_none.return_value = mock_user

        # Make the API call
        response = authenticated_client.get(f"/api/v1/users/organizations/{organization_id}/context")

        # Assert successful response
        assert response.status_code == 200

        data = response.json()
        assert data["id"] == organization_id
        assert data["name"] == "Specific Organization"
        assert data["user_role"] == "admin"
        assert data["agents_count"] == 15
        assert data["can_add_agent"] is True
        assert data["user_permissions"]["manage_billing"] is True


@pytest.mark.api
def test_user_context_endpoint_handles_database_errors(client: TestClient, authenticated_client: TestClient, mock_db_session):
    """Test that the user context endpoint handles database errors gracefully."""
    with patch('app.services.database_service.db_service') as mock_db_service:
        # Simulate a database error
        mock_db_service.get_user_organizations.side_effect = Exception("Database connection failed")

        # Make the API call
        response = authenticated_client.get("/api/v1/users/context")

        # Assert error response
        assert response.status_code == 500
        assert "Error retrieving user context" in response.json()["detail"]


@pytest.mark.api
def test_user_context_endpoint_empty_organizations(client: TestClient, authenticated_client: TestClient, mock_db_session):
    """Test user context endpoint when user has no organizations."""
    with patch('app.services.database_service.db_service') as mock_db_service:
        # Mock empty organizations
        mock_db_service.get_user_organizations.return_value = []

        # Make the API call
        response = authenticated_client.get("/api/v1/users/context")

        # Assert successful response with empty organizations
        assert response.status_code == 200

        data = response.json()
        assert data["organizations"] == []
        assert data["default_organization_id"] is None


@pytest.mark.api
def test_unauthenticated_access_to_user_endpoints(client: TestClient):
    """Test that unauthenticated requests are properly rejected."""
    # Test user context endpoint
    response = client.get("/api/v1/users/context")
    assert response.status_code in [401, 403]  # Unauthorized or Forbidden

    # Test user organizations endpoint
    response = client.get("/api/v1/users/organizations")
    assert response.status_code in [401, 403]

    # Test organization context endpoint
    response = client.get("/api/v1/users/organizations/1/context")
    assert response.status_code in [401, 403]


@pytest.mark.api
def test_user_context_with_inactive_organizations(client: TestClient, authenticated_client: TestClient, mock_db_session):
    """Test that inactive organizations are filtered out from user context."""
    with patch('app.services.database_service.db_service') as mock_db_service:
        # Mock user organizations with one active and one inactive
        mock_user_organizations = [
            AsyncMock(
                is_active=True,
                organization=AsyncMock(
                    id=1,
                    name="Active Organization",
                    slug="active-org",
                    plan="pro",
                    max_agents=10,
                    max_users=5,
                    max_documents_per_agent=100,
                    is_active=True
                ),
                can_manage_users=True,
                can_manage_agents=True,
                can_view_agents=True,
                can_manage_billing=False,
                can_delete_organization=False,
                role="admin"
            ),
            AsyncMock(
                is_active=True,
                organization=AsyncMock(
                    id=2,
                    name="Inactive Organization",
                    slug="inactive-org",
                    plan="free",
                    max_agents=5,
                    max_users=3,
                    max_documents_per_agent=50,
                    is_active=False  # This organization is inactive
                ),
                can_manage_users=True,
                can_manage_agents=True,
                can_view_agents=True,
                can_manage_billing=False,
                can_delete_organization=False,
                role="owner"
            )
        ]

        mock_db_service.get_user_organizations.return_value = mock_user_organizations
        mock_db_service.count_organization_agents.return_value = 2
        mock_db_service.count_organization_users.return_value = 1

        # Make the API call
        response = authenticated_client.get("/api/v1/users/context")

        # Assert successful response with only active organization
        assert response.status_code == 200

        data = response.json()
        assert len(data["organizations"]) == 1
        assert data["organizations"][0]["name"] == "Active Organization"
        assert data["organizations"][0]["id"] == 1


@pytest.mark.api
def test_organization_limits_logic(client: TestClient, authenticated_client: TestClient, mock_db_session):
    """Test that organization limits are correctly calculated."""
    with patch('app.services.database_service.db_service') as mock_db_service:
        # Mock organization at limits
        mock_user_organizations = [
            AsyncMock(
                is_active=True,
                organization=AsyncMock(
                    id=1,
                    name="At Limits Organization",
                    slug="at-limits-org",
                    plan="starter",
                    max_agents=5,
                    max_users=3,
                    max_documents_per_agent=50,
                    is_active=True
                ),
                can_manage_users=True,
                can_manage_agents=True,
                can_view_agents=True,
                can_manage_billing=True,
                can_delete_organization=False,
                role="owner"
            )
        ]

        # Set counts to be at the limits
        mock_db_service.get_user_organizations.return_value = mock_user_organizations
        mock_db_service.count_organization_agents.return_value = 5  # At max_agents limit
        mock_db_service.count_organization_users.return_value = 3   # At max_users limit

        # Make the API call
        response = authenticated_client.get("/api/v1/users/context")

        # Assert successful response
        assert response.status_code == 200

        data = response.json()
        org = data["organizations"][0]
        assert org["can_add_agent"] is False  # 5 >= 5 (at limit)
        assert org["can_add_user"] is False   # 3 >= 3 (at limit)
        assert org["agents_count"] == 5
        assert org["active_users_count"] == 3


@pytest.mark.api
def test_unlimited_organization_limits(client: TestClient, authenticated_client: TestClient, mock_db_session):
    """Test that unlimited organizations (-1 limits) work correctly."""
    with patch('app.services.database_service.db_service') as mock_db_service:
        # Mock organization with unlimited limits
        mock_user_organizations = [
            AsyncMock(
                is_active=True,
                organization=AsyncMock(
                    id=1,
                    name="Unlimited Organization",
                    slug="unlimited-org",
                    plan="enterprise",
                    max_agents=-1,  # Unlimited
                    max_users=-1,   # Unlimited
                    max_documents_per_agent=1000,
                    is_active=True
                ),
                can_manage_users=True,
                can_manage_agents=True,
                can_view_agents=True,
                can_manage_billing=True,
                can_delete_organization=True,
                role="owner"
            )
        ]

        mock_db_service.get_user_organizations.return_value = mock_user_organizations
        mock_db_service.count_organization_agents.return_value = 100  # High count
        mock_db_service.count_organization_users.return_value = 50    # High count

        # Make the API call
        response = authenticated_client.get("/api/v1/users/context")

        # Assert successful response
        assert response.status_code == 200

        data = response.json()
        org = data["organizations"][0]
        assert org["can_add_agent"] is True   # -1 means unlimited
        assert org["can_add_user"] is True    # -1 means unlimited
        assert org["agents_count"] == 100
        assert org["active_users_count"] == 50