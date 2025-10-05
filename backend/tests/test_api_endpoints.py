"""
API endpoint tests for critical user and organization functionality.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, Mock

from app.models.user import User
from app.models.organization import Organization
from app.models.agent import Agent
from app.models.user_organization import UserOrganization


@pytest.mark.api
def test_user_context_endpoint_success(client: TestClient, authenticated_client: TestClient, mock_db_session):
    """Test the user context endpoint that was causing the original public_id error."""
    with patch('app.services.database_service.db_service') as mock_db_service:
        # Mock the database service responses
        mock_org = Mock(spec=Organization)
        mock_org.id = 1
        mock_org.name = "Test Organization"
        mock_org.slug = "test-org"
        mock_org.plan = "pro"
        mock_org.max_agents = 10
        mock_org.max_users = 5
        mock_org.max_documents_per_agent = 100
        mock_org.is_active = True

        mock_user_org = Mock(spec=UserOrganization)
        mock_user_org.is_active = True
        mock_user_org.organization = mock_org
        mock_user_org.can_manage_users = True
        mock_user_org.can_manage_agents = True
        mock_user_org.can_view_agents = True
        mock_user_org.can_manage_billing = False
        mock_user_org.can_delete_organization = False
        mock_user_org.role = "admin"

        mock_user_organizations = [mock_user_org]

        mock_db_service.get_user_organizations = AsyncMock(return_value=mock_user_organizations)
        mock_db_service.count_organization_agents = AsyncMock(return_value=3)
        mock_db_service.count_organization_users = AsyncMock(return_value=2)

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
        mock_organization = Mock(spec=Organization)
        mock_organization.id = 1
        mock_organization.name = "Test Organization"
        mock_organization.slug = "test-org"
        mock_organization.plan = "enterprise"
        mock_organization.max_agents = 100
        mock_organization.max_users = 50
        mock_organization.max_documents_per_agent = 1000
        mock_organization.is_active = True

        mock_user_organization = Mock(spec=UserOrganization)
        mock_user_organization.is_active = True
        mock_user_organization.organization = mock_organization
        mock_user_organization.can_manage_users = True
        mock_user_organization.can_manage_agents = True
        mock_user_organization.can_view_agents = True
        mock_user_organization.can_manage_billing = True
        mock_user_organization.can_delete_organization = True
        mock_user_organization.role = "owner"

        # Mock agents with public_id fields
        mock_agent1 = Mock(spec=Agent)
        mock_agent1.id = 1
        mock_agent1.name = "Agent 1"
        mock_agent1.public_id = "agent-1-public-id"
        mock_agent1.organization_id = 1
        mock_agent1.is_active = True

        mock_agent2 = Mock(spec=Agent)
        mock_agent2.id = 2
        mock_agent2.name = "Agent 2"
        mock_agent2.public_id = "agent-2-public-id"
        mock_agent2.organization_id = 1
        mock_agent2.is_active = True

        mock_agents = [mock_agent1, mock_agent2]

        mock_db_service.get_user_organizations = AsyncMock(return_value=[mock_user_organization])
        mock_db_service.count_organization_agents = AsyncMock(return_value=len(mock_agents))
        mock_db_service.count_organization_users = AsyncMock(return_value=5)

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
    # Create proper mock objects
    mock_org = Mock(spec=Organization)
    mock_org.id = 1
    mock_org.name = "Test Organization"
    mock_org.slug = "test-org"
    mock_org.plan = "pro"
    mock_org.max_agents = 10
    mock_org.max_users = 5
    mock_org.max_documents_per_agent = 100
    mock_org.is_active = True
    # Mock properties and methods
    mock_org.agents_count = 2
    mock_org.active_users_count = 3
    mock_org.can_add_agent = Mock(return_value=True)
    mock_org.can_add_user = Mock(return_value=True)

    mock_user_org = Mock(spec=UserOrganization)
    mock_user_org.is_active = True
    mock_user_org.organization = mock_org
    mock_user_org.can_manage_users = False
    mock_user_org.can_manage_agents = True
    mock_user_org.can_view_agents = True
    mock_user_org.can_manage_billing = False
    mock_user_org.can_delete_organization = False
    mock_user_org.role = "member"

    mock_user = Mock(spec=User)
    mock_user.id = 1
    mock_user.user_organizations = [mock_user_org]

    # Mock the database execution
    mock_result = Mock()
    mock_result.scalar_one_or_none.return_value = mock_user

    # Make the API call
    with patch.object(mock_db_session, 'execute', new=AsyncMock(return_value=mock_result)):
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

    # Create proper mock objects
    mock_org = Mock(spec=Organization)
    mock_org.id = organization_id
    mock_org.name = "Specific Organization"
    mock_org.slug = "specific-org"
    mock_org.plan = "enterprise"
    mock_org.max_agents = 100
    mock_org.max_users = 25
    mock_org.max_documents_per_agent = 500
    mock_org.is_active = True
    # Mock properties and methods
    mock_org.agents_count = 15
    mock_org.active_users_count = 12
    mock_org.can_add_agent = Mock(return_value=True)
    mock_org.can_add_user = Mock(return_value=True)

    mock_user_org = Mock(spec=UserOrganization)
    mock_user_org.organization_id = organization_id
    mock_user_org.is_active = True
    mock_user_org.organization = mock_org
    mock_user_org.can_manage_users = True
    mock_user_org.can_manage_agents = True
    mock_user_org.can_view_agents = True
    mock_user_org.can_manage_billing = True
    mock_user_org.can_delete_organization = False
    mock_user_org.role = "admin"

    mock_user = Mock(spec=User)
    mock_user.id = 1
    mock_user.user_organizations = [mock_user_org]

    # Mock the database execution
    mock_result = Mock()
    mock_result.scalar_one_or_none.return_value = mock_user

    # Make the API call
    with patch.object(mock_db_session, 'execute', new=AsyncMock(return_value=mock_result)):
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
        response_data = response.json()
        # Check if error message is in detail or message field
        error_message = response_data.get("detail") or response_data.get("message") or str(response_data)
        assert "Error retrieving user context" in error_message


@pytest.mark.api
def test_user_context_endpoint_empty_organizations(client: TestClient, authenticated_client: TestClient, mock_db_session):
    """Test user context endpoint when user has no organizations."""
    with patch('app.services.database_service.db_service') as mock_db_service:
        # Mock empty organizations
        mock_db_service.get_user_organizations = AsyncMock(return_value=[])

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
        # Mock active organization
        mock_active_org = Mock(spec=Organization)
        mock_active_org.id = 1
        mock_active_org.name = "Active Organization"
        mock_active_org.slug = "active-org"
        mock_active_org.plan = "pro"
        mock_active_org.max_agents = 10
        mock_active_org.max_users = 5
        mock_active_org.max_documents_per_agent = 100
        mock_active_org.is_active = True

        mock_active_user_org = Mock(spec=UserOrganization)
        mock_active_user_org.is_active = True
        mock_active_user_org.organization = mock_active_org
        mock_active_user_org.can_manage_users = True
        mock_active_user_org.can_manage_agents = True
        mock_active_user_org.can_view_agents = True
        mock_active_user_org.can_manage_billing = False
        mock_active_user_org.can_delete_organization = False
        mock_active_user_org.role = "admin"

        # Mock inactive organization
        mock_inactive_org = Mock(spec=Organization)
        mock_inactive_org.id = 2
        mock_inactive_org.name = "Inactive Organization"
        mock_inactive_org.slug = "inactive-org"
        mock_inactive_org.plan = "free"
        mock_inactive_org.max_agents = 5
        mock_inactive_org.max_users = 3
        mock_inactive_org.max_documents_per_agent = 50
        mock_inactive_org.is_active = False

        mock_inactive_user_org = Mock(spec=UserOrganization)
        mock_inactive_user_org.is_active = True
        mock_inactive_user_org.organization = mock_inactive_org
        mock_inactive_user_org.can_manage_users = True
        mock_inactive_user_org.can_manage_agents = True
        mock_inactive_user_org.can_view_agents = True
        mock_inactive_user_org.can_manage_billing = False
        mock_inactive_user_org.can_delete_organization = False
        mock_inactive_user_org.role = "owner"

        mock_user_organizations = [mock_active_user_org, mock_inactive_user_org]

        mock_db_service.get_user_organizations = AsyncMock(return_value=mock_user_organizations)
        mock_db_service.count_organization_agents = AsyncMock(return_value=2)
        mock_db_service.count_organization_users = AsyncMock(return_value=1)

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
        mock_org = Mock(spec=Organization)
        mock_org.id = 1
        mock_org.name = "At Limits Organization"
        mock_org.slug = "at-limits-org"
        mock_org.plan = "starter"
        mock_org.max_agents = 5
        mock_org.max_users = 3
        mock_org.max_documents_per_agent = 50
        mock_org.is_active = True

        mock_user_org = Mock(spec=UserOrganization)
        mock_user_org.is_active = True
        mock_user_org.organization = mock_org
        mock_user_org.can_manage_users = True
        mock_user_org.can_manage_agents = True
        mock_user_org.can_view_agents = True
        mock_user_org.can_manage_billing = True
        mock_user_org.can_delete_organization = False
        mock_user_org.role = "owner"

        mock_user_organizations = [mock_user_org]

        # Set counts to be at the limits
        mock_db_service.get_user_organizations = AsyncMock(return_value=mock_user_organizations)
        mock_db_service.count_organization_agents = AsyncMock(return_value=5)  # At max_agents limit
        mock_db_service.count_organization_users = AsyncMock(return_value=3)   # At max_users limit

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
        mock_org = Mock(spec=Organization)
        mock_org.id = 1
        mock_org.name = "Unlimited Organization"
        mock_org.slug = "unlimited-org"
        mock_org.plan = "enterprise"
        mock_org.max_agents = -1  # Unlimited
        mock_org.max_users = -1   # Unlimited
        mock_org.max_documents_per_agent = 1000
        mock_org.is_active = True

        mock_user_org = Mock(spec=UserOrganization)
        mock_user_org.is_active = True
        mock_user_org.organization = mock_org
        mock_user_org.can_manage_users = True
        mock_user_org.can_manage_agents = True
        mock_user_org.can_view_agents = True
        mock_user_org.can_manage_billing = True
        mock_user_org.can_delete_organization = True
        mock_user_org.role = "owner"

        mock_user_organizations = [mock_user_org]

        mock_db_service.get_user_organizations = AsyncMock(return_value=mock_user_organizations)
        mock_db_service.count_organization_agents = AsyncMock(return_value=100)  # High count
        mock_db_service.count_organization_users = AsyncMock(return_value=50)    # High count

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