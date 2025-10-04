"""
Comprehensive tests for production-ready improvements.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.responses import StandardResponse, success_response, error_response
from app.core.exceptions import (
    NotFoundException, ValidationException, ForbiddenException,
    ConflictException, BusinessLogicException
)
from app.core.validation import (
    AgentValidator, UserValidator, OrganizationValidator,
    ChatValidator, ConfigValidator, validate_input
)
from app.services.agent_service import AgentService
from app.models.agent import Agent, AgentTier
from app.models.user import User
from app.models.organization import Organization


class TestResponseModels:
    """Test standardized response models"""
    
    def test_success_response(self):
        """Test success response creation"""
        data = {"test": "data"}
        response = success_response(data, "Test message")
        
        assert response.status == "success"
        assert response.data == data
        assert response.message == "Test message"
        assert response.errors is None
    
    def test_error_response(self):
        """Test error response creation"""
        errors = ["Error 1", "Error 2"]
        response = error_response("Test error", errors, "TEST_ERROR")
        
        assert response.status == "error"
        assert response.message == "Test error"
        assert response.errors == errors
        assert response.meta["code"] == "TEST_ERROR"


class TestExceptionHandling:
    """Test custom exception classes"""
    
    def test_not_found_exception(self):
        """Test NotFoundException"""
        exc = NotFoundException("Resource not found", "agent", 123)
        
        assert exc.message == "Resource not found"
        assert exc.code == "NOT_FOUND"
        assert exc.details["resource_type"] == "agent"
        assert exc.details["resource_id"] == 123
    
    def test_validation_exception(self):
        """Test ValidationException"""
        field_errors = ["Field 1 error", "Field 2 error"]
        exc = ValidationException("Validation failed", field_errors)
        
        assert exc.message == "Validation failed"
        assert exc.code == "VALIDATION_ERROR"
        assert exc.field_errors == field_errors
        assert exc.details["field_errors"] == field_errors
    
    def test_forbidden_exception(self):
        """Test ForbiddenException"""
        exc = ForbiddenException("Access denied")
        
        assert exc.message == "Access denied"
        assert exc.code == "FORBIDDEN"


class TestValidationSystem:
    """Test comprehensive validation system"""
    
    def test_agent_validator_success(self):
        """Test successful agent validation"""
        data = {
            "name": "Test Agent",
            "description": "A test agent",
            "system_prompt": "You are a helpful assistant"
        }
        
        validated = AgentValidator(**data)
        assert validated.name == "Test Agent"
        assert validated.description == "A test agent"
        assert validated.system_prompt == "You are a helpful assistant"
    
    def test_agent_validator_invalid_name(self):
        """Test agent validation with invalid name"""
        data = {
            "name": "",  # Empty name should fail
            "description": "A test agent",
            "system_prompt": "You are a helpful assistant"
        }
        
        with pytest.raises(ValueError, match="Agent name"):
            AgentValidator(**data)
    
    def test_user_validator_success(self):
        """Test successful user validation"""
        data = {
            "email": "test@gmail.com",
            "password": "SecurePass123!",
            "name": "Test User"
        }
        
        validated = UserValidator(**data)
        assert validated.email == "test@gmail.com"
        assert validated.password == "SecurePass123!"
        assert validated.name == "Test User"
    
    def test_user_validator_weak_password(self):
        """Test user validation with weak password"""
        data = {
            "email": "test@gmail.com",
            "password": "weak",  # Too weak
            "name": "Test User"
        }
        
        with pytest.raises(ValueError, match="Password must be at least 8 characters"):
            UserValidator(**data)
    
    def test_organization_validator_success(self):
        """Test successful organization validation"""
        data = {
            "name": "Test Organization",
            "slug": "test-org",
            "description": "A test organization",
            "website": "https://example.com"
        }
        
        validated = OrganizationValidator(**data)
        assert validated.name == "Test Organization"
        assert validated.slug == "test-org"
        assert validated.website == "https://example.com"
    
    def test_chat_validator_success(self):
        """Test successful chat validation"""
        data = {
            "message": "Hello, how are you?",
            "visitor_id": "visitor_123",
            "session_context": {"page": "home"}
        }
        
        validated = ChatValidator(**data)
        assert validated.message == "Hello, how are you?"
        assert validated.visitor_id == "visitor_123"
    
    def test_chat_validator_harmful_content(self):
        """Test chat validation with harmful content"""
        data = {
            "message": "<script>alert('xss')</script>",
            "visitor_id": "visitor_123"
        }
        
        with pytest.raises(ValueError, match="potentially harmful content"):
            ChatValidator(**data)
    
    def test_config_validator_agent_config(self):
        """Test agent configuration validation"""
        config = {
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        validated = ConfigValidator.validate_agent_config(config)
        assert validated["model"] == "gpt-4"
        assert validated["temperature"] == 0.7
        assert validated["max_tokens"] == 1000
    
    def test_config_validator_invalid_model(self):
        """Test agent configuration with invalid model"""
        config = {
            "model": "invalid-model",
            "temperature": 0.7
        }
        
        with pytest.raises(ValueError, match="Model must be one of"):
            ConfigValidator.validate_agent_config(config)
    
    def test_validate_input_function(self):
        """Test validate_input utility function"""
        data = {"name": "Test", "description": "Test description"}
        
        validated = validate_input(data, AgentValidator)
        assert isinstance(validated, AgentValidator)
        assert validated.name == "Test"


class TestAgentService:
    """Test enhanced agent service"""
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        return AsyncMock(spec=AsyncSession)
    
    @pytest.fixture
    def agent_service(self, mock_db_session):
        """Create agent service with mocked session"""
        return AgentService(mock_db_session)
    
    @pytest.fixture
    def mock_agent(self):
        """Mock agent object"""
        agent = MagicMock()
        agent.id = 1
        agent.name = "Test Agent"
        agent.description = "Test description"
        agent.user_id = 1
        agent.organization_id = 1
        agent.is_active = True
        agent.api_key = "test_api_key_123"
        return agent
    
    @pytest.fixture
    def mock_user(self):
        """Mock user object"""
        user = MagicMock()
        user.id = 1
        user.email = "test@example.com"
        user.name = "Test User"
        user.is_active = True
        return user
    
    @pytest.fixture
    def mock_organization(self):
        """Mock organization object"""
        org = MagicMock()
        org.id = 1
        org.name = "Test Organization"
        org.max_agents = 10
        return org
    
    async def test_get_agent_by_id_success(self, agent_service, mock_db_session, mock_agent):
        """Test successful agent retrieval by ID"""
        # Mock database query result
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_agent
        mock_db_session.execute.return_value = mock_result
        
        result = await agent_service.get_agent_by_id(1)
        
        assert result == mock_agent
        mock_db_session.execute.assert_called_once()
    
    async def test_get_agent_by_id_not_found(self, agent_service, mock_db_session):
        """Test agent retrieval when not found"""
        # Mock database query result
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result
        
        result = await agent_service.get_agent_by_id(999)
        
        assert result is None
    
    async def test_create_agent_success(self, agent_service, mock_db_session, mock_organization):
        """Test successful agent creation"""
        # Mock organization lookup
        mock_org_result = MagicMock()
        mock_org_result.scalar_one_or_none.return_value = mock_organization
        mock_db_session.execute.return_value = mock_org_result
        
        agent_data = {
            "name": "New Agent",
            "description": "A new agent",
            "system_prompt": "You are helpful"
        }
        
        result = await agent_service.create_agent(
            user_id=1,
            organization_id=1,
            agent_data=agent_data
        )
        
        assert result is not None
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once()
    
    async def test_create_agent_validation_error(self, agent_service, mock_db_session):
        """Test agent creation with validation error"""
        agent_data = {
            "name": "",  # Invalid name
            "description": "A new agent"
        }
        
        with pytest.raises(ValidationException):
            await agent_service.create_agent(
                user_id=1,
                organization_id=1,
                agent_data=agent_data
            )
    
    async def test_verify_agent_ownership_success(self, agent_service, mock_agent):
        """Test successful ownership verification"""
        with patch.object(agent_service, 'get_agent_by_id', return_value=mock_agent):
            result = await agent_service.verify_agent_ownership(1, 1)
            assert result is True
    
    async def test_verify_agent_ownership_failure(self, agent_service, mock_agent):
        """Test failed ownership verification"""
        mock_agent.user_id = 2  # Different user
        
        with patch.object(agent_service, 'get_agent_by_id', return_value=mock_agent):
            result = await agent_service.verify_agent_ownership(1, 1)
            assert result is False
    
    async def test_delete_agent_success(self, agent_service, mock_db_session, mock_agent):
        """Test successful agent deletion"""
        with patch.object(agent_service, 'get_agent_by_id', return_value=mock_agent):
            result = await agent_service.delete_agent(1)
            
            assert result is True
            mock_db_session.delete.assert_called_once_with(mock_agent)
            mock_db_session.commit.assert_called_once()


class TestSecurityFeatures:
    """Test security improvements"""
    
    def test_api_key_masking(self):
        """Test API key masking functionality"""
        from app.api.v1.agents import serialize_agent
        
        # Mock agent with API key
        mock_agent = MagicMock()
        mock_agent.id = 1
        mock_agent.public_id = "test-public-id"
        mock_agent.name = "Test Agent"
        mock_agent.description = "Test description"
        mock_agent.system_prompt = "Test prompt"
        mock_agent.is_active = True
        mock_agent.config = {}
        mock_agent.widget_config = {}
        mock_agent.api_key = "agent_1234567890abcdef"
        mock_agent.created_at = None
        mock_agent.updated_at = None
        mock_agent.tier = AgentTier.basic
        mock_agent.domain_expertise_type = None
        mock_agent.domain_expertise_enabled = False
        mock_agent.personality_profile = {}
        mock_agent.expertise_level = 0.7
        mock_agent.domain_knowledge_sources = []
        mock_agent.web_search_enabled = False
        mock_agent.custom_training_data = {}
        mock_agent.expert_context = None
        mock_agent.tool_policy = {}
        mock_agent.grounding_mode = "blended"
        
        # Test masked API key (default behavior)
        result = serialize_agent(mock_agent)
        assert result.api_key == "agent_123...cdef"
        
        # Test unmasked API key (when explicitly requested)
        result_unmasked = serialize_agent(mock_agent, include_api_key=True)
        assert result_unmasked.api_key == "agent_1234567890abcdef"
    
    def test_password_strength_validation(self):
        """Test password strength validation"""
        # Strong password
        strong_password = "SecurePass123!"
        assert UserValidator.validate_password_strength(strong_password) == strong_password
        
        # Weak passwords
        weak_passwords = [
            "weak",  # Too short
            "weakpassword",  # No uppercase, numbers, or special chars
            "WEAKPASSWORD",  # No lowercase, numbers, or special chars
            "WeakPassword",  # No numbers or special chars
            "WeakPassword123",  # No special chars
        ]
        
        for weak_password in weak_passwords:
            with pytest.raises(ValueError):
                UserValidator.validate_password_strength(weak_password)


class TestDatabaseMigrations:
    """Test database migration functionality"""
    
    def test_idempotency_key_migration(self):
        """Test that idempotency_key field is properly added"""
        from app.models.agent import Agent
        
        # Check that the field exists in the model
        assert hasattr(Agent, 'idempotency_key')
        
        # Check field properties
        idempotency_field = Agent.__table__.columns['idempotency_key']
        assert idempotency_field.nullable is True
        assert idempotency_field.type.length == 255


@pytest.mark.asyncio
class TestIntegrationScenarios:
    """Test integration scenarios"""
    
    async def test_agent_creation_workflow(self):
        """Test complete agent creation workflow"""
        # This would test the full workflow from API endpoint to database
        # For now, we'll test the key components
        
        # 1. Validate input
        agent_data = {
            "name": "Integration Test Agent",
            "description": "Test agent for integration",
            "system_prompt": "You are a helpful test assistant"
        }
        validated_data = AgentValidator(**agent_data)
        
        # 2. Create response
        response = success_response(validated_data, "Agent created successfully")
        
        # 3. Verify response structure
        assert response.status == "success"
        assert response.message == "Agent created successfully"
        assert isinstance(response.data, AgentValidator)
    
    async def test_error_handling_workflow(self):
        """Test error handling workflow"""
        # 1. Simulate validation error
        try:
            invalid_data = {"name": "", "description": "Test"}
            AgentValidator(**invalid_data)
        except ValueError as e:
            # 2. Convert to custom exception
            validation_exc = ValidationException(f"Validation failed: {str(e)}")
            
            # 3. Create error response
            error_resp = error_response(
                validation_exc.message,
                [str(e)],
                validation_exc.code
            )
            
            # 4. Verify error response
            assert error_resp.status == "error"
            assert error_resp.code == "VALIDATION_ERROR"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
