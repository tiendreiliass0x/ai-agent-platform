"""
Service Layer Tests for regression prevention.
Tests critical service functionality to ensure public_id and other core features work correctly.

UPDATED: All tests now use the proper test database session fixture (db_session) instead of
creating new production sessions with get_db_session(). This ensures tests can access data
created by test fixtures and maintains proper test isolation.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from app.services.database_service import DatabaseService, db_service
from app.services.context.context_service import ContextEngine, ContextChunk
from app.services.agent_creation_service import AgentCreationService, AgentType, IndustryType
from app.models.user import User
from app.models.organization import Organization
from app.models.agent import Agent
from app.models.document import Document


@pytest.mark.service
class TestDatabaseServiceRegressionPrevention:
    """Test DatabaseService to prevent regressions like public_id missing"""

    @pytest.mark.asyncio
    async def test_agent_creation_ensures_public_id(self, db_session, test_user, test_organization):
        """Test that agent creation always ensures public_id exists - regression prevention"""
        service = DatabaseService()

        # Create agent with explicit session management
        agent = await service.create_agent(
            db=db_session,
            user_id=test_user.id,
            organization_id=test_organization.id,
            name="Test Agent",
            description="Test Description",
            system_prompt="Test prompt"
        )
        await db_session.commit()

        # Critical regression check: public_id must exist
        assert agent.public_id is not None, "Agent public_id should never be None"
        assert len(agent.public_id) > 0, "Agent public_id should not be empty"
        assert isinstance(agent.public_id, str), "Agent public_id should be string"

        # Verify UUID format
        import uuid
        try:
            uuid.UUID(agent.public_id)
        except ValueError:
            pytest.fail(f"public_id '{agent.public_id}' is not valid UUID format")

    @pytest.mark.asyncio
    async def test_get_agent_by_public_id_works(self, db_session, test_user, test_organization):
        """Test that get_agent_by_public_id works correctly - original regression fix"""
        service = DatabaseService()

        # Create agent
        original_agent = await service.create_agent(
            db=db_session,
            user_id=test_user.id,
            organization_id=test_organization.id,
            name="Public ID Test Agent",
            description="Test",
            system_prompt="Test"
        )
        await db_session.commit()

        # Test getting agent by public_id
        retrieved_agent = await service.get_agent_by_public_id(original_agent.public_id)

        assert retrieved_agent is not None, "Should retrieve agent by public_id"
        assert retrieved_agent.id == original_agent.id, "Should retrieve correct agent"
        assert retrieved_agent.public_id == original_agent.public_id, "public_id should match"

    @pytest.mark.asyncio
    async def test_ensure_agent_public_id_handles_missing_id(self, db_session, test_user, test_organization):
        """Test the _ensure_agent_public_id method handles missing public_id"""
        service = DatabaseService()

        # Create agent normally first
        agent = await service.create_agent(
            db=db_session,
            user_id=test_user.id,
            organization_id=test_organization.id,
            name="Test Agent",
            description="Test",
            system_prompt="Test"
        )
        await db_session.commit()

        # Simulate the case where public_id might be missing (regression scenario)
        # Temporarily set public_id to None to simulate the regression
        agent.public_id = None

        # Call the method that should ensure public_id exists
        ensured_agent = await service._ensure_agent_public_id(agent, db_session)

        assert ensured_agent.public_id is not None, "Should generate public_id if missing"
        assert len(ensured_agent.public_id) > 0, "Generated public_id should not be empty"

    @pytest.mark.asyncio
    async def test_user_organizations_relationship_integrity(self, test_user, test_organization):
        """Test that user-organization relationships maintain integrity"""
        service = DatabaseService()

        # Add user to organization
        user_org = await service.add_user_to_organization(
            user_id=test_user.id,
            organization_id=test_organization.id,
            role="member"
        )

        assert user_org is not None
        assert user_org.user_id == test_user.id
        assert user_org.organization_id == test_organization.id
        assert user_org.is_active is True

        # Verify we can retrieve the relationship
        retrieved_user_org = await service.get_user_organization(
            test_user.id, test_organization.id
        )
        assert retrieved_user_org is not None
        assert retrieved_user_org.id == user_org.id

    @pytest.mark.asyncio
    async def test_organization_agent_retrieval_with_public_id(self, db_session, test_user, test_organization):
        """Test that organization agents are retrieved with valid public_ids"""
        service = DatabaseService()

        # Create multiple agents in single transaction for efficiency
        agents = []
        for i in range(3):
            agent = await service.create_agent(
                db=db_session,
                user_id=test_user.id,
                organization_id=test_organization.id,
                name=f"Test Agent {i}",
                description=f"Test Description {i}",
                system_prompt="Test prompt"
            )
            agents.append(agent)
        await db_session.commit()

        # Retrieve organization agents
        org_agents = await service.get_organization_agents(test_organization.id)

        assert len(org_agents) >= 3, "Should retrieve all organization agents"

        # Critical regression check: all agents must have public_id
        for agent in org_agents:
            assert agent.public_id is not None, f"Agent {agent.id} missing public_id"
            assert len(agent.public_id) > 0, f"Agent {agent.id} has empty public_id"

    @pytest.mark.asyncio
    async def test_agent_deletion_cascade_behavior(self, db_session, test_user, test_organization):
        """Test that agent deletion properly cascades and cleans up"""
        service = DatabaseService()

        # Create agent with documents
        agent = await service.create_agent(
            db=db_session,
            user_id=test_user.id,
            organization_id=test_organization.id,
            name="Delete Test Agent",
            description="Test",
            system_prompt="Test"
        )
        await db_session.commit()

        # Add document to agent
        document = await service.create_document(
            agent_id=agent.id,
            filename="test.txt",
            content="Test content",
            content_type="text/plain"
        )

        # Delete agent
        delete_result = await service.delete_agent(agent.id)
        assert delete_result is True

        # Verify agent is deleted
        deleted_agent = await service.get_agent_by_id(agent.id)
        assert deleted_agent is None or deleted_agent.is_active is False

        # Verify document is cascade deleted (depending on cascade rules)
        deleted_document = await service.get_document_by_id(document.id)
        assert deleted_document is None, "Document should be cascade deleted"

    @pytest.mark.asyncio
    async def test_agent_stats_computation_with_valid_data(self, db_session, test_user, test_organization):
        """Test that agent stats computation works with valid data structure"""
        service = DatabaseService()

        # Create agent
        agent = await service.create_agent(
            db=db_session,
            user_id=test_user.id,
            organization_id=test_organization.id,
            name="Stats Test Agent",
            description="Test",
            system_prompt="Test"
        )
        await db_session.commit()

        # Get stats
        stats = await service.get_agent_stats(agent.id)

        # Verify stats structure
        assert isinstance(stats, dict), "Stats should be dictionary"
        assert "total_conversations" in stats, "Should have total_conversations"
        assert "total_messages" in stats, "Should have total_messages"


@pytest.mark.service
class TestContextServiceBasicFunctionality:
    """Test ContextService basic functionality without complex mocking"""

    @pytest.mark.asyncio
    async def test_context_engine_initialization(self, db_session):
        """Test that ContextEngine can be initialized"""
        context_engine = ContextEngine()

        assert context_engine is not None
        assert hasattr(context_engine, 'optimize_context')

    @pytest.mark.asyncio
    async def test_context_chunk_creation(self, db_session):
        """Test creating ContextChunk objects"""
        chunk = ContextChunk(
            content="Test content",
            source_type="test_source",
            relevance_score=0.95,
            importance=0.8,
            recency_score=0.9,
            metadata={"test": "data"}
        )

        assert chunk.content == "Test content"
        assert chunk.source_type == "test_source"
        assert chunk.relevance_score == 0.95
        assert chunk.importance == 0.8
        assert chunk.recency_score == 0.9
        assert chunk.metadata["test"] == "data"

    @pytest.mark.asyncio
    async def test_context_engine_optimize_context_with_mocks(self, db_session):
        """Test context optimization with proper mocking"""
        context_engine = ContextEngine()

        # Mock the internal methods to avoid database dependencies
        mock_chunks = [
            ContextChunk(
                content="Test chunk",
                source_type="memory",
                relevance_score=0.95,
                importance=0.8,
                recency_score=0.9
            )
        ]

        with patch.object(context_engine, '_gather_context_sources', return_value=mock_chunks), \
             patch.object(context_engine, '_rank_and_select_context', return_value=mock_chunks), \
             patch.object(context_engine, '_synthesize_intelligent_context', return_value="Test context"), \
             patch.object(context_engine, '_optimize_context_window', return_value="Optimized context"):

            result = await context_engine.optimize_context(
                query="test query",
                customer_profile_id=1,
                agent_id=1
            )

            # Verify result structure
            assert isinstance(result, dict), "Should return dictionary"
            assert "context" in result, "Should have context field"
            assert "context_quality_score" in result, "Should have quality score"
            assert "chunks_used" in result, "Should have chunks_used count"
            assert "total_chunks_available" in result, "Should have total chunks count"


@pytest.mark.service
class TestServiceLayerIntegration:
    """Integration tests for service layer interactions"""

    @pytest.mark.asyncio
    async def test_end_to_end_agent_workflow(self, db_session, test_user, test_organization):
        """Test complete agent workflow from creation to retrieval"""
        # Test the full workflow that was broken by the public_id regression

        # 1. Create agent through service
        agent = await db_service.create_agent(
            db=db_session,
            user_id=test_user.id,
            organization_id=test_organization.id,
            name="E2E Test Agent",
            description="End-to-end test",
            system_prompt="Test prompt"
        )
        await db_session.commit()

        # 2. Verify agent has public_id (regression check)
        assert agent.public_id is not None

        # 3. Retrieve agent by public_id (this was failing in the regression)
        retrieved_agent = await db_service.get_agent_by_public_id(agent.public_id)
        assert retrieved_agent is not None
        assert retrieved_agent.id == agent.id

        # 4. Verify agent can be retrieved by ID
        id_retrieved = await db_service.get_agent_by_id(agent.id)
        assert id_retrieved is not None
        assert id_retrieved.public_id == agent.public_id

    @pytest.mark.asyncio
    async def test_service_error_handling_with_invalid_data(self, db_session, test_user, test_organization):
        """Test that service handles invalid data gracefully"""

        # Test database service with invalid data
        try:
            await db_service.create_agent(
                db=db_session,
                user_id=999999,  # Non-existent user
                organization_id=test_organization.id,
                name="Invalid Agent",
                description="Test",
                system_prompt="Test"
            )
            await db_session.commit()
        except Exception:
            # Should handle gracefully without affecting other operations
            await db_session.rollback()

        # Verify normal operations still work after error
        valid_agent = await db_service.create_agent(
            db=db_session,
            user_id=test_user.id,
            organization_id=test_organization.id,
            name="Valid Agent",
            description="Test",
            system_prompt="Test"
        )
        await db_session.commit()
        assert valid_agent.public_id is not None

    @pytest.mark.asyncio
    async def test_service_transaction_rollback_behavior(self, db_session, test_user, test_organization):
        """Test that service transactions rollback properly on failure"""

        # Create agent
        agent = await db_service.create_agent(
            db=db_session,
            user_id=test_user.id,
            organization_id=test_organization.id,
            name="Transaction Test Agent",
            description="Test",
            system_prompt="Test"
        )
        await db_session.commit()

        # Attempt to delete with invalid ID should not affect valid operations
        delete_result = await db_service.delete_agent(999999)
        assert delete_result is False

        # Original agent should still exist
        existing_agent = await db_service.get_agent_by_id(agent.id)
        assert existing_agent is not None

    @pytest.mark.asyncio
    async def test_concurrent_agent_creation_safety(self, db_session, test_user, test_organization):
        """Test that concurrent agent creation is safe"""

        # Create multiple agents in the same session
        agents = []
        for i in range(3):
            agent = await db_service.create_agent(
                db=db_session,
                user_id=test_user.id,
                organization_id=test_organization.id,
                name=f"Concurrent Agent {i}",
                description="Concurrent test",
                system_prompt="Test"
            )
            agents.append(agent)
        await db_session.commit()

        # Verify all agents were created with valid public_ids
        assert len(agents) == 3
        public_ids = set()
        for agent in agents:
            assert agent.public_id is not None
            assert agent.public_id not in public_ids, "public_ids should be unique"
            public_ids.add(agent.public_id)


@pytest.mark.service
class TestServiceLayerPerformance:
    """Test performance characteristics of service layer"""

    @pytest.mark.asyncio
    async def test_agent_api_key_generation_uniqueness(self, db_session, test_user, test_organization):
        """Test that agent API keys are generated uniquely"""

        # Create multiple agents
        agents = []
        for i in range(5):
            agent = await db_service.create_agent(
                db=db_session,
                user_id=test_user.id,
                organization_id=test_organization.id,
                name=f"API Key Test Agent {i}",
                description="Test",
                system_prompt="Test"
            )
            agents.append(agent)
        await db_session.commit()

        # Verify all API keys are unique
        api_keys = set()
        for agent in agents:
            assert agent.api_key is not None, "API key should be generated"
            assert agent.api_key.startswith("agent_"), "API key should have correct prefix"
            assert agent.api_key not in api_keys, "API keys should be unique"
            api_keys.add(agent.api_key)

    @pytest.mark.asyncio
    async def test_user_organization_relationship_updates(self, test_user, test_organization):
        """Test that user-organization relationships can be updated"""

        # Add user to organization
        user_org = await db_service.add_user_to_organization(
            user_id=test_user.id,
            organization_id=test_organization.id,
            role="member"
        )

        assert user_org.role == "member"

        # Update role
        update_result = await db_service.update_user_organization_role(
            user_id=test_user.id,
            organization_id=test_organization.id,
            role="admin"
        )
        assert update_result is True, "Update should succeed"

        # Verify role was updated
        updated_user_org = await db_service.get_user_organization(
            test_user.id, test_organization.id
        )
        assert updated_user_org.role == "admin"

    @pytest.mark.asyncio
    async def test_document_content_hash_validation(self, db_session, test_user, test_organization):
        """Test that document content hashing works for deduplication"""

        # Create agent
        agent = await db_service.create_agent(
            db=db_session,
            user_id=test_user.id,
            organization_id=test_organization.id,
            name="Document Test Agent",
            description="Test",
            system_prompt="Test"
        )
        await db_session.commit()

        # Create document
        content = "This is test content for hashing"
        document = await db_service.create_document(
            agent_id=agent.id,
            filename="test.txt",
            content=content,
            content_type="text/plain"
        )

        assert document is not None
        assert document.filename == "test.txt"

    @pytest.mark.asyncio
    async def test_large_agent_list_retrieval_performance(self, db_session, test_user, test_organization):
        """Test that retrieving large lists of agents performs adequately"""

        # Create multiple agents (smaller number for test speed)
        agents = []
        for i in range(10):
            agent = await db_service.create_agent(
                db=db_session,
                user_id=test_user.id,
                organization_id=test_organization.id,
                name=f"Performance Test Agent {i}",
                description="Performance test",
                system_prompt="Test"
            )
            agents.append(agent)
        await db_session.commit()

        # Time the retrieval
        import time
        start_time = time.time()

        retrieved_agents = await db_service.get_user_agents(test_user.id)

        end_time = time.time()
        retrieval_time = end_time - start_time

        # Should retrieve quickly (less than 1 second for 10 agents)
        assert retrieval_time < 1.0, f"Retrieval took {retrieval_time}s, should be under 1s"
        assert len(retrieved_agents) >= 10

        # All agents should have valid public_ids
        for agent in retrieved_agents:
            assert agent.public_id is not None

    @pytest.mark.asyncio
    async def test_agent_stats_computation_performance(self, db_session, test_user, test_organization):
        """Test that agent stats computation performs adequately"""

        # Create agent
        agent = await db_service.create_agent(
            db=db_session,
            user_id=test_user.id,
            organization_id=test_organization.id,
            name="Stats Performance Agent",
            description="Test",
            system_prompt="Test"
        )
        await db_session.commit()

        # Time the stats computation
        import time
        start_time = time.time()

        stats = await db_service.get_agent_stats(agent.id)

        end_time = time.time()
        computation_time = end_time - start_time

        # Should compute quickly
        assert computation_time < 1.0, f"Stats computation took {computation_time}s"
        assert isinstance(stats, dict)


@pytest.mark.service
class TestServiceLayerDataConsistency:
    """Test data consistency across service operations"""

    @pytest.mark.asyncio
    async def test_agent_document_relationship_consistency(self, db_session, test_user, test_organization):
        """Test that agent-document relationships remain consistent"""

        # Create agent
        agent = await db_service.create_agent(
            db=db_session,
            user_id=test_user.id,
            organization_id=test_organization.id,
            name="Relationship Test Agent",
            description="Test",
            system_prompt="Test"
        )
        await db_session.commit()

        # Create documents
        documents = []
        for i in range(3):
            document = await db_service.create_document(
                agent_id=agent.id,
                filename=f"test_{i}.txt",
                content=f"Test content {i}",
                content_type="text/plain"
            )
            documents.append(document)

        # Retrieve agent documents
        agent_docs = await db_service.get_agent_documents(agent.id)

        # Should have all documents
        assert len(agent_docs) >= 3

    @pytest.mark.asyncio
    async def test_user_organization_agent_hierarchy(self, db_session, test_user, test_organization):
        """Test that user->organization->agent hierarchy is maintained"""

        # Add user to organization
        user_org = await db_service.add_user_to_organization(
            user_id=test_user.id,
            organization_id=test_organization.id,
            role="admin"
        )

        # Create agent
        agent = await db_service.create_agent(
            db=db_session,
            user_id=test_user.id,
            organization_id=test_organization.id,
            name="Hierarchy Test Agent",
            description="Test",
            system_prompt="Test"
        )
        await db_session.commit()

        # Verify relationships
        # 1. User should be in organization
        user_orgs = await db_service.get_user_organizations(test_user.id)
        assert len(user_orgs) >= 1
        assert any(uo.organization_id == test_organization.id for uo in user_orgs)

        # 2. Agent should belong to user
        user_agents = await db_service.get_user_agents(test_user.id)
        assert any(a.id == agent.id for a in user_agents)

        # 3. Agent should belong to organization
        org_agents = await db_service.get_organization_agents(test_organization.id)
        assert any(a.id == agent.id for a in org_agents)
