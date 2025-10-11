"""
Tests for Query Router - RAG vs Agentic routing

Tests the intelligent routing system that dispatches queries to
RAG (information retrieval) or Agentic workflow (tool execution).
"""

import pytest
from app.services.intent_classifier import IntentClassifier, QueryIntent
from app.services.query_router import QueryRouter


class StubRAGService:
    async def generate_response(self, **kwargs):
        return {
            "response": "stub",
            "confidence_score": 0.8,
            "sources": [],
        }


class TestIntentClassifier:
    """Test intent classification"""

    def test_classify_information_query_heuristic(self):
        """Test RAG classification for information queries"""
        classifier = IntentClassifier()

        # Information queries
        info_queries = [
            "What are your pricing plans?",
            "How does your product work?",
            "Tell me about enterprise features",
            "Explain the refund policy",
            "Where can I find documentation?"
        ]

        for query in info_queries:
            result = classifier._classify_heuristic(query)
            assert result.intent == QueryIntent.RAG, f"Failed for: {query}"
            assert result.confidence > 0.5

    def test_classify_action_query_heuristic(self):
        """Test AGENTIC classification for action queries"""
        classifier = IntentClassifier()

        # Action queries
        action_queries = [
            "Create a Salesforce lead for John Doe",
            "Send him a welcome email",
            "Update the CRM ticket status",
            "Schedule a demo for tomorrow",
            "Create a support ticket for order 12345"
        ]

        for query in action_queries:
            result = classifier._classify_heuristic(query)
            assert result.intent == QueryIntent.AGENTIC, f"Failed for: {query}"
            assert result.confidence > 0.5
            assert len(result.detected_actions) > 0

    def test_detect_external_systems(self):
        """Test detection of external systems"""
        classifier = IntentClassifier()

        result = classifier._classify_heuristic("Create a lead in Salesforce")
        assert "salesforce" in result.detected_entities or "crm" in result.detected_actions

        result = classifier._classify_heuristic("Send email via SendGrid")
        assert "email" in result.detected_actions or "sendgrid" in result.detected_entities

    def test_detect_multi_step(self):
        """Test detection of multi-step workflows"""
        classifier = IntentClassifier()

        result = classifier._classify_heuristic(
            "Create a lead and then send them a welcome email"
        )
        assert result.intent == QueryIntent.AGENTIC
        assert result.confidence > 0.7  # Multi-step should increase confidence

    def test_is_agentic_helper(self):
        """Test quick is_agentic helper method"""
        classifier = IntentClassifier()

        assert classifier.is_agentic("Create a CRM ticket") is True
        assert classifier.is_agentic("What is your pricing?") is False


class TestQueryRouter:
    """Test query routing logic"""

    def test_router_initialization_without_orchestrator(self):
        """Test router can work without orchestrator (RAG only)"""
        router = QueryRouter(enable_agentic=False, rag_service=StubRAGService())

        assert router.orchestrator is None
        assert router.enable_agentic is False

    def test_router_initialization_with_orchestrator(self):
        """Test router with mock orchestrator"""

        class MockOrchestrator:
            pass

        mock_orch = MockOrchestrator()
        router = QueryRouter(orchestrator=mock_orch, enable_agentic=True, rag_service=StubRAGService())

        assert router.orchestrator is not None
        assert router.enable_agentic is True

    @pytest.mark.asyncio
    async def test_should_use_agentic_disabled(self):
        """Test agentic routing when disabled"""
        router = QueryRouter(enable_agentic=False, rag_service=StubRAGService())

        result = await router.should_use_agentic("Create a CRM lead")
        assert result is False  # Disabled regardless of intent

    @pytest.mark.asyncio
    async def test_should_use_agentic_no_orchestrator(self):
        """Test agentic routing without orchestrator"""
        router = QueryRouter(orchestrator=None, enable_agentic=True, rag_service=StubRAGService())

        result = await router.should_use_agentic("Create a CRM lead")
        assert result is False  # No orchestrator available

    def test_get_agent_permissions_default(self):
        """Test default permission extraction"""

        class MockAgent:
            id = 1
            config = {}

        router = QueryRouter(rag_service=StubRAGService(), enable_agentic=False)
        permissions = router._get_agent_permissions(MockAgent())

        assert permissions == set()

    def test_get_agent_permissions_from_config(self):
        """Test permission extraction from agent config"""

        class MockAgent:
            id = 1
            config = {
                "permissions": ["crm.create_lead", "email.send_transactional"]
            }

        router = QueryRouter(rag_service=StubRAGService(), enable_agentic=False)
        permissions = router._get_agent_permissions(MockAgent())

        assert "crm.create_lead" in permissions
        assert "email.send_transactional" in permissions


class TestIntegrationScenarios:
    """Integration test scenarios"""

    @pytest.mark.asyncio
    async def test_classify_with_llm_fallback(self):
        """Test LLM classification with fallback to heuristic"""

        # Classifier without LLM (should use heuristic)
        classifier = IntentClassifier(llm_service=None)

        result = await classifier.classify("Create a Salesforce lead")
        assert result.intent == QueryIntent.AGENTIC
        assert result.reasoning  # Should have reasoning

    def test_edge_case_ambiguous_query(self):
        """Test ambiguous queries"""
        classifier = IntentClassifier()

        # Could be either - checking it doesn't crash
        result = classifier._classify_heuristic("Check the status")
        assert result.intent in [QueryIntent.RAG, QueryIntent.AGENTIC]
        assert 0.0 <= result.confidence <= 1.0

    def test_edge_case_empty_query(self):
        """Test empty query handling"""
        classifier = IntentClassifier()

        result = classifier._classify_heuristic("")
        assert result.intent == QueryIntent.RAG  # Default to RAG
        assert result.confidence >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
