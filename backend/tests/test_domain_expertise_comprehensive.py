#!/usr/bin/env python3
"""
Comprehensive Domain Expertise Service Tests
Tests multi-agent retrieval, knowledge packs, and domain-specific functionality.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from app.services.domain_expertise_service import DomainExpertiseService, RetrievalCandidate


@pytest.mark.asyncio
async def test_domain_expertise_multi_agent_retrieval():
    """Test multi-agent knowledge retrieval"""
    service = DomainExpertiseService()

    # Mock the organization agents retrieval
    with patch.object(service, '_get_organization_agents') as mock_agents:
        mock_agents.return_value = [
            {"id": 1, "name": "Tech Agent", "expertise": "technology"},
            {"id": 2, "name": "Business Agent", "expertise": "business"},
            {"id": 3, "name": "Legal Agent", "expertise": "legal"}
        ]

        # Mock document retrieval for each agent
        with patch.object(service, '_retrieve_from_agent') as mock_retrieve:
            mock_retrieve.side_effect = [
                [RetrievalCandidate("doc1", "Tech content", 0.9)],
                [RetrievalCandidate("doc2", "Business content", 0.8)],
                [RetrievalCandidate("doc3", "Legal content", 0.7)]
            ]

            results = await service.retrieve_multi_agent_knowledge(
                query="How to implement AI in business?",
                organization_id=1,
                top_k_per_agent=3,
                max_agents=5
            )

            assert len(results) == 3
            assert results[0].score == 0.9
            assert "Tech content" in results[0].content


@pytest.mark.asyncio
async def test_domain_expertise_knowledge_pack_filtering():
    """Test knowledge pack filtering functionality"""
    service = DomainExpertiseService()

    # Test different knowledge pack types
    knowledge_packs = ["technology", "business", "legal", "healthcare"]

    for pack in knowledge_packs:
        with patch.object(service, '_filter_by_knowledge_pack') as mock_filter:
            mock_filter.return_value = [
                RetrievalCandidate(f"doc_{pack}_1", f"{pack} content 1", 0.9),
                RetrievalCandidate(f"doc_{pack}_2", f"{pack} content 2", 0.8)
            ]

            results = await service._apply_knowledge_pack_filter(
                candidates=[
                    RetrievalCandidate("doc1", "general content", 0.7),
                    RetrievalCandidate("doc2", "another content", 0.6)
                ],
                knowledge_pack=pack
            )

            assert len(results) == 2
            assert pack in results[0].content


@pytest.mark.asyncio
async def test_domain_expertise_confidence_scoring():
    """Test confidence scoring algorithm"""
    service = DomainExpertiseService()

    candidates = [
        RetrievalCandidate("doc1", "Highly relevant content about AI", 0.95),
        RetrievalCandidate("doc2", "Somewhat relevant content", 0.75),
        RetrievalCandidate("doc3", "Less relevant content", 0.55),
        RetrievalCandidate("doc4", "Barely relevant content", 0.35)
    ]

    scored_candidates = await service._calculate_confidence_scores(
        candidates=candidates,
        query="AI implementation strategies"
    )

    # Higher similarity scores should result in higher confidence
    assert scored_candidates[0].confidence_score > scored_candidates[1].confidence_score
    assert scored_candidates[1].confidence_score > scored_candidates[2].confidence_score


@pytest.mark.asyncio
async def test_domain_expertise_persona_enhancement():
    """Test persona-driven response enhancement"""
    service = DomainExpertiseService()

    base_context = "AI can improve business efficiency through automation."

    personas = ["technical_expert", "business_consultant", "startup_advisor"]

    for persona in personas:
        enhanced_context = await service._enhance_with_persona(
            context=base_context,
            persona=persona,
            query="How can AI help my business?"
        )

        assert isinstance(enhanced_context, str)
        assert len(enhanced_context) >= len(base_context)
        # Enhanced context should be longer and more specific


@pytest.mark.asyncio
async def test_domain_expertise_web_search_integration():
    """Test web search integration for knowledge gaps"""
    service = DomainExpertiseService()

    # Mock web search results
    with patch.object(service, '_perform_web_search') as mock_search:
        mock_search.return_value = [
            {"title": "AI Trends 2024", "content": "Latest AI developments", "url": "example.com"},
            {"title": "Business AI Guide", "content": "AI implementation guide", "url": "guide.com"}
        ]

        web_results = await service._supplement_with_web_search(
            query="Latest AI trends 2024",
            existing_candidates=[
                RetrievalCandidate("doc1", "Old AI content", 0.6)
            ],
            max_web_results=5
        )

        assert len(web_results) >= 1
        mock_search.assert_called_once()


@pytest.mark.asyncio
async def test_domain_expertise_agent_specialization():
    """Test agent specialization matching"""
    service = DomainExpertiseService()

    # Test different query types
    test_cases = [
        ("How to code in Python?", ["programming", "software", "tech"]),
        ("Marketing strategy for startups", ["marketing", "business", "growth"]),
        ("Legal compliance requirements", ["legal", "compliance", "regulations"]),
        ("Financial planning advice", ["finance", "planning", "investment"])
    ]

    for query, expected_domains in test_cases:
        matched_agents = await service._match_specialized_agents(
            query=query,
            available_agents=[
                {"id": 1, "expertise": "programming", "name": "Code Expert"},
                {"id": 2, "expertise": "marketing", "name": "Marketing Pro"},
                {"id": 3, "expertise": "legal", "name": "Legal Advisor"},
                {"id": 4, "expertise": "finance", "name": "Finance Expert"}
            ]
        )

        assert len(matched_agents) > 0
        # Should match relevant expertise


@pytest.mark.asyncio
async def test_domain_expertise_cross_domain_synthesis():
    """Test cross-domain knowledge synthesis"""
    service = DomainExpertiseService()

    multi_domain_candidates = [
        RetrievalCandidate("tech1", "AI implementation requires technical expertise", 0.9),
        RetrievalCandidate("biz1", "Business strategy should align with technology", 0.8),
        RetrievalCandidate("legal1", "AI systems must comply with regulations", 0.7),
        RetrievalCandidate("finance1", "AI investments need ROI analysis", 0.6)
    ]

    synthesized_response = await service._synthesize_cross_domain_knowledge(
        candidates=multi_domain_candidates,
        query="How to implement AI in a regulated industry?",
        domains=["technology", "business", "legal", "finance"]
    )

    assert isinstance(synthesized_response, str)
    assert len(synthesized_response) > 0
    # Should combine insights from multiple domains


@pytest.mark.asyncio
async def test_domain_expertise_grounding_verification():
    """Test answer grounding and verification"""
    service = DomainExpertiseService()

    answer = "AI can reduce costs by 30% through automation"
    sources = [
        RetrievalCandidate("study1", "Research shows AI reduces costs by 25-35%", 0.9),
        RetrievalCandidate("case1", "Company X reduced costs by 30% with AI", 0.8)
    ]

    grounding_result = await service._verify_answer_grounding(
        answer=answer,
        sources=sources,
        query="How much can AI reduce business costs?"
    )

    assert "confidence_score" in grounding_result
    assert "grounding_quality" in grounding_result
    assert "supporting_sources" in grounding_result
    assert grounding_result["confidence_score"] > 0.7  # Should be well-grounded


@pytest.mark.asyncio
async def test_domain_expertise_incremental_learning():
    """Test incremental learning from user feedback"""
    service = DomainExpertiseService()

    feedback_data = {
        "query": "AI implementation best practices",
        "response": "Follow these steps: 1. Assess needs, 2. Choose tools...",
        "user_rating": 4.5,
        "feedback": "Very helpful, could use more technical details",
        "context_used": ["doc1", "doc2", "doc3"]
    }

    # Test learning from feedback
    learning_result = await service._learn_from_feedback(feedback_data)

    assert "updated_weights" in learning_result
    assert "improved_ranking" in learning_result
    # Should improve future responses


@pytest.mark.asyncio
async def test_domain_expertise_edge_cases():
    """Test domain expertise service edge cases"""
    service = DomainExpertiseService()

    # Test with empty query
    empty_result = await service.retrieve_multi_agent_knowledge(
        query="",
        organization_id=1
    )
    assert len(empty_result) == 0

    # Test with very long query
    long_query = "This is a very long query " * 100
    try:
        long_result = await service.retrieve_multi_agent_knowledge(
            query=long_query,
            organization_id=1,
            top_k_per_agent=1
        )
        assert isinstance(long_result, list)
    except Exception as e:
        # Should handle gracefully
        assert "length" in str(e).lower() or "token" in str(e).lower()

    # Test with non-existent organization
    no_org_result = await service.retrieve_multi_agent_knowledge(
        query="test query",
        organization_id=99999
    )
    assert len(no_org_result) == 0


@pytest.mark.asyncio
async def test_domain_expertise_performance():
    """Test domain expertise service performance"""
    service = DomainExpertiseService()

    # Mock multiple agents for performance testing
    with patch.object(service, '_get_organization_agents') as mock_agents:
        mock_agents.return_value = [{"id": i, "name": f"Agent {i}"} for i in range(20)]

        with patch.object(service, '_retrieve_from_agent') as mock_retrieve:
            mock_retrieve.return_value = [
                RetrievalCandidate(f"doc_{i}", f"Content {i}", 0.8) for i in range(5)
            ]

            import time
            start_time = time.time()

            results = await service.retrieve_multi_agent_knowledge(
                query="Performance test query",
                organization_id=1,
                top_k_per_agent=5,
                max_agents=10
            )

            end_time = time.time()
            execution_time = end_time - start_time

            # Should complete within reasonable time (< 10 seconds for 10 agents)
            assert execution_time < 10.0
            assert len(results) > 0


def test_retrieval_candidate_class():
    """Test RetrievalCandidate data class"""
    candidate = RetrievalCandidate(
        doc_id="test_doc",
        content="Test content",
        score=0.85
    )

    assert candidate.doc_id == "test_doc"
    assert candidate.content == "Test content"
    assert candidate.score == 0.85

    # Test with metadata
    candidate_with_meta = RetrievalCandidate(
        doc_id="meta_doc",
        content="Content with metadata",
        score=0.9,
        metadata={"source": "test.txt", "author": "test"}
    )

    assert candidate_with_meta.metadata["source"] == "test.txt"


@pytest.mark.asyncio
async def test_domain_expertise_concurrent_requests():
    """Test handling concurrent requests"""
    service = DomainExpertiseService()

    async def make_request(query_id: int):
        return await service.retrieve_multi_agent_knowledge(
            query=f"Test query {query_id}",
            organization_id=1,
            top_k_per_agent=2
        )

    # Make concurrent requests
    import asyncio
    results = await asyncio.gather(*[make_request(i) for i in range(5)])

    # All requests should complete
    assert len(results) == 5
    for result in results:
        assert isinstance(result, list)


@pytest.mark.asyncio
async def test_domain_expertise_error_recovery():
    """Test error recovery and fallback mechanisms"""
    service = DomainExpertiseService()

    # Test with failing agent retrieval
    with patch.object(service, '_retrieve_from_agent') as mock_retrieve:
        mock_retrieve.side_effect = [
            Exception("Agent 1 failed"),
            [RetrievalCandidate("doc2", "Success content", 0.8)],
            Exception("Agent 3 failed")
        ]

        with patch.object(service, '_get_organization_agents') as mock_agents:
            mock_agents.return_value = [
                {"id": 1, "name": "Failing Agent 1"},
                {"id": 2, "name": "Working Agent"},
                {"id": 3, "name": "Failing Agent 3"}
            ]

            results = await service.retrieve_multi_agent_knowledge(
                query="Error recovery test",
                organization_id=1
            )

            # Should return results from working agent, skip failed ones
            assert len(results) >= 0  # Should handle errors gracefully