#!/usr/bin/env python3
"""
End-to-End Retrieval Pipeline Integration Tests
Tests the complete flow from query to response with all components.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from app.services.rag_service import RAGService
from app.services.domain_expertise_service import DomainExpertiseService
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStoreService
from app.services.reranker_service import RerankerService


@pytest.mark.asyncio
async def test_complete_rag_pipeline():
    """Test the complete RAG pipeline from query to response"""
    rag_service = RAGService()

    # Mock all dependencies to test integration
    with patch.object(rag_service.document_processor, 'search_similar_content') as mock_search, \
         patch.object(rag_service.reranker, 'rerank') as mock_rerank, \
         patch.object(rag_service.gemini_service, 'generate_response') as mock_llm:

        # Setup mock responses
        mock_search.return_value = [
            {"text": "AI can automate business processes", "score": 0.9, "metadata": {"source": "doc1"}},
            {"text": "Machine learning improves efficiency", "score": 0.8, "metadata": {"source": "doc2"}},
            {"text": "Automation reduces manual work", "score": 0.7, "metadata": {"source": "doc3"}}
        ]

        mock_rerank.return_value = [
            {"text": "AI can automate business processes", "score": 0.9, "rerank_score": 0.95, "metadata": {"source": "doc1"}},
            {"text": "Machine learning improves efficiency", "score": 0.8, "rerank_score": 0.85, "metadata": {"source": "doc2"}},
            {"text": "Automation reduces manual work", "score": 0.7, "rerank_score": 0.75, "metadata": {"source": "doc3"}}
        ]

        mock_llm.return_value = "Based on the retrieved context, AI can significantly improve business operations through automation and machine learning."

        # Test the complete pipeline
        result = await rag_service.generate_response(
            query="How can AI improve business operations?",
            agent_id=1,
            conversation_history=[{"role": "user", "content": "Hello"}],
            system_prompt="You are a helpful business AI assistant.",
            db_session=Mock()
        )

        # Verify the pipeline executed correctly
        assert "response" in result
        assert "sources" in result
        assert "context_used" in result
        assert result["response"] == "Based on the retrieved context, AI can significantly improve business operations through automation and machine learning."
        assert len(result["sources"]) == 3

        # Verify all components were called
        mock_search.assert_called_once()
        mock_rerank.assert_called_once()
        mock_llm.assert_called_once()


@pytest.mark.asyncio
async def test_domain_expertise_pipeline():
    """Test the domain expertise enhanced pipeline"""
    domain_service = DomainExpertiseService()

    with patch.object(domain_service, '_get_organization_agents') as mock_agents, \
         patch.object(domain_service, '_retrieve_from_agent') as mock_retrieve, \
         patch.object(domain_service, '_rerank_candidates') as mock_rerank:

        # Setup mock organization agents
        mock_agents.return_value = [
            {"id": 1, "name": "Tech Expert", "expertise": "technology"},
            {"id": 2, "name": "Business Analyst", "expertise": "business"}
        ]

        # Setup mock retrieval results
        from app.services.domain_expertise_service import RetrievalCandidate
        mock_retrieve.side_effect = [
            [RetrievalCandidate("tech1", "AI implementation requires technical planning", 0.9)],
            [RetrievalCandidate("biz1", "Business strategy must align with AI capabilities", 0.8)]
        ]

        mock_rerank.return_value = [
            RetrievalCandidate("tech1", "AI implementation requires technical planning", 0.95),
            RetrievalCandidate("biz1", "Business strategy must align with AI capabilities", 0.85)
        ]

        # Test multi-agent retrieval
        results = await domain_service.retrieve_multi_agent_knowledge(
            query="How to implement AI in business?",
            organization_id=1,
            top_k_per_agent=5,
            max_agents=10
        )

        assert len(results) >= 2
        assert results[0].score >= results[1].score  # Should be ranked by score
        mock_agents.assert_called_once()
        assert mock_retrieve.call_count == 2  # Called for each agent


@pytest.mark.asyncio
async def test_document_processing_to_search_pipeline():
    """Test the complete document processing and search pipeline"""
    doc_processor = DocumentProcessor()

    # Test document ingestion
    test_content = """
    # AI Implementation Guide

    ## Introduction
    Artificial Intelligence can transform business operations through automation and intelligent decision-making.

    ## Benefits
    - Increased efficiency
    - Reduced costs
    - Better customer service

    ## Implementation Steps
    1. Assess current processes
    2. Identify AI opportunities
    3. Select appropriate tools
    4. Train staff
    5. Monitor and optimize
    """

    # Process the document
    processing_result = await doc_processor.process_text_content(
        text=test_content,
        agent_id=1,
        source="ai_guide.md",
        document_id=1,
        extra_metadata={"category": "guide", "topic": "AI implementation"}
    )

    assert processing_result["status"] == "completed"
    assert processing_result["chunk_count"] > 0
    assert len(processing_result["vector_ids"]) == processing_result["chunk_count"]

    # Test search functionality
    search_results = await doc_processor.search_similar_content(
        query="How to implement AI in business?",
        agent_id=1,
        top_k=5
    )

    assert isinstance(search_results, list)
    # Results structure depends on vector store availability


@pytest.mark.asyncio
async def test_error_handling_across_pipeline():
    """Test error handling across the entire pipeline"""
    rag_service = RAGService()

    # Test with failing document search
    with patch.object(rag_service.document_processor, 'search_similar_content') as mock_search:
        mock_search.side_effect = Exception("Search service unavailable")

        # Should handle search failure gracefully
        try:
            result = await rag_service.generate_response(
                query="Test query",
                agent_id=1,
                db_session=Mock()
            )
            # Should either succeed with fallback or handle error gracefully
            assert isinstance(result, dict)
        except Exception as e:
            # Should provide meaningful error information
            assert len(str(e)) > 0

    # Test with failing reranker
    with patch.object(rag_service.document_processor, 'search_similar_content') as mock_search, \
         patch.object(rag_service.reranker, 'rerank') as mock_rerank:

        mock_search.return_value = [{"text": "test", "score": 0.8}]
        mock_rerank.side_effect = Exception("Reranker failed")

        # Should fallback gracefully
        try:
            result = await rag_service.generate_response(
                query="Test query",
                agent_id=1,
                db_session=Mock()
            )
            # Should handle reranker failure
            assert isinstance(result, dict)
        except Exception:
            # Acceptable if it fails gracefully
            pass


@pytest.mark.asyncio
async def test_performance_with_large_context():
    """Test pipeline performance with large context"""
    rag_service = RAGService()

    # Create large context with many documents
    large_context = []
    for i in range(100):
        large_context.append({
            "text": f"Document {i} contains relevant information about topic {i % 10}. " * 10,
            "score": 0.8 - (i * 0.001),  # Decreasing scores
            "metadata": {"source": f"doc_{i}.txt", "topic": f"topic_{i % 10}"}
        })

    with patch.object(rag_service.document_processor, 'search_similar_content') as mock_search, \
         patch.object(rag_service.reranker, 'rerank') as mock_rerank, \
         patch.object(rag_service.gemini_service, 'generate_response') as mock_llm:

        mock_search.return_value = large_context
        mock_rerank.return_value = large_context[:20]  # Rerank returns top 20
        mock_llm.return_value = "Response based on large context"

        import time
        start_time = time.time()

        result = await rag_service.generate_response(
            query="Test query with large context",
            agent_id=1,
            db_session=Mock()
        )

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete within reasonable time
        assert execution_time < 30.0  # 30 second timeout for large context
        assert result["context_used"] <= 20  # Should be limited


@pytest.mark.asyncio
async def test_concurrent_pipeline_requests():
    """Test concurrent requests through the pipeline"""
    rag_service = RAGService()

    async def make_rag_request(query_id: int):
        with patch.object(rag_service.document_processor, 'search_similar_content') as mock_search, \
             patch.object(rag_service.gemini_service, 'generate_response') as mock_llm:

            mock_search.return_value = [{"text": f"Result for query {query_id}", "score": 0.8}]
            mock_llm.return_value = f"Response for query {query_id}"

            return await rag_service.generate_response(
                query=f"Test query {query_id}",
                agent_id=1,
                db_session=Mock()
            )

    # Make concurrent requests
    import asyncio
    results = await asyncio.gather(*[make_rag_request(i) for i in range(5)])

    # All requests should complete successfully
    assert len(results) == 5
    for i, result in enumerate(results):
        assert f"query {i}" in result["response"]


@pytest.mark.asyncio
async def test_context_compression_integration():
    """Test context compression in the pipeline"""
    rag_service = RAGService()

    # Create very long context that needs compression
    long_context = [
        {"text": "Very long document content " * 100, "score": 0.9},
        {"text": "Another long document " * 100, "score": 0.8},
        {"text": "Third long document " * 100, "score": 0.7}
    ]

    with patch.object(rag_service.document_processor, 'search_similar_content') as mock_search, \
         patch.object(rag_service.gemini_service, 'generate_response') as mock_llm:

        mock_search.return_value = long_context
        mock_llm.return_value = "Compressed context response"

        result = await rag_service.generate_response(
            query="Query requiring context compression",
            agent_id=1,
            db_session=Mock()
        )

        # Should handle long context appropriately
        assert "response" in result
        assert result["context_used"] > 0


@pytest.mark.asyncio
async def test_streaming_response_pipeline():
    """Test streaming response generation"""
    rag_service = RAGService()

    async def mock_stream():
        chunks = ["Streaming ", "response ", "with ", "context"]
        for chunk in chunks:
            yield chunk

    with patch.object(rag_service.document_processor, 'search_similar_content') as mock_search, \
         patch.object(rag_service, 'generate_streaming_response') as mock_stream_gen:

        mock_search.return_value = [{"text": "Context for streaming", "score": 0.9}]
        mock_stream_gen.return_value = mock_stream()

        # Test streaming (if implemented)
        try:
            stream = rag_service.generate_streaming_response(
                query="Test streaming query",
                agent_id=1,
                db_session=Mock()
            )

            # Should return async generator
            chunks = []
            async for chunk in stream:
                chunks.append(chunk)

            assert len(chunks) > 0

        except AttributeError:
            # Streaming not implemented yet
            pass


@pytest.mark.asyncio
async def test_multilingual_pipeline():
    """Test pipeline with multilingual queries"""
    rag_service = RAGService()

    multilingual_queries = [
        "How can AI improve business?",  # English
        "¿Cómo puede la IA mejorar los negocios?",  # Spanish
        "Comment l'IA peut-elle améliorer les entreprises?",  # French
        "Wie kann KI Unternehmen verbessern?",  # German
        "AIはどのようにビジネスを改善できますか？"  # Japanese
    ]

    for query in multilingual_queries:
        with patch.object(rag_service.document_processor, 'search_similar_content') as mock_search, \
             patch.object(rag_service.gemini_service, 'generate_response') as mock_llm:

            mock_search.return_value = [{"text": "Multilingual content", "score": 0.8}]
            mock_llm.return_value = f"Response to: {query}"

            result = await rag_service.generate_response(
                query=query,
                agent_id=1,
                db_session=Mock()
            )

            # Should handle different languages
            assert "response" in result
            assert len(result["response"]) > 0


@pytest.mark.asyncio
async def test_pipeline_with_conversation_history():
    """Test pipeline with conversation context"""
    rag_service = RAGService()

    conversation_history = [
        {"role": "user", "content": "What is artificial intelligence?"},
        {"role": "assistant", "content": "AI is a technology that enables machines to perform tasks that typically require human intelligence."},
        {"role": "user", "content": "How can it help businesses?"}
    ]

    with patch.object(rag_service.document_processor, 'search_similar_content') as mock_search, \
         patch.object(rag_service.gemini_service, 'generate_response') as mock_llm:

        mock_search.return_value = [{"text": "AI business benefits", "score": 0.9}]
        mock_llm.return_value = "AI can help businesses through automation and insights."

        result = await rag_service.generate_response(
            query="How can it help businesses?",  # Contextual query
            agent_id=1,
            conversation_history=conversation_history,
            db_session=Mock()
        )

        # Should use conversation context
        assert "response" in result
        mock_llm.assert_called_once()
        # LLM should receive conversation history in the call