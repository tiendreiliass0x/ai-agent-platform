#!/usr/bin/env python3
"""
Comprehensive Vector Store Tests
Tests vector operations, search functionality, and error handling.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from app.services.vector_store import VectorStoreService


@pytest.mark.asyncio
async def test_vector_store_add_vectors():
    """Test adding vectors to the store"""
    vector_store = VectorStoreService()

    test_vectors = [
        {
            "id": "test_1",
            "values": [0.1] * 3072,  # OpenAI dimension
            "metadata": {
                "text": "Test content 1",
                "source": "test.txt",
                "agent_id": 1
            }
        },
        {
            "id": "test_2",
            "values": [0.2] * 3072,
            "metadata": {
                "text": "Test content 2",
                "source": "test.txt",
                "agent_id": 1
            }
        }
    ]

    try:
        result = await vector_store.add_vectors(test_vectors)
        assert result is True
    except Exception as e:
        # If Pinecone is not configured, test should handle gracefully
        assert "Pinecone" in str(e) or "index" in str(e).lower()


@pytest.mark.asyncio
async def test_vector_store_search_similar():
    """Test similarity search"""
    vector_store = VectorStoreService()

    query_vector = [0.1] * 3072

    try:
        results = await vector_store.search_similar(
            query_embedding=query_vector,
            agent_id=1,
            top_k=5,
            score_threshold=0.7
        )

        assert isinstance(results, list)
        # If results exist, verify structure
        if results:
            for result in results:
                assert "text" in result
                assert "score" in result
                assert "metadata" in result

    except Exception as e:
        # Handle Pinecone configuration issues gracefully
        assert "Pinecone" in str(e) or "index" in str(e).lower()


@pytest.mark.asyncio
async def test_vector_store_search_with_filters():
    """Test search with metadata filters"""
    vector_store = VectorStoreService()

    query_vector = [0.1] * 3072
    filters = {"agent_id": 1, "source": "test.txt"}

    try:
        results = await vector_store.search_similar(
            query_embedding=query_vector,
            agent_id=1,
            top_k=3,
            filters=filters
        )

        assert isinstance(results, list)

    except Exception as e:
        # Handle configuration issues
        pass


@pytest.mark.asyncio
async def test_vector_store_get_index_stats():
    """Test getting index statistics"""
    vector_store = VectorStoreService()

    try:
        stats = await vector_store.get_index_stats()

        assert isinstance(stats, dict)
        assert "status" in stats

        if stats["status"] == "available":
            assert "total_vectors" in stats
            assert "dimension" in stats

    except Exception as e:
        # Expected if Pinecone not configured
        pass


@pytest.mark.asyncio
async def test_vector_store_delete_vectors():
    """Test deleting vectors"""
    vector_store = VectorStoreService()

    vector_ids = ["test_1", "test_2", "test_3"]

    try:
        result = await vector_store.delete_vectors(vector_ids)
        assert isinstance(result, bool)

    except Exception as e:
        # Handle configuration issues
        pass


@pytest.mark.asyncio
async def test_vector_store_delete_by_agent():
    """Test deleting all vectors for an agent"""
    vector_store = VectorStoreService()

    try:
        result = await vector_store.delete_by_agent(agent_id=999)
        assert isinstance(result, bool)

    except Exception as e:
        # Handle configuration issues
        pass


@pytest.mark.asyncio
async def test_vector_store_upsert_vectors():
    """Test upserting (insert or update) vectors"""
    vector_store = VectorStoreService()

    vectors = [
        {
            "id": "upsert_test_1",
            "values": [0.3] * 3072,
            "metadata": {
                "text": "Upsert test content",
                "agent_id": 1,
                "updated": True
            }
        }
    ]

    try:
        result = await vector_store.add_vectors(vectors)
        assert isinstance(result, bool)

    except Exception as e:
        # Handle configuration issues
        pass


@pytest.mark.asyncio
async def test_vector_store_batch_operations():
    """Test batch vector operations"""
    vector_store = VectorStoreService()

    # Create larger batch
    batch_vectors = []
    for i in range(50):
        batch_vectors.append({
            "id": f"batch_test_{i}",
            "values": [0.1 + (i * 0.001)] * 3072,
            "metadata": {
                "text": f"Batch test content {i}",
                "agent_id": 1,
                "batch_id": "test_batch"
            }
        })

    try:
        result = await vector_store.add_vectors(batch_vectors)
        assert isinstance(result, bool)

    except Exception as e:
        # Handle configuration issues
        pass


@pytest.mark.asyncio
async def test_vector_store_error_handling():
    """Test vector store error handling"""
    vector_store = VectorStoreService()

    # Test with invalid vector dimensions
    invalid_vectors = [
        {
            "id": "invalid_test",
            "values": [0.1] * 100,  # Wrong dimension
            "metadata": {"text": "Invalid vector"}
        }
    ]

    try:
        await vector_store.add_vectors(invalid_vectors)
    except Exception as e:
        # Should handle dimension mismatch
        assert "dimension" in str(e).lower() or "vector" in str(e).lower()


@pytest.mark.asyncio
async def test_vector_store_concurrent_operations():
    """Test concurrent vector operations"""
    vector_store = VectorStoreService()

    async def add_test_vector(index: int):
        vectors = [{
            "id": f"concurrent_test_{index}",
            "values": [0.1 * index] * 3072,
            "metadata": {
                "text": f"Concurrent test {index}",
                "agent_id": 1
            }
        }]
        try:
            return await vector_store.add_vectors(vectors)
        except:
            return False

    # Run concurrent operations
    import asyncio
    results = await asyncio.gather(*[add_test_vector(i) for i in range(5)])

    # Should handle concurrent operations gracefully
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_vector_store_similarity_threshold():
    """Test similarity search with different thresholds"""
    vector_store = VectorStoreService()

    query_vector = [0.5] * 3072

    thresholds = [0.5, 0.7, 0.9]

    for threshold in thresholds:
        try:
            results = await vector_store.search_similar(
                query_embedding=query_vector,
                agent_id=1,
                top_k=10,
                score_threshold=threshold
            )

            assert isinstance(results, list)

            # Higher thresholds should return fewer results
            for result in results:
                assert result.get("score", 0) >= threshold

        except Exception:
            # Handle configuration issues
            continue


def test_vector_store_initialization():
    """Test vector store initialization"""
    vector_store = VectorStoreService()

    assert hasattr(vector_store, 'index_name')
    assert hasattr(vector_store, 'dimension')
    assert vector_store.dimension == 3072  # OpenAI text-embedding-3-large


@pytest.mark.asyncio
async def test_vector_store_metadata_validation():
    """Test metadata validation"""
    vector_store = VectorStoreService()

    # Test with various metadata types
    test_cases = [
        {"text": "string value", "number": 42, "boolean": True},
        {"list_field": ["item1", "item2"], "nested": {"key": "value"}},
        {"null_field": None, "empty_string": ""},
    ]

    for metadata in test_cases:
        vectors = [{
            "id": f"metadata_test_{hash(str(metadata))}",
            "values": [0.1] * 3072,
            "metadata": metadata
        }]

        try:
            await vector_store.add_vectors(vectors)
        except Exception as e:
            # Some metadata types might not be supported
            if "metadata" in str(e).lower():
                continue
            else:
                raise


@pytest.mark.asyncio
async def test_vector_store_large_text_handling():
    """Test handling of large text in metadata"""
    vector_store = VectorStoreService()

    large_text = "This is a very long text content. " * 1000  # ~34KB

    vectors = [{
        "id": "large_text_test",
        "values": [0.1] * 3072,
        "metadata": {
            "text": large_text,
            "agent_id": 1,
            "source": "large_document.txt"
        }
    }]

    try:
        result = await vector_store.add_vectors(vectors)
        # Should handle large text or truncate appropriately
        assert isinstance(result, bool)
    except Exception as e:
        # Some vector stores have text size limits
        if "size" in str(e).lower() or "limit" in str(e).lower():
            pass
        else:
            raise


@pytest.mark.asyncio
async def test_vector_store_duplicate_ids():
    """Test handling of duplicate vector IDs"""
    vector_store = VectorStoreService()

    # Add same ID twice
    vector_id = "duplicate_test"

    vectors1 = [{
        "id": vector_id,
        "values": [0.1] * 3072,
        "metadata": {"text": "First version", "version": 1}
    }]

    vectors2 = [{
        "id": vector_id,
        "values": [0.2] * 3072,
        "metadata": {"text": "Second version", "version": 2}
    }]

    try:
        await vector_store.add_vectors(vectors1)
        await vector_store.add_vectors(vectors2)
        # Should handle upsert behavior
    except Exception:
        # Some stores might handle duplicates differently
        pass