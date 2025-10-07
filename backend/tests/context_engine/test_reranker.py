"""
Tests for Reranker

Validates fallback reranking (cross-encoder tests skipped by default).
"""

import pytest
from app.context_engine.reranker import Reranker, RerankResult


def test_fallback_reranking():
    """Test fallback reranking without cross-encoder"""
    reranker = Reranker(use_cross_encoder=False, top_k=3)

    query = "Python programming tutorial"
    documents = [
        "JavaScript is a web programming language used for frontend development",
        "Python is a versatile programming language. This comprehensive tutorial covers basics to advanced topics",
        "Learn Python programming step by step with examples and exercises",
        "Java programming language for enterprise applications",
    ]

    original_scores = [0.3, 0.8, 0.9, 0.2]

    results = reranker.rerank(query, documents, original_scores)

    assert len(results) <= 3
    assert isinstance(results[0], RerankResult)

    # Should prioritize Python-related docs
    assert "Python" in results[0].text or "python" in results[0].text.lower()

    # Scores should be populated
    assert results[0].final_score > 0
    assert results[0].rank == 0

    print(f"\n✅ Fallback reranking works:")
    for i, result in enumerate(results):
        print(f"  {i+1}. Score: {result.final_score:.3f} - {result.text[:50]}...")


def test_query_term_coverage():
    """Test that reranking considers query term coverage"""
    reranker = Reranker(use_cross_encoder=False)

    query = "machine learning neural networks"
    documents = [
        "Deep learning uses neural networks for pattern recognition",
        "This is about gardening and plants",
        "Machine learning and neural networks are key to AI",
    ]

    original_scores = [0.5, 0.5, 0.5]  # All equal initially
    results = reranker.rerank(query, documents, original_scores)

    # Doc with more query terms should rank higher
    top_result_text = results[0].text.lower()
    assert "neural" in top_result_text or "machine" in top_result_text

    print(f"\n✅ Query term coverage affects ranking")


def test_empty_documents():
    """Test handling of empty document list"""
    reranker = Reranker(use_cross_encoder=False)

    results = reranker.rerank("test query", [])

    assert results == []

    print("\n✅ Empty documents handled gracefully")


def test_top_k_limiting():
    """Test that top_k limits results"""
    reranker = Reranker(use_cross_encoder=False, top_k=2)

    documents = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    results = reranker.rerank("query", documents)

    assert len(results) == 2

    print(f"\n✅ Top-k limiting works (returned {len(results)} of {len(documents)})")


def test_rerank_result_structure():
    """Test RerankResult data structure"""
    reranker = Reranker(use_cross_encoder=False)

    documents = ["Test document about Python programming"]
    metadata = [{"source": "doc1", "timestamp": "2024-01-01"}]

    results = reranker.rerank(
        "Python",
        documents,
        original_scores=[0.8],
        metadata=metadata
    )

    result = results[0]

    assert hasattr(result, 'text')
    assert hasattr(result, 'original_score')
    assert hasattr(result, 'rerank_score')
    assert hasattr(result, 'final_score')
    assert hasattr(result, 'rank')
    assert hasattr(result, 'metadata')

    assert result.metadata == metadata[0]
    assert result.original_score == 0.8

    print("\n✅ RerankResult structure is correct")


def test_batch_reranking():
    """Test batch reranking multiple queries"""
    reranker = Reranker(use_cross_encoder=False, top_k=2)

    queries = ["Python programming", "JavaScript web"]
    documents_list = [
        ["Python tutorial", "Java guide", "Python advanced"],
        ["JavaScript basics", "Python intro", "JavaScript DOM"]
    ]

    results = reranker.batch_rerank(queries, documents_list)

    assert len(results) == 2  # One result list per query
    assert len(results[0]) <= 2  # Top-k limit
    assert len(results[1]) <= 2

    print(f"\n✅ Batch reranking works:")
    for i, query_results in enumerate(results):
        print(f"  Query {i+1}: {len(query_results)} results")


def test_score_combination():
    """Test original + rerank score combination"""
    reranker = Reranker(use_cross_encoder=False, score_weight=0.5)

    query = "test"
    documents = ["test document"]
    original_scores = [0.6]

    results = reranker.rerank(query, documents, original_scores)

    # Final score should be influenced by both original and rerank
    assert 0.0 <= results[0].final_score <= 2.0

    print(f"\n✅ Score combination works:")
    print(f"   Original: {results[0].original_score:.3f}")
    print(f"   Rerank: {results[0].rerank_score:.3f}")
    print(f"   Final: {results[0].final_score:.3f}")


def test_model_info():
    """Test getting model information"""
    reranker = Reranker(use_cross_encoder=False)

    info = reranker.get_model_info()

    assert "model_name" in info
    assert "using_cross_encoder" in info
    assert "top_k" in info
    assert info["using_cross_encoder"] == False

    print(f"\n✅ Model info: {info}")


@pytest.mark.skipif(
    True,  # Skip by default - requires model download
    reason="Requires transformers and model download"
)
def test_cross_encoder_reranking():
    """Test cross-encoder reranking (slow, requires model)"""
    reranker = Reranker(
        use_cross_encoder=True,
        model_name="jinaai/jina-reranker-v2-base-multilingual",
        top_k=3
    )

    query = "How to train neural networks?"
    documents = [
        "Neural networks require labeled training data and backpropagation",
        "The weather is sunny today",
        "Deep learning models use gradient descent for training neural networks",
        "Cooking pasta requires boiling water",
    ]

    results = reranker.rerank(query, documents)

    assert len(results) <= 3

    # Neural network docs should rank higher
    top_text = results[0].text.lower()
    assert "neural" in top_text or "training" in top_text

    print(f"\n✅ Cross-encoder reranking works:")
    for i, result in enumerate(results):
        print(f"  {i+1}. Score: {result.final_score:.3f}")
        print(f"     {result.text[:60]}...")


if __name__ == "__main__":
    # Run tests
    test_fallback_reranking()
    test_query_term_coverage()
    test_empty_documents()
    test_top_k_limiting()
    test_rerank_result_structure()
    test_batch_reranking()
    test_score_combination()
    test_model_info()

    print("\n" + "="*60)
    print("✅ All reranker tests passed!")
    print("="*60)
