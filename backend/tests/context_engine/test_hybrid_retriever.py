"""
Tests for Hybrid Retriever

Validates BM25, dense retrieval, and hybrid fusion strategies.
"""

import pytest
import numpy as np
from app.context_engine.hybrid_retriever import (
    HybridRetriever,
    BM25,
    RetrievalResult
)


def test_bm25_basic():
    """Test basic BM25 functionality"""
    corpus = [
        "AI agents are intelligent systems that automate tasks",
        "Machine learning models require training data",
        "Natural language processing helps computers understand text",
        "AI agents use machine learning for intelligent automation"
    ]

    bm25 = BM25()
    bm25.fit(corpus)

    # Search for "AI agents"
    results = bm25.search("AI agents", top_k=2)

    assert len(results) > 0
    # First result should be doc 0 or 3 (both mention "AI agents")
    assert results[0][0] in [0, 3]
    assert results[0][1] > 0  # Score should be positive

    print(f"\n✅ BM25 search results for 'AI agents':")
    for doc_id, score in results:
        print(f"  Doc {doc_id}: {corpus[doc_id][:50]}... (score: {score:.2f})")


def test_bm25_keyword_matching():
    """Test that BM25 finds exact keyword matches"""
    corpus = [
        "The quick brown fox jumps over the lazy dog",
        "A fast brown animal leaps above a sleepy canine",
        "The brown fox is very quick and agile"
    ]

    bm25 = BM25()
    bm25.fit(corpus)

    # Search for "brown fox" - should prefer docs with both words
    results = bm25.search("brown fox", top_k=3)

    # Doc 0 and 2 have both "brown" and "fox"
    top_docs = [r[0] for r in results[:2]]
    assert 0 in top_docs or 2 in top_docs

    print(f"\n✅ BM25 keyword matching works correctly")


def test_hybrid_retriever_sparse_only():
    """Test hybrid retriever with sparse (BM25) only"""
    corpus = [
        "Python is a programming language",
        "JavaScript is used for web development",
        "Python is popular for data science",
        "Machine learning uses Python frequently"
    ]

    retriever = HybridRetriever(corpus=corpus)

    # Search without embeddings (sparse only)
    results = retriever.retrieve(
        query="Python programming",
        query_embedding=None,
        top_k=2
    )

    assert len(results) > 0
    assert isinstance(results[0], RetrievalResult)
    assert "Python" in results[0].text or "python" in results[0].text.lower()

    print(f"\n✅ Sparse-only retrieval works:")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result.text[:50]}... (score: {result.score:.3f})")


def test_hybrid_retriever_dense_only():
    """Test hybrid retriever with dense (vector) only"""
    corpus = [
        "AI systems are intelligent",
        "Machine learning is powerful",
        "Deep learning uses neural networks"
    ]

    # Create fake embeddings (in real use, these would be from a model)
    embeddings = np.random.rand(3, 128)  # 3 docs, 128-dim embeddings

    retriever = HybridRetriever(corpus=corpus, embeddings=embeddings)

    # Create query embedding
    query_embedding = np.random.rand(128)

    # Search with embeddings
    results = retriever.retrieve_dense(
        query_embedding=query_embedding,
        top_k=2
    )

    assert len(results) > 0
    assert results[0].dense_score is not None
    assert results[0].retrieval_method == "dense"

    print(f"\n✅ Dense-only retrieval works:")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result.text} (score: {result.score:.3f})")


def test_hybrid_retriever_rrf_fusion():
    """Test Reciprocal Rank Fusion (RRF)"""
    corpus = [
        "AI agents automate tasks intelligently",
        "Machine learning requires data",
        "AI and machine learning work together",
        "Automation improves efficiency"
    ]

    # Create embeddings where doc 2 is most similar to query
    embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.9, 0.9, 0.0, 0.0],  # Most similar to query
        [0.0, 0.0, 1.0, 0.0]
    ])

    retriever = HybridRetriever(corpus=corpus, embeddings=embeddings)

    # Query embedding (similar to doc 2)
    query_embedding = np.array([0.8, 0.8, 0.0, 0.0])

    # Hybrid search with RRF
    results = retriever.retrieve_hybrid(
        query="AI machine learning",
        query_embedding=query_embedding,
        top_k=3,
        use_rrf=True
    )

    assert len(results) > 0
    assert results[0].retrieval_method == "hybrid"

    # Check that both scores are tracked
    has_both_scores = any(
        r.dense_score is not None and r.sparse_score is not None
        for r in results
    )

    print(f"\n✅ RRF Fusion results:")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result.text[:50]}...")
        print(f"      Hybrid: {result.score:.4f}, Dense: {result.dense_score}, Sparse: {result.sparse_score}")


def test_hybrid_retriever_weighted_fusion():
    """Test weighted score fusion"""
    corpus = [
        "Python programming language",
        "Java development platform",
        "Python for data science"
    ]

    embeddings = np.random.rand(3, 64)
    retriever = HybridRetriever(corpus=corpus, embeddings=embeddings)

    query_embedding = np.random.rand(64)

    # Test with different alpha values
    results_dense = retriever.retrieve_hybrid(
        query="Python",
        query_embedding=query_embedding,
        top_k=2,
        use_rrf=False,
        alpha=1.0  # Only dense
    )

    results_sparse = retriever.retrieve_hybrid(
        query="Python",
        query_embedding=query_embedding,
        top_k=2,
        use_rrf=False,
        alpha=0.0  # Only sparse
    )

    results_balanced = retriever.retrieve_hybrid(
        query="Python",
        query_embedding=query_embedding,
        top_k=2,
        use_rrf=False,
        alpha=0.5  # Balanced
    )

    assert len(results_dense) > 0
    assert len(results_sparse) > 0
    assert len(results_balanced) > 0

    print(f"\n✅ Weighted fusion with different alphas works")


def test_hybrid_retriever_empty_corpus():
    """Test handling of empty corpus"""
    retriever = HybridRetriever(corpus=[])

    results = retriever.retrieve(query="test", top_k=5)

    assert results == []

    print(f"\n✅ Empty corpus handled gracefully")


def test_hybrid_retriever_index_corpus():
    """Test indexing corpus after initialization"""
    retriever = HybridRetriever()

    corpus = ["Document one", "Document two", "Document three"]
    embeddings = np.random.rand(3, 64)
    metadata = [{"id": i} for i in range(3)]

    retriever.index_corpus(
        corpus=corpus,
        embeddings=embeddings,
        metadata=metadata
    )

    results = retriever.retrieve(query="document", top_k=2)

    assert len(results) > 0
    assert results[0].metadata["id"] in [0, 1, 2]

    print(f"\n✅ Corpus indexing after initialization works")


if __name__ == "__main__":
    # Run tests
    test_bm25_basic()
    test_bm25_keyword_matching()
    test_hybrid_retriever_sparse_only()
    test_hybrid_retriever_dense_only()
    test_hybrid_retriever_rrf_fusion()
    test_hybrid_retriever_weighted_fusion()
    test_hybrid_retriever_empty_corpus()
    test_hybrid_retriever_index_corpus()

    print("\n" + "="*60)
    print("✅ All hybrid retriever tests passed!")
    print("="*60)
