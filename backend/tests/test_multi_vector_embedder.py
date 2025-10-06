"""
Tests for Multi-Vector Embedder

Validates semantic, keyword, and structural embeddings with fusion strategies.
"""

import pytest
import numpy as np
from app.context_engine.multi_vector_embedder import (
    MultiVectorEmbedder,
    MultiVectorEmbedding
)


def test_keyword_embeddings_only():
    """Test keyword (TF-IDF) embeddings without semantic model"""
    embedder = MultiVectorEmbedder(
        use_semantic=False,
        use_keyword=True,
        use_structural=False
    )

    chunks = [
        ("Python is a programming language", "chunk1", {}),
        ("JavaScript is used for web development", "chunk2", {}),
        ("Python is popular for data science", "chunk3", {})
    ]

    embeddings = embedder.embed_chunks(chunks)

    assert len(embeddings) == 3
    assert embeddings[0].keyword_embedding is not None
    assert embeddings[0].semantic_embedding is None

    # Keyword embeddings should be vectors
    assert len(embeddings[0].keyword_embedding) > 0

    print(f"\n✅ Keyword embeddings generated:")
    print(f"  Vocabulary size: {len(embeddings[0].keyword_embedding)}")
    print(f"  Non-zero features: {np.count_nonzero(embeddings[0].keyword_embedding)}")


def test_structural_features():
    """Test structural feature generation"""
    embedder = MultiVectorEmbedder(
        use_semantic=False,
        use_keyword=False,
        use_structural=True
    )

    # Chunk with structural metadata
    metadata = {
        "level": 1,  # Subsection
        "position": 0.2,  # Near start
        "topic": "Introduction"
    }

    embedding = embedder.embed_chunk(
        text="This is an introduction to AI agents",
        chunk_id="chunk1",
        metadata=metadata
    )

    features = embedding.structural_features

    assert 'level' in features
    assert 'position' in features
    assert 'is_topic' in features
    assert 'start_bias' in features

    assert features['level'] == 1.0
    assert features['position'] == 0.2
    assert features['is_topic'] == 1.0  # Has topic
    assert features['start_bias'] == 0.8  # 1.0 - position

    print(f"\n✅ Structural features extracted:")
    for feature, value in features.items():
        print(f"  {feature}: {value:.3f}")


def test_combined_embeddings():
    """Test combining semantic + keyword + structural"""
    embedder = MultiVectorEmbedder(
        use_semantic=False,  # Skip semantic for speed
        use_keyword=True,
        use_structural=True
    )

    chunks = [
        ("AI agents are intelligent systems", "chunk1", {"level": 0, "position": 0.1}),
        ("Machine learning requires data", "chunk2", {"level": 1, "position": 0.5})
    ]

    embeddings = embedder.embed_chunks(chunks)

    # Combined embedding should include keyword + structural
    assert embeddings[0].combined_embedding is not None

    # Combined should be longer than keyword alone
    keyword_dim = len(embeddings[0].keyword_embedding)
    combined_dim = len(embeddings[0].combined_embedding)

    assert combined_dim > keyword_dim  # Includes structural features

    # Combined should be normalized to unit length
    norm = np.linalg.norm(embeddings[0].combined_embedding)
    assert abs(norm - 1.0) < 0.001  # Very close to 1.0

    print(f"\n✅ Combined embeddings created:")
    print(f"  Keyword dimensions: {keyword_dim}")
    print(f"  Combined dimensions: {combined_dim}")
    print(f"  Norm: {norm:.6f}")


def test_query_embedding():
    """Test query embedding (no structural features)"""
    embedder = MultiVectorEmbedder(
        use_semantic=False,
        use_keyword=True,
        use_structural=True
    )

    # Fit on corpus first
    corpus = [
        ("Python programming language", "c1", {}),
        ("JavaScript web development", "c2", {})
    ]
    embedder.embed_chunks(corpus)

    # Embed query
    query_emb = embedder.embed_query("Python coding")

    assert query_emb.chunk_id == "query"
    assert query_emb.keyword_embedding is not None
    # Structural features should not be generated for queries
    assert len(query_emb.structural_features) == 0

    print(f"\n✅ Query embedding generated (no structural features)")


def test_similarity_search():
    """Test multi-vector similarity search"""
    embedder = MultiVectorEmbedder(
        use_semantic=False,
        use_keyword=True,
        use_structural=False
    )

    chunks = [
        ("Python is a programming language", "chunk1", {}),
        ("JavaScript is for web development", "chunk2", {}),
        ("Python is popular for data science", "chunk3", {}),
        ("Machine learning uses Python", "chunk4", {})
    ]

    chunk_embeddings = embedder.embed_chunks(chunks)
    query_embedding = embedder.embed_query("Python programming")

    # Search
    results = embedder.similarity_search(
        query_embedding=query_embedding,
        chunk_embeddings=chunk_embeddings,
        top_k=2,
        semantic_weight=0.0,  # Only keyword
        keyword_weight=1.0
    )

    assert len(results) == 2
    assert isinstance(results[0], tuple)
    assert isinstance(results[0][0], str)  # chunk_id
    assert isinstance(results[0][1], float)  # score

    # chunk1 or chunk3 should be top (both mention "Python")
    top_chunks = [r[0] for r in results]
    assert "chunk1" in top_chunks or "chunk3" in top_chunks

    print(f"\n✅ Similarity search results:")
    for chunk_id, score in results:
        chunk_text = next(c.text for c in chunk_embeddings if c.chunk_id == chunk_id)
        print(f"  {chunk_id}: {chunk_text[:50]}... (score: {score:.3f})")


def test_empty_corpus():
    """Test handling of empty corpus"""
    embedder = MultiVectorEmbedder(
        use_semantic=False,
        use_keyword=True,
        use_structural=False
    )

    embeddings = embedder.embed_chunks([])

    assert embeddings == []

    print(f"\n✅ Empty corpus handled gracefully")


def test_multi_vector_to_dict():
    """Test serialization to dictionary"""
    embedder = MultiVectorEmbedder(
        use_semantic=False,
        use_keyword=True,
        use_structural=True
    )

    chunks = [("Test content", "chunk1", {"level": 0})]
    embeddings = embedder.embed_chunks(chunks)

    result_dict = embeddings[0].to_dict()

    assert 'text' in result_dict
    assert 'chunk_id' in result_dict
    assert 'keyword_embedding' in result_dict
    assert 'structural_features' in result_dict

    # Embeddings should be serialized to lists
    assert isinstance(result_dict['keyword_embedding'], list)

    print(f"\n✅ Serialization to dict works correctly")


@pytest.mark.skipif(
    True,  # Skip by default - requires model download
    reason="Requires sentence-transformers model download"
)
def test_semantic_embeddings():
    """Test semantic embeddings with sentence transformers (slow)"""
    embedder = MultiVectorEmbedder(
        use_semantic=True,
        use_keyword=False,
        use_structural=False
    )

    chunks = [
        ("AI agents are intelligent systems", "chunk1", {}),
        ("Artificial intelligence for automation", "chunk2", {}),
        ("Pricing and billing information", "chunk3", {})
    ]

    embeddings = embedder.embed_chunks(chunks)

    assert embeddings[0].semantic_embedding is not None
    assert len(embeddings[0].semantic_embedding) == 384  # MiniLM dimension

    # Semantic similarity: chunk1 should be closer to chunk2 than chunk3
    query_emb = embedder.embed_query("intelligent AI systems")

    results = embedder.similarity_search(
        query_embedding=query_emb,
        chunk_embeddings=embeddings,
        top_k=2,
        semantic_weight=1.0,
        keyword_weight=0.0
    )

    # chunk1 or chunk2 should be top
    assert results[0][0] in ["chunk1", "chunk2"]

    print(f"\n✅ Semantic embeddings work correctly")


@pytest.mark.skipif(
    True,  # Skip by default - requires model download
    reason="Requires sentence-transformers model download"
)
def test_full_multi_vector_search():
    """Test complete multi-vector search with all features (slow)"""
    embedder = MultiVectorEmbedder(
        use_semantic=True,
        use_keyword=True,
        use_structural=True
    )

    chunks = [
        ("AI agents automate business tasks", "chunk1", {"level": 0, "position": 0.0, "topic": "Overview"}),
        ("Our pricing starts at $99 per month", "chunk2", {"level": 1, "position": 0.5}),
        ("Intelligent agents use machine learning", "chunk3", {"level": 1, "position": 0.3})
    ]

    chunk_embeddings = embedder.embed_chunks(chunks)

    # Query: "AI agent pricing"
    # Should match chunk2 (keyword: pricing) and chunk1/chunk3 (semantic: AI agents)
    query_emb = embedder.embed_query("AI agent pricing")

    results = embedder.similarity_search(
        query_embedding=query_emb,
        chunk_embeddings=chunk_embeddings,
        top_k=3,
        semantic_weight=0.5,
        keyword_weight=0.5
    )

    assert len(results) == 3

    print(f"\n✅ Full multi-vector search results:")
    for chunk_id, score in results:
        chunk = next(c for c in chunk_embeddings if c.chunk_id == chunk_id)
        print(f"  {chunk_id}: {chunk.text} (score: {score:.3f})")


if __name__ == "__main__":
    # Run tests
    test_keyword_embeddings_only()
    test_structural_features()
    test_combined_embeddings()
    test_query_embedding()
    test_similarity_search()
    test_empty_corpus()
    test_multi_vector_to_dict()

    print("\n" + "="*60)
    print("✅ All multi-vector embedder tests passed!")
    print("="*60)
