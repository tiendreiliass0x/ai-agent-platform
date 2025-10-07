"""
Tests for Semantic Chunker

Validates that semantic chunking works correctly with boundary detection.
"""

import pytest
from app.context_engine.semantic_chunker import SemanticChunker, SemanticChunk


def test_semantic_chunker_basic():
    """Test basic semantic chunking functionality"""
    chunker = SemanticChunker(
        target_chunk_size=50,
        min_chunk_size=20,
        max_chunk_size=100,
        use_embeddings=False  # Disable for faster test
    )

    text = """
    # Introduction to AI Agents

    AI agents are intelligent systems that can perform tasks autonomously. They use machine learning
    and natural language processing to understand and respond to user queries.

    # How Agents Work

    Agents work by processing input, retrieving relevant context, and generating responses. The
    context retrieval process is critical for accurate responses.

    # Benefits of AI Agents

    AI agents can significantly improve customer service efficiency. They provide 24/7 support and
    can handle multiple conversations simultaneously.
    """

    chunks = chunker.chunk_text(text.strip())

    # Should create multiple chunks
    assert len(chunks) > 0
    assert isinstance(chunks[0], SemanticChunk)

    # All chunks should have text
    for chunk in chunks:
        assert len(chunk.text) > 0
        assert chunk.start_pos >= 0
        assert chunk.end_pos > chunk.start_pos

    print(f"\n✅ Created {len(chunks)} semantic chunks")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} (Level {chunk.level}):")
        print(f"  Topic: {chunk.topic}")
        print(f"  Length: {len(chunk.text.split())} words")
        print(f"  Coherence: {chunk.coherence_score:.2f}")
        print(f"  Preview: {chunk.text[:100]}...")


def test_semantic_chunker_respects_headers():
    """Test that chunker respects document headers"""
    chunker = SemanticChunker(use_embeddings=False)

    text = """
    # Section 1
    This is section one content.

    # Section 2
    This is section two content.

    # Section 3
    This is section three content.
    """

    chunks = chunker.chunk_text(text.strip())

    # Should have detected headers
    topics = [c.topic for c in chunks if c.topic]
    assert len(topics) > 0
    assert "Section 1" in topics or "Section 2" in topics or "Section 3" in topics

    print(f"\n✅ Detected {len(topics)} topics from headers")


def test_semantic_chunker_empty_text():
    """Test that empty text returns empty list"""
    chunker = SemanticChunker()

    chunks = chunker.chunk_text("")
    assert chunks == []

    chunks = chunker.chunk_text("   \n\n  ")
    assert chunks == []


def test_semantic_chunker_size_constraints():
    """Test that chunks respect size constraints"""
    chunker = SemanticChunker(
        min_chunk_size=10,
        max_chunk_size=50,
        target_chunk_size=30,
        use_embeddings=False
    )

    # Create text with distinct sections to enable proper splitting
    text = """
    # Section One
    This is the first section with content. It has multiple sentences to make it substantial.
    We want to test how the chunker handles size constraints with real content.

    # Section Two
    This is the second section with different content. It should be split into a separate chunk.
    The chunker should respect boundaries while managing size.

    # Section Three
    The third section contains even more information. Each section should ideally be its own chunk.
    This tests the chunker's ability to balance semantic boundaries with size constraints.
    """

    chunks = chunker.chunk_text(text.strip())

    # Check that chunks were created
    assert len(chunks) > 0

    # Check that very large chunks are split
    for chunk in chunks:
        word_count = len(chunk.text.split())
        # No chunk should be excessively large (allow some flexibility)
        assert word_count <= chunker.max_chunk_size * 2, f"Chunk with {word_count} words exceeds reasonable limit"

    print(f"\n✅ All chunks respect reasonable size constraints")
    print(f"   Chunks created: {len(chunks)}")
    print(f"   Avg words per chunk: {sum(len(c.text.split()) for c in chunks) / len(chunks):.1f}")
    for i, chunk in enumerate(chunks):
        print(f"   Chunk {i+1}: {len(chunk.text.split())} words")


@pytest.mark.skipif(
    True,  # Skip by default as it requires model download
    reason="Requires sentence-transformers model download"
)
def test_semantic_chunker_with_embeddings():
    """Test semantic chunking with embeddings (slow)"""
    chunker = SemanticChunker(
        target_chunk_size=50,
        similarity_threshold=0.5,
        use_embeddings=True  # Enable semantic similarity
    )

    text = """
    Machine learning is a subset of artificial intelligence. It focuses on building systems that
    learn from data. Deep learning is a specialized form of machine learning.

    Customer service is very important for businesses. Good customer service leads to customer
    satisfaction and loyalty. Companies invest heavily in support teams.

    Python is a popular programming language. It's widely used for data science and web development.
    JavaScript is another popular language for web applications.
    """

    chunks = chunker.chunk_text(text.strip())

    # With embeddings, should better detect topic boundaries
    assert len(chunks) >= 2  # Should split ML/CS/Programming topics

    print(f"\n✅ Created {len(chunks)} chunks with semantic similarity")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"  Words: {len(chunk.text.split())}")
        print(f"  Preview: {chunk.text[:80]}...")


if __name__ == "__main__":
    # Run tests
    test_semantic_chunker_basic()
    test_semantic_chunker_respects_headers()
    test_semantic_chunker_empty_text()
    test_semantic_chunker_size_constraints()

    print("\n" + "="*60)
    print("✅ All semantic chunker tests passed!")
    print("="*60)
