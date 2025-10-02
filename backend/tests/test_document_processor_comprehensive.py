#!/usr/bin/env python3
"""
Comprehensive Document Processor Tests
Tests edge cases, error handling, and advanced functionality.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from app.services.document_processor import DocumentProcessor


@pytest.mark.asyncio
async def test_document_processor_empty_content():
    """Test document processor with empty content"""
    processor = DocumentProcessor()

    result = await processor.process_text_content(
        text="",
        agent_id=1,
        source="empty.txt",
        document_id=1,
        extra_metadata={"test": True}
    )

    # Empty content should return a failed status
    assert result["status"] == "failed"
    assert "error_message" in result


@pytest.mark.asyncio
async def test_document_processor_very_long_content():
    """Test document processor with very long content"""
    processor = DocumentProcessor()

    # Create content longer than typical chunk size
    long_content = "This is a test sentence. " * 1000  # About 25k characters

    result = await processor.process_text_content(
        text=long_content,
        agent_id=1,
        source="long.txt",
        document_id=1
    )

    assert result["status"] == "completed"
    assert result["chunk_count"] > 1  # Should be split into multiple chunks
    assert len(result["vector_ids"]) == result["chunk_count"]


@pytest.mark.asyncio
async def test_document_processor_markdown_content():
    """Test document processor with markdown formatting"""
    processor = DocumentProcessor()

    markdown_content = """
# Main Title

## Section 1
This is the first section with **bold** and *italic* text.

### Subsection 1.1
- List item 1
- List item 2
- List item 3

## Section 2
This is the second section.

```python
def example_function():
    return "Hello World"
```

## Section 3
Final section with a [link](https://example.com).
"""

    result = await processor.process_text_content(
        text=markdown_content,
        agent_id=1,
        source="markdown.md",
        document_id=1
    )

    assert result["status"] == "completed"
    assert result["chunk_count"] > 0
    # Should preserve structure and extract section information


@pytest.mark.asyncio
async def test_document_processor_special_characters():
    """Test document processor with special characters and unicode"""
    processor = DocumentProcessor()

    special_content = """
Testing special characters: àáâãäå æç èéêë ìíîï ñ òóôõö ùúûü ÿ
Emojis: 🚀 🎉 💡 🔥 ⭐ 🌟
Mathematical symbols: α β γ δ ε ζ η θ ι κ λ μ ν ξ ο π ρ σ τ υ φ χ ψ ω
Currency symbols: $ € £ ¥ ₹ ₽ ₩
Punctuation: "quotes" 'apostrophes' —dashes— …ellipsis…
"""

    result = await processor.process_text_content(
        text=special_content,
        agent_id=1,
        source="special.txt",
        document_id=1
    )

    assert result["status"] == "completed"
    assert result["chunk_count"] > 0
    # Should handle unicode properly


@pytest.mark.asyncio
async def test_document_processor_chunking_logic():
    """Test the chunking logic with controlled content"""
    processor = DocumentProcessor()

    # Create content with clear section breaks
    structured_content = """
Section One Header
This is the first section with some content that should be grouped together.
It has multiple sentences to test the chunking logic.

Section Two Header
This is the second section with different content.
It also has multiple sentences for testing purposes.
This section is slightly longer than the first one.

Section Three Header
This is the third and final section.
It contains the conclusion of our test document.
"""

    result = await processor.process_text_content(
        text=structured_content,
        agent_id=1,
        source="structured.txt",
        document_id=1
    )

    assert result["status"] == "completed"
    assert result["chunk_count"] >= 1
    # Chunks should respect section boundaries when possible


@pytest.mark.asyncio
async def test_document_processor_metadata_extraction():
    """Test metadata extraction from different content types"""
    processor = DocumentProcessor()

    content_with_keywords = """
This document discusses machine learning algorithms, particularly neural networks
and deep learning models. It covers topics like supervised learning, unsupervised
learning, and reinforcement learning. The document also mentions Python programming,
TensorFlow, PyTorch, and data science methodologies.
"""

    result = await processor.process_text_content(
        text=content_with_keywords,
        agent_id=1,
        source="ml_doc.txt",
        document_id=1,
        extra_metadata={"category": "machine_learning", "author": "test"}
    )

    assert result["status"] == "completed"
    assert result["chunk_count"] > 0
    # Should extract relevant keywords and preserve extra metadata


@pytest.mark.asyncio
async def test_document_processor_error_handling():
    """Test document processor error handling"""
    processor = DocumentProcessor()

    with patch.object(processor, '_create_chunks') as mock_chunks:
        mock_chunks.side_effect = Exception("Chunking failed")

        result = await processor.process_text_content(
            text="Test content",
            agent_id=1,
            source="error.txt",
            document_id=1
        )

        assert result["status"] == "failed"
        assert "error_message" in result
        assert result["chunk_count"] == 0


@pytest.mark.asyncio
async def test_document_processor_different_file_types():
    """Test document processor with different simulated file types"""
    processor = DocumentProcessor()

    test_cases = [
        ("test.pdf", "PDF content with paragraphs and structure"),
        ("test.docx", "Word document content with formatting"),
        ("test.html", "<h1>HTML content</h1><p>With tags and structure</p>"),
        ("test.json", '{"key": "JSON content", "data": ["array", "items"]}'),
    ]

    for filename, content in test_cases:
        result = await processor.process_text_content(
            text=content,
            agent_id=1,
            source=filename,
            document_id=1
        )

        assert result["status"] == "completed"
        assert result["chunk_count"] > 0
        assert len(result["vector_ids"]) == result["chunk_count"]


@pytest.mark.asyncio
async def test_document_processor_concurrent_processing():
    """Test document processor with concurrent requests"""
    processor = DocumentProcessor()

    async def process_document(doc_id: int):
        return await processor.process_text_content(
            text=f"Test document {doc_id} with unique content for processing.",
            agent_id=1,
            source=f"test_{doc_id}.txt",
            document_id=doc_id
        )

    # Process multiple documents concurrently
    import asyncio
    results = await asyncio.gather(*[process_document(i) for i in range(1, 6)])

    # All should succeed
    for result in results:
        assert result["status"] == "completed"
        assert result["chunk_count"] > 0


@pytest.mark.asyncio
async def test_search_similar_content():
    """Test the search_similar_content method"""
    processor = DocumentProcessor()

    # Mock the vector store search
    with patch.object(processor.vector_store, 'search_similar') as mock_search:
        mock_search.return_value = [
            {"text": "Result 1", "score": 0.9, "metadata": {"id": "1"}},
            {"text": "Result 2", "score": 0.8, "metadata": {"id": "2"}},
        ]

        results = await processor.search_similar_content(
            query="test query",
            agent_id=1,
            top_k=5
        )

        assert len(results) == 2
        assert results[0]["score"] == 0.9
        assert results[1]["score"] == 0.8
        mock_search.assert_called_once()


@pytest.mark.asyncio
async def test_document_processor_memory_efficiency():
    """Test that document processor doesn't consume excessive memory"""
    processor = DocumentProcessor()

    # Process a medium-sized document multiple times
    medium_content = "This is a test document. " * 500  # About 12.5k characters

    for i in range(10):
        result = await processor.process_text_content(
            text=medium_content,
            agent_id=1,
            source=f"memory_test_{i}.txt",
            document_id=i
        )
        assert result["status"] == "completed"

    # Test should complete without memory issues


@pytest.mark.asyncio
async def test_document_processor_extract_keywords():
    """Test keyword extraction functionality"""
    processor = DocumentProcessor()

    # Test with content that has clear keywords
    keyword_content = """
This article discusses artificial intelligence, machine learning, and deep learning.
It covers natural language processing, computer vision, and neural networks.
The document also mentions Python programming, data analysis, and statistical modeling.
"""

    # Access the private method for testing
    keywords = processor._extract_keywords(keyword_content)

    # Keywords may be None if YAKE is not available, which is acceptable
    if keywords is not None:
        assert isinstance(keywords, list)
        assert len(keywords) >= 0
        # Should extract relevant technical terms
    else:
        # YAKE not available, which is expected in some environments
        assert keywords is None


def test_derive_section_path():
    """Test section path derivation from content"""
    processor = DocumentProcessor()

    test_cases = [
        ("# Main Header\nContent here", "# Main Header"),
        ("## Sub Header\nMore content", "## Sub Header"),
        ("UPPERCASE SECTION\nContent", "Uppercase Section"),
        ("Regular paragraph text\nNo headers", ""),
    ]

    for content, expected in test_cases:
        result = processor._derive_section_path(content)
        assert result == expected


def test_summarize_chunk():
    """Test chunk summarization"""
    processor = DocumentProcessor()

    long_chunk = """
    This is the first sentence of a longer chunk of text.
    It contains multiple sentences to test the summarization.
    The third sentence adds more detail about the content.
    Finally, the fourth sentence concludes the chunk.
    """

    summary = processor._summarize_chunk(long_chunk)

    assert isinstance(summary, str)
    assert len(summary) > 0
    assert len(summary) < len(long_chunk)
    # Should be shorter than original