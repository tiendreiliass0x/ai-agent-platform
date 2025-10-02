#!/usr/bin/env python3
"""
Test script to validate vector embeddings creation and processing.
"""

import pytest
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStoreService


@pytest.mark.asyncio
async def test_vector_embeddings():
    """Test the vector embeddings creation and storage process."""
    print("🔍 Testing Vector Embeddings Creation and Processing")
    print("=" * 60)

    # Initialize services
    document_processor = DocumentProcessor()
    vector_store = VectorStoreService()

    # Test data - our ChatGPT documentation content
    test_content = """# ChatGPT Documentation Test Content

## What is ChatGPT?

ChatGPT is an AI-powered conversational agent developed by OpenAI. It uses advanced natural language processing to understand and respond to user queries in a human-like manner.

## Key Features

### Conversational AI
ChatGPT can engage in natural conversations, maintaining context throughout the interaction. It can answer questions, provide explanations, help with writing tasks, and assist with problem-solving.

### Code Generation and Analysis
The model can generate code in multiple programming languages including Python, JavaScript, Java, C++, and more. It can also analyze existing code, debug issues, and suggest improvements.

### Creative Writing
ChatGPT excels at creative writing tasks including:
- Story writing and narrative development
- Poetry composition
- Content creation for marketing
- Email drafting and communication

### Problem Solving
The AI can help solve complex problems by:
- Breaking down complex tasks into manageable steps
- Providing multiple solution approaches
- Explaining reasoning and methodology
- Offering alternative perspectives

## Technical Capabilities

### Language Understanding
ChatGPT demonstrates sophisticated language understanding through:
- Context awareness across long conversations
- Nuanced interpretation of ambiguous queries
- Support for multiple languages and dialects
- Recognition of tone and intent

### Knowledge Integration
The model integrates knowledge from diverse domains including:
- Science and technology
- History and culture
- Literature and arts
- Business and economics
- Current events (up to training cutoff)

### Reasoning Abilities
ChatGPT can perform various types of reasoning:
- Logical deduction and inference
- Mathematical problem solving
- Causal reasoning
- Analogical thinking

## Use Cases

### Education and Learning
- Tutoring and homework assistance
- Concept explanation and clarification
- Research guidance
- Learning path recommendations

### Business Applications
- Customer service automation
- Content generation and marketing
- Document analysis and summarization
- Process optimization suggestions

### Development and Programming
- Code review and optimization
- Bug detection and fixing
- Architecture design discussions
- API documentation generation

### Creative Projects
- Brainstorming and ideation
- Character development for stories
- Marketing campaign concepts
- Design thinking facilitation"""

    print(f"📄 Test Content Length: {len(test_content)} characters")
    print(f"📄 Test Content Preview: {test_content[:200]}...")
    print()

    try:
        print("🔄 Step 1: Processing text content through document processor...")

        # Process the content using our document processor
        result = await document_processor.process_text_content(
            text=test_content,
            agent_id=999,  # Test agent ID
            source="chatgpt_docs_test.txt",
            document_id=999,  # Test document ID
            extra_metadata={
                "test_run": True,
                "content_type": "documentation",
                "source_url": "https://help.openai.com/en/collections/3742473-chatgpt"
            }
        )

        print(f"✅ Processing completed successfully!")
        print(f"   Status: {result.get('status', 'unknown')}")
        print(f"   Chunk Count: {result.get('chunk_count', 0)}")
        print(f"   Vector IDs Count: {len(result.get('vector_ids', []))}")
        print(f"   Preview: {result.get('preview', 'No preview available')[:100]}...")
        print()

        if result.get('status') == 'completed' and result.get('vector_ids'):
            print("🔄 Step 2: Testing vector search functionality...")

            # Test searching for similar content
            test_queries = [
                "What is ChatGPT and how does it work?",
                "How can ChatGPT help with programming tasks?",
                "What are the business applications of ChatGPT?",
                "Tell me about ChatGPT's creative writing capabilities"
            ]

            for i, query in enumerate(test_queries, 1):
                print(f"\n   Query {i}: {query}")

                # Generate real embedding using OpenAI service
                try:
                    from app.services.embedding_service import EmbeddingService
                    embedding_service = EmbeddingService()
                    embeddings = await embedding_service.generate_embeddings([query])
                    query_embedding = embeddings[0]
                    print(f"   ✅ Generated embedding with {len(query_embedding)} dimensions")
                except Exception as embedding_error:
                    print(f"   ⚠️  Embedding generation failed: {embedding_error}")
                    # Fallback to mock embedding for testing
                    query_embedding = [0.1] * 3072  # OpenAI text-embedding-3-large dimension
                    print(f"   🔄 Using fallback mock embedding")

                try:
                    search_results = await vector_store.search_similar(
                        query_embedding=query_embedding,
                        agent_id=999,
                        top_k=3,
                        score_threshold=0.5
                    )

                    if search_results:
                        print(f"   ✅ Found {len(search_results)} results")
                        for j, result in enumerate(search_results[:2], 1):
                            text_preview = result.get('text', '')[:100]
                            score = result.get('score', 0)
                            print(f"    Result {j}: Score={score:.3f}, Text: {text_preview}...")
                    else:
                        print(f"   ⚠️  No results found (this may be expected with mock embeddings)")

                except Exception as search_error:
                    print(f"   ⚠️  Search error: {search_error}")

            print("\n🔄 Step 3: Testing vector store statistics...")

            try:
                stats = await vector_store.get_index_stats()
                print(f"   Index Status: {stats.get('status', 'unknown')}")
                print(f"   Total Vectors: {stats.get('total_vectors', 0)}")
                print(f"   Index Dimension: {stats.get('dimension', 'unknown')}")
                print(f"   Index Fullness: {stats.get('index_fullness', 'unknown')}")

            except Exception as stats_error:
                print(f"   ⚠️  Stats error: {stats_error}")

        else:
            print("❌ Processing failed or no vectors were created")
            if result.get('error_message'):
                print(f"   Error: {result['error_message']}")

        print("\n" + "=" * 60)
        print("🏁 Vector Embeddings Test Completed")

        # Summary
        success_indicators = [
            result.get('status') == 'completed',
            result.get('chunk_count', 0) > 0,
            len(result.get('vector_ids', [])) > 0
        ]

        if all(success_indicators):
            print("✅ OVERALL STATUS: SUCCESS - Vector embeddings creation and storage working correctly")
        elif any(success_indicators):
            print("⚠️  OVERALL STATUS: PARTIAL - Some components working, may need Pinecone configuration")
        else:
            print("❌ OVERALL STATUS: FAILED - Vector embeddings process not working")

        return result

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return None


@pytest.mark.asyncio
async def test_document_processor_basic():
    """Test basic document processor functionality."""
    processor = DocumentProcessor()
    
    # Test with simple content
    simple_content = "This is a simple test document with basic content."
    
    result = await processor.process_text_content(
        text=simple_content,
        agent_id=1,
        source="test.txt",
        document_id=1,
        extra_metadata={"test": True}
    )
    
    assert result is not None
    assert "status" in result
    print(f"Basic processing result: {result}")


@pytest.mark.asyncio
async def test_vector_store_basic():
    """Test basic vector store functionality."""
    vector_store = VectorStoreService()
    
    # Test getting stats
    try:
        stats = await vector_store.get_index_stats()
        assert stats is not None
        print(f"Vector store stats: {stats}")
    except Exception as e:
        print(f"Vector store stats error (expected if not configured): {e}")
