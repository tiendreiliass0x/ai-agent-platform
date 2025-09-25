#!/usr/bin/env python3
"""
Quick test script to verify Gemini integration is working.
"""

import asyncio
import sys
from pathlib import Path

# Add the backend directory to the path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from app.services.gemini_service import gemini_service


async def test_gemini():
    """Test basic Gemini functionality"""

    print("🧪 Testing Gemini 2.0 Flash Integration...")
    print("=" * 50)

    # Test 1: Connection
    print("\n1️⃣ Testing connection...")
    connection_result = await gemini_service.test_connection()
    print(f"Status: {connection_result['status']}")
    if connection_result['status'] == 'success':
        print(f"✅ Text generation: {connection_result['text_generation']}")
        print(f"✅ Embeddings: {connection_result['embeddings']}")
        print(f"📝 Sample response: {connection_result['test_response']}")
        print(f"🔢 Embedding dimensions: {connection_result['embedding_dimensions']}")
    else:
        print(f"❌ Error: {connection_result.get('error', 'Unknown error')}")
        return False

    # Test 2: Simple text generation
    print("\n2️⃣ Testing text generation...")
    response = await gemini_service.generate_response(
        prompt="What is AI? Explain in one sentence.",
        temperature=0.3
    )
    print(f"Response: {response}")

    # Test 3: System prompt
    print("\n3️⃣ Testing with system prompt...")
    response = await gemini_service.generate_response(
        prompt="What is the capital of France?",
        system_prompt="You are a geography expert. Always provide detailed explanations.",
        temperature=0.1
    )
    print(f"Response: {response}")

    # Test 4: Embeddings
    print("\n4️⃣ Testing embeddings...")
    embeddings = await gemini_service.generate_embeddings([
        "This is a test document about AI",
        "Machine learning is a subset of artificial intelligence"
    ])
    print(f"Generated {len(embeddings)} embeddings")
    print(f"First embedding dimensions: {len(embeddings[0]) if embeddings else 0}")

    # Test 5: Query embedding
    print("\n5️⃣ Testing query embedding...")
    query_embedding = await gemini_service.generate_query_embedding("What is machine learning?")
    print(f"Query embedding dimensions: {len(query_embedding)}")

    # Test 6: Streaming response
    print("\n6️⃣ Testing streaming response...")
    print("Streaming output: ", end="")
    async for chunk in gemini_service.generate_streaming_response(
        prompt="Count from 1 to 5, one number per response",
        temperature=0.1
    ):
        print(chunk, end="", flush=True)
    print("\n")

    print("\n✅ All Gemini tests completed successfully!")
    return True


async def test_rag_with_gemini():
    """Test RAG service with Gemini"""

    print("\n🔧 Testing RAG Service with Gemini...")
    print("=" * 50)

    try:
        from app.services.rag_service import RAGService

        rag_service = RAGService()

        # Test simple response generation
        print("\n1️⃣ Testing RAG response generation...")

        # Note: This will use mock data since we don't have documents loaded
        response = await rag_service.generate_response(
            query="What is your return policy?",
            agent_id=1,
            system_prompt="You are a helpful customer service assistant."
        )

        print(f"Model used: {response.get('model', 'unknown')}")
        print(f"Response: {response.get('content', 'No response')}")
        print(f"Token usage: {response.get('token_usage', {})}")

        print("\n✅ RAG with Gemini test completed!")
        return True

    except Exception as e:
        print(f"❌ RAG test failed: {e}")
        return False


if __name__ == "__main__":
    async def main():
        # Test Gemini service
        gemini_success = await test_gemini()

        if gemini_success:
            # Test RAG integration
            await test_rag_with_gemini()
        else:
            print("❌ Gemini tests failed, skipping RAG tests")

    asyncio.run(main())