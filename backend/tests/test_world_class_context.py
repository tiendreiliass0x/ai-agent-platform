"""
Test script for world-class context engineering system.
Validates advanced context ranking, semantic similarity, and optimization.
"""

import asyncio
import json
import httpx
from datetime import datetime

# Test configuration
BASE_URL = "http://127.0.0.1:8001"
AGENT_ID = 4  # Using the agent we created earlier
VISITOR_ID = "test_context_visitor_456"

async def test_advanced_context_system():
    """Test the advanced context optimization system"""

    async with httpx.AsyncClient() as client:
        print("🌟 Testing Advanced Context System")
        print("=" * 50)

        # Test 1: Build rich customer profile through conversation
        print("\n1. Building Rich Customer Profile")
        response1 = await client.post(
            f"{BASE_URL}/api/v1/agents/{AGENT_ID}/chat",
            json={
                "message": "Hi! I'm Sarah Chen, CTO at DataFlow Inc. We're a Series B fintech startup with 50 engineers. I need help architecting our microservices for PCI compliance while maintaining high availability. We're currently processing 10M transactions daily.",
                "visitor_id": VISITOR_ID,
                "session_context": {
                    "page_url": "https://example.com/enterprise-architecture",
                    "referrer": "https://google.com/search?q=pci+compliant+microservices",
                    "device_info": {
                        "user_agent": "Mozilla/5.0 (Professional)",
                        "screen_resolution": "2560x1440"
                    }
                }
            }
        )

        if response1.status_code == 200:
            data1 = response1.json()
            print(f"✅ Response: {data1['response'][:150]}...")
            print(f"📊 Context Quality: {data1['customer_context'].get('context_quality_score', 'N/A')}")
            print(f"🧩 Context Chunks: {data1['customer_context'].get('context_chunks_used', 'N/A')}")
            print(f"⚡ Context Efficiency: {data1['customer_context'].get('context_efficiency', 'N/A'):.2%}")
        else:
            print(f"❌ Error: {response1.status_code} - {response1.text}")
            return

        # Test 2: Add more behavioral data
        print("\n2. Adding Behavioral Context")
        response2 = await client.post(
            f"{BASE_URL}/api/v1/agents/{AGENT_ID}/chat",
            json={
                "message": "Actually, I prefer detailed technical explanations with specific implementation examples. We use Kubernetes, Istio service mesh, and PostgreSQL. Our team is very experienced with distributed systems, so don't worry about being too technical.",
                "visitor_id": VISITOR_ID,
                "session_context": {
                    "page_url": "https://example.com/kubernetes-guide",
                }
            }
        )

        if response2.status_code == 200:
            data2 = response2.json()
            print(f"✅ Response: {data2['response'][:150]}...")
            print(f"📊 Context Quality: {data2['customer_context'].get('context_quality_score', 'N/A')}")
            print(f"🧩 Context Chunks: {data2['customer_context'].get('context_chunks_used', 'N/A')}")
            print(f"🔄 Returning Customer: {data2['customer_context'].get('returning_customer', False)}")
        else:
            print(f"❌ Error: {response2.status_code} - {response2.text}")

        # Test 3: Complex technical query to test context ranking
        print("\n3. Complex Technical Query - Context Ranking Test")
        response3 = await client.post(
            f"{BASE_URL}/api/v1/agents/{AGENT_ID}/chat",
            json={
                "message": "What are the security implications of implementing API gateways in our PCI-compliant microservices architecture? We need to consider both data tokenization and network segmentation.",
                "visitor_id": VISITOR_ID,
                "session_context": {
                    "page_url": "https://example.com/api-gateway-security",
                }
            }
        )

        if response3.status_code == 200:
            data3 = response3.json()
            print(f"✅ Response: {data3['response'][:150]}...")
            print(f"📊 Context Quality: {data3['customer_context'].get('context_quality_score', 'N/A')}")
            print(f"🧩 Context Chunks: {data3['customer_context'].get('context_chunks_used', 'N/A')}")
            print(f"⚡ Context Efficiency: {data3['customer_context'].get('context_efficiency', 'N/A'):.2%}")
            print(f"🎯 Engagement Level: {data3['customer_context'].get('engagement_level', 'unknown')}")
        else:
            print(f"❌ Error: {response3.status_code} - {response3.text}")

        # Test 4: Follow-up with domain-specific context
        print("\n4. Domain-Specific Follow-up - Memory Synthesis Test")
        response4 = await client.post(
            f"{BASE_URL}/api/v1/agents/{AGENT_ID}/chat",
            json={
                "message": "Given our transaction volume and the fact that we're a fintech, what specific PCI DSS requirements should we focus on first?",
                "visitor_id": VISITOR_ID,
                "session_context": {
                    "page_url": "https://example.com/pci-dss-compliance",
                }
            }
        )

        if response4.status_code == 200:
            data4 = response4.json()
            print(f"✅ Response: {data4['response'][:150]}...")
            print(f"📊 Context Quality: {data4['customer_context'].get('context_quality_score', 'N/A')}")
            print(f"🧩 Context Chunks: {data4['customer_context'].get('context_chunks_used', 'N/A')}")
            print(f"🎭 Personalization: {data4['customer_context'].get('personalization_applied', False)}")
        else:
            print(f"❌ Error: {response4.status_code} - {response4.text}")

        print("\n" + "=" * 50)
        print("✅ Advanced Context System Test Complete!")
        print("\n🏆 CONTEXT METRICS:")

        if response4.status_code == 200:
            final_context = data4['customer_context']
            print(f"📊 Final Context Quality: {final_context.get('context_quality_score', 'N/A')}")
            print(f"🧩 Context Chunks Used: {final_context.get('context_chunks_used', 'N/A')}")
            print(f"⚡ Context Efficiency: {final_context.get('context_efficiency', 'N/A'):.2%}")
            print(f"🎯 Customer Engagement: {final_context.get('engagement_level', 'unknown')}")
            print(f"🔄 Cross-session Memory: {final_context.get('returning_customer', False)}")
            print(f"🎭 Personalization Applied: {final_context.get('personalization_applied', False)}")

if __name__ == "__main__":
    asyncio.run(test_advanced_context_system())