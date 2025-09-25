"""
Test script for the intelligent chat system with memory and personalization.
"""

import asyncio
import json
import httpx
from datetime import datetime

# Test configuration
BASE_URL = "http://127.0.0.1:8001"
AGENT_ID = 4  # Using the agent we created earlier
VISITOR_ID = "test_visitor_123"

async def test_intelligent_chat():
    """Test the intelligent chat endpoint with memory features"""

    async with httpx.AsyncClient() as client:
        print("🧪 Testing Intelligent Chat System with Memory")
        print("=" * 50)

        # Test 1: First interaction
        print("\n1. First conversation - Introduction")
        response1 = await client.post(
            f"{BASE_URL}/api/v1/agents/{AGENT_ID}/chat",
            json={
                "message": "Hi, I'm John Smith, a software engineer at Google. I'm looking for help with setting up a new deployment pipeline.",
                "visitor_id": VISITOR_ID,
                "session_context": {
                    "page_url": "https://example.com/deployment-guide",
                    "referrer": "https://google.com",
                    "device_info": {
                        "user_agent": "Mozilla/5.0...",
                        "screen_resolution": "1920x1080"
                    }
                }
            }
        )

        if response1.status_code == 200:
            data1 = response1.json()
            print(f"✅ Response: {data1['response'][:100]}...")
            print(f"📊 Customer Context: {json.dumps(data1['customer_context'], indent=2)}")
            print(f"🤖 Model: {data1['model']}")
        else:
            print(f"❌ Error: {response1.status_code} - {response1.text}")
            return

        # Test 2: Follow-up conversation - should remember context
        print("\n2. Follow-up conversation - Memory test")
        response2 = await client.post(
            f"{BASE_URL}/api/v1/agents/{AGENT_ID}/chat",
            json={
                "message": "Actually, I prefer detailed explanations. Can you give me more specifics about CI/CD best practices?",
                "visitor_id": VISITOR_ID,
                "session_context": {
                    "page_url": "https://example.com/ci-cd-best-practices",
                }
            }
        )

        if response2.status_code == 200:
            data2 = response2.json()
            print(f"✅ Response: {data2['response'][:100]}...")
            print(f"📊 Customer Context: {json.dumps(data2['customer_context'], indent=2)}")
            print(f"🧠 Memory entries used: {data2['customer_context'].get('memory_entries_used', 0)}")
        else:
            print(f"❌ Error: {response2.status_code} - {response2.text}")

        # Test 3: Third interaction - personalization should be applied
        print("\n3. Third conversation - Personalization test")
        response3 = await client.post(
            f"{BASE_URL}/api/v1/agents/{AGENT_ID}/chat",
            json={
                "message": "What about Docker security?",
                "visitor_id": VISITOR_ID,
                "session_context": {
                    "page_url": "https://example.com/docker-security",
                }
            }
        )

        if response3.status_code == 200:
            data3 = response3.json()
            print(f"✅ Response: {data3['response'][:100]}...")
            print(f"📊 Customer Context: {json.dumps(data3['customer_context'], indent=2)}")
            print(f"🔄 Returning customer: {data3['customer_context'].get('returning_customer', False)}")
            print(f"🎯 Engagement level: {data3['customer_context'].get('engagement_level', 'unknown')}")
        else:
            print(f"❌ Error: {response3.status_code} - {response3.text}")

        print("\n" + "=" * 50)
        print("✅ Intelligent Chat Test Complete!")

if __name__ == "__main__":
    asyncio.run(test_intelligent_chat())