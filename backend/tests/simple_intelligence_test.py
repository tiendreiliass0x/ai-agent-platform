import asyncio
import httpx

async def simple_test():
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "http://127.0.0.1:8001/api/v1/agents/5/chat",
            json={
                "message": "I am really frustrated with this service!",
                "visitor_id": "test_user_123"
            }
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {data.get('response', '')[:100]}...")
            intelligence = data.get('customer_context', {}).get('user_intelligence', {})
            if intelligence:
                print(f"Emotional State: {intelligence.get('emotional_state')}")
                print(f"Urgency: {intelligence.get('urgency_level')}")
                print(f"Intent: {intelligence.get('intent_category')}")
            else:
                print("No intelligence data found")
        else:
            print(f"Error: {response.text}")

if __name__ == "__main__":
    asyncio.run(simple_test())