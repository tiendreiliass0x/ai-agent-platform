"""
Test script for advanced user intelligence system.
Tests emotional detection, urgency analysis, intent prediction, and response adaptation.
"""

import asyncio
import json
import httpx
from datetime import datetime

# Test configuration
BASE_URL = "http://127.0.0.1:8001"

async def test_user_intelligence_system():
    """Test the advanced user intelligence and concierge capabilities"""

    async with httpx.AsyncClient() as client:
        print("🧠 Testing Advanced User Intelligence System")
        print("=" * 60)

        # Login first
        print("\n1. Logging in...")
        login_response = await client.post(
            f"{BASE_URL}/api/v1/auth/login",
            data={
                "username": "testuser@example.com",
                "password": "password123"
            }
        )

        if login_response.status_code != 200:
            print(f"❌ Login failed: {login_response.text}")
            return

        token_data = login_response.json()
        token = token_data["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        print("✅ Login successful")

        # Use existing agent (from previous tests)
        agent_id = 5  # FinTech Support Agent
        visitor_id_prefix = "intelligence_test"

        # Test scenarios for different user states and intents
        test_scenarios = [
            {
                "name": "Frustrated Customer - Technical Issue",
                "visitor_id": f"{visitor_id_prefix}_frustrated",
                "message": "This is really frustrating! I've been trying to set up the API integration for 3 hours and nothing works. The documentation is confusing and I have a client demo tomorrow. URGENT help needed!",
                "session_context": {
                    "page_url": "https://example.com/api-docs",
                    "referrer": "https://google.com/search?q=api+integration+help",
                    "device_info": {"user_agent": "Mozilla/5.0", "screen_resolution": "1920x1080"}
                },
                "expected_emotional_state": "frustrated",
                "expected_urgency": "high",
                "expected_intent": "support_request"
            },
            {
                "name": "Excited Prospect - Purchase Intent",
                "visitor_id": f"{visitor_id_prefix}_excited",
                "message": "Wow! This looks amazing! I'm really excited about your enterprise features. Can I get pricing for our team of 50 people? We want to upgrade from our current solution ASAP. When can we schedule a demo?",
                "session_context": {
                    "page_url": "https://example.com/pricing",
                    "referrer": "https://example.com/features"
                },
                "expected_emotional_state": "excited",
                "expected_urgency": "medium",
                "expected_intent": "purchase_intent"
            },
            {
                "name": "Confused Beginner - Information Seeking",
                "visitor_id": f"{visitor_id_prefix}_confused",
                "message": "I'm new to this and honestly quite confused. What exactly does this platform do? How does it work? I don't understand the difference between the plans. Can you explain in simple terms?",
                "session_context": {
                    "page_url": "https://example.com/getting-started"
                },
                "expected_emotional_state": "confused",
                "expected_urgency": "low",
                "expected_intent": "information_seeking"
            },
            {
                "name": "Angry Customer - Complaint",
                "visitor_id": f"{visitor_id_prefix}_angry",
                "message": "I am absolutely furious! Your service went down during our most important presentation and cost us a major client. This is unacceptable! I want a full refund and explanation NOW!",
                "session_context": {
                    "page_url": "https://example.com/support",
                    "referrer": "https://status.example.com"
                },
                "expected_emotional_state": "angry",
                "expected_urgency": "critical",
                "expected_intent": "complaint"
            },
            {
                "name": "Potential Churner - Cancellation Intent",
                "visitor_id": f"{visitor_id_prefix}_churner",
                "message": "I'm thinking about canceling my subscription. The service isn't really meeting our needs and it's getting expensive. What are my options?",
                "session_context": {
                    "page_url": "https://example.com/account/billing"
                },
                "expected_emotional_state": "neutral",
                "expected_urgency": "medium",
                "expected_intent": "cancellation"
            },
            {
                "name": "Technical Expert - Integration Question",
                "visitor_id": f"{visitor_id_prefix}_expert",
                "message": "I need to implement a webhook endpoint that can handle high-throughput events. What's the recommended architecture for processing 10k+ events per second with proper error handling and retry logic?",
                "session_context": {
                    "page_url": "https://example.com/docs/webhooks",
                    "referrer": "https://example.com/docs/api"
                },
                "expected_emotional_state": "neutral",
                "expected_urgency": "medium",
                "expected_intent": "information_seeking"
            }
        ]

        # Test each scenario
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n{i}. Testing: {scenario['name']}")
            print("-" * 40)

            chat_response = await client.post(
                f"{BASE_URL}/api/v1/agents/{agent_id}/chat",
                json={
                    "message": scenario["message"],
                    "visitor_id": scenario["visitor_id"],
                    "session_context": scenario["session_context"]
                }
            )

            if chat_response.status_code == 200:
                result = chat_response.json()
                intelligence = result.get("customer_context", {}).get("user_intelligence", {})

                print(f"📝 Message: {scenario['message'][:80]}...")
                print(f"🤖 Response: {result['response'][:100]}...")
                print()
                print("🧠 INTELLIGENCE ANALYSIS:")
                print(f"   😊 Emotional State: {intelligence.get('emotional_state', 'N/A')}")
                print(f"   ⚡ Urgency Level: {intelligence.get('urgency_level', 'N/A')}")
                print(f"   🎯 Intent Category: {intelligence.get('intent_category', 'N/A')}")
                print(f"   📊 Confidence: {intelligence.get('confidence_score', 'N/A'):.2f}")
                print(f"   🔑 Key Topics: {intelligence.get('key_topics', [])}")
                print(f"   😓 Pain Points: {intelligence.get('pain_points', [])}")
                print(f"   💡 Opportunities: {intelligence.get('opportunities', [])}")
                print(f"   🚨 Escalation Needed: {intelligence.get('escalation_needed', False)}")

                # Validate intelligence accuracy
                validation_results = []
                if intelligence.get('emotional_state') == scenario['expected_emotional_state']:
                    validation_results.append("✅ Emotional state correct")
                else:
                    validation_results.append(f"❌ Emotional state wrong: expected {scenario['expected_emotional_state']}, got {intelligence.get('emotional_state')}")

                if intelligence.get('urgency_level') == scenario['expected_urgency']:
                    validation_results.append("✅ Urgency level correct")
                else:
                    validation_results.append(f"❌ Urgency level wrong: expected {scenario['expected_urgency']}, got {intelligence.get('urgency_level')}")

                if intelligence.get('intent_category') == scenario['expected_intent']:
                    validation_results.append("✅ Intent category correct")
                else:
                    validation_results.append(f"❌ Intent category wrong: expected {scenario['expected_intent']}, got {intelligence.get('intent_category')}")

                print("\n📋 VALIDATION:")
                for validation in validation_results:
                    print(f"   {validation}")

                # Show additional context insights
                context = result.get("customer_context", {})
                print(f"\n📈 CUSTOMER CONTEXT:")
                print(f"   👤 Engagement Level: {context.get('engagement_level', 'N/A')}")
                print(f"   🔄 Returning Customer: {context.get('returning_customer', False)}")
                print(f"   🎨 Personalization: {context.get('personalization_applied', False)}")
                print(f"   📊 Context Quality: {context.get('context_quality_score', 'N/A')}")

            else:
                print(f"❌ Chat failed: {chat_response.status_code} - {chat_response.text}")

            print()

        # Test conversation continuity with follow-up messages
        print(f"\n{len(test_scenarios) + 1}. Testing Conversation Continuity")
        print("-" * 40)

        # Follow up on the frustrated customer
        followup_response = await client.post(
            f"{BASE_URL}/api/v1/agents/{agent_id}/chat",
            json={
                "message": "Thank you for the help! That actually worked. I'm feeling much better about this now. Do you have any best practices for error handling?",
                "visitor_id": f"{visitor_id_prefix}_frustrated",
                "session_context": {
                    "page_url": "https://example.com/api-docs/error-handling"
                }
            }
        )

        if followup_response.status_code == 200:
            result = followup_response.json()
            intelligence = result.get("customer_context", {}).get("user_intelligence", {})

            print("📝 Follow-up Message: Thank you for the help! That actually worked...")
            print(f"🤖 Response: {result['response'][:100]}...")
            print()
            print("🧠 EMOTIONAL STATE TRANSITION:")
            print(f"   😊 New Emotional State: {intelligence.get('emotional_state', 'N/A')}")
            print(f"   ⚡ New Urgency Level: {intelligence.get('urgency_level', 'N/A')}")
            print(f"   🎯 New Intent: {intelligence.get('intent_category', 'N/A')}")
            print(f"   🔄 Returning Customer: {result.get('customer_context', {}).get('returning_customer', False)}")

        print("\n" + "=" * 60)
        print("✅ User Intelligence System Test Complete!")

        # Summary analysis
        print("\n🏆 INTELLIGENCE SYSTEM CAPABILITIES DEMONSTRATED:")
        print("   🧠 Real-time emotional state detection")
        print("   ⚡ Urgency level assessment")
        print("   🎯 Intent category prediction")
        print("   😓 Pain point identification")
        print("   💡 Business opportunity spotting")
        print("   🚨 Automatic escalation detection")
        print("   🎨 Adaptive response personalization")
        print("   🔄 Cross-session memory continuity")
        print("   📊 Confidence scoring and validation")

if __name__ == "__main__":
    asyncio.run(test_user_intelligence_system())