"""
Test WebSocket Real-time Chat with Intelligent Concierge Agent
Tests the world-class conversational experience.
"""

import asyncio
import json
import websockets
import time
from datetime import datetime


async def test_websocket_chat():
    """Test real-time WebSocket chat with intelligence integration"""

    agent_id = 8  # Use the agent we created earlier
    visitor_id = f"test_visitor_{int(time.time())}"
    uri = f"ws://127.0.0.1:8001/api/v1/agents/{agent_id}/ws?visitor_id={visitor_id}"

    print("🚀 Testing Real-time WebSocket Chat with Intelligent Concierge Agent")
    print("=" * 70)

    try:
        async with websockets.connect(uri) as websocket:
            print(f"✅ Connected to agent {agent_id} with visitor ID: {visitor_id}")

            # Start listening for messages in the background
            async def listen_for_messages():
                """Listen for all incoming messages"""
                try:
                    async for message in websocket:
                        data = json.loads(message)
                        message_type = data.get("type", "unknown")
                        timestamp = data.get("timestamp", "")

                        if message_type == "connection_established":
                            print(f"🔗 {data['message']}")

                        elif message_type == "agent_info":
                            print(f"🤖 Agent: {data['agent_name']}")
                            print(f"📝 Description: {data['agent_description']}")
                            print(f"⚡ Capabilities: {', '.join(data['capabilities'])}")

                        elif message_type == "typing_indicator":
                            if data['is_typing']:
                                print(f"💭 {data['agent_name']} is typing...")
                            else:
                                print(f"✅ {data['agent_name']} finished typing")

                        elif message_type == "status":
                            stage = data.get('stage', '')
                            print(f"⏳ Status [{stage}]: {data['message']}")

                        elif message_type == "intelligence_analysis":
                            intelligence = data['data']
                            print(f"🧠 Intelligence Analysis:")
                            print(f"   😊 Emotional State: {intelligence['emotional_state']}")
                            print(f"   ⚡ Urgency: {intelligence['urgency_level']}")
                            print(f"   🎯 Intent: {intelligence['intent_category']}")
                            print(f"   📊 Confidence: {intelligence['confidence_score']:.2f}")
                            if intelligence.get('escalation_needed'):
                                print(f"   🚨 Escalation needed!")

                        elif message_type == "response_start":
                            print(f"🤖 Agent Response (streaming {data['estimated_words']} words):")
                            print("   ", end="", flush=True)

                        elif message_type == "response_chunk":
                            print(data['content'], end=" ", flush=True)
                            if data['is_final']:
                                print()  # New line after final chunk

                        elif message_type == "response_complete":
                            print(f"✅ Response complete")
                            context = data.get('customer_context', {})
                            if context:
                                print(f"📊 Customer Context: {context}")

                        elif message_type == "error":
                            print(f"❌ Error: {data['message']}")

                        elif message_type == "pong":
                            print(f"🏓 Pong received")

                        else:
                            print(f"📨 {message_type}: {data}")

                except websockets.exceptions.ConnectionClosed:
                    print("🔌 Connection closed")
                except Exception as e:
                    print(f"❌ Listen error: {e}")

            # Start the listener
            listen_task = asyncio.create_task(listen_for_messages())

            # Wait a moment for connection setup
            await asyncio.sleep(2)

            # Test different conversation scenarios
            test_messages = [
                {
                    "type": "message",
                    "message": "Hi there! I'm really frustrated with my API integration. It's been 3 hours and nothing works!",
                    "session_context": {
                        "page_url": "https://example.com/api-docs",
                        "referrer": "https://google.com/search?q=api+help",
                        "device_info": {"user_agent": "Mozilla/5.0", "screen_resolution": "1920x1080"}
                    }
                },
                {
                    "type": "message",
                    "message": "Actually, that was really helpful! I'm feeling much better now. Do you have any advanced tips?",
                    "session_context": {
                        "page_url": "https://example.com/api-docs/advanced",
                        "referrer": "https://example.com/api-docs"
                    }
                },
                {
                    "type": "message",
                    "message": "Wow, this is amazing! Can I get pricing for our team? We want to upgrade ASAP!",
                    "session_context": {
                        "page_url": "https://example.com/pricing",
                        "referrer": "https://example.com/features"
                    }
                }
            ]

            for i, message_data in enumerate(test_messages, 1):
                print(f"\n📤 Test Message {i}: {message_data['message'][:50]}...")
                await websocket.send(json.dumps(message_data))

                # Wait for response to complete
                await asyncio.sleep(8)

                # Send a ping to test keep-alive
                if i == 2:
                    print(f"\n🏓 Sending ping...")
                    await websocket.send(json.dumps({"type": "ping"}))
                    await asyncio.sleep(1)

            print(f"\n✅ All test messages sent successfully!")

            # Keep listening for a bit more
            await asyncio.sleep(5)

            # Cancel the listener
            listen_task.cancel()

    except websockets.exceptions.ConnectionRefused:
        print("❌ Connection refused. Make sure the server is running on port 8001")
    except Exception as e:
        print(f"❌ WebSocket test error: {e}")

    print("\n" + "=" * 70)
    print("🏆 Real-time WebSocket Chat Test Complete!")


if __name__ == "__main__":
    asyncio.run(test_websocket_chat())