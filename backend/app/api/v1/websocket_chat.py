"""
Real-time WebSocket Chat with Intelligent Concierge Agent
World-class chat experience with streaming responses and live intelligence analysis.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.database import get_db
from ...services.database_service import db_service
from ...services.memory_service import memory_service
from ...services.intelligent_rag_service import intelligent_rag_service
from ...services.user_intelligence_service import user_intelligence_service
from ...services.gemini_service import gemini_service


class ConnectionManager:
    """Manage WebSocket connections with session tracking"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, session_id: str, agent_id: int, visitor_id: str):
        """Accept WebSocket connection and initialize session"""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.user_sessions[session_id] = {
            "agent_id": agent_id,
            "visitor_id": visitor_id,
            "connected_at": datetime.now(),
            "last_activity": datetime.now(),
            "typing": False,
            "emotional_state": "neutral",
            "conversation_context": []
        }

        # Send welcome message with connection status
        await self.send_system_message(session_id, {
            "type": "connection_established",
            "message": "Connected to intelligent concierge agent",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        })

    def disconnect(self, session_id: str):
        """Remove connection and clean up session"""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.user_sessions:
            del self.user_sessions[session_id]

    async def send_personal_message(self, session_id: str, message: dict):
        """Send message to specific session"""
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            try:
                await websocket.send_text(json.dumps(message))
                return True
            except Exception:
                # Connection closed, clean up
                self.disconnect(session_id)
                return False
        return False

    async def send_system_message(self, session_id: str, message: dict):
        """Send system message with enhanced formatting"""
        system_message = {
            **message,
            "is_system": True,
            "timestamp": datetime.now().isoformat()
        }
        return await self.send_personal_message(session_id, system_message)

    async def send_typing_indicator(self, session_id: str, is_typing: bool, agent_name: str = "Agent"):
        """Send typing indicator"""
        await self.send_personal_message(session_id, {
            "type": "typing_indicator",
            "is_typing": is_typing,
            "agent_name": agent_name,
            "timestamp": datetime.now().isoformat()
        })

    async def send_intelligence_update(self, session_id: str, intelligence_data: dict):
        """Send real-time intelligence analysis (for admin/debug)"""
        await self.send_personal_message(session_id, {
            "type": "intelligence_analysis",
            "data": intelligence_data,
            "timestamp": datetime.now().isoformat()
        })

    def update_session_context(self, session_id: str, context_update: dict):
        """Update session context"""
        if session_id in self.user_sessions:
            self.user_sessions[session_id].update(context_update)
            self.user_sessions[session_id]["last_activity"] = datetime.now()


# Global connection manager
connection_manager = ConnectionManager()


class StreamingIntelligentRAG:
    """Streaming version of Intelligent RAG with real-time updates"""

    def __init__(self):
        self.chunk_size = 50  # Words per chunk for streaming

    async def stream_intelligent_response(
        self,
        session_id: str,
        message: str,
        agent_id: int,
        visitor_id: str,
        session_context: Optional[Dict[str, Any]] = None
    ):
        """Generate streaming response with live intelligence analysis"""

        try:
            # Step 1: Send typing indicator
            await connection_manager.send_typing_indicator(session_id, True, "Concierge Agent")

            # Step 2: Send analysis status
            await connection_manager.send_personal_message(session_id, {
                "type": "status",
                "message": "Analyzing your message...",
                "stage": "analysis",
                "timestamp": datetime.now().isoformat()
            })

            # Step 3: Get agent and verify
            agent = await db_service.get_agent_by_id(agent_id)
            if not agent or not agent.is_active:
                await connection_manager.send_personal_message(session_id, {
                    "type": "error",
                    "message": "Agent not available",
                    "timestamp": datetime.now().isoformat()
                })
                return

            # Step 4: Get or create customer profile and conversation
            conversation_session_id = f"{visitor_id}_{agent_id}"
            conversation = await db_service.get_conversation_by_session(agent_id, conversation_session_id)

            if not conversation:
                conversation = await db_service.create_conversation(
                    agent_id=agent_id,
                    session_id=conversation_session_id,
                    metadata={"visitor_id": visitor_id, "session_context": session_context}
                )

            # Get or create customer profile via memory service
            customer_profile = await memory_service.get_or_create_customer_profile(
                visitor_id=visitor_id,
                agent_id=agent_id,
                initial_context=session_context or {}
            )

            # Step 5: Real-time intelligence analysis
            await connection_manager.send_personal_message(session_id, {
                "type": "status",
                "message": "Understanding your emotional state and intent...",
                "stage": "intelligence",
                "timestamp": datetime.now().isoformat()
            })

            # Get conversation history
            conversation_history = await db_service.get_conversation_messages(conversation.id)
            history_for_analysis = [
                {"role": msg.role, "content": msg.content}
                for msg in conversation_history[-10:]  # Last 10 messages
            ]

            # Parallel intelligence analysis
            user_analysis = await user_intelligence_service.analyze_user_message(
                message=message,
                customer_profile_id=customer_profile.id,
                conversation_history=history_for_analysis,
                session_context=session_context or {}
            )

            # Send intelligence update to client (for premium experience)
            await connection_manager.send_intelligence_update(session_id, {
                "emotional_state": user_analysis.emotional_state.value,
                "urgency_level": user_analysis.urgency_level.value,
                "intent_category": user_analysis.intent_category.value,
                "confidence_score": user_analysis.confidence_score,
                "key_topics": user_analysis.key_topics,
                "escalation_needed": any("escalat" in action.lower() for action in user_analysis.next_best_actions)
            })

            # Update session context with intelligence
            connection_manager.update_session_context(session_id, {
                "emotional_state": user_analysis.emotional_state.value,
                "urgency_level": user_analysis.urgency_level.value,
                "intent_category": user_analysis.intent_category.value
            })

            # Step 6: Generate response using intelligent RAG
            await connection_manager.send_personal_message(session_id, {
                "type": "status",
                "message": "Crafting personalized response...",
                "stage": "generation",
                "timestamp": datetime.now().isoformat()
            })

            # Get RAG response (we'll stream this)
            rag_response = await intelligent_rag_service.generate_intelligent_response(
                query=message,
                agent_id=agent_id,
                conversation_id=conversation.id,
                visitor_id=visitor_id,
                session_context=session_context or {},
                system_prompt=agent.system_prompt,
                agent_config=agent.config or {},
                agent_profile=agent
            )

            # Step 7: Stream the response word by word
            await self._stream_response_content(
                session_id=session_id,
                response_content=rag_response["content"],
                conversation_id=conversation.id,
                customer_context=rag_response.get("customer_context", {}),
                user_analysis=user_analysis
            )

            # Step 8: Save message to conversation
            await db_service.create_message(
                conversation_id=conversation.id,
                role="user",
                content=message
            )

            await db_service.create_message(
                conversation_id=conversation.id,
                role="assistant",
                content=rag_response["content"]
            )

            # Step 9: Send completion signal
            await connection_manager.send_typing_indicator(session_id, False)
            await connection_manager.send_personal_message(session_id, {
                "type": "response_complete",
                "conversation_id": conversation.id,
                "customer_context": rag_response.get("customer_context", {}),
                "model_info": {
                    "model": rag_response.get("model", "gemini-2.0-flash-exp"),
                    "usage": rag_response.get("usage", {})
                },
                "timestamp": datetime.now().isoformat()
            })

            # Notify client if an escalation was created
            esc = (rag_response.get("customer_context", {}) or {}).get("escalation")
            if esc and esc.get("created"):
                await connection_manager.send_personal_message(session_id, {
                    "type": "escalation",
                    "status": "created",
                    "escalation_id": esc.get("id"),
                    "priority": esc.get("priority"),
                    "conversation_id": conversation.id,
                    "timestamp": datetime.now().isoformat()
                })

        except Exception as e:
            # Send error with typing indicator off
            await connection_manager.send_typing_indicator(session_id, False)
            await connection_manager.send_personal_message(session_id, {
                "type": "error",
                "message": f"I apologize, but I encountered an issue processing your message. Please try again.",
                "error_id": f"err_{int(time.time())}",
                "timestamp": datetime.now().isoformat()
            })
            print(f"Streaming chat error: {e}")

    async def _stream_response_content(
        self,
        session_id: str,
        response_content: str,
        conversation_id: int,
        customer_context: dict,
        user_analysis: Any
    ):
        """Stream response content word by word for natural typing effect"""

        words = response_content.split()
        current_chunk = ""

        # Start response stream
        await connection_manager.send_personal_message(session_id, {
            "type": "response_start",
            "conversation_id": conversation_id,
            "estimated_words": len(words),
            "timestamp": datetime.now().isoformat()
        })

        # Stream words in chunks
        for i, word in enumerate(words):
            current_chunk += word + " "

            # Send chunk every N words or at the end
            if (i + 1) % self.chunk_size == 0 or i == len(words) - 1:
                await connection_manager.send_personal_message(session_id, {
                    "type": "response_chunk",
                    "content": current_chunk.strip(),
                    "chunk_index": i // self.chunk_size,
                    "is_final": i == len(words) - 1,
                    "timestamp": datetime.now().isoformat()
                })
                current_chunk = ""

                # Add natural typing delay based on urgency
                urgency_delay = {
                    "critical": 0.05,  # Very fast for urgent matters
                    "high": 0.08,
                    "medium": 0.12,
                    "low": 0.15
                }.get(user_analysis.urgency_level.value, 0.12)

                await asyncio.sleep(urgency_delay)


# Initialize streaming service
streaming_rag = StreamingIntelligentRAG()


async def handle_websocket_chat(websocket: WebSocket, agent_id: int):
    """Main WebSocket chat handler with full intelligence integration"""

    # Extract session info from query params or generate
    visitor_id = None
    session_id = None

    try:
        # Get connection info from query parameters
        query_params = dict(websocket.query_params)
        visitor_id = query_params.get("visitor_id", f"visitor_{int(time.time())}")
        session_id = f"{visitor_id}_{agent_id}_{int(time.time())}"

        # Connect to session manager
        await connection_manager.connect(websocket, session_id, agent_id, visitor_id)

        # Send welcome message with agent info
        agent = await db_service.get_agent_by_id(agent_id)
        if agent:
            await connection_manager.send_personal_message(session_id, {
                "type": "agent_info",
                "agent_name": agent.name,
                "agent_description": agent.description,
                "capabilities": [
                    "Emotional intelligence analysis",
                    "Real-time personalization",
                    "Memory across conversations",
                    "Contextual understanding",
                    "Instant knowledge retrieval"
                ],
                "timestamp": datetime.now().isoformat()
            })

        # Main message loop
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message_data = json.loads(data)

                message_type = message_data.get("type", "message")

                if message_type == "message":
                    # Handle chat message with streaming response
                    message_content = message_data.get("message", "")
                    session_context = message_data.get("session_context", {})

                    if message_content.strip():
                        # Process with streaming intelligent RAG
                        await streaming_rag.stream_intelligent_response(
                            session_id=session_id,
                            message=message_content,
                            agent_id=agent_id,
                            visitor_id=visitor_id,
                            session_context=session_context
                        )

                elif message_type == "typing":
                    # Handle typing indicators from user
                    is_typing = message_data.get("is_typing", False)
                    connection_manager.update_session_context(session_id, {"user_typing": is_typing})

                elif message_type == "ping":
                    # Handle keep-alive pings
                    await connection_manager.send_personal_message(session_id, {
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })

                elif message_type == "session_update":
                    # Handle session context updates (page changes, etc.)
                    context_update = message_data.get("context", {})
                    connection_manager.update_session_context(session_id, context_update)

            except json.JSONDecodeError:
                await connection_manager.send_personal_message(session_id, {
                    "type": "error",
                    "message": "Invalid message format. Please send valid JSON.",
                    "timestamp": datetime.now().isoformat()
                })

            except Exception as inner_e:
                print(f"Inner WebSocket error: {inner_e}")
                await connection_manager.send_personal_message(session_id, {
                    "type": "error",
                    "message": "An error occurred processing your message.",
                    "timestamp": datetime.now().isoformat()
                })

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session {session_id}")
        if session_id:
            connection_manager.disconnect(session_id)

    except Exception as e:
        print(f"WebSocket error: {e}")
        if session_id:
            connection_manager.disconnect(session_id)
        try:
            await websocket.close()
        except:
            pass
