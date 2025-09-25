from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import json

from app.core.database import get_db
from app.models.agent import Agent
from app.models.conversation import Conversation, Message
from app.services.domain_expertise_service import domain_expertise_service
from app.services.concierge_intelligence_service import concierge_intelligence
from app.services.customer_profile_service import customer_profile_service

router = APIRouter()

class ChatMessage(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_context: Dict[str, Any] = {}

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    confidence_score: Optional[float] = None
    sources: List[Dict[str, Any]] = []
    grounding_mode: Optional[str] = None
    persona_applied: Optional[str] = None
    escalation_suggested: bool = False
    web_search_used: bool = False

class ConversationHistory(BaseModel):
    id: int
    role: str  # user, assistant
    content: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

@router.post("/{agent_id}", response_model=ChatResponse)
async def chat_with_agent(
    agent_id: int,
    chat_data: ChatMessage,
    db: Session = Depends(get_db)
):
    """Chat with an agent using revolutionary domain expertise"""

    # Get agent with organization
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    try:
        # Create or get customer profile
        customer_profile_id = await _get_or_create_customer_profile(
            user_id=chat_data.user_id or f"anon_{agent_id}",
            agent_id=agent_id,
            db=db
        )

        # Mock organization object for domain expertise service
        class MockOrganization:
            def __init__(self, org_id):
                self.id = org_id

        organization = MockOrganization(agent.organization_id)

        # Use domain expertise service if enabled
        if agent.domain_expertise_enabled:
            domain_response = await domain_expertise_service.answer_with_domain_expertise(
                message=chat_data.message,
                agent=agent,
                organization=organization,
                conversation_context=chat_data.session_context
            )

            response = ChatResponse(
                response=domain_response.answer,
                conversation_id=chat_data.conversation_id or f"conv_{agent_id}_{int(datetime.now().timestamp())}",
                confidence_score=domain_response.confidence_score,
                sources=domain_response.sources,
                grounding_mode=domain_response.grounding_mode,
                persona_applied=domain_response.persona_applied,
                escalation_suggested=domain_response.escalation_suggested,
                web_search_used=domain_response.web_search_used
            )

        else:
            # Fallback to concierge intelligence service
            concierge_case = await concierge_intelligence.build_concierge_case(
                customer_profile_id=customer_profile_id,
                current_message=chat_data.message,
                session_context=chat_data.session_context,
                agent_id=agent_id
            )

            # Generate response using existing concierge intelligence
            llm_context = await concierge_intelligence.generate_llm_context(concierge_case)

            # Simple response generation (in production, integrate with your LLM service)
            from app.services.gemini_service import gemini_service
            ai_response = await gemini_service.generate_response(
                prompt=f"Context: {llm_context}\n\nUser: {chat_data.message}",
                system_prompt=agent.system_prompt or "You are a helpful assistant.",
                temperature=0.7
            )

            response = ChatResponse(
                response=ai_response,
                conversation_id=chat_data.conversation_id or f"conv_{agent_id}_{int(datetime.now().timestamp())}",
                confidence_score=0.7,
                sources=[],
                grounding_mode="blended",
                persona_applied="Standard"
            )

        # Save conversation (simplified)
        await _save_conversation_message(
            agent_id=agent_id,
            conversation_id=response.conversation_id,
            user_message=chat_data.message,
            ai_response=response.response,
            metadata={
                "confidence_score": response.confidence_score,
                "persona_applied": response.persona_applied,
                "sources_count": len(response.sources)
            },
            db=db
        )

        return response

    except Exception as e:
        print(f"Chat error: {e}")
        # Return fallback response
        return ChatResponse(
            response="I apologize, but I'm having trouble processing your request right now. Please try again or contact support.",
            conversation_id=chat_data.conversation_id or f"conv_{agent_id}_error",
            confidence_score=0.2,
            escalation_suggested=True
        )

@router.get("/{agent_id}/conversations/{conversation_id}", response_model=List[ConversationHistory])
async def get_conversation_history(
    agent_id: int,
    conversation_id: str,
    db: AsyncSession = Depends(get_async_session)
):
    # TODO: Get conversation history
    return [
        {
            "id": 1,
            "role": "user",
            "content": "Hello",
            "timestamp": "2024-01-01T12:00:00Z"
        },
        {
            "id": 2,
            "role": "assistant",
            "content": "Hi! How can I help you?",
            "timestamp": "2024-01-01T12:00:01Z"
        }
    ]

@router.websocket("/{agent_id}/ws")
async def websocket_chat(
    websocket: WebSocket,
    agent_id: int
):
    await websocket.accept()
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)

            # TODO: Process message and generate streaming response

            # Send response back
            response = {
                "type": "message",
                "content": f"Echo from agent {agent_id}: {message_data.get('message', '')}",
                "conversation_id": "conv_ws_123"
            }
            await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for agent {agent_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()


# Helper functions
async def _get_or_create_customer_profile(user_id: str, agent_id: int, db: Session) -> int:
    """Get or create customer profile for chat user"""
    try:
        # This would integrate with your customer profile service
        # For now, return a mock profile ID
        return hash(f"{user_id}_{agent_id}") % 1000000
    except Exception:
        return 0


async def _save_conversation_message(
    agent_id: int,
    conversation_id: str,
    user_message: str,
    ai_response: str,
    metadata: Dict[str, Any],
    db: Session
):
    """Save conversation messages to database"""
    try:
        # This would save to your conversation/message tables
        # For now, just print for debugging
        print(f"Saving conversation {conversation_id}: User: {user_message[:50]}... -> AI: {ai_response[:50]}...")
    except Exception as e:
        print(f"Error saving conversation: {e}")