import asyncio
import json
import logging
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Depends, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db as get_async_db
from app.core.rate_limiter import rate_limiter
from app.models.agent import Agent
from app.models.conversation import Conversation
from app.models.message import Message
from app.services.agent_token_service import verify_agent_session_token, AgentTokenError
from app.services.concierge_intelligence_service import concierge_intelligence
from app.services.customer_data_service import customer_data_service
from app.services.domain_expertise_service import domain_expertise_service
from app.services.intelligent_fallback_service import intelligent_fallback_service
from app.services.rag_service import RAGService
from app.services.database_service import db_service

router = APIRouter()

logger = logging.getLogger(__name__)

class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="User message (max 4000 characters)")
    conversation_id: Optional[str] = Field(None, max_length=100)
    user_id: Optional[str] = Field(None, max_length=100)
    session_context: Dict[str, Any] = Field(default_factory=dict)

    @validator('message')
    def validate_message(cls, v):
        """Validate message is not just whitespace"""
        if not v or not v.strip():
            raise ValueError('Message cannot be empty or whitespace only')
        return v.strip()

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    confidence_score: Optional[float] = None
    sources: List[Dict[str, Any]] = []
    grounding_mode: Optional[str] = None
    persona_applied: Optional[str] = None
    escalation_suggested: bool = False
    web_search_used: bool = False
    customer_context: Dict[str, Any] = {}

class ConversationHistory(BaseModel):
    id: int
    role: str  # user, assistant
    content: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

def _extract_agent_api_key(authorization_header: Optional[str], explicit_key: Optional[str]) -> Optional[str]:
    """Normalize agent API key from incoming headers."""
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()
    if authorization_header and authorization_header.startswith("Bearer "):
        return authorization_header.split(" ", 1)[1].strip()
    return None


async def _authorize_agent_request(
    agent_public_id: str,
    authorization: Optional[str],
    agent_api_key_header: Optional[str],
    agent_session_header: Optional[str]
) -> Agent:
    session_token_value: Optional[str] = None
    provided_key: Optional[str] = None
    token_agent_public_id: Optional[str] = None

    if agent_session_header and agent_session_header.strip():
        session_token_value = agent_session_header.strip()

    bearer_value: Optional[str] = None
    if authorization and authorization.startswith("Bearer "):
        bearer_value = authorization.split(" ", 1)[1].strip()

    if session_token_value:
        try:
            token_agent_public_id = verify_agent_session_token(session_token_value)
        except AgentTokenError as exc:
            raise HTTPException(status_code=401, detail=str(exc))

        if token_agent_public_id != agent_public_id:
            raise HTTPException(status_code=403, detail="Session token does not match agent")

        agent = await db_service.get_agent_by_public_id(token_agent_public_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        return agent

    if bearer_value:
        try:
            token_agent_public_id = verify_agent_session_token(bearer_value)
            if token_agent_public_id == agent_public_id:
                agent = await db_service.get_agent_by_public_id(token_agent_public_id)
                if not agent:
                    raise HTTPException(status_code=404, detail="Agent not found")
                return agent
        except AgentTokenError:
            provided_key = bearer_value

    provided_key = provided_key or _extract_agent_api_key(authorization, agent_api_key_header)

    agent = await db_service.get_agent_by_public_id(agent_public_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    if not provided_key or provided_key != agent.api_key:
        raise HTTPException(status_code=403, detail="Invalid or missing agent API key")

    return agent


async def _prepare_customer_context(
    agent: Agent,
    chat_data: ChatMessage,
    db: AsyncSession
) -> Dict[str, Any]:
    agent_id = agent.id
    visitor_id = chat_data.user_id or f"anon_{agent_id}"

    customer_profile_id = await _get_or_create_customer_profile(
        user_id=visitor_id,
        agent_id=agent_id,
        db=db
    )

    customer_context = await customer_data_service.get_customer_context(
        visitor_id=visitor_id,
        agent_id=agent_id,
        session_context=chat_data.session_context,
        db=db
    )

    product_context = await customer_data_service.get_product_context(agent_id, db)

    analysis = await customer_data_service.enrich_with_langextract(chat_data.message, customer_context)

    return {
        "visitor_id": visitor_id,
        "customer_profile_id": customer_profile_id,
        "customer_context": customer_context,
        "product_context": product_context,
        "analysis": analysis,
    }


def _sse_event(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


async def check_rate_limit_dependency(
    agent_public_id: str,
    chat_data: ChatMessage,
) -> None:
    """
    Dependency to check rate limiting before processing request.
    Rate limits per user_id or agent_public_id.
    """
    # Use user_id if provided, otherwise use agent_public_id
    identifier = chat_data.user_id or f"agent:{agent_public_id}"

    # Check rate limit (10 requests per minute for streaming)
    is_limited, requests_made, remaining = await rate_limiter.is_rate_limited(
        identifier=identifier,
        max_requests=10,  # 10 requests per minute
        window_seconds=60
    )

    if is_limited:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "requests_made": requests_made,
                "requests_remaining": remaining,
                "message": "Too many requests. Please wait before sending another message."
            }
        )


@router.post("/{agent_public_id}", response_model=ChatResponse)
async def chat_with_agent(
    agent_public_id: str,
    chat_data: ChatMessage,
    db: AsyncSession = Depends(get_async_db),
    authorization: Optional[str] = Header(None),
    agent_api_key_header: Optional[str] = Header(None, alias="X-Agent-API-Key"),
    agent_session_header: Optional[str] = Header(None, alias="X-Agent-Session")
):
    """Chat with an agent using revolutionary domain expertise"""

    # Check rate limit
    await check_rate_limit_dependency(agent_public_id, chat_data)

    agent = await _authorize_agent_request(
        agent_public_id,
        authorization,
        agent_api_key_header,
        agent_session_header,
    )

    try:
        session_details = await _prepare_customer_context(agent, chat_data, db)
        agent_id = agent.id
        customer_profile_id = session_details["customer_profile_id"]
        customer_context = session_details["customer_context"]
        product_context = session_details["product_context"]
        visitor_id = session_details["visitor_id"]
        analysis = session_details.get("analysis")
        insights_payload = analysis or {}

        # Use domain expertise service if enabled
        if agent.domain_expertise_enabled:
            domain_response = await domain_expertise_service.answer_with_domain_expertise(
                message=chat_data.message,
                agent=agent,
                organization=agent.organization,
                conversation_context=chat_data.session_context
            )

            response = ChatResponse(
                response=domain_response.answer,
                conversation_id=chat_data.conversation_id or f"conv_{agent.public_id}_{int(datetime.now().timestamp())}",
                confidence_score=domain_response.confidence_score,
                sources=domain_response.sources,
                grounding_mode=domain_response.grounding_mode,
                persona_applied=domain_response.persona_applied,
                escalation_suggested=domain_response.escalation_suggested,
                web_search_used=domain_response.web_search_used,
                customer_context={
                    "insights": insights_payload,
                    "sentiment": customer_context.sentiment,
                    "entities": customer_context.named_entities,
                    "user_intelligence": insights_payload,
                    "escalation": {
                        "created": domain_response.escalation_suggested,
                        "priority": None
                    }
                }
            )

        else:
            # Use enhanced customer data and intelligent fallback system
            rag_service = RAGService()

            # Determine fallback strategy for unknown/minimal context users
            fallback_strategy = intelligent_fallback_service.determine_fallback_strategy(
                customer_context=customer_context,
                product_context=product_context,
                user_message=chat_data.message
            )

            # Apply fallback strategy to enhance response
            fallback_response = intelligent_fallback_service.apply_fallback_strategy(
                strategy=fallback_strategy,
                customer_context=customer_context,
                product_context=product_context,
                user_message=chat_data.message,
                base_response=""  # We'll let RAG generate the base response
            )

            # Create context-enriched system prompt
            enhanced_system_prompt = intelligent_fallback_service.create_context_enriched_prompt(
                base_system_prompt=agent.system_prompt or "You are a helpful assistant.",
                customer_context=customer_context,
                product_context=product_context,
                fallback_response=fallback_response
            )

            # Generate enhanced response with all context
            rag_response = await rag_service.generate_response(
                query=chat_data.message,
                agent_id=agent_id,
                conversation_history=customer_context.conversation_history or [],
                system_prompt=enhanced_system_prompt,
                agent_config=agent.config or {},
                db_session=db
            )

            # Create or update customer profile based on this interaction
            await customer_data_service.create_or_update_customer_profile(
                visitor_id=visitor_id,
                agent_id=agent_id,
                session_context=chat_data.session_context,
                interaction_data={
                    "message_count": 1,
                    "interests": customer_context.current_interests,
                    "communication_style": customer_context.communication_style
                },
                db=db
            )

            response = ChatResponse(
                response=rag_response["response"],
                conversation_id=chat_data.conversation_id or f"conv_{agent.public_id}_{int(datetime.now().timestamp())}",
                confidence_score=customer_context.confidence_score,
                sources=rag_response.get("sources", []),
                grounding_mode="blended",
                persona_applied=f"Enhanced ({fallback_strategy.value})" if rag_response.get("personality_applied", False) else f"Fallback ({fallback_strategy.value})",
                web_search_used=False,
                customer_context={
                    "insights": insights_payload,
                    "sentiment": customer_context.sentiment,
                    "entities": customer_context.named_entities,
                    "user_intelligence": insights_payload,
                    "escalation": {"created": False, "priority": None}
                }
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
                "sources_count": len(response.sources),
                "sources": response.sources,
                "sentiment": customer_context.sentiment,
                "entities": customer_context.named_entities,
                "insights": insights_payload,
                "visitor_id": visitor_id,
                "customer_profile_id": customer_profile_id,
                "web_search_used": response.web_search_used
            },
            db=db
        )

        return response

    except Exception as e:
        print(f"Chat error: {e}")
        # Return fallback response
        return ChatResponse(
            response="I apologize, but I'm having trouble processing your request right now. Please try again or contact support.",
            conversation_id=chat_data.conversation_id or f"conv_{agent.public_id}_error",
            confidence_score=0.2,
            escalation_suggested=True,
            customer_context={"escalation": {"created": True}, "insights": None}
        )

@router.post("/{agent_public_id}/stream")
async def chat_with_agent_stream(
    agent_public_id: str,
    chat_data: ChatMessage,
    db: AsyncSession = Depends(get_async_db),
    authorization: Optional[str] = Header(None),
    agent_api_key_header: Optional[str] = Header(None, alias="X-Agent-API-Key"),
    agent_session_header: Optional[str] = Header(None, alias="X-Agent-Session")
):
    # Check rate limit
    await check_rate_limit_dependency(agent_public_id, chat_data)

    agent = await _authorize_agent_request(
        agent_public_id,
        authorization,
        agent_api_key_header,
        agent_session_header,
    )

    session_details = await _prepare_customer_context(agent, chat_data, db)
    agent_id = agent.id
    visitor_id = session_details["visitor_id"]
    customer_context = session_details["customer_context"]
    product_context = session_details["product_context"]
    analysis = session_details.get("analysis")
    insights_payload = analysis or {}

    conversation_id = chat_data.conversation_id or f"conv_{agent.public_id}_{int(datetime.now().timestamp())}"

    async def event_generator():
        aggregated_response = ""
        sources_count = 0
        try:
            yield _sse_event({"type": "conversation", "conversation_id": conversation_id})

            if agent.domain_expertise_enabled:
                domain_response = await domain_expertise_service.answer_with_domain_expertise(
                    message=chat_data.message,
                    agent=agent,
                    organization=agent.organization,
                    conversation_context=chat_data.session_context
                )

                aggregated_response = domain_response.answer or ""
                sources = domain_response.sources or []
                sources_count = len(sources)

                if sources:
                    yield _sse_event({
                        "type": "metadata",
                        "sources": sources,
                        "conversation_id": conversation_id
                    })

                if aggregated_response:
                    yield _sse_event({"type": "content", "content": aggregated_response})

                await customer_data_service.create_or_update_customer_profile(
                    visitor_id=visitor_id,
                    agent_id=agent_id,
                    session_context=chat_data.session_context,
                    interaction_data={
                        "message_count": 1,
                        "interests": customer_context.current_interests,
                        "communication_style": customer_context.communication_style
                    },
                    db=db
                )

                await _save_conversation_message(
                    agent_id=agent_id,
                    conversation_id=conversation_id,
                    user_message=chat_data.message,
                    ai_response=aggregated_response,
                    metadata={
                        "confidence_score": domain_response.confidence_score,
                        "persona_applied": domain_response.persona_applied,
                        "sources_count": sources_count,
                        "sources": sources,
                        "sentiment": customer_context.sentiment,
                        "entities": customer_context.named_entities,
                        "insights": insights_payload,
                        "visitor_id": visitor_id,
                        "customer_profile_id": session_details.get("customer_profile_id"),
                        "web_search_used": domain_response.web_search_used,
                    },
                    db=db
                )

                yield _sse_event({
                    "type": "done",
                    "conversation_id": conversation_id,
                    "confidence_score": domain_response.confidence_score,
                    "persona_applied": domain_response.persona_applied,
                    "sources_count": sources_count,
                    "sources": sources,
                    "customer_context": {
                        "escalation": {
                            "created": domain_response.escalation_suggested,
                            "priority": None
                        },
                        "insights": insights_payload,
                        "sentiment": customer_context.sentiment,
                        "entities": customer_context.named_entities,
                        "user_intelligence": insights_payload
                    }
                })

            else:
                rag_service = RAGService()

                fallback_strategy = intelligent_fallback_service.determine_fallback_strategy(
                    customer_context=customer_context,
                    product_context=product_context,
                    user_message=chat_data.message
                )

                fallback_response = intelligent_fallback_service.apply_fallback_strategy(
                    strategy=fallback_strategy,
                    customer_context=customer_context,
                    product_context=product_context,
                    user_message=chat_data.message,
                    base_response=""
                )

                enhanced_system_prompt = intelligent_fallback_service.create_context_enriched_prompt(
                    base_system_prompt=agent.system_prompt or "You are a helpful assistant.",
                    customer_context=customer_context,
                    product_context=product_context,
                    fallback_response=fallback_response
                )

                persona_label = (
                    f"Enhanced ({fallback_strategy.value})"
                    if getattr(fallback_strategy, "value", None)
                    else str(fallback_strategy)
                )

                collected_sources: List[Dict[str, Any]] = []

                async for chunk in rag_service.generate_streaming_response(
                    query=chat_data.message,
                    agent_id=agent_id,
                    conversation_history=customer_context.conversation_history or [],
                    system_prompt=enhanced_system_prompt,
                    agent_config=agent.config or {},
                    db_session=db
                ):
                    if chunk["type"] == "metadata":
                        sources = chunk.get("sources", [])
                        sources_count = len(sources)
                        if sources:
                            collected_sources = sources
                        chunk["conversation_id"] = conversation_id
                        yield _sse_event(chunk)
                    elif chunk["type"] == "content":
                        aggregated_response += chunk.get("content", "")
                        yield _sse_event(chunk)
                    elif chunk["type"] == "done":
                        await customer_data_service.create_or_update_customer_profile(
                            visitor_id=visitor_id,
                            agent_id=agent_id,
                            session_context=chat_data.session_context,
                            interaction_data={
                                "message_count": 1,
                                "interests": customer_context.current_interests,
                                "communication_style": customer_context.communication_style
                            },
                            db=db
                        )

                        await _save_conversation_message(
                            agent_id=agent_id,
                            conversation_id=conversation_id,
                            user_message=chat_data.message,
                            ai_response=aggregated_response,
                            metadata={
                                "confidence_score": customer_context.confidence_score,
                                "persona_applied": persona_label,
                                "sources_count": sources_count,
                                "sources": collected_sources,
                                "sentiment": customer_context.sentiment,
                                "entities": customer_context.named_entities,
                                "insights": insights_payload,
                                "visitor_id": visitor_id,
                                "customer_profile_id": session_details.get("customer_profile_id"),
                                "web_search_used": chunk.get("web_search_used")
                            },
                            db=db
                        )

                        chunk.update({
                            "conversation_id": conversation_id,
                            "confidence_score": customer_context.confidence_score,
                            "persona_applied": persona_label,
                            "sources_count": sources_count,
                            "sources": collected_sources,
                            "customer_context": {
                                "escalation": {
                                    "created": False,
                                    "priority": None
                                },
                                "insights": insights_payload,
                                "sentiment": customer_context.sentiment,
                                "entities": customer_context.named_entities,
                                "user_intelligence": insights_payload
                            }
                        })

                        yield _sse_event(chunk)
                    else:
                        yield _sse_event(chunk)

        except asyncio.TimeoutError:
            yield _sse_event({
                "type": "error",
                "message": "Request timeout: Streaming response exceeded 30 second limit"
            })
        except Exception as exc:
            yield _sse_event({"type": "error", "message": str(exc)})

    async def timeout_wrapper():
        """Wrap event generator with timeout protection"""
        try:
            async with asyncio.timeout(30):  # 30 second timeout
                async for event in event_generator():
                    yield event
        except asyncio.TimeoutError:
            yield _sse_event({
                "type": "error",
                "message": "Request timeout: Streaming response exceeded 30 second limit"
            })

    return StreamingResponse(
        timeout_wrapper(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"}
    )


@router.get("/{agent_public_id}/conversations/{conversation_id}", response_model=List[ConversationHistory])
async def get_conversation_history(
    agent_public_id: str,
    conversation_id: str,
    db: AsyncSession = Depends(get_async_db),
    authorization: Optional[str] = Header(None),
    agent_api_key_header: Optional[str] = Header(None, alias="X-Agent-API-Key"),
    agent_session_header: Optional[str] = Header(None, alias="X-Agent-Session")
):
    """Get conversation history for a specific conversation"""
    agent = await _authorize_agent_request(
        agent_public_id,
        authorization,
        agent_api_key_header,
        agent_session_header,
    )

    # Query conversation to verify it belongs to this agent
    result = await db.execute(
        select(Conversation)
        .where(
            Conversation.session_id == conversation_id,
            Conversation.agent_id == agent.id
        )
    )
    conversation = result.scalar_one_or_none()

    if not conversation:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation '{conversation_id}' not found for this agent"
        )

    # Query all messages in chronological order
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.created_at.asc())
    )
    messages = result.scalars().all()

    # Format response
    return [
        ConversationHistory(
            id=msg.id,
            role=msg.role,
            content=msg.content,
            timestamp=msg.created_at.isoformat() if msg.created_at else None,
            metadata=msg.msg_metadata
        )
        for msg in messages
    ]

# Helper functions
async def _get_or_create_customer_profile(user_id: str, agent_id: int, db: AsyncSession) -> int:
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
    db: AsyncSession
):
    """Save conversation messages to database"""
    try:
        metadata = metadata or {}
        visitor_id = metadata.get("visitor_id")
        sources = metadata.get("sources") or []
        sentiment = metadata.get("sentiment")
        entities = metadata.get("entities")
        insights = metadata.get("insights")
        confidence_score = metadata.get("confidence_score")
        customer_profile_id = metadata.get("customer_profile_id")

        result = await db.execute(
            select(Conversation).where(
                Conversation.session_id == conversation_id,
                Conversation.agent_id == agent_id
            )
        )
        conversation = result.scalar_one_or_none()

        if not conversation:
            conversation = Conversation(
                agent_id=agent_id,
                session_id=conversation_id,
                conv_metadata={
                    "visitor_id": visitor_id,
                    "started_at": datetime.utcnow().isoformat()
                }
            )
            db.add(conversation)
            await db.flush()

        if customer_profile_id and not conversation.customer_profile_id:
            conversation.customer_profile_id = customer_profile_id

        def estimate_tokens(text: str) -> int:
            if not text:
                return 0
            return len(re.findall(r"\w+", text))

        user_tokens = estimate_tokens(user_message)
        assistant_tokens = estimate_tokens(ai_response)

        conv_meta = conversation.conv_metadata or {}
        conv_meta.setdefault("visitor_id", visitor_id)
        conv_meta.setdefault("total_messages", 0)
        conv_meta.setdefault("total_tokens", 0)
        conv_meta["total_messages"] += 2
        conv_meta["total_tokens"] += user_tokens + assistant_tokens
        conv_meta["last_user_message"] = user_message[:500]
        conv_meta["last_assistant_message"] = ai_response[:500]
        if confidence_score is not None:
            conv_meta["last_confidence_score"] = confidence_score
        if sentiment:
            conv_meta["last_sentiment"] = sentiment
        if entities:
            conv_meta["last_entities"] = entities
        if sources:
            conv_meta["sources"] = sources
        if insights:
            conv_meta["insights"] = insights
        if metadata.get("persona_applied"):
            conv_meta["persona_applied"] = metadata["persona_applied"]
        if metadata.get("web_search_used") is not None:
            conv_meta["web_search_used"] = metadata["web_search_used"]

        conv_meta["last_updated_at"] = datetime.utcnow().isoformat()
        conversation.conv_metadata = conv_meta
        conversation.updated_at = datetime.utcnow()

        user_msg_metadata = {
            "role": "user",
            "sentiment": sentiment,
            "entities": entities,
            "visitor_id": visitor_id
        }

        assistant_msg_metadata = {
            "role": "assistant",
            "confidence_score": confidence_score,
            "sources": sources,
            "insights": insights,
            "web_search_used": metadata.get("web_search_used"),
            "persona_applied": metadata.get("persona_applied")
        }

        user_message_row = Message(
            conversation_id=conversation.id,
            role="user",
            content=user_message,
            msg_metadata=user_msg_metadata,
            token_count=user_tokens
        )

        assistant_message_row = Message(
            conversation_id=conversation.id,
            role="assistant",
            content=ai_response,
            msg_metadata=assistant_msg_metadata,
            token_count=assistant_tokens
        )

        db.add_all([user_message_row, assistant_message_row])
        await db.commit()

    except Exception as e:
        logger.error(f"Error saving conversation {conversation_id}: {e}")
