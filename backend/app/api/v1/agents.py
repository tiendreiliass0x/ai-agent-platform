"""
API endpoints for agent management.
"""

from typing import List, Dict, Any, Optional, Union
from fastapi import APIRouter, Depends, HTTPException, status, WebSocket
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from ...core.database import get_db
from ...core.auth import get_current_user
from ...services.database_service import db_service
from ...services.agent_creation_service import agent_creation_service, AgentType, IndustryType
from ...models.agent import Agent, AgentTier, DomainExpertiseType
from ...models.user import User
from ...services.escalation_service import escalation_service

# Import intelligent RAG service dynamically to avoid dependency issues
try:
    from ...services.intelligent_rag_service import intelligent_rag_service
except ImportError:
    intelligent_rag_service = None

# Import WebSocket handler dynamically
try:
    from .websocket_chat import handle_websocket_chat
except ImportError:
    handle_websocket_chat = None

router = APIRouter()


class AgentResponse(BaseModel):
    id: int
    name: str
    description: str
    system_prompt: str
    is_active: bool
    config: Dict[str, Any]
    widget_config: Dict[str, Any]
    api_key: Optional[str]
    created_at: str
    updated_at: Optional[str]
    tier: Optional[str]
    domain_expertise_type: Optional[str]
    domain_expertise_enabled: bool
    personality_profile: Dict[str, Any]
    expertise_level: float
    domain_knowledge_sources: List[Any]
    web_search_enabled: bool
    custom_training_data: Dict[str, Any]
    expert_context: Optional[str]
    tool_policy: Dict[str, Any]
    grounding_mode: str

    class Config:
        from_attributes = True


def serialize_agent(agent: Agent) -> AgentResponse:
    domain_sources = getattr(agent, "domain_knowledge_sources", []) or []
    if isinstance(domain_sources, list):
        normalized_sources: List[int] = []
        for source in domain_sources:
            try:
                normalized_sources.append(int(source))
            except (TypeError, ValueError):
                continue
        domain_sources = normalized_sources
    else:
        domain_sources = []

    return AgentResponse(
        id=agent.id,
        name=agent.name,
        description=agent.description or "",
        system_prompt=agent.system_prompt or "",
        is_active=agent.is_active,
        config=agent.config or {},
        widget_config=agent.widget_config or {},
        api_key=agent.api_key,
        created_at=agent.created_at.isoformat() if agent.created_at else "",
        updated_at=agent.updated_at.isoformat() if agent.updated_at else None,
        tier=agent.tier.value if getattr(agent, "tier", None) else None,
        domain_expertise_type=agent.domain_expertise_type.value if getattr(agent, "domain_expertise_type", None) else None,
        domain_expertise_enabled=getattr(agent, "domain_expertise_enabled", False) or False,
        personality_profile=getattr(agent, "personality_profile", {}) or {},
        expertise_level=float(getattr(agent, "expertise_level", 0.7) or 0.7),
        domain_knowledge_sources=domain_sources,
        web_search_enabled=getattr(agent, "web_search_enabled", False) or False,
        custom_training_data=getattr(agent, "custom_training_data", {}) or {},
        expert_context=getattr(agent, "expert_context", None),
        tool_policy=getattr(agent, "tool_policy", {}) or {},
        grounding_mode=getattr(agent, "grounding_mode", "blended") or "blended"
    )


BASE_PERSONAS: Dict[str, Dict[str, Any]] = {
    "sales_rep": {
        "name": "Sales Representative",
        "system_prompt": (
            "You are a senior B2B sales representative. Diagnose the prospect's needs, "
            "articulate differentiated value, and recommend the next step. Always ground claims in cited sources."
        ),
        "tactics": {
            "style": "executive",
            "steps": [
                "Qualify the prospect's role, pain, and timeline",
                "Frame business impact with concise value bullets",
                "Cite relevant proof points or customer examples",
                "Close with a clear next step or CTA"
            ],
            "tips": [
                "Mirror the customer's language and priorities",
                "Handle objections with empathy and data",
                "Highlight ROI or cost savings when possible"
            ]
        }
    },
    "solution_engineer": {
        "name": "Solutions Engineer",
        "system_prompt": (
            "You are a pragmatic solutions engineer. Map requirements to architecture, outline trade-offs, "
            "and provide implementation guidance grounded in cited sources."
        ),
        "tactics": {
            "style": "technical",
            "steps": [
                "Clarify goals, constraints, and existing stack",
                "Recommend an architecture with annotated diagram steps",
                "Surface trade-offs, risks, and mitigation strategies",
                "Outline a minimal viable implementation plan"
            ],
            "tips": [
                "Cite limits or SLAs for any critical component",
                "Offer integration checklists or pseudo-code when helpful"
            ]
        }
    },
    "support_expert": {
        "name": "Support Expert",
        "system_prompt": (
            "You are a tier-2 support expert. Diagnose issues methodically, confirm reproduction, "
            "and present precise resolutions. Cite sources for known fixes or knowledge base articles."
        ),
        "tactics": {
            "style": "concise",
            "steps": [
                "Confirm environment and version details",
                "Gather relevant logs or telemetry",
                "Identify known issues or root causes",
                "Provide step-by-step resolution and prevention tips"
            ],
            "tips": [
                "If uncertain, offer hypotheses with required validation",
                "List escalation criteria when self-service is insufficient"
            ]
        }
    },
    "domain_specialist": {
        "name": "Domain Specialist",
        "system_prompt": (
            "You are a domain mentor. Provide best practices, tips, and actionable insights drawn from curated knowledge "
            "and trusted sources."
        ),
        "tactics": {
            "style": "friendly",
            "steps": [
                "Diagnose the user's current level",
                "Share practical tips and guardrails",
                "Suggest next actions or resources",
                "Offer advanced tricks when appropriate"
            ],
            "tips": [
                "Use relatable analogies when introducing new concepts",
                "Encourage experimentation with clear boundaries"
            ]
        }
    }
}

PERSONA_ENUM_MAP: Dict[str, DomainExpertiseType] = {
    "sales_rep": DomainExpertiseType.sales_rep,
    "solution_engineer": DomainExpertiseType.solution_engineer,
    "support_expert": DomainExpertiseType.support_expert,
    "domain_specialist": DomainExpertiseType.domain_specialist,
    "product_expert": DomainExpertiseType.product_expert,
}

GROUNDING_MODES = {"strict", "blended"}


class EnhancedAgentResponse(BaseModel):
    agent: AgentResponse
    embed_code: str
    setup_guide: Dict[str, Any]
    optimization_applied: bool
    template_used: Optional[str]
    recommendations: List[Dict[str, Any]]


class AgentCreate(BaseModel):
    name: str
    description: str
    system_prompt: str = ""
    config: Dict[str, Any] = {}
    widget_config: Dict[str, Any] = {}
    agent_type: AgentType = AgentType.CUSTOM
    industry: IndustryType = IndustryType.GENERAL
    auto_optimize: bool = True


class AgentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    is_active: Optional[bool] = None
    config: Optional[Dict[str, Any]] = None
    widget_config: Optional[Dict[str, Any]] = None


class ChatMessage(BaseModel):
    message: str
    visitor_id: str
    session_context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: Optional[int] = None
    customer_context: Dict[str, Any]
    model: str
    usage: Dict[str, Any]


class CRMConfig(BaseModel):
    provider: Optional[str] = None
    enabled: Optional[bool] = None
    auth: Optional[Dict[str, Any]] = None
    field_map: Optional[Dict[str, Any]] = None
    stage_map: Optional[Dict[str, Any]] = None
    webhook_secret: Optional[str] = None
    last_sync_at: Optional[str] = None


class DomainExpertiseUpdate(BaseModel):
    enabled: Optional[bool] = None
    persona_key: Optional[str] = None
    persona_overrides: Optional[Dict[str, Any]] = None
    custom_persona: Optional[Dict[str, Any]] = None
    knowledge_document_ids: Optional[List[int]] = None
    web_search_enabled: Optional[bool] = None
    site_whitelist: Optional[List[str]] = None
    grounding_mode: Optional[str] = None
    expertise_level: Optional[float] = None
    additional_context: Optional[str] = None


@router.get("/templates/types")
async def get_agent_types():
    """Get available agent types"""
    return {
        "agent_types": [
            {"value": agent_type.value, "label": agent_type.value.replace("_", " ").title()}
            for agent_type in AgentType
        ]
    }


@router.get("/templates/industries")
async def get_industries():
    """Get available industries"""
    return {
        "industries": [
            {"value": industry.value, "label": industry.value.replace("_", " ").title()}
            for industry in IndustryType
        ]
    }


@router.get("/", response_model=List[AgentResponse])
async def get_agents(
    organization_id: int,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get all agents for an organization"""
    try:
        # Verify user has access to organization using new auth system
        from ...core.auth import verify_organization_access
        if not verify_organization_access(organization_id, current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied - user does not have access to this organization"
            )

        agents = await db_service.get_organization_agents(organization_id)
        return [serialize_agent(agent) for agent in agents]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching agents: {str(e)}"
        )


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific agent by ID"""
    try:
        agent = await db_service.get_agent_by_id(agent_id)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent not found"
            )

        # Verify user owns this agent
        if agent.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )

        return serialize_agent(agent)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching agent: {str(e)}"
        )


@router.post("/", response_model=EnhancedAgentResponse)
async def create_agent(
    organization_id: int,
    agent_data: AgentCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new agent with intelligent optimization and templates"""
    try:
        # Verify user can manage agents in organization
        user_org = await db_service.get_user_organization(current_user.id, organization_id)
        if not user_org or not user_org.can_manage_agents:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )

        # Check organization limits
        organization = await db_service.get_organization_by_id(organization_id)
        if not organization:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found"
            )

        # Check agent limits
        current_agent_count = await db_service.count_organization_agents(organization_id)
        if organization.max_agents != -1 and current_agent_count >= organization.max_agents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Organization has reached agent limit"
            )

        # Use advanced agent creation service
        creation_result = await agent_creation_service.create_intelligent_agent(
            user_id=current_user.id,
            organization_id=organization_id,
            agent_data={
                "name": agent_data.name,
                "description": agent_data.description,
                "system_prompt": agent_data.system_prompt,
                "config": agent_data.config,
                "widget_config": agent_data.widget_config
            },
            agent_type=agent_data.agent_type,
            industry=agent_data.industry,
            auto_optimize=agent_data.auto_optimize
        )

        agent = creation_result["agent"]

        return EnhancedAgentResponse(
            agent=serialize_agent(agent),
            embed_code=creation_result["embed_code"],
            setup_guide=creation_result["setup_guide"],
            optimization_applied=creation_result["optimization_applied"],
            template_used=creation_result["template_used"],
            recommendations=creation_result["recommendations"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating agent: {str(e)}"
        )


@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: int,
    agent_data: AgentUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update an agent"""
    try:
        # Verify user owns this agent
        existing_agent = await db_service.get_agent_by_id(agent_id)
        if not existing_agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent not found"
            )

        if existing_agent.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )

        # Build update dict with only non-None values
        update_data = {
            key: value for key, value in agent_data.dict().items()
            if value is not None
        }

        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No data provided for update"
            )

        agent = await db_service.update_agent(agent_id, **update_data)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent not found"
            )

        return serialize_agent(agent)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating agent: {str(e)}"
        )


@router.get("/{agent_id}/stats")
async def get_agent_stats(
    agent_id: int,
    time_range: str = "30d",
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get comprehensive stats for an agent"""
    try:
        # Verify agent exists and user owns it
        agent = await db_service.get_agent_by_id(agent_id)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent not found"
            )

        if agent.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )

        stats = await db_service.get_agent_stats(agent_id, time_range=time_range)
        return stats
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching agent stats: {str(e)}"
        )


@router.patch("/{agent_id}/domain-expertise", response_model=AgentResponse)
async def update_agent_domain_expertise(
    agent_id: int,
    domain_update: DomainExpertiseUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    agent = await db_service.get_agent_by_id(agent_id)
    if not agent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")

    if agent.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    update_fields: Dict[str, Any] = {}

    # Normalize existing enum values
    existing_tier = agent.tier
    if existing_tier and not isinstance(existing_tier, AgentTier):
        try:
            existing_tier = AgentTier(existing_tier)
        except ValueError:
            existing_tier = AgentTier.basic
    if not existing_tier:
        existing_tier = AgentTier.basic

    domain_type = agent.domain_expertise_type
    if domain_type and not isinstance(domain_type, DomainExpertiseType):
        try:
            domain_type = DomainExpertiseType(domain_type)
        except ValueError:
            domain_type = None

    # Persona handling
    persona_config = agent.personality_profile or {}
    persona_changed = False

    if domain_update.persona_key:
        key = domain_update.persona_key
        if key not in BASE_PERSONAS:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unknown persona key")
        persona_config = BASE_PERSONAS[key].copy()
        if domain_update.persona_overrides:
            persona_config.update(domain_update.persona_overrides)
        domain_type = PERSONA_ENUM_MAP.get(key, DomainExpertiseType.domain_specialist)
        persona_changed = True

    if domain_update.custom_persona:
        persona_config = domain_update.custom_persona
        domain_type = DomainExpertiseType.domain_specialist
        persona_changed = True

    if domain_update.persona_overrides and not domain_update.persona_key and not domain_update.custom_persona:
        persona_config = {**persona_config, **domain_update.persona_overrides}

    if persona_config:
        update_fields["personality_profile"] = persona_config
    if persona_changed:
        update_fields["domain_expertise_type"] = domain_type

    # Knowledge sources
    if domain_update.knowledge_document_ids is not None:
        doc_ids = list({int(doc_id) for doc_id in domain_update.knowledge_document_ids})
        documents = await db_service.get_documents_by_ids(doc_ids)
        if len(documents) != len(doc_ids):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="One or more documents not found")
        for document in documents:
            if document.agent_id != agent_id:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Document does not belong to this agent")
        update_fields["domain_knowledge_sources"] = doc_ids

    # Tool policy / web search
    tool_policy = agent.tool_policy or {}
    if domain_update.web_search_enabled is not None:
        tool_policy["web_search"] = domain_update.web_search_enabled
        update_fields["web_search_enabled"] = domain_update.web_search_enabled
    if domain_update.site_whitelist is not None:
        whitelist = [item.strip() for item in domain_update.site_whitelist if item.strip()]
        tool_policy["site_whitelist"] = whitelist
    if tool_policy:
        update_fields["tool_policy"] = tool_policy

    # Grounding mode
    if domain_update.grounding_mode:
        mode = domain_update.grounding_mode.lower()
        if mode not in GROUNDING_MODES:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid grounding mode")
        update_fields["grounding_mode"] = mode

    # Expertise level
    if domain_update.expertise_level is not None:
        level = max(0.0, min(1.0, float(domain_update.expertise_level)))
        update_fields["expertise_level"] = level

    # Additional context
    if domain_update.additional_context is not None:
        update_fields["expert_context"] = domain_update.additional_context.strip() or None

    # Enable / disable
    if domain_update.enabled is not None:
        update_fields["domain_expertise_enabled"] = domain_update.enabled
        if domain_update.enabled:
            update_fields["tier"] = AgentTier.professional if existing_tier != AgentTier.enterprise else existing_tier
        else:
            if existing_tier == AgentTier.professional:
                update_fields["tier"] = AgentTier.basic

    # Persist persona overrides in custom training data for audit
    custom_training_data = agent.custom_training_data or {}
    if domain_update.persona_overrides is not None:
        custom_training_data["persona_overrides"] = domain_update.persona_overrides
    if domain_update.custom_persona is not None:
        custom_training_data["custom_persona"] = domain_update.custom_persona
    if custom_training_data:
        update_fields["custom_training_data"] = custom_training_data

    updated_agent = await db_service.update_agent(
        agent_id,
        **update_fields
    )

    return serialize_agent(updated_agent)


@router.get("/{agent_id}/conversations")
async def list_agent_conversations(
    agent_id: int,
    limit: int = 10,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List recent conversations for an agent"""
    agent = await db_service.get_agent_by_id(agent_id)
    if not agent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")

    if agent.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    conversations = await db_service.get_agent_conversations(agent_id=agent_id, limit=limit, offset=offset)

    items = []
    for convo in conversations:
        messages = await db_service.get_conversation_messages(convo.id, limit=1, ascending=False)
        last_message = None
        if messages:
            msg = messages[0]
            last_message = {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at.isoformat() if msg.created_at else None,
            }

        profile = convo.customer_profile
        items.append({
            "id": convo.id,
            "session_id": convo.session_id,
            "created_at": convo.created_at.isoformat() if convo.created_at else None,
            "updated_at": convo.updated_at.isoformat() if convo.updated_at else None,
            "customer_profile": {
                "id": profile.id,
                "name": profile.display_name,
                "visitor_id": profile.visitor_id,
                "is_vip": profile.is_vip,
                "primary_interests": profile.primary_interests or [],
            } if profile else None,
            "last_message": last_message,
            "total_messages": await db_service.count_conversation_messages(convo.id),
        })

    return {"items": items, "limit": limit, "offset": offset}


@router.post("/{agent_id}/chat", response_model=ChatResponse)
async def chat_with_agent(
    agent_id: int,
    chat_data: ChatMessage,
    db: AsyncSession = Depends(get_db)
):
    """
    Chat with an agent using intelligent RAG with memory and personalization.
    This endpoint is public and doesn't require authentication (for embedding in customer websites).
    """
    try:
        # Verify agent exists and is active
        agent = await db_service.get_agent_by_id(agent_id)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent not found"
            )

        if not agent.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Agent is not active"
            )

        # Get or create conversation
        session_id = f"{chat_data.visitor_id}_{agent_id}"
        conversation = await db_service.get_conversation_by_session(agent_id, session_id)

        if not conversation:
            conversation = await db_service.create_conversation(
                agent_id=agent_id,
                session_id=session_id,
                metadata={
                    "visitor_id": chat_data.visitor_id,
                    "session_context": chat_data.session_context
                }
            )

        # Use intelligent RAG service for response generation
        if intelligent_rag_service is None:
            # Fallback response when intelligent RAG is not available
            response_data = {
                "content": f"Hello! I'm {agent.name}. I'm currently in basic mode. You said: {chat_data.message}",
                "customer_context": {
                    "visitor_id": chat_data.visitor_id,
                    "personalization_applied": False,
                    "basic_mode": True
                },
                "model": "fallback",
                "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
            }
        else:
            response_data = await intelligent_rag_service.generate_intelligent_response(
                query=chat_data.message,
                agent_id=agent_id,
                visitor_id=chat_data.visitor_id,
                conversation_id=conversation.id,
                session_context=chat_data.session_context,
                system_prompt=agent.system_prompt,
                agent_config=agent.config or {},
                agent_profile=agent
            )

        # Store the conversation messages
        await db_service.create_message(
            conversation_id=conversation.id,
            role="user",
            content=chat_data.message
        )

        await db_service.create_message(
            conversation_id=conversation.id,
            role="assistant",
            content=response_data["content"]
        )

        return ChatResponse(
            response=response_data["content"],
            conversation_id=conversation.id,
            customer_context=response_data.get("customer_context", {}),
            model=response_data.get("model", "gemini-2.0-flash-exp"),
            usage=response_data.get("usage", {})
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat message: {str(e)}"
        )


@router.get("/{agent_id}/insights")
async def get_agent_customer_insights(
    agent_id: int,
    time_range: str = "30d",
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get analytics insights for the dashboard (top questions and satisfaction mix).
    """
    try:
        # Verify agent exists and user owns it
        agent = await db_service.get_agent_by_id(agent_id)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent not found"
            )

        if agent.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )

        insights = await db_service.get_agent_insights(agent_id, time_range=time_range)
        return insights

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching customer insights: {str(e)}"
        )


@router.websocket("/{agent_id}/ws")
async def websocket_chat_endpoint(websocket: WebSocket, agent_id: int):
    """
    Real-time WebSocket chat with intelligent concierge agent.

    Provides world-class conversational experience with:
    - Streaming responses with typing indicators
    - Real-time emotional intelligence analysis
    - Live personalization and memory integration
    - Context-aware response adaptation

    Query parameters:
    - visitor_id: Unique identifier for the visitor (optional, auto-generated if not provided)

    WebSocket message format:
    {
        "type": "message" | "typing" | "ping" | "session_update",
        "message": "User message content",
        "session_context": {
            "page_url": "https://example.com/page",
            "referrer": "https://google.com",
            "device_info": {...}
        }
    }

    Response formats:
    - response_chunk: Streaming response content
    - typing_indicator: Agent typing status
    - intelligence_analysis: Real-time user analysis
    - status: Processing status updates
    - error: Error messages
    """
    if handle_websocket_chat is None:
        await websocket.close(code=1000, reason="WebSocket chat not available")
        return

    await handle_websocket_chat(websocket, agent_id)


@router.get("/{agent_id}/escalations")
async def list_agent_escalations(
    agent_id: int,
    status: str = None,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List recent escalation events for an agent (admin view)."""
    # Verify agent exists and user owns it
    agent = await db_service.get_agent_by_id(agent_id)
    if not agent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
    if agent.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    escalations = await escalation_service.list_escalations(agent_id=agent_id, status=status, limit=limit)
    def to_dict(e):
        return {
            "id": e.id,
            "status": e.status,
            "priority": e.priority,
            "reason": e.reason,
            "summary": e.summary,
            "conversation_id": e.conversation_id,
            "customer_profile_id": e.customer_profile_id,
            "details": e.details or {},
            "created_at": e.created_at.isoformat() if e.created_at else None,
        }
    return {"items": [to_dict(e) for e in escalations]}


@router.post("/{agent_id}/escalations/{escalation_id}/resolve")
async def resolve_agent_escalation(
    agent_id: int,
    escalation_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Resolve a specific escalation for an agent."""
    agent = await db_service.get_agent_by_id(agent_id)
    if not agent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
    if agent.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    ok = await escalation_service.resolve_escalation(escalation_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Escalation not found")

    return {"success": True}


@router.get("/{agent_id}/integrations/crm")
async def get_agent_crm_integration(
    agent_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get per-agent CRM override config (falls back to organization-level in usage)."""
    agent = await db_service.get_agent_by_id(agent_id)
    if not agent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
    if agent.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    cfg = (agent.config or {}).get("integrations", {}).get("crm", {})
    return {"crm": cfg}


@router.put("/{agent_id}/integrations/crm")
async def update_agent_crm_integration(
    agent_id: int,
    cfg: CRMConfig,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update per-agent CRM override config (stored in agent.config.integrations.crm)."""
    agent = await db_service.get_agent_by_id(agent_id)
    if not agent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
    if agent.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    config = agent.config or {}
    integrations = config.get("integrations", {})
    crm_cfg = integrations.get("crm", {})
    patch = {k: v for k, v in cfg.dict().items() if v is not None}
    crm_cfg.update(patch)
    integrations["crm"] = crm_cfg
    config["integrations"] = integrations

    updated = await db_service.update_agent(agent_id, config=config)
    return {"crm": (updated.config or {}).get("integrations", {}).get("crm", {})}
