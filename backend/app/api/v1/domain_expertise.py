"""
Domain Expertise API Endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from enum import Enum

from app.core.database import get_db
from app.core.auth import get_current_user
from app.models.persona import Persona, KnowledgePack, KnowledgePackSource
from app.models.agent import Agent
from app.services.persona_templates import PersonaTemplates, get_persona_seeds
from app.services.domain_expertise_service import domain_expertise_service

router = APIRouter()


# Validation Enums
class GroundingMode(str, Enum):
    STRICT = "strict"
    BLENDED = "blended"

class FreshnessPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class RecrawlStrategy(str, Enum):
    NEVER = "never"
    CHANGED = "changed"
    SCHEDULED = "scheduled"


# Request/Response Models
class PersonaCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    system_prompt: str = Field(..., min_length=10, max_length=5000)
    tactics: Dict[str, Any] = Field(default_factory=dict)
    communication_style: Dict[str, Any] = Field(default_factory=dict)
    response_patterns: Dict[str, Any] = Field(default_factory=dict)


class PersonaResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    system_prompt: str
    tactics: Dict[str, Any]
    communication_style: Dict[str, Any]
    response_patterns: Dict[str, Any]
    is_built_in: bool
    template_name: Optional[str]
    usage_count: int

    model_config = {
        "from_attributes": True,
    }


class KnowledgePackCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    grounding_mode: GroundingMode = GroundingMode.BLENDED
    freshness_policy: Dict[str, Any] = Field(default_factory=lambda: {
        "ttl_days": 30,
        "recrawl": "changed",
        "priority": "medium"
    })

    @validator('freshness_policy')
    def validate_freshness_policy(cls, v):
        required_keys = {'ttl_days', 'recrawl', 'priority'}
        if not all(key in v for key in required_keys):
            raise ValueError(f'freshness_policy must contain keys: {required_keys}')

        if not isinstance(v['ttl_days'], int) or v['ttl_days'] < 1:
            raise ValueError('ttl_days must be a positive integer')

        if v['recrawl'] not in ['never', 'changed', 'scheduled']:
            raise ValueError('recrawl must be one of: never, changed, scheduled')

        if v['priority'] not in ['low', 'medium', 'high']:
            raise ValueError('priority must be one of: low, medium, high')

        return v


class KnowledgePackResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    grounding_mode: str
    freshness_policy: Dict[str, Any]
    document_count: int
    last_crawl_at: Optional[str]
    coverage_score: Optional[str]
    usage_count: int

    model_config = {
        "from_attributes": True,
    }


class DomainExpertiseUpdate(BaseModel):
    persona_id: Optional[int] = Field(None, ge=1)
    knowledge_pack_id: Optional[int] = Field(None, ge=1)
    tool_policy: Dict[str, Any] = Field(default_factory=lambda: {
        "web_search": False,
        "site_search": [],
        "code_exec": False
    })
    grounding_mode: GroundingMode = GroundingMode.BLENDED

    @validator('tool_policy')
    def validate_tool_policy(cls, v):
        required_keys = {'web_search', 'site_search', 'code_exec'}
        if not all(key in v for key in required_keys):
            raise ValueError(f'tool_policy must contain keys: {required_keys}')

        if not isinstance(v['web_search'], bool):
            raise ValueError('web_search must be a boolean')

        if not isinstance(v['site_search'], list):
            raise ValueError('site_search must be a list')

        if not isinstance(v['code_exec'], bool):
            raise ValueError('code_exec must be a boolean')

        # Validate site_search URLs
        import re
        url_pattern = re.compile(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        for site in v['site_search']:
            if not isinstance(site, str) or not url_pattern.match(site):
                raise ValueError(f'Invalid site domain: {site}')

        return v


class TestQueryRequest(BaseModel):
    query: str
    agent_id: int


class TestQueryResponse(BaseModel):
    answer: str
    confidence_score: float
    sources: List[Dict[str, Any]]
    grounding_mode: str
    persona_applied: str
    escalation_suggested: bool
    web_search_used: bool


# Persona Endpoints
@router.get("/personas", response_model=List[PersonaResponse])
async def list_personas(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all personas for the organization"""

    personas = db.query(Persona).filter(
        Persona.organization_id == current_user["organization_id"]
    ).all()

    return personas


@router.post("/personas", response_model=PersonaResponse)
async def create_persona(
    persona_data: PersonaCreate,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new custom persona"""

    persona = Persona(
        organization_id=current_user["organization_id"],
        name=persona_data.name,
        description=persona_data.description,
        system_prompt=persona_data.system_prompt,
        tactics=persona_data.tactics,
        communication_style=persona_data.communication_style,
        response_patterns=persona_data.response_patterns,
        is_built_in=False
    )

    db.add(persona)
    db.commit()
    db.refresh(persona)

    return persona


@router.get("/personas/templates")
async def get_persona_templates():
    """Get all built-in persona templates"""

    templates = PersonaTemplates.get_all_templates()
    return {
        "templates": templates,
        "available_types": list(templates.keys())
    }


@router.post("/personas/from-template")
async def create_persona_from_template(
    template_name: str,
    custom_name: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a persona from a built-in template"""

    templates = PersonaTemplates.get_all_templates()
    if template_name not in templates:
        raise HTTPException(
            status_code=404,
            detail=f"Template '{template_name}' not found"
        )

    template = templates[template_name]

    persona = Persona(
        organization_id=current_user["organization_id"],
        name=custom_name or template["name"],
        description=template["description"],
        system_prompt=template["system_prompt"],
        tactics=template["tactics"],
        communication_style=template["communication_style"],
        response_patterns=template["response_patterns"],
        is_built_in=True,
        template_name=template_name
    )

    db.add(persona)
    db.commit()
    db.refresh(persona)

    return persona


# Knowledge Pack Endpoints
@router.get("/knowledge-packs", response_model=List[KnowledgePackResponse])
async def list_knowledge_packs(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all knowledge packs for the organization"""

    packs = db.query(KnowledgePack).filter(
        KnowledgePack.organization_id == current_user["organization_id"]
    ).all()

    return packs


@router.post("/knowledge-packs", response_model=KnowledgePackResponse)
async def create_knowledge_pack(
    pack_data: KnowledgePackCreate,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new knowledge pack"""

    pack = KnowledgePack(
        organization_id=current_user["organization_id"],
        name=pack_data.name,
        description=pack_data.description,
        grounding_mode=pack_data.grounding_mode,
        freshness_policy=pack_data.freshness_policy,
        document_count=0
    )

    db.add(pack)
    db.commit()
    db.refresh(pack)

    return pack


@router.post("/knowledge-packs/{pack_id}/sources")
async def add_sources_to_pack(
    pack_id: int,
    source_ids: List[int],
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Add document sources to a knowledge pack"""

    # Verify pack ownership
    pack = db.query(KnowledgePack).filter(
        KnowledgePack.id == pack_id,
        KnowledgePack.organization_id == current_user["organization_id"]
    ).first()

    if not pack:
        raise HTTPException(status_code=404, detail="Knowledge pack not found")

    # Add sources
    added_sources = []
    for source_id in source_ids:
        # Check if source already exists
        existing = db.query(KnowledgePackSource).filter(
            KnowledgePackSource.pack_id == pack_id,
            KnowledgePackSource.source_id == source_id
        ).first()

        if not existing:
            pack_source = KnowledgePackSource(
                pack_id=pack_id,
                source_id=source_id,
                source_type="document",
                is_active=True,
                processing_status="pending"
            )
            db.add(pack_source)
            added_sources.append(source_id)

    # Update document count
    pack.document_count = db.query(KnowledgePackSource).filter(
        KnowledgePackSource.pack_id == pack_id,
        KnowledgePackSource.is_active == True
    ).count()

    db.commit()

    return {
        "added_sources": added_sources,
        "total_sources": pack.document_count,
        "message": f"Added {len(added_sources)} new sources to knowledge pack"
    }


# Agent Domain Expertise Configuration
@router.patch("/agents/{agent_id}/domain-expertise")
async def update_agent_domain_expertise(
    agent_id: int,
    domain_config: DomainExpertiseUpdate,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update agent's domain expertise configuration"""

    # Verify agent ownership
    agent = db.query(Agent).filter(
        Agent.id == agent_id,
        Agent.organization_id == current_user["organization_id"]
    ).first()

    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Verify persona ownership if provided
    if domain_config.persona_id:
        persona = db.query(Persona).filter(
            Persona.id == domain_config.persona_id,
            Persona.organization_id == current_user["organization_id"]
        ).first()
        if not persona:
            raise HTTPException(status_code=404, detail="Persona not found")

    # Verify knowledge pack ownership if provided
    if domain_config.knowledge_pack_id:
        pack = db.query(KnowledgePack).filter(
            KnowledgePack.id == domain_config.knowledge_pack_id,
            KnowledgePack.organization_id == current_user["organization_id"]
        ).first()
        if not pack:
            raise HTTPException(status_code=404, detail="Knowledge pack not found")

    # Update agent configuration
    agent.persona_id = domain_config.persona_id
    agent.knowledge_pack_id = domain_config.knowledge_pack_id
    agent.tool_policy = domain_config.tool_policy
    agent.grounding_mode = domain_config.grounding_mode
    agent.domain_expertise_enabled = bool(domain_config.persona_id or domain_config.knowledge_pack_id)

    db.commit()
    db.refresh(agent)

    return {
        "agent_id": agent.id,
        "domain_expertise_enabled": agent.domain_expertise_enabled,
        "persona_id": agent.persona_id,
        "knowledge_pack_id": agent.knowledge_pack_id,
        "tool_policy": agent.tool_policy,
        "grounding_mode": agent.grounding_mode,
        "message": "Domain expertise configuration updated successfully"
    }


@router.get("/agents/{agent_id}/domain-expertise")
async def get_agent_domain_expertise(
    agent_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get agent's current domain expertise configuration"""

    agent = db.query(Agent).filter(
        Agent.id == agent_id,
        Agent.organization_id == current_user["organization_id"]
    ).first()

    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Get related objects
    persona = None
    if agent.persona_id:
        persona = db.query(Persona).filter(Persona.id == agent.persona_id).first()

    knowledge_pack = None
    if agent.knowledge_pack_id:
        knowledge_pack = db.query(KnowledgePack).filter(
            KnowledgePack.id == agent.knowledge_pack_id
        ).first()

    return {
        "agent_id": agent.id,
        "domain_expertise_enabled": agent.domain_expertise_enabled,
        "persona": PersonaResponse.from_orm(persona) if persona else None,
        "knowledge_pack": KnowledgePackResponse.from_orm(knowledge_pack) if knowledge_pack else None,
        "tool_policy": agent.tool_policy,
        "grounding_mode": agent.grounding_mode
    }


# Testing and Preview
@router.post("/test-query", response_model=TestQueryResponse)
async def test_domain_expertise_query(
    test_request: TestQueryRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Test a query with domain expertise configuration"""

    # Verify agent ownership
    agent = db.query(Agent).filter(
        Agent.id == test_request.agent_id,
        Agent.organization_id == current_user["organization_id"]
    ).first()

    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Mock organization object for service
    class MockOrg:
        def __init__(self, org_id):
            self.id = org_id

    organization = MockOrg(current_user["organization_id"])

    try:
        # Use domain expertise service to generate response
        response = await domain_expertise_service.answer_with_domain_expertise(
            message=test_request.query,
            agent=agent,
            organization=organization,
            conversation_context={}
        )

        return TestQueryResponse(
            answer=response.answer,
            confidence_score=response.confidence_score,
            sources=response.sources,
            grounding_mode=response.grounding_mode,
            persona_applied=response.persona_applied,
            escalation_suggested=response.escalation_suggested,
            web_search_used=response.web_search_used
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing test query: {str(e)}"
        )


@router.get("/agents/{agent_id}/capabilities")
async def get_agent_capabilities(
    agent_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get detailed agent capabilities and configuration"""

    agent = db.query(Agent).filter(
        Agent.id == agent_id,
        Agent.organization_id == current_user["organization_id"]
    ).first()

    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    capabilities = {
        "basic_chat": True,
        "knowledge_base_access": bool(agent.documents),
        "domain_expertise": agent.domain_expertise_enabled,
        "web_search": agent.tool_policy.get("web_search", False),
        "site_search": len(agent.tool_policy.get("site_search", [])) > 0,
        "persona_driven": bool(agent.persona_id),
        "knowledge_pack_enhanced": bool(agent.knowledge_pack_id),
        "grounding_mode": agent.grounding_mode
    }

    return {
        "agent_id": agent.id,
        "capabilities": capabilities,
        "configuration_completeness": sum(capabilities.values()) / len(capabilities),
        "upgrade_suggestions": _get_upgrade_suggestions(capabilities)
    }


def _get_upgrade_suggestions(capabilities: Dict[str, Any]) -> List[str]:
    """Generate upgrade suggestions based on current capabilities"""

    suggestions = []

    if not capabilities["domain_expertise"]:
        suggestions.append("Enable domain expertise for more intelligent responses")

    if not capabilities["persona_driven"]:
        suggestions.append("Add a persona to give your agent professional personality")

    if not capabilities["knowledge_pack_enhanced"]:
        suggestions.append("Create knowledge packs for organized domain knowledge")

    if not capabilities["web_search"]:
        suggestions.append("Enable web search for real-time information")

    if capabilities["grounding_mode"] != "strict":
        suggestions.append("Consider strict grounding mode for higher accuracy")

    return suggestions
