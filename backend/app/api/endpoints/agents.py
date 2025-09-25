from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import List, Optional
from app.models import get_async_session

router = APIRouter()

class AgentCreate(BaseModel):
    name: str
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    config: Optional[dict] = {}
    widget_config: Optional[dict] = {}

class AgentResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    is_active: bool
    api_key: str
    config: dict
    widget_config: dict

    class Config:
        from_attributes = True

class AgentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    is_active: Optional[bool] = None
    config: Optional[dict] = None
    widget_config: Optional[dict] = None

@router.get("/", response_model=List[AgentResponse])
async def get_agents(
    db: AsyncSession = Depends(get_async_session)
):
    # TODO: Get user's agents
    return []

@router.post("/", response_model=AgentResponse)
async def create_agent(
    agent_data: AgentCreate,
    db: AsyncSession = Depends(get_async_session)
):
    # TODO: Create new agent
    return {
        "id": 1,
        "name": agent_data.name,
        "description": agent_data.description,
        "system_prompt": agent_data.system_prompt,
        "is_active": True,
        "api_key": "agent_api_key_123",
        "config": agent_data.config,
        "widget_config": agent_data.widget_config
    }

@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: int,
    db: AsyncSession = Depends(get_async_session)
):
    # TODO: Get specific agent
    return {
        "id": agent_id,
        "name": "Sample Agent",
        "description": "A sample agent",
        "system_prompt": "You are a helpful assistant",
        "is_active": True,
        "api_key": "agent_api_key_123",
        "config": {},
        "widget_config": {}
    }

@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: int,
    updates: AgentUpdate,
    db: AsyncSession = Depends(get_async_session)
):
    # TODO: Update agent
    return {
        "id": agent_id,
        "name": "Updated Agent",
        "description": "Updated description",
        "system_prompt": "You are a helpful assistant",
        "is_active": True,
        "api_key": "agent_api_key_123",
        "config": {},
        "widget_config": {}
    }

@router.delete("/{agent_id}")
async def delete_agent(
    agent_id: int,
    db: AsyncSession = Depends(get_async_session)
):
    # TODO: Delete agent
    return {"message": "Agent deleted successfully"}

@router.get("/{agent_id}/embed-code")
async def get_embed_code(
    agent_id: int,
    db: AsyncSession = Depends(get_async_session)
):
    # TODO: Generate embed code for agent
    embed_code = f"""
    <script>
      (function() {{
        const script = document.createElement('script');
        script.src = 'https://your-domain.com/widget.js';
        script.setAttribute('data-agent-id', '{agent_id}');
        document.head.appendChild(script);
      }})();
    </script>
    """
    return {"embed_code": embed_code.strip()}