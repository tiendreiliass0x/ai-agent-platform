"""
Enhanced Agent Service Layer for production-ready operations.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, desc
from sqlalchemy.orm import selectinload

from ..core.exceptions import (
    NotFoundException, ForbiddenException, ValidationException,
    ConflictException, BusinessLogicException
)
from ..models.agent import Agent, AgentTier
from ..models.user import User
from ..models.organization import Organization
from ..models.user_organization import UserOrganization

logger = logging.getLogger(__name__)


class AgentService:
    """Enhanced service for agent operations with proper error handling and validation"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
    
    async def get_agent_by_id(self, agent_id: int, include_relations: bool = False) -> Optional[Agent]:
        """Get agent by ID with optional relations"""
        try:
            query = select(Agent).where(Agent.id == agent_id)
            
            if include_relations:
                query = query.options(
                    selectinload(Agent.user),
                    selectinload(Agent.organization),
                    selectinload(Agent.documents),
                    selectinload(Agent.conversations)
                )
            
            result = await self.db.execute(query)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error fetching agent {agent_id}: {e}")
            raise BusinessLogicException(f"Failed to fetch agent: {str(e)}", "get_agent_by_id")
    
    async def get_agent_by_public_id(self, public_id: str) -> Optional[Agent]:
        """Get agent by public ID"""
        try:
            result = await self.db.execute(
                select(Agent).where(Agent.public_id == public_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error fetching agent by public_id {public_id}: {e}")
            raise BusinessLogicException(f"Failed to fetch agent: {str(e)}", "get_agent_by_public_id")
    
    async def get_agent_by_idempotency_key(
        self, 
        user_id: int, 
        organization_id: int, 
        idempotency_key: str
    ) -> Optional[Agent]:
        """Get agent by idempotency key"""
        try:
            result = await self.db.execute(
                select(Agent).where(
                    and_(
                        Agent.user_id == user_id,
                        Agent.organization_id == organization_id,
                        Agent.idempotency_key == idempotency_key
                    )
                )
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error fetching agent by idempotency key: {e}")
            raise BusinessLogicException(f"Failed to fetch agent: {str(e)}", "get_agent_by_idempotency_key")
    
    async def get_organization_agents(
        self, 
        organization_id: int, 
        limit: int = 50, 
        offset: int = 0,
        include_inactive: bool = False
    ) -> List[Agent]:
        """Get agents for an organization with pagination"""
        try:
            query = select(Agent).where(Agent.organization_id == organization_id)
            
            if not include_inactive:
                query = query.where(Agent.is_active == True)
            
            query = query.order_by(desc(Agent.created_at)).limit(limit).offset(offset)
            
            result = await self.db.execute(query)
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error fetching organization agents: {e}")
            raise BusinessLogicException(f"Failed to fetch agents: {str(e)}", "get_organization_agents")
    
    async def count_organization_agents(self, organization_id: int, include_inactive: bool = False) -> int:
        """Count agents for an organization"""
        try:
            query = select(func.count(Agent.id)).where(Agent.organization_id == organization_id)
            
            if not include_inactive:
                query = query.where(Agent.is_active == True)
            
            result = await self.db.execute(query)
            return result.scalar() or 0
        except Exception as e:
            logger.error(f"Error counting organization agents: {e}")
            raise BusinessLogicException(f"Failed to count agents: {str(e)}", "count_organization_agents")
    
    async def create_agent(
        self,
        user_id: int,
        organization_id: int,
        agent_data: Dict[str, Any],
        api_key: Optional[str] = None
    ) -> Agent:
        """Create a new agent with validation"""
        try:
            # Validate required fields
            required_fields = ['name', 'description']
            for field in required_fields:
                if not agent_data.get(field):
                    raise ValidationException(f"Field '{field}' is required")
            
            # Check organization limits
            await self._check_organization_limits(organization_id)
            
            # Generate API key if not provided
            if not api_key:
                import secrets
                api_key = f"agent_{secrets.token_urlsafe(32)}"
            
            # Create agent
            agent = Agent(
                user_id=user_id,
                organization_id=organization_id,
                name=agent_data['name'],
                description=agent_data['description'],
                system_prompt=agent_data.get('system_prompt', ''),
                config=agent_data.get('config', {}),
                widget_config=agent_data.get('widget_config', {}),
                api_key=api_key,
                idempotency_key=agent_data.get('idempotency_key'),
                is_active=True
            )
            
            self.db.add(agent)
            await self.db.commit()
            await self.db.refresh(agent)
            
            logger.info(f"Created agent {agent.id} for user {user_id} in organization {organization_id}")
            return agent
            
        except ValidationException:
            raise
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error creating agent: {e}")
            raise BusinessLogicException(f"Failed to create agent: {str(e)}", "create_agent")
    
    async def update_agent(self, agent_id: int, update_data: Dict[str, Any]) -> Optional[Agent]:
        """Update an agent with validation"""
        try:
            # Get existing agent
            agent = await self.get_agent_by_id(agent_id)
            if not agent:
                raise NotFoundException("Agent not found", "agent", agent_id)
            
            # Update fields
            for key, value in update_data.items():
                if hasattr(agent, key) and value is not None:
                    setattr(agent, key, value)
            
            await self.db.commit()
            await self.db.refresh(agent)
            
            logger.info(f"Updated agent {agent_id}")
            return agent
            
        except NotFoundException:
            raise
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error updating agent {agent_id}: {e}")
            raise BusinessLogicException(f"Failed to update agent: {str(e)}", "update_agent")
    
    async def delete_agent(self, agent_id: int) -> bool:
        """Delete an agent and all associated data"""
        try:
            # Get agent to verify it exists
            agent = await self.get_agent_by_id(agent_id)
            if not agent:
                raise NotFoundException("Agent not found", "agent", agent_id)
            
            # Delete agent (cascade will handle related data)
            await self.db.delete(agent)
            await self.db.commit()
            
            logger.info(f"Deleted agent {agent_id}")
            return True
            
        except NotFoundException:
            raise
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error deleting agent {agent_id}: {e}")
            raise BusinessLogicException(f"Failed to delete agent: {str(e)}", "delete_agent")
    
    async def verify_agent_ownership(self, agent_id: int, user_id: int) -> bool:
        """Verify that a user owns an agent"""
        try:
            agent = await self.get_agent_by_id(agent_id)
            if not agent:
                raise NotFoundException("Agent not found", "agent", agent_id)
            
            return agent.user_id == user_id
            
        except NotFoundException:
            raise
        except Exception as e:
            logger.error(f"Error verifying agent ownership: {e}")
            raise BusinessLogicException(f"Failed to verify ownership: {str(e)}", "verify_agent_ownership")
    
    async def verify_organization_access(self, organization_id: int, user_id: int) -> bool:
        """Verify that a user has access to an organization"""
        try:
            result = await self.db.execute(
                select(UserOrganization).where(
                    and_(
                        UserOrganization.user_id == user_id,
                        UserOrganization.organization_id == organization_id,
                        UserOrganization.is_active == True
                    )
                )
            )
            user_org = result.scalar_one_or_none()
            return user_org is not None
            
        except Exception as e:
            logger.error(f"Error verifying organization access: {e}")
            raise BusinessLogicException(f"Failed to verify access: {str(e)}", "verify_organization_access")
    
    async def _check_organization_limits(self, organization_id: int) -> None:
        """Check if organization has reached agent limits"""
        try:
            # Get organization
            result = await self.db.execute(
                select(Organization).where(Organization.id == organization_id)
            )
            organization = result.scalar_one_or_none()
            
            if not organization:
                raise NotFoundException("Organization not found", "organization", organization_id)
            
            # Count current agents
            current_count = await self.count_organization_agents(organization_id)
            
            # Check limits
            if organization.max_agents > 0 and current_count >= organization.max_agents:
                raise ConflictException(
                    f"Organization has reached the maximum number of agents ({organization.max_agents})",
                    "agent_limit"
                )
                
        except NotFoundException:
            raise
        except ConflictException:
            raise
        except Exception as e:
            logger.error(f"Error checking organization limits: {e}")
            raise BusinessLogicException(f"Failed to check limits: {str(e)}", "check_organization_limits")
    
    async def get_agent_stats(self, agent_id: int, time_range: str = "30d") -> Dict[str, Any]:
        """Get comprehensive stats for an agent"""
        try:
            agent = await self.get_agent_by_id(agent_id)
            if not agent:
                raise NotFoundException("Agent not found", "agent", agent_id)
            
            # This would be implemented with actual statistics queries
            # For now, return basic stats
            stats = {
                "agent_id": agent_id,
                "total_conversations": agent.total_conversations or 0,
                "total_messages": agent.total_messages or 0,
                "last_used_at": agent.last_used_at.isoformat() if agent.last_used_at else None,
                "time_range": time_range,
                "is_active": agent.is_active,
                "tier": agent.tier.value if agent.tier else "basic"
            }
            
            return stats
            
        except NotFoundException:
            raise
        except Exception as e:
            logger.error(f"Error getting agent stats: {e}")
            raise BusinessLogicException(f"Failed to get stats: {str(e)}", "get_agent_stats")


# Service factory function
def create_agent_service(db_session: AsyncSession) -> AgentService:
    """Create an agent service instance"""
    return AgentService(db_session)
