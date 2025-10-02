"""
User context API endpoints.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from ...core.database import get_db
from ...core.auth import get_current_user
from ...models.user import User

router = APIRouter()


class OrganizationContextResponse(BaseModel):
    id: int
    name: str
    slug: str
    plan: str
    max_agents: int
    max_users: int
    max_documents_per_agent: int
    agents_count: int
    active_users_count: int
    can_add_agent: bool
    can_add_user: bool
    user_role: str
    user_permissions: dict

    model_config = {
        "from_attributes": True,
    }


class UserContextResponse(BaseModel):
    id: int
    email: str
    name: str
    plan: str
    is_active: bool
    created_at: str
    organizations: List[OrganizationContextResponse]
    default_organization_id: Optional[int] = None

    model_config = {
        "from_attributes": True,
    }


@router.get("/context", response_model=UserContextResponse)
async def get_user_context(
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get comprehensive user context including organization information"""
    try:
        # Get user organizations using the database service
        from ...services.database_service import db_service

        user_organizations = await db_service.get_user_organizations(current_user.id)

        # Get user's organizations with context
        organizations = []
        default_org_id = None

        for uo in user_organizations:
            if uo.is_active and uo.organization.is_active:
                org = uo.organization

                # Set first active organization as default
                if default_org_id is None:
                    default_org_id = org.id

                # Build user permissions for this organization
                permissions = {
                    "manage_users": uo.can_manage_users,
                    "manage_agents": uo.can_manage_agents,
                    "view_agents": uo.can_view_agents,
                    "manage_billing": uo.can_manage_billing,
                    "delete_organization": uo.can_delete_organization
                }

                # Get counts using database service to avoid lazy loading
                agents_count = await db_service.count_organization_agents(org.id)
                active_users_count = await db_service.count_organization_users(org.id)

                # Calculate can_add flags
                can_add_agent = org.max_agents == -1 or agents_count < org.max_agents
                can_add_user = org.max_users == -1 or active_users_count < org.max_users

                org_context = OrganizationContextResponse(
                    id=org.id,
                    name=org.name,
                    slug=org.slug,
                    plan=org.plan,
                    max_agents=org.max_agents,
                    max_users=org.max_users,
                    max_documents_per_agent=org.max_documents_per_agent,
                    agents_count=agents_count,
                    active_users_count=active_users_count,
                    can_add_agent=can_add_agent,
                    can_add_user=can_add_user,
                    user_role=uo.role,
                    user_permissions=permissions
                )
                organizations.append(org_context)

        return UserContextResponse(
            id=current_user.id,
            email=current_user.email,
            name=current_user.name,
            plan=getattr(current_user, 'plan', 'free'),
            is_active=getattr(current_user, 'is_active', True),
            created_at=getattr(current_user, 'created_at', ''),
            organizations=organizations,
            default_organization_id=default_org_id
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving user context: {str(e)}"
        )


@router.get("/organizations", response_model=List[OrganizationContextResponse])
async def get_user_organizations(
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user's organizations with detailed context"""
    try:
        # Get full user data from database using the authenticated user's ID with organization relationships
        from sqlalchemy.future import select
        from sqlalchemy.orm import selectinload
        from ...models.user_organization import UserOrganization

        result = await db.execute(
            select(User)
            .options(selectinload(User.user_organizations).selectinload(UserOrganization.organization))
            .where(User.id == current_user.id)
        )
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        organizations = []

        for uo in user.user_organizations:
            if uo.is_active and uo.organization.is_active:
                org = uo.organization

                permissions = {
                    "manage_users": uo.can_manage_users,
                    "manage_agents": uo.can_manage_agents,
                    "view_agents": uo.can_view_agents,
                    "manage_billing": uo.can_manage_billing,
                    "delete_organization": uo.can_delete_organization
                }

                org_context = OrganizationContextResponse(
                    id=org.id,
                    name=org.name,
                    slug=org.slug,
                    plan=org.plan,
                    max_agents=org.max_agents,
                    max_users=org.max_users,
                    max_documents_per_agent=org.max_documents_per_agent,
                    agents_count=org.agents_count,
                    active_users_count=org.active_users_count,
                    can_add_agent=org.can_add_agent(),
                    can_add_user=org.can_add_user(),
                    user_role=uo.role,
                    user_permissions=permissions
                )
                organizations.append(org_context)

        return organizations

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving user organizations: {str(e)}"
        )


@router.get("/organizations/{organization_id}/context", response_model=OrganizationContextResponse)
async def get_organization_context(
    organization_id: int,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get detailed context for a specific organization"""
    try:
        # Get full user data from database using the authenticated user's ID with organization relationships
        from sqlalchemy.future import select
        from sqlalchemy.orm import selectinload
        from ...models.user_organization import UserOrganization

        result = await db.execute(
            select(User)
            .options(selectinload(User.user_organizations).selectinload(UserOrganization.organization))
            .where(User.id == current_user.id)
        )
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        # Find user's membership in this organization
        uo = None
        for user_org in user.user_organizations:
            if user_org.organization_id == organization_id and user_org.is_active:
                uo = user_org
                break

        if not uo or not uo.organization.is_active:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found or access denied"
            )

        org = uo.organization

        permissions = {
            "manage_users": uo.can_manage_users,
            "manage_agents": uo.can_manage_agents,
            "view_agents": uo.can_view_agents,
            "manage_billing": uo.can_manage_billing,
            "delete_organization": uo.can_delete_organization
        }

        return OrganizationContextResponse(
            id=org.id,
            name=org.name,
            slug=org.slug,
            plan=org.plan,
            max_agents=org.max_agents,
            max_users=org.max_users,
            max_documents_per_agent=org.max_documents_per_agent,
            agents_count=org.agents_count,
            active_users_count=org.active_users_count,
            can_add_agent=org.can_add_agent(),
            can_add_user=org.can_add_user(),
            user_role=uo.role,
            user_permissions=permissions
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving organization context: {str(e)}"
        )
