"""
API endpoints for organization management.
"""

import secrets
import string
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from ...core.database import get_db
from ...core.auth import get_current_user
from ...services.database_service import db_service
from ...models.user import User
from ...models.organization import Organization
from ...models.user_organization import UserOrganization, OrganizationRole
from ...services.integrations.crm_service import crm_service

router = APIRouter()


class OrganizationCreate(BaseModel):
    name: str
    slug: str
    description: Optional[str] = None
    website: Optional[str] = None
    plan: str = "free"


class OrganizationUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    website: Optional[str] = None
    logo_url: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None


class OrganizationResponse(BaseModel):
    id: int
    name: str
    slug: str
    description: Optional[str]
    website: Optional[str]
    logo_url: Optional[str]
    plan: str
    subscription_status: str
    settings: Dict[str, Any]
    is_active: bool
    max_agents: int
    max_users: int
    max_documents_per_agent: int
    agents_count: int
    active_users_count: int
    created_at: str
    updated_at: Optional[str]

    class Config:
        from_attributes = True


class CRMConfig(BaseModel):
    provider: Optional[str] = None  # 'hubspot' | 'salesforce' | 'custom'
    enabled: Optional[bool] = None
    auth: Optional[Dict[str, Any]] = None
    field_map: Optional[Dict[str, Any]] = None
    stage_map: Optional[Dict[str, Any]] = None
    webhook_secret: Optional[str] = None
    last_sync_at: Optional[str] = None


class UserOrganizationResponse(BaseModel):
    id: int
    user_id: int
    organization_id: int
    role: str
    is_active: bool
    created_at: str
    user_name: str
    user_email: str

    class Config:
        from_attributes = True


class InviteUserRequest(BaseModel):
    email: str
    role: str = OrganizationRole.VIEWER.value


class AcceptInviteRequest(BaseModel):
    invitation_token: str


def generate_slug_from_name(name: str) -> str:
    """Generate URL-friendly slug from organization name"""
    slug = name.lower().replace(" ", "-").replace("_", "-")
    # Remove special characters except hyphens
    slug = "".join(c for c in slug if c.isalnum() or c == "-")
    # Remove multiple consecutive hyphens
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


def generate_invitation_token() -> str:
    """Generate secure invitation token"""
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(32))


@router.get("/", response_model=List[OrganizationResponse])
async def get_user_organizations(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get all organizations for current user"""
    try:
        # Get user organizations through database service
        user_orgs = await db_service.get_user_organizations(current_user.id)

        organizations = []
        for uo in user_orgs:
            if uo.is_active and uo.organization.is_active:
                org = uo.organization
                # Get counts separately
                agents_count = await db_service.count_organization_agents(org.id)
                users_count = await db_service.count_organization_users(org.id)

                organizations.append(OrganizationResponse(
                    id=org.id,
                    name=org.name,
                    slug=org.slug,
                    description=org.description,
                    website=org.website,
                    logo_url=org.logo_url,
                    plan=org.plan,
                    subscription_status=org.subscription_status,
                    settings=org.settings or {},
                    is_active=org.is_active,
                    max_agents=org.max_agents,
                    max_users=org.max_users,
                    max_documents_per_agent=org.max_documents_per_agent,
                    agents_count=agents_count,
                    active_users_count=users_count,
                    created_at=org.created_at.isoformat(),
                    updated_at=org.updated_at.isoformat() if org.updated_at else None
                ))
        return organizations
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching organizations: {str(e)}"
        )


@router.post("/", response_model=OrganizationResponse)
async def create_organization(
    org_data: OrganizationCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new organization"""
    try:
        # Check if slug is already taken
        existing_org = await db_service.get_organization_by_slug(org_data.slug)
        if existing_org:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Organization slug already exists"
            )

        # Set plan limits
        plan_limits = {
            "free": {"max_agents": 3, "max_users": 1, "max_documents_per_agent": 10},
            "pro": {"max_agents": 50, "max_users": 10, "max_documents_per_agent": 1000},
            "enterprise": {"max_agents": -1, "max_users": -1, "max_documents_per_agent": -1}
        }
        limits = plan_limits.get(org_data.plan, plan_limits["free"])

        # Create organization
        organization = await db_service.create_organization(
            name=org_data.name,
            slug=org_data.slug,
            description=org_data.description,
            website=org_data.website,
            plan=org_data.plan,
            max_agents=limits["max_agents"],
            max_users=limits["max_users"],
            max_documents_per_agent=limits["max_documents_per_agent"]
        )

        # Add user as owner
        await db_service.add_user_to_organization(
            user_id=current_user.id,
            organization_id=organization.id,
            role=OrganizationRole.OWNER.value
        )

        return OrganizationResponse(
            id=organization.id,
            name=organization.name,
            slug=organization.slug,
            description=organization.description,
            website=organization.website,
            logo_url=organization.logo_url,
            plan=organization.plan,
            subscription_status=organization.subscription_status,
            settings=organization.settings or {},
            is_active=organization.is_active,
            max_agents=organization.max_agents,
            max_users=organization.max_users,
            max_documents_per_agent=organization.max_documents_per_agent,
            agents_count=0,
            active_users_count=1,
            created_at=organization.created_at.isoformat(),
            updated_at=organization.updated_at.isoformat() if organization.updated_at else None
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating organization: {str(e)}"
        )


@router.get("/{organization_id}", response_model=OrganizationResponse)
async def get_organization(
    organization_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get specific organization details"""
    try:
        # Verify user has access to organization
        user_org = await db_service.get_user_organization(current_user.id, organization_id)
        if not user_org or not user_org.can_view_agents:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )

        organization = await db_service.get_organization_by_id(organization_id)
        if not organization:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found"
            )

        # Get counts using explicit queries to avoid lazy loading issues
        agents_count = await db_service.count_organization_agents(organization.id)
        users_count = await db_service.count_organization_users(organization.id)

        return OrganizationResponse(
            id=organization.id,
            name=organization.name,
            slug=organization.slug,
            description=organization.description,
            website=organization.website,
            logo_url=organization.logo_url,
            plan=organization.plan,
            subscription_status=organization.subscription_status,
            settings=organization.settings or {},
            is_active=organization.is_active,
            max_agents=organization.max_agents,
            max_users=organization.max_users,
            max_documents_per_agent=organization.max_documents_per_agent,
            agents_count=agents_count,
            active_users_count=users_count,
            created_at=organization.created_at.isoformat(),
            updated_at=organization.updated_at.isoformat() if organization.updated_at else None
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching organization: {str(e)}"
        )


@router.put("/{organization_id}", response_model=OrganizationResponse)
async def update_organization(
    organization_id: int,
    org_data: OrganizationUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update organization"""
    try:
        # Verify user can manage organization
        if not current_user.has_organization_permission(organization_id, "manage_users"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )

        # Build update dict
        update_data = {
            key: value for key, value in org_data.dict().items()
            if value is not None
        }

        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No data provided for update"
            )

        organization = await db_service.update_organization(organization_id, **update_data)
        if not organization:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found"
            )

        # Get counts using explicit queries to avoid lazy loading issues
        agents_count = await db_service.count_organization_agents(organization.id)
        users_count = await db_service.count_organization_users(organization.id)

        return OrganizationResponse(
            id=organization.id,
            name=organization.name,
            slug=organization.slug,
            description=organization.description,
            website=organization.website,
            logo_url=organization.logo_url,
            plan=organization.plan,
            subscription_status=organization.subscription_status,
            settings=organization.settings or {},
            is_active=organization.is_active,
            max_agents=organization.max_agents,
            max_users=organization.max_users,
            max_documents_per_agent=organization.max_documents_per_agent,
            agents_count=agents_count,
            active_users_count=users_count,
            created_at=organization.created_at.isoformat(),
            updated_at=organization.updated_at.isoformat() if organization.updated_at else None
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating organization: {str(e)}"
        )


@router.get("/{organization_id}/integrations/crm")
async def get_org_crm_integration(
    organization_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Return CRM integration configuration for the organization."""
    # Verify access
    user_org = await db_service.get_user_organization(current_user.id, organization_id)
    if not user_org or not user_org.can_manage_agents:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    org = await db_service.get_organization_by_id(organization_id)
    if not org:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")

    crm_cfg = (org.settings or {}).get("crm", {})
    return {"crm": crm_cfg}


@router.put("/{organization_id}/integrations/crm")
async def update_org_crm_integration(
    organization_id: int,
    cfg: CRMConfig,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update CRM integration configuration for the organization."""
    # Verify access
    user_org = await db_service.get_user_organization(current_user.id, organization_id)
    if not user_org or not user_org.can_manage_agents:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    org = await db_service.get_organization_by_id(organization_id)
    if not org:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")

    settings = org.settings or {}
    current_crm = settings.get("crm", {})
    # Merge patch
    patch = {k: v for k, v in cfg.dict().items() if v is not None}
    current_crm.update(patch)
    settings["crm"] = current_crm

    updated = await db_service.update_organization(organization_id, settings=settings)
    return {"crm": updated.settings.get("crm", {})}


@router.post("/{organization_id}/integrations/crm/test-connection")
async def test_org_crm_connection(
    organization_id: int,
    cfg: CRMConfig = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Validate CRM configuration (no external calls in this environment)."""
    user_org = await db_service.get_user_organization(current_user.id, organization_id)
    if not user_org or not user_org.can_manage_agents:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    org = await db_service.get_organization_by_id(organization_id)
    if not org:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")

    settings = org.settings or {}
    effective_cfg = (cfg.dict(exclude_none=True) if cfg else {}) or settings.get("crm", {})
    result = await crm_service.test_connection(effective_cfg)
    return result


@router.post("/{organization_id}/integrations/crm/sync")
async def sync_org_crm(
    organization_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Trigger a (mock) CRM sync and record last_sync_at."""
    user_org = await db_service.get_user_organization(current_user.id, organization_id)
    if not user_org or not user_org.can_manage_agents:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    org = await db_service.get_organization_by_id(organization_id)
    if not org:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")

    result = await crm_service.sync(organization_id)

    # Update last_sync_at in settings.crm
    settings = org.settings or {}
    crm_cfg = settings.get("crm", {})
    crm_cfg["last_sync_at"] = result.get("last_sync_at")
    settings["crm"] = crm_cfg
    await db_service.update_organization(organization_id, settings=settings)

    return result


@router.get("/{organization_id}/members", response_model=List[UserOrganizationResponse])
async def get_organization_members(
    organization_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get all members of an organization"""
    try:
        # Verify user has access to organization
        user_org = await db_service.get_user_organization(current_user.id, organization_id)
        if not user_org or not user_org.can_view_agents:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )

        members = await db_service.get_organization_members(organization_id)
        return [
            UserOrganizationResponse(
                id=uo.id,
                user_id=uo.user_id,
                organization_id=uo.organization_id,
                role=uo.role,
                is_active=uo.is_active,
                created_at=uo.created_at.isoformat(),
                user_name=uo.user.name,
                user_email=uo.user.email
            )
            for uo in members
        ]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching organization members: {str(e)}"
        )


@router.post("/{organization_id}/invite")
async def invite_user_to_organization(
    organization_id: int,
    invite_data: InviteUserRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Invite a user to join organization"""
    try:
        # Verify user can manage users
        user_org = await db_service.get_user_organization(current_user.id, organization_id)
        if not user_org or not user_org.can_manage_users:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )

        # Verify organization exists and check limits
        organization = await db_service.get_organization_by_id(organization_id)
        if not organization:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found"
            )

        # Check user limits
        current_user_count = await db_service.count_organization_users(organization_id)
        if organization.max_users != -1 and current_user_count >= organization.max_users:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Organization has reached user limit"
            )

        # Check if user already exists
        invited_user = await db_service.get_user_by_email(invite_data.email)
        if invited_user:
            # Check if already member
            existing_membership = await db_service.get_user_organization(
                invited_user.id, organization_id
            )
            if existing_membership:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="User is already a member of this organization"
                )

        # Generate invitation token
        invitation_token = generate_invitation_token()
        expires_at = datetime.utcnow() + timedelta(days=7)  # 7 days expiry

        # Create invitation
        await db_service.create_organization_invitation(
            email=invite_data.email,
            organization_id=organization_id,
            role=invite_data.role,
            invited_by_user_id=current_user.id,
            invitation_token=invitation_token,
            expires_at=expires_at
        )

        # TODO: Send invitation email

        return {
            "message": "Invitation sent successfully",
            "invitation_token": invitation_token,  # Remove this in production
            "expires_at": expires_at.isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error sending invitation: {str(e)}"
        )


@router.post("/accept-invite")
async def accept_organization_invitation(
    accept_data: AcceptInviteRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Accept organization invitation"""
    try:
        # Find and validate invitation
        invitation = await db_service.get_organization_invitation_by_token(
            accept_data.invitation_token
        )

        if not invitation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invalid invitation token"
            )

        if invitation.invitation_accepted_at:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invitation already accepted"
            )

        if invitation.invitation_expires_at < datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invitation has expired"
            )

        # Verify user email matches invitation
        if current_user.email != invitation.user.email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invitation email does not match current user"
            )

        # Accept invitation
        await db_service.accept_organization_invitation(
            invitation.id,
            current_user.id
        )

        return {"message": "Invitation accepted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error accepting invitation: {str(e)}"
        )


@router.delete("/{organization_id}/members/{user_id}")
async def remove_user_from_organization(
    organization_id: int,
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Remove user from organization"""
    try:
        # Verify user can manage users
        user_org = await db_service.get_user_organization(current_user.id, organization_id)
        if not user_org or not user_org.can_manage_users:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )

        # Can't remove yourself if you're the only owner
        if user_id == current_user.id:
            owners_count = await db_service.count_organization_owners(organization_id)
            if owners_count <= 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot remove the last owner of the organization"
                )

        success = await db_service.remove_user_from_organization(user_id, organization_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found in organization"
            )

        return {"message": "User removed from organization successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error removing user from organization: {str(e)}"
        )


@router.put("/{organization_id}/members/{user_id}/role")
async def update_user_role(
    organization_id: int,
    user_id: int,
    role_data: dict,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update user role in organization"""
    try:
        # Verify user can manage users
        user_org = await db_service.get_user_organization(current_user.id, organization_id)
        if not user_org or not user_org.can_manage_users:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )

        new_role = role_data.get("role")
        if new_role not in [role.value for role in OrganizationRole]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid role"
            )

        # Can't change your own role if you're the only owner
        if user_id == current_user.id and new_role != OrganizationRole.OWNER.value:
            owners_count = await db_service.count_organization_owners(organization_id)
            if owners_count <= 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot change role of the last owner"
                )

        success = await db_service.update_user_organization_role(
            user_id, organization_id, new_role
        )
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found in organization"
            )

        return {"message": "User role updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating user role: {str(e)}"
        )
