"""
UserOrganization model for multi-user organization support.
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from enum import Enum
from ..core.database import Base


class OrganizationRole(str, Enum):
    """Organization roles with hierarchical permissions"""
    OWNER = "owner"           # Full access, can delete org, manage billing
    ADMIN = "admin"           # Can manage users, agents, settings (except billing)
    EDITOR = "editor"         # Can create/edit agents and documents
    VIEWER = "viewer"         # Read-only access to agents and documents


class UserOrganization(Base):
    __tablename__ = "user_organizations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # Allow NULL for pending invitations
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)

    # Role and permissions
    role = Column(String(20), nullable=False, default=OrganizationRole.VIEWER.value)
    is_active = Column(Boolean, default=True, nullable=False)

    # Invitation tracking
    invited_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    invitation_token = Column(String(255), nullable=True, unique=True)
    invitation_expires_at = Column(DateTime(timezone=True), nullable=True)
    invitation_accepted_at = Column(DateTime(timezone=True), nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)

    # Relationships
    user = relationship("User", foreign_keys=[user_id], back_populates="user_organizations")
    organization = relationship("Organization", back_populates="user_organizations")
    invited_by = relationship("User", foreign_keys=[invited_by_user_id])

    # Constraints
    __table_args__ = (
        UniqueConstraint('user_id', 'organization_id', name='unique_user_organization'),
    )

    def __repr__(self):
        return f"<UserOrganization(user_id={self.user_id}, org_id={self.organization_id}, role='{self.role}')>"

    @property
    def can_manage_users(self):
        """Check if user can manage other users in organization"""
        return self.role in [OrganizationRole.OWNER.value, OrganizationRole.ADMIN.value]

    @property
    def can_manage_agents(self):
        """Check if user can create/edit agents"""
        return self.role in [OrganizationRole.OWNER.value, OrganizationRole.ADMIN.value, OrganizationRole.EDITOR.value]

    @property
    def can_view_agents(self):
        """Check if user can view agents"""
        return self.role in [OrganizationRole.OWNER.value, OrganizationRole.ADMIN.value, OrganizationRole.EDITOR.value, OrganizationRole.VIEWER.value]

    @property
    def can_manage_billing(self):
        """Check if user can manage billing and subscription"""
        return self.role == OrganizationRole.OWNER.value

    @property
    def can_delete_organization(self):
        """Check if user can delete the organization"""
        return self.role == OrganizationRole.OWNER.value

    @property
    def is_pending_invitation(self):
        """Check if this is a pending invitation"""
        return (
            self.invitation_token is not None and
            self.invitation_accepted_at is None and
            self.invitation_expires_at is not None and
            self.invitation_expires_at > func.now()
        )