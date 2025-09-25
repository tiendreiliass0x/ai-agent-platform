from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from ..core.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)  # Changed from hashed_password
    name = Column(String, nullable=True)  # Changed from full_name
    plan = Column(String, default="free")  # Added plan field
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    agents = relationship("Agent", back_populates="user")  # Changed from owner
    user_organizations = relationship("UserOrganization", foreign_keys="UserOrganization.user_id", back_populates="user", cascade="all, delete-orphan")

    @property
    def active_organizations(self):
        """Get all active organizations for this user"""
        return [uo.organization for uo in self.user_organizations if uo.is_active and uo.organization.is_active]

    @property
    def owned_organizations(self):
        """Get organizations where user is owner"""
        return [uo.organization for uo in self.user_organizations if uo.role == "owner" and uo.is_active]

    def get_organization_role(self, organization_id: int):
        """Get user's role in a specific organization"""
        for uo in self.user_organizations:
            if uo.organization_id == organization_id and uo.is_active:
                return uo.role
        return None

    def has_organization_permission(self, organization_id: int, permission: str):
        """Check if user has specific permission in organization"""
        uo = next((uo for uo in self.user_organizations if uo.organization_id == organization_id and uo.is_active), None)
        if not uo:
            return False

        if permission == "manage_users":
            return uo.can_manage_users
        elif permission == "manage_agents":
            return uo.can_manage_agents
        elif permission == "view_agents":
            return uo.can_view_agents
        elif permission == "manage_billing":
            return uo.can_manage_billing
        elif permission == "delete_organization":
            return uo.can_delete_organization

        return False