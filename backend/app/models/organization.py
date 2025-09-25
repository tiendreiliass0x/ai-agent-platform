"""
Organization model for multi-tenant support.
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..core.database import Base


class Organization(Base):
    __tablename__ = "organizations"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    website = Column(String(255), nullable=True)
    logo_url = Column(String(500), nullable=True)

    # Billing and subscription
    plan = Column(String(50), nullable=False, default="free")
    billing_email = Column(String(255), nullable=True)
    stripe_customer_id = Column(String(255), nullable=True)
    subscription_status = Column(String(50), nullable=False, default="active")

    # Organization settings
    settings = Column(JSON, nullable=True, default=dict)
    is_active = Column(Boolean, default=True, nullable=False)

    # Limits based on plan
    max_agents = Column(Integer, default=3)  # free: 3, pro: 50, enterprise: unlimited
    max_users = Column(Integer, default=1)   # free: 1, pro: 10, enterprise: unlimited
    max_documents_per_agent = Column(Integer, default=10)  # free: 10, pro: 1000, enterprise: unlimited

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)

    # Relationships
    user_organizations = relationship("UserOrganization", back_populates="organization", cascade="all, delete-orphan")
    agents = relationship("Agent", back_populates="organization", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Organization(id={self.id}, name='{self.name}', slug='{self.slug}')>"

    @property
    def active_users_count(self):
        """Get count of active users in organization"""
        return len([uo for uo in self.user_organizations if uo.is_active])

    @property
    def agents_count(self):
        """Get count of agents in organization"""
        return len([agent for agent in self.agents if agent.is_active])

    def can_add_user(self):
        """Check if organization can add more users"""
        if self.max_users == -1:  # unlimited
            return True
        return self.active_users_count < self.max_users

    def can_add_agent(self):
        """Check if organization can add more agents"""
        if self.max_agents == -1:  # unlimited
            return True
        return self.agents_count < self.max_agents