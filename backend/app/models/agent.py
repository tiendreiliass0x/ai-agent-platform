from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey, JSON, Float, Enum as SQLEnum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from enum import Enum as PyEnum
from ..core.database import Base

class AgentTier(PyEnum):
    """Agent subscription tiers"""
    BASIC = "basic"           # Standard knowledge base only
    PROFESSIONAL = "professional"  # + Domain expertise + web search
    ENTERPRISE = "enterprise"      # + Custom training + advanced analytics

class DomainExpertiseType(PyEnum):
    """Types of domain expertise"""
    SALES_REP = "sales_rep"               # Sales representative with product knowledge
    SOLUTION_ENGINEER = "solution_engineer" # Technical solution specialist
    SUPPORT_EXPERT = "support_expert"      # Expert support with deep troubleshooting
    DOMAIN_SPECIALIST = "domain_specialist" # Industry/niche domain expert
    PRODUCT_EXPERT = "product_expert"      # Deep product knowledge specialist

class Agent(Base):
    __tablename__ = "agents"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    system_prompt = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)

    # Tier and Expertise Configuration
    tier = Column(SQLEnum(AgentTier), default=AgentTier.BASIC)
    domain_expertise_enabled = Column(Boolean, default=False)

    # Domain Expertise Links
    persona_id = Column(Integer, ForeignKey("personas.id"), nullable=True)
    knowledge_pack_id = Column(Integer, ForeignKey("knowledge_packs.id"), nullable=True)

    # Tool and Grounding Configuration
    tool_policy = Column(JSON, default=lambda: {
        "web_search": False,
        "site_search": [],
        "code_exec": False
    })
    grounding_mode = Column(String, default="blended")  # "strict" or "blended"

    # Configuration
    config = Column(JSON, default=dict)  # Model settings, temperature, etc.
    widget_config = Column(JSON, default=dict)  # Widget appearance, position, etc.

    # Owner and Organization
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)  # Created by user
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)  # Belongs to organization

    # API key for embedding
    api_key = Column(String, unique=True, index=True)

    # Usage tracking
    total_conversations = Column(Integer, default=0)
    total_messages = Column(Integer, default=0)
    last_used_at = Column(DateTime(timezone=True), nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="agents")  # Created by user
    organization = relationship("Organization", back_populates="agents")  # Belongs to organization
    documents = relationship("Document", back_populates="agent")
    conversations = relationship("Conversation", back_populates="agent")
    persona = relationship("Persona", back_populates="agents")
    knowledge_pack = relationship("KnowledgePack", back_populates="agents")
