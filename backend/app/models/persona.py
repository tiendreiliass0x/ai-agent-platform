"""
Persona Models - Role-based agent personalities and tactics
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from ..core.database import Base


class Persona(Base):
    """Templated roles with specific tactics and behavior patterns"""
    __tablename__ = "personas"

    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)

    # Persona Definition
    name = Column(String, nullable=False)  # "Sales Rep", "Solutions Engineer", etc.
    description = Column(Text, nullable=True)
    system_prompt = Column(Text, nullable=False)  # Base role prompt

    # Behavior Configuration
    tactics = Column(JSON, nullable=False, default={})  # {"style":"executive", "steps":[...]}
    communication_style = Column(JSON, default={})     # Tone, formality, depth
    response_patterns = Column(JSON, default={})       # How they structure responses

    # Built-in vs Custom
    is_built_in = Column(String, default=False)  # True for our templates
    template_name = Column(String, nullable=True)  # "sales_rep", "solutions_engineer", etc.

    # Usage Tracking
    usage_count = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    organization = relationship("Organization", back_populates="personas")
    agents = relationship("Agent", back_populates="persona")


class KnowledgePack(Base):
    """Named bundle of knowledge sources for domain expertise"""
    __tablename__ = "knowledge_packs"

    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)

    # Pack Definition
    name = Column(String, nullable=False)  # "Acme Sales Playbook"
    description = Column(Text, nullable=True)

    # Grounding Configuration
    grounding_mode = Column(String, default="blended")  # "strict" or "blended"

    # Freshness Policy
    freshness_policy = Column(JSON, default={
        "ttl_days": 30,
        "recrawl": "changed",
        "priority": "medium"
    })

    # Health Metrics
    document_count = Column(Integer, default=0)
    last_crawl_at = Column(DateTime(timezone=True), nullable=True)
    broken_links_count = Column(Integer, default=0)
    coverage_score = Column(String, nullable=True)  # Health indicator

    # Usage Tracking
    usage_count = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    organization = relationship("Organization", back_populates="knowledge_packs")
    sources = relationship("KnowledgePackSource", back_populates="knowledge_pack")
    agents = relationship("Agent", back_populates="knowledge_pack")


class KnowledgePackSource(Base):
    """Links knowledge packs to document sources"""
    __tablename__ = "knowledge_pack_sources"

    id = Column(Integer, primary_key=True, index=True)
    pack_id = Column(Integer, ForeignKey("knowledge_packs.id"), nullable=False)
    source_id = Column(Integer, nullable=False)  # Points to documents table

    # Source Metadata
    source_type = Column(String, default="document")  # "document", "url", "manual"
    importance_weight = Column(String, default=1.0)

    # Processing Status
    is_active = Column(String, default=True)
    last_processed_at = Column(DateTime(timezone=True), nullable=True)
    processing_status = Column(String, default="pending")  # pending, processed, failed

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    knowledge_pack = relationship("KnowledgePack", back_populates="sources")