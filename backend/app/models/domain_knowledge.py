"""
Domain Knowledge Model
Stores domain-specific expertise, custom training data, and specialized knowledge sources
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey, JSON, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from enum import Enum as PyEnum
from ..core.database import Base


class KnowledgeSourceType(PyEnum):
    """Types of knowledge sources"""
    INDUSTRY_DATA = "industry_data"           # Industry-specific datasets
    PRODUCT_SPECS = "product_specs"           # Detailed product specifications
    COMPETITIVE_INTEL = "competitive_intel"   # Competitive analysis data
    CASE_STUDIES = "case_studies"             # Customer success stories
    TECHNICAL_DOCS = "technical_docs"         # Deep technical documentation
    SALES_MATERIALS = "sales_materials"       # Sales playbooks, objection handling
    SUPPORT_KNOWLEDGE = "support_knowledge"   # Advanced troubleshooting guides
    CUSTOM_TRAINING = "custom_training"       # Custom domain training data


class DomainKnowledge(Base):
    """Domain-specific knowledge for premium agents"""
    __tablename__ = "domain_knowledge"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False)

    # Knowledge Source Details
    source_type = Column(String, nullable=False)  # KnowledgeSourceType
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    content = Column(Text, nullable=False)  # The actual knowledge content

    # Vector Embeddings for Semantic Search
    embedding = Column(JSON, nullable=True)  # Vector representation
    embedding_model = Column(String, default="text-embedding-004")

    # Metadata
    source_url = Column(String, nullable=True)
    confidence_score = Column(Float, default=0.8)  # Confidence in this knowledge
    importance_weight = Column(Float, default=1.0)  # Relative importance

    # Usage Statistics
    usage_count = Column(Integer, default=0)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    effectiveness_score = Column(Float, nullable=True)  # User feedback score

    # Processing Status
    is_processed = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    processing_metadata = Column(JSON, default={})

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    agent = relationship("Agent", back_populates="domain_knowledge")


class ExpertPersona(Base):
    """Defines expert personality and behavior patterns"""
    __tablename__ = "expert_personas"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False)

    # Personality Configuration
    persona_name = Column(String, nullable=False)  # e.g., "Senior Solutions Engineer"
    expertise_areas = Column(JSON, default=[])     # List of expertise areas
    communication_style = Column(JSON, default={}) # Tone, formality, technical depth

    # Behavioral Patterns
    response_patterns = Column(JSON, default={})   # How they structure responses
    preferred_examples = Column(JSON, default=[])  # Types of examples they give
    escalation_triggers = Column(JSON, default=[]) # When to escalate or refer

    # Knowledge Synthesis Preferences
    information_priorities = Column(JSON, default={})  # What info to prioritize
    context_weighting = Column(JSON, default={})       # How to weight different contexts

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class WebSearchQuery(Base):
    """Tracks web search queries and results for domain experts"""
    __tablename__ = "web_search_queries"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False)

    # Query Details
    original_query = Column(Text, nullable=False)    # User's original question
    search_query = Column(Text, nullable=False)      # Optimized search query
    search_provider = Column(String, default="serpapi")  # Which search API used

    # Results
    search_results = Column(JSON, default=[])        # Raw search results
    processed_results = Column(JSON, default=[])     # Filtered and processed
    relevance_scores = Column(JSON, default=[])      # Relevance scoring

    # Context
    conversation_context = Column(JSON, default={})  # Context from conversation
    domain_context = Column(JSON, default={})        # Domain-specific context

    # Usage
    was_helpful = Column(Boolean, nullable=True)     # User feedback
    response_quality = Column(Float, nullable=True)  # Quality score

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())