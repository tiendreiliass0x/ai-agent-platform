"""
Customer Memory model for multi-layer memory architecture.
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Float, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy import ForeignKey
from enum import Enum
from ..core.database import Base


class MemoryType(str, Enum):
    """Types of memory entries"""
    FACTUAL = "factual"              # Hard facts about customer
    PREFERENCE = "preference"        # Learned preferences
    BEHAVIORAL = "behavioral"        # Behavior patterns
    CONTEXTUAL = "contextual"        # Situational context
    EPISODIC = "episodic"           # Specific interactions/events
    PROCEDURAL = "procedural"        # Learned workflows/patterns
    EMOTIONAL = "emotional"          # Sentiment and emotional context


class MemoryImportance(str, Enum):
    """Importance levels for memory decay"""
    CRITICAL = "critical"    # Never decay
    HIGH = "high"           # Slow decay
    MEDIUM = "medium"       # Normal decay
    LOW = "low"            # Fast decay


class CustomerMemory(Base):
    __tablename__ = "customer_memory"

    id = Column(Integer, primary_key=True, index=True)

    # Associations
    customer_profile_id = Column(Integer, ForeignKey("customer_profiles.id"), nullable=False)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=True)  # Source conversation

    # Memory Content
    memory_type = Column(String(20), nullable=False, default=MemoryType.FACTUAL.value)
    key = Column(String(255), nullable=False, index=True)  # Memory key/identifier
    value = Column(Text, nullable=False)  # Memory content
    context = Column(JSON, default=dict)  # Additional context

    # Memory Metadata
    importance = Column(String(20), default=MemoryImportance.MEDIUM.value)
    confidence_score = Column(Float, default=1.0)  # 0-1, how confident we are
    source = Column(String(100), default="conversation")  # Where this memory came from
    tags = Column(JSON, default=list)  # Tags for categorization

    # Temporal Data
    valid_from = Column(DateTime(timezone=True), server_default=func.now())
    valid_until = Column(DateTime(timezone=True), nullable=True)  # For time-sensitive info
    last_accessed = Column(DateTime(timezone=True), server_default=func.now())
    access_count = Column(Integer, default=0)

    # Memory State
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)  # Has been confirmed by customer
    decay_rate = Column(Float, default=0.1)  # How fast this memory decays

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)

    # Relationships
    customer_profile = relationship("CustomerProfile", back_populates="memory_entries")
    conversation = relationship("Conversation")

    def __repr__(self):
        return f"<CustomerMemory(key='{self.key}', type='{self.memory_type}', importance='{self.importance}')>"

    def access(self):
        """Update access tracking when memory is retrieved"""
        from datetime import datetime
        self.last_accessed = datetime.utcnow()
        self.access_count += 1

    def verify(self):
        """Mark memory as verified by customer"""
        self.is_verified = True
        self.confidence_score = 1.0

    def invalidate(self):
        """Mark memory as invalid/outdated"""
        self.is_active = False

    def update_confidence(self, new_confidence: float):
        """Update confidence score"""
        self.confidence_score = max(0.0, min(1.0, new_confidence))

    @property
    def is_expired(self):
        """Check if memory has expired"""
        if not self.valid_until:
            return False
        from datetime import datetime
        return datetime.utcnow() > self.valid_until.replace(tzinfo=None)

    @property
    def relevance_score(self):
        """Calculate relevance based on recency, frequency, and importance"""
        # Recency score (more recent = higher score)
        import datetime
        now = datetime.datetime.utcnow()
        hours_since_access = (now - self.last_accessed.replace(tzinfo=None)).total_seconds() / 3600
        recency_score = max(0, 1 - (hours_since_access / (24 * 7)))  # Decay over a week

        # Frequency score
        frequency_score = min(1.0, self.access_count / 10)  # Cap at 10 accesses

        # Importance weight
        importance_weights = {
            MemoryImportance.CRITICAL.value: 1.0,
            MemoryImportance.HIGH.value: 0.8,
            MemoryImportance.MEDIUM.value: 0.6,
            MemoryImportance.LOW.value: 0.4
        }
        importance_weight = importance_weights.get(self.importance, 0.6)

        return (recency_score * 0.4 + frequency_score * 0.3 + importance_weight * 0.3) * self.confidence_score

    def add_tag(self, tag: str):
        """Add a tag to this memory"""
        if not self.tags:
            self.tags = []
        if tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str):
        """Remove a tag from this memory"""
        if self.tags and tag in self.tags:
            self.tags.remove(tag)