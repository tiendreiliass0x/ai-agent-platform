"""
Customer Profile model for advanced visitor tracking and personalization.
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, JSON, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy import ForeignKey
from ..core.database import Base


class CustomerProfile(Base):
    __tablename__ = "customer_profiles"

    id = Column(Integer, primary_key=True, index=True)

    # Identity & Tracking
    visitor_id = Column(String(255), unique=True, nullable=False, index=True)  # Anonymous visitor ID
    email = Column(String(255), nullable=True, index=True)  # If known
    phone = Column(String(50), nullable=True)
    name = Column(String(255), nullable=True)

    # Agent association
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False)

    # Behavioral Data
    communication_style = Column(String(50), default="neutral")  # formal, casual, technical, friendly
    preferred_language = Column(String(10), default="en")
    response_length_preference = Column(String(20), default="medium")  # brief, medium, detailed
    technical_level = Column(String(20), default="intermediate")  # beginner, intermediate, expert

    # Engagement Metrics
    total_conversations = Column(Integer, default=0)
    total_messages = Column(Integer, default=0)
    avg_session_duration = Column(Float, default=0.0)  # minutes
    satisfaction_score = Column(Float, nullable=True)  # 1-5 rating

    # Context & Intent
    primary_interests = Column(JSON, default=list)  # ["pricing", "features", "support"]
    pain_points = Column(JSON, default=list)  # Identified issues
    goals = Column(JSON, default=list)  # Customer objectives
    current_journey_stage = Column(String(50), default="awareness")  # awareness, consideration, decision, retention

    # Personalization Data
    personality_traits = Column(JSON, default=dict)  # Inferred traits
    preferences = Column(JSON, default=dict)  # User preferences
    context_memory = Column(JSON, default=dict)  # Long-term context

    # Session Data
    last_seen_at = Column(DateTime(timezone=True), nullable=True)
    last_page_visited = Column(String(500), nullable=True)
    referrer_source = Column(String(500), nullable=True)
    device_info = Column(JSON, default=dict)
    location_data = Column(JSON, default=dict)

    # Status
    is_active = Column(Boolean, default=True)
    is_vip = Column(Boolean, default=False)
    status = Column(String(50), default="active")  # active, inactive, blocked

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)

    # Relationships
    agent = relationship("Agent")
    conversations = relationship("Conversation", back_populates="customer_profile")
    memory_entries = relationship("CustomerMemory", back_populates="customer_profile", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<CustomerProfile(visitor_id='{self.visitor_id}', agent_id={self.agent_id})>"

    @property
    def display_name(self):
        """Get display name for the customer"""
        if self.name:
            return self.name
        elif self.email:
            return self.email.split('@')[0]
        else:
            return f"Visitor #{self.visitor_id[:8]}"

    @property
    def engagement_level(self):
        """Calculate engagement level based on interactions"""
        if self.total_conversations == 0:
            return "new"
        elif self.total_conversations < 3:
            return "exploring"
        elif self.total_conversations < 10:
            return "engaged"
        else:
            return "highly_engaged"

    @property
    def is_returning_customer(self):
        """Check if this is a returning customer"""
        return self.total_conversations > 1

    def update_engagement_metrics(self, session_duration: float = None, message_count: int = 1):
        """Update engagement metrics after interaction"""
        self.total_messages += message_count
        if session_duration:
            # Update average session duration
            total_duration = self.avg_session_duration * self.total_conversations
            self.total_conversations += 1
            self.avg_session_duration = (total_duration + session_duration) / self.total_conversations

        from datetime import datetime
        self.last_seen_at = datetime.utcnow()

    def add_interest(self, interest: str):
        """Add to primary interests if not already present"""
        if not self.primary_interests:
            self.primary_interests = []
        if interest not in self.primary_interests:
            self.primary_interests.append(interest)

    def add_pain_point(self, pain_point: str):
        """Add identified pain point"""
        if not self.pain_points:
            self.pain_points = []
        if pain_point not in self.pain_points:
            self.pain_points.append(pain_point)

    def set_preference(self, key: str, value: any):
        """Set a user preference"""
        if not self.preferences:
            self.preferences = {}
        self.preferences[key] = value

    def get_preference(self, key: str, default=None):
        """Get a user preference"""
        if not self.preferences:
            return default
        return self.preferences.get(key, default)