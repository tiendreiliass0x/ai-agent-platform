"""
Escalation model for auto-escalation to human support with context.
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from ..core.database import Base


class Escalation(Base):
    __tablename__ = "escalations"

    id = Column(Integer, primary_key=True, index=True)

    # Foreign references
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=True)
    customer_profile_id = Column(Integer, ForeignKey("customer_profiles.id"), nullable=True)

    # Escalation details
    priority = Column(String(20), default="normal")  # normal, high, critical
    status = Column(String(20), default="open")      # open, in_progress, resolved
    reason = Column(String(100), nullable=True)       # auto_detection, user_request, vip, etc.
    summary = Column(Text, nullable=True)
    details = Column(JSON, default=dict)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships (optional)
    agent = relationship("Agent")
    conversation = relationship("Conversation")
    customer_profile = relationship("CustomerProfile")

    def __repr__(self):
        return f"<Escalation(id={self.id}, agent_id={self.agent_id}, status='{self.status}', priority='{self.priority}')>"

