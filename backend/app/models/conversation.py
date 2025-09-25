from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from ..core.database import Base

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)

    # Participants
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # Optional for anonymous
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False)
    customer_profile_id = Column(Integer, ForeignKey("customer_profiles.id"), nullable=True)  # Customer profile

    # Metadata
    conv_metadata = Column(JSON, default={})
    user_ip = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    referrer = Column(String, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User")  # Removed back_populates since User doesn't have conversations
    agent = relationship("Agent", back_populates="conversations")
    customer_profile = relationship("CustomerProfile", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")