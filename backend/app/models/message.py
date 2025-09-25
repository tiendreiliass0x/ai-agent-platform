from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from ..core.database import Base

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    role = Column(String, nullable=False)  # user, assistant, system

    # Conversation
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)

    # Metadata
    msg_metadata = Column(JSON, default={})
    token_count = Column(Integer, nullable=True)
    response_time = Column(Integer, nullable=True)  # milliseconds

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")