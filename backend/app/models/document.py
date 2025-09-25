from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from ..core.database import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    content_type = Column(String, nullable=False)  # Changed from file_type
    size = Column(Integer, nullable=False)  # Changed from file_size
    content = Column(Text, nullable=True)  # Extracted text content
    content_hash = Column(String, nullable=True)  # For deduplication
    url = Column(String, nullable=True)  # For web pages

    # Vector storage info
    vector_ids = Column(JSON, default=[])  # List of vector IDs in Pinecone
    chunk_count = Column(Integer, default=0)

    # Processing status
    status = Column(String, default="pending")  # pending, processing, completed, failed
    error_message = Column(Text, nullable=True)

    # Metadata
    doc_metadata = Column(JSON, default={})

    # Agent relationship
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    agent = relationship("Agent", back_populates="documents")