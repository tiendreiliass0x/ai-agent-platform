from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    JSON,
    Boolean,
    UniqueConstraint,
)
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from ..core.database import Base


class Tool(Base):
    __tablename__ = "tools"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    display_name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    auth = Column(JSON, default=dict)
    governance = Column(JSON, default=dict)
    rate_limits = Column(JSON, default=dict)
    schemas = Column(JSON, default=dict)
    latest_version = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    manifests = relationship("ToolManifest", back_populates="tool", cascade="all, delete-orphan")


class ToolManifest(Base):
    __tablename__ = "tool_manifests"

    id = Column(Integer, primary_key=True)
    tool_id = Column(Integer, ForeignKey("tools.id", ondelete="CASCADE"), nullable=False)
    version = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    manifest = Column(JSON, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    tool = relationship("Tool", back_populates="manifests")
    operations = relationship("ToolOperation", back_populates="manifest", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("tool_id", "version", name="uq_tool_manifest_version"),
    )


class ToolOperation(Base):
    __tablename__ = "tool_operations"

    id = Column(Integer, primary_key=True)
    manifest_id = Column(Integer, ForeignKey("tool_manifests.id", ondelete="CASCADE"), nullable=False)
    op_id = Column(String, nullable=False)
    method = Column(String, nullable=False)
    path = Column(String, nullable=False)
    side_effect = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    args_schema = Column(JSON, default=dict)
    args_schema_hash = Column(String, nullable=True)
    returns_schema = Column(JSON, default=dict)
    preconditions = Column(JSON, default=list)
    postconditions = Column(JSON, default=list)
    idempotency_header = Column(String, nullable=True)
    requires_approval = Column(Boolean, default=False)
    compensation = Column(JSON, default=dict)
    errors = Column(JSON, default=list)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    manifest = relationship("ToolManifest", back_populates="operations")

    __table_args__ = (
        UniqueConstraint("manifest_id", "op_id", name="uq_tool_operation"),
    )
