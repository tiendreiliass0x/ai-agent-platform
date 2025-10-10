"""Create tool registry tables"""

from collections.abc import Sequence
from typing import Optional

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "2025_10_01_1230"
down_revision: Optional[str] = "2025_10_01_1200"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "tools",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("display_name", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("auth", sa.JSON(), nullable=False),
        sa.Column("governance", sa.JSON(), nullable=False),
        sa.Column("rate_limits", sa.JSON(), nullable=False),
        sa.Column("schemas", sa.JSON(), nullable=False),
        sa.Column("latest_version", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_tools_name", "tools", ["name"], unique=True)

    op.create_table(
        "tool_manifests",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("tool_id", sa.Integer(), sa.ForeignKey("tools.id", ondelete="CASCADE"), nullable=False),
        sa.Column("version", sa.String(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("manifest", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("tool_id", "version", name="uq_tool_manifest_version"),
    )

    op.create_table(
        "tool_operations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("manifest_id", sa.Integer(), sa.ForeignKey("tool_manifests.id", ondelete="CASCADE"), nullable=False),
        sa.Column("op_id", sa.String(), nullable=False),
        sa.Column("method", sa.String(), nullable=False),
        sa.Column("path", sa.String(), nullable=False),
        sa.Column("side_effect", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("args_schema", sa.JSON(), nullable=False),
        sa.Column("args_schema_hash", sa.String(length=64), nullable=True),
        sa.Column("returns_schema", sa.JSON(), nullable=False),
        sa.Column("preconditions", sa.JSON(), nullable=False),
        sa.Column("postconditions", sa.JSON(), nullable=False),
        sa.Column("idempotency_header", sa.String(), nullable=True),
        sa.Column("requires_approval", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("compensation", sa.JSON(), nullable=False),
        sa.Column("errors", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("manifest_id", "op_id", name="uq_tool_operation"),
    )


def downgrade() -> None:
    op.drop_table("tool_operations")
    op.drop_table("tool_manifests")
    op.drop_index("ix_tools_name", table_name="tools")
    op.drop_table("tools")
