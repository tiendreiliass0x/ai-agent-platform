"""Add domain expertise tables

Revision ID: 439c9624fda1
Revises: add_domain_expertise_fields
Create Date: 2025-09-25 18:03:22.475495+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '439c9624fda1'
down_revision: Union[str, None] = 'add_domain_expertise_fields'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create personas table
    op.create_table(
        'personas',
        sa.Column('id', sa.Integer, primary_key=True, index=True),
        sa.Column('name', sa.String, nullable=False),
        sa.Column('template_name', sa.String, nullable=True, index=True),
        sa.Column('system_prompt', sa.Text, nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('tactics', sa.JSON, nullable=True),
        sa.Column('communication_style', sa.JSON, nullable=True),
        sa.Column('response_patterns', sa.JSON, nullable=True),
        sa.Column('is_built_in', sa.Boolean, default=False),
        sa.Column('organization_id', sa.Integer, sa.ForeignKey('organizations.id'), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now())
    )

    # Create knowledge_packs table
    op.create_table(
        'knowledge_packs',
        sa.Column('id', sa.Integer, primary_key=True, index=True),
        sa.Column('name', sa.String, nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('freshness_policy', sa.JSON, nullable=True),
        sa.Column('organization_id', sa.Integer, sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now())
    )

    # Create knowledge_pack_sources junction table
    op.create_table(
        'knowledge_pack_sources',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('knowledge_pack_id', sa.Integer, sa.ForeignKey('knowledge_packs.id'), nullable=False),
        sa.Column('source_id', sa.Integer, nullable=False),  # Document ID or other source
        sa.Column('source_type', sa.String, default='document'),  # 'document', 'url', etc.
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now())
    )

    # Add missing columns to agents table
    op.add_column('agents', sa.Column('persona_id', sa.Integer, nullable=True))
    op.add_column('agents', sa.Column('knowledge_pack_id', sa.Integer, nullable=True))

    # Add foreign key constraints to agents table
    op.create_foreign_key('fk_agents_persona_id', 'agents', 'personas', ['persona_id'], ['id'])
    op.create_foreign_key('fk_agents_knowledge_pack_id', 'agents', 'knowledge_packs', ['knowledge_pack_id'], ['id'])


def downgrade() -> None:
    # Drop foreign keys first
    op.drop_constraint('fk_agents_knowledge_pack_id', 'agents')
    op.drop_constraint('fk_agents_persona_id', 'agents')

    # Drop columns from agents table
    op.drop_column('agents', 'knowledge_pack_id')
    op.drop_column('agents', 'persona_id')

    # Drop tables
    op.drop_table('knowledge_pack_sources')
    op.drop_table('knowledge_packs')
    op.drop_table('personas')