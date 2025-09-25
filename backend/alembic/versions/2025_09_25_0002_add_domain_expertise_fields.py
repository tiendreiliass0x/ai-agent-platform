"""add domain expertise fields to agents

Revision ID: add_domain_expertise_fields
Revises: add_escalations_table_0001
Create Date: 2025-09-25 00:02:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'add_domain_expertise_fields'
down_revision = 'add_escalations_table_0001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    agent_tier_enum = sa.Enum('basic', 'professional', 'enterprise', name='agenttier')
    domain_expertise_enum = sa.Enum(
        'sales_rep',
        'solution_engineer',
        'support_expert',
        'domain_specialist',
        'product_expert',
        name='domainexpertisetype'
    )

    bind = op.get_bind()
    agent_tier_enum.create(bind, checkfirst=True)
    domain_expertise_enum.create(bind, checkfirst=True)

    op.add_column('agents', sa.Column('tier', agent_tier_enum, nullable=True, server_default='basic'))
    op.add_column('agents', sa.Column('domain_expertise_type', domain_expertise_enum, nullable=True))
    op.add_column('agents', sa.Column('personality_profile', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")))
    op.add_column('agents', sa.Column('expertise_level', sa.Float(), nullable=False, server_default='0.7'))
    op.add_column('agents', sa.Column('domain_knowledge_sources', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")))
    op.add_column('agents', sa.Column('web_search_enabled', sa.Boolean(), nullable=False, server_default=sa.text('false')))
    op.add_column('agents', sa.Column('custom_training_data', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")))
    op.add_column('agents', sa.Column('expert_context', sa.Text(), nullable=True))
    op.add_column('agents', sa.Column('tool_policy', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")))
    op.add_column('agents', sa.Column('grounding_mode', sa.String(), nullable=False, server_default='blended'))
    op.add_column('agents', sa.Column('domain_expertise_enabled', sa.Boolean(), nullable=False, server_default=sa.text('false')))

    # Ensure defaults persisted for existing rows
    op.execute("UPDATE agents SET tier = 'basic' WHERE tier IS NULL")
    op.execute("UPDATE agents SET personality_profile = '{}'::jsonb WHERE personality_profile IS NULL")
    op.execute("UPDATE agents SET domain_knowledge_sources = '[]'::jsonb WHERE domain_knowledge_sources IS NULL")
    op.execute("UPDATE agents SET custom_training_data = '{}'::jsonb WHERE custom_training_data IS NULL")
    op.execute("UPDATE agents SET tool_policy = '{}'::jsonb WHERE tool_policy IS NULL")
    op.execute("UPDATE agents SET grounding_mode = 'blended' WHERE grounding_mode IS NULL")
    op.execute("UPDATE agents SET expertise_level = 0.7 WHERE expertise_level IS NULL")
    op.execute("UPDATE agents SET web_search_enabled = false WHERE web_search_enabled IS NULL")
    op.execute("UPDATE agents SET domain_expertise_enabled = false WHERE domain_expertise_enabled IS NULL")


def downgrade() -> None:
    op.drop_column('agents', 'domain_expertise_enabled')
    op.drop_column('agents', 'grounding_mode')
    op.drop_column('agents', 'tool_policy')
    op.drop_column('agents', 'expert_context')
    op.drop_column('agents', 'custom_training_data')
    op.drop_column('agents', 'web_search_enabled')
    op.drop_column('agents', 'domain_knowledge_sources')
    op.drop_column('agents', 'expertise_level')
    op.drop_column('agents', 'personality_profile')
    op.drop_column('agents', 'domain_expertise_type')
    op.drop_column('agents', 'tier')

    bind = op.get_bind()
    sa.Enum(name='domainexpertisetype').drop(bind, checkfirst=True)
    sa.Enum(name='agenttier').drop(bind, checkfirst=True)
