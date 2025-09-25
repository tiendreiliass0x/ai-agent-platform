"""
add escalations table

Revision ID: add_escalations_table_0001
Revises: a56a8e8a9ecd
Create Date: 2025-09-24 00:01:00
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_escalations_table_0001'
down_revision = 'a56a8e8a9ecd'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'escalations',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('agent_id', sa.Integer(), sa.ForeignKey('agents.id'), nullable=False),
        sa.Column('conversation_id', sa.Integer(), sa.ForeignKey('conversations.id'), nullable=True),
        sa.Column('customer_profile_id', sa.Integer(), sa.ForeignKey('customer_profiles.id'), nullable=True),
        sa.Column('priority', sa.String(length=20), nullable=False, server_default='normal'),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='open'),
        sa.Column('reason', sa.String(length=100), nullable=True),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('details', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True)
    )


def downgrade() -> None:
    op.drop_table('escalations')

