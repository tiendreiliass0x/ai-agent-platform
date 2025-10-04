"""add idempotency_key to agents

Revision ID: add_idempotency_key_to_agents
Revises: bb84a883744f
Create Date: 2025-01-03 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_idempotency_key_to_agents'
down_revision = 'bb84a883744f'
branch_labels = None
depends_on = None


def upgrade():
    """Add idempotency_key column to agents table"""
    # Add the idempotency_key column
    op.add_column('agents', sa.Column('idempotency_key', sa.String(255), nullable=True))
    
    # Create index for better performance
    op.create_index('ix_agents_idempotency_key', 'agents', ['idempotency_key'])


def downgrade():
    """Remove idempotency_key column from agents table"""
    # Drop the index first
    op.drop_index('ix_agents_idempotency_key', table_name='agents')
    
    # Drop the column
    op.drop_column('agents', 'idempotency_key')
