"""add_public_id_to_agents

Revision ID: bb84a883744f
Revises: 2958e8d7d2b0
Create Date: 2025-09-30 00:01:58.619247+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'bb84a883744f'
down_revision: Union[str, None] = '2958e8d7d2b0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add public_id column to agents table
    op.add_column('agents', sa.Column('public_id', sa.String(), nullable=True))

    # Create unique index on public_id
    op.create_index('ix_agents_public_id', 'agents', ['public_id'], unique=True)

    # Generate UUIDs for existing agents
    op.execute("""
        UPDATE agents
        SET public_id = gen_random_uuid()::text
        WHERE public_id IS NULL
    """)

    # Make public_id NOT NULL after populating existing records
    op.alter_column('agents', 'public_id', nullable=False)


def downgrade() -> None:
    # Remove public_id column from agents table
    op.drop_index('ix_agents_public_id', table_name='agents')
    op.drop_column('agents', 'public_id')