"""add_user_organization_permissions

Revision ID: 2958e8d7d2b0
Revises: 439c9624fda1
Create Date: 2025-09-29 07:00:29.802163+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2958e8d7d2b0'
down_revision: Union[str, None] = '439c9624fda1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add permission columns to user_organizations table
    op.add_column('user_organizations', sa.Column('can_manage_users', sa.Boolean(), nullable=False, server_default='false'))
    op.add_column('user_organizations', sa.Column('can_manage_agents', sa.Boolean(), nullable=False, server_default='false'))
    op.add_column('user_organizations', sa.Column('can_view_agents', sa.Boolean(), nullable=False, server_default='true'))
    op.add_column('user_organizations', sa.Column('can_manage_billing', sa.Boolean(), nullable=False, server_default='false'))
    op.add_column('user_organizations', sa.Column('can_delete_organization', sa.Boolean(), nullable=False, server_default='false'))

    # Update existing records based on role
    # Owners get all permissions
    op.execute("""
        UPDATE user_organizations
        SET can_manage_users = true,
            can_manage_agents = true,
            can_view_agents = true,
            can_manage_billing = true,
            can_delete_organization = true
        WHERE role = 'owner'
    """)

    # Admins get most permissions except delete organization
    op.execute("""
        UPDATE user_organizations
        SET can_manage_users = true,
            can_manage_agents = true,
            can_view_agents = true,
            can_manage_billing = true,
            can_delete_organization = false
        WHERE role = 'admin'
    """)

    # Members get limited permissions
    op.execute("""
        UPDATE user_organizations
        SET can_manage_users = false,
            can_manage_agents = false,
            can_view_agents = true,
            can_manage_billing = false,
            can_delete_organization = false
        WHERE role = 'member'
    """)


def downgrade() -> None:
    # Remove permission columns from user_organizations table
    op.drop_column('user_organizations', 'can_delete_organization')
    op.drop_column('user_organizations', 'can_manage_billing')
    op.drop_column('user_organizations', 'can_view_agents')
    op.drop_column('user_organizations', 'can_manage_agents')
    op.drop_column('user_organizations', 'can_manage_users')