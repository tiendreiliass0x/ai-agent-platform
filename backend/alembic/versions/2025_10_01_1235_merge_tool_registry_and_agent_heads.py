"""Merge heads for tool registry and agent idempotency."""

from collections.abc import Sequence
from typing import Optional

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "2025_10_01_1235"
down_revision = ("add_idempotency_key_to_agents", "2025_10_01_1230")
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
