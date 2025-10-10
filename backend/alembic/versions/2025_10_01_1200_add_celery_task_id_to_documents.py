"""Add celery_task_id column to documents for indexed lookups."""

from collections.abc import Sequence
from typing import Optional

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "2025_10_01_1200"
down_revision: Optional[str] = "bb84a883744f"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("celery_task_id", sa.String(length=36), nullable=True),
    )
    op.create_unique_constraint(
        "uq_documents_celery_task_id",
        "documents",
        ["celery_task_id"],
    )

    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        bind.execute(
            sa.text(
                """
                UPDATE documents
                SET celery_task_id = (doc_metadata::jsonb ->> 'celery_task_id')
                WHERE doc_metadata::jsonb ? 'celery_task_id'
                  AND doc_metadata::jsonb ->> 'celery_task_id' IS NOT NULL
                """
            )
        )


def downgrade() -> None:
    op.drop_constraint("uq_documents_celery_task_id", "documents", type_="unique")
    op.drop_column("documents", "celery_task_id")
