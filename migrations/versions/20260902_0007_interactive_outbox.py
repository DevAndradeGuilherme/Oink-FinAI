"""Add provider-neutral interactive outbox fields."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260902_0007"
down_revision: str | None = "20260902_0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "outbound_messages",
        sa.Column("content_type", sa.String(20), server_default="TEXT", nullable=False),
    )
    op.add_column(
        "outbound_messages",
        sa.Column("actions", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column(
        "outbound_messages",
        sa.Column("fallback_content", sa.Text(), nullable=True),
    )
    op.alter_column("outbound_messages", "content_type", server_default=None)


def downgrade() -> None:
    op.drop_column("outbound_messages", "fallback_content")
    op.drop_column("outbound_messages", "actions")
    op.drop_column("outbound_messages", "content_type")
