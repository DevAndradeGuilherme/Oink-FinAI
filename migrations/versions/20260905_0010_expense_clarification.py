"""Add durable expense clarification conversation status."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260905_0010"
down_revision: str | None = "20260904_0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TYPE conversation_status ADD VALUE IF NOT EXISTS 'WAITING_EXPENSE_CLARIFICATION'"
    )
    op.add_column(
        "processed_messages",
        sa.Column("clarification_origin_message_id", sa.Uuid(), nullable=True),
    )
    op.create_foreign_key(
        "fk_processed_messages_clarification_origin",
        "processed_messages",
        "processed_messages",
        ["clarification_origin_message_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_processed_messages_clarification_origin_message_id",
        "processed_messages",
        ["clarification_origin_message_id"],
    )


def downgrade() -> None:
    """Forward-only: existing clarification states and history must remain readable."""
