"""Add durable Gemini retry scheduling and terminal failure notifications."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260902_0005"
down_revision: str | None = "20260901_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TYPE outbound_message_kind ADD VALUE IF NOT EXISTS 'PROCESSING_FAILURE'")
    op.add_column(
        "processed_messages",
        sa.Column("processing_attempts", sa.Integer(), server_default="0", nullable=False),
    )
    op.add_column(
        "processed_messages",
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column("processed_messages", sa.Column("last_error_code", sa.String(64), nullable=True))
    op.execute(
        "UPDATE processed_messages SET last_error_code = error_code "
        "WHERE error_code IS NOT NULL AND last_error_code IS NULL"
    )
    op.alter_column("processed_messages", "processing_attempts", server_default=None)
    op.create_index(
        "ix_processed_messages_retry",
        "processed_messages",
        ["status", "next_attempt_at", "created_at"],
    )


def downgrade() -> None:
    pass
