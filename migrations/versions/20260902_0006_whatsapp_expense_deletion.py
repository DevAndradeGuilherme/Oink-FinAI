"""Add durable WhatsApp expense deletion workflow."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260902_0006"
down_revision: str | None = "20260902_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TYPE conversation_status ADD VALUE IF NOT EXISTS 'WAITING_EXPENSE_DELETE_CONFIRM'"
    )
    op.execute("ALTER TYPE expense_history_action ADD VALUE IF NOT EXISTS 'DELETE'")
    for value in (
        "DELETE_CONFIRMATION_REQUEST",
        "EXPENSE_DELETED",
        "ACTION_CANCELLED",
        "ACTION_ERROR",
    ):
        op.execute(f"ALTER TYPE outbound_message_kind ADD VALUE IF NOT EXISTS '{value}'")
    op.add_column(
        "conversation_states",
        sa.Column(
            "context",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )
    op.add_column(
        "conversation_states",
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("conversation_states", "expires_at")
    op.drop_column("conversation_states", "context")
