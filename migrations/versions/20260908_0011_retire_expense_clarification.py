"""Retire chained expense clarification for new processing."""

from collections.abc import Sequence

from alembic import op

revision: str = "20260908_0011"
down_revision: str | None = "20260905_0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TYPE outbound_message_kind ADD VALUE IF NOT EXISTS 'INCOMPLETE_EXPENSE'")
    op.execute(
        """
        UPDATE conversation_states
        SET status = 'IDLE',
            active_expense_id = NULL,
            context = NULL,
            expires_at = NULL
        WHERE status::text = 'WAITING_EXPENSE_CLARIFICATION'
        """
    )


def downgrade() -> None:
    """Forward-only: historical enum values and records remain readable."""
