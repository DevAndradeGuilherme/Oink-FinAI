"""Reconcile databases that received an earlier durable-pipeline migration."""

from collections.abc import Sequence

from alembic import op

revision: str = "20260901_0004"
down_revision: str | None = "20260901_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TYPE outbound_message_status ADD VALUE IF NOT EXISTS 'CLAIMED' BEFORE 'SENDING'"
    )
    op.execute(
        "ALTER TABLE outbound_messages ADD COLUMN IF NOT EXISTS claimed_at TIMESTAMP WITH TIME ZONE"
    )
    op.execute(
        "ALTER TABLE outbound_messages ADD COLUMN IF NOT EXISTS sending_at TIMESTAMP WITH TIME ZONE"
    )
    op.execute("ALTER TABLE outbound_messages ADD COLUMN IF NOT EXISTS claim_token UUID")
    op.execute(
        "UPDATE outbound_messages SET sending_at = COALESCE(locked_at, updated_at, created_at) "
        "WHERE status = 'SENDING' AND sending_at IS NULL"
    )
    op.execute(
        "UPDATE processed_messages SET status = 'PROCESSED' "
        "WHERE status = 'PENDING' AND accepted_text = ''"
    )


def downgrade() -> None:
    pass
