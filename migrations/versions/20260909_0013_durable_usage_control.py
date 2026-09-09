"""Add content-free durable usage admission ledger."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260909_0013"
down_revision: str | None = "20260908_0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    usage_operation = postgresql.ENUM(
        "INBOUND_MESSAGE",
        "TEXT_INTERPRETATION",
        "QUERY_INTERPRETATION",
        "IMAGE_ANALYSIS",
        "AUDIO_TRANSCRIPTION",
        name="usage_operation",
        create_type=False,
    )
    usage_state = postgresql.ENUM(
        "RESERVED",
        "COMPLETED",
        "AMBIGUOUS",
        "RELEASED",
        name="usage_reservation_state",
        create_type=False,
    )
    usage_operation.create(op.get_bind())
    usage_state.create(op.get_bind())
    op.execute("ALTER TYPE outbound_message_kind ADD VALUE IF NOT EXISTS 'RATE_LIMIT_GUIDANCE'")

    op.create_table(
        "usage_ledger",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("processed_message_id", sa.Uuid(), nullable=False),
        sa.Column("operation", usage_operation, nullable=False),
        sa.Column("model", sa.String(length=128), nullable=True),
        sa.Column("durable_attempt", sa.Integer(), nullable=False),
        sa.Column("window_minute_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_day_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("state", usage_state, nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("audio_seconds", sa.Numeric(12, 3), nullable=True),
        sa.Column("transmitted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("released_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "durable_attempt >= 0", name="ck_usage_ledger_usage_ledger_attempt_non_negative"
        ),
        sa.CheckConstraint(
            "input_tokens IS NULL OR input_tokens >= 0",
            name="ck_usage_ledger_usage_ledger_input_tokens_non_negative",
        ),
        sa.CheckConstraint(
            "output_tokens IS NULL OR output_tokens >= 0",
            name="ck_usage_ledger_usage_ledger_output_tokens_non_negative",
        ),
        sa.CheckConstraint(
            "audio_seconds IS NULL OR audio_seconds >= 0",
            name="ck_usage_ledger_usage_ledger_audio_seconds_non_negative",
        ),
        sa.ForeignKeyConstraint(
            ["processed_message_id"], ["processed_messages.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id", name="pk_usage_ledger"),
        sa.UniqueConstraint(
            "processed_message_id",
            "operation",
            "durable_attempt",
            name="uq_usage_ledger_message_operation_attempt",
        ),
    )
    op.create_index("ix_usage_ledger_state", "usage_ledger", ["state"])
    op.create_index(
        "ix_usage_ledger_window_operation",
        "usage_ledger",
        ["window_day_start", "operation", "state"],
    )
    op.create_index(
        "ix_usage_ledger_user_window", "usage_ledger", ["user_id", "window_day_start", "state"]
    )
    op.create_index(
        "ix_usage_ledger_reconciliation", "usage_ledger", ["state", "transmitted_at", "created_at"]
    )


def downgrade() -> None:
    """Forward-only: durable usage and admission history must be preserved."""
