"""Add durable financial query checkpoints and ordered outbox pages."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260908_0012"
down_revision: str | None = "20260908_0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    expense_intent = postgresql.ENUM(
        "CREATE_EXPENSE",
        "QUERY",
        "NOT_EXPENSE",
        "UNCLEAR",
        name="expense_message_intent",
        create_type=False,
    )
    expense_intent.create(op.get_bind())
    op.execute("ALTER TYPE outbound_message_kind ADD VALUE IF NOT EXISTS 'QUERY_RESULT'")
    op.execute("ALTER TYPE outbound_message_kind ADD VALUE IF NOT EXISTS 'QUERY_GUIDANCE'")

    op.add_column(
        "processed_messages", sa.Column("classification_intent", expense_intent, nullable=True)
    )
    op.add_column(
        "processed_messages",
        sa.Column("classification_checkpoint", postgresql.JSONB(), nullable=True),
    )
    op.add_column(
        "processed_messages", sa.Column("classified_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column(
        "processed_messages",
        sa.Column("query_plan_checkpoint", postgresql.JSONB(), nullable=True),
    )
    op.add_column(
        "processed_messages",
        sa.Column("query_plan_created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "processed_messages",
        sa.Column("query_result_checkpoint", postgresql.JSONB(), nullable=True),
    )
    op.add_column(
        "processed_messages",
        sa.Column("query_executed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column("processed_messages", sa.Column("query_page_count", sa.Integer(), nullable=True))
    op.create_check_constraint(
        "processed_message_query_page_count_valid",
        "processed_messages",
        "query_page_count IS NULL OR query_page_count >= 1",
    )

    op.add_column(
        "outbound_messages",
        sa.Column(
            "processed_message_id",
            sa.Uuid(),
            sa.ForeignKey("processed_messages.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.add_column("outbound_messages", sa.Column("sequence_no", sa.Integer(), nullable=True))
    op.add_column("outbound_messages", sa.Column("sequence_count", sa.Integer(), nullable=True))
    op.create_index(
        "ix_outbound_messages_processed_message_id",
        "outbound_messages",
        ["processed_message_id"],
    )
    op.create_unique_constraint(
        "uq_outbound_message_processed_kind_sequence",
        "outbound_messages",
        ["processed_message_id", "kind", "sequence_no"],
    )
    op.create_check_constraint(
        "outbound_message_sequence_valid",
        "outbound_messages",
        "(sequence_no IS NULL AND sequence_count IS NULL) OR "
        "(sequence_no >= 1 AND sequence_count >= sequence_no)",
    )


def downgrade() -> None:
    """Forward-only: query checkpoints and ordered delivery history must be preserved."""
