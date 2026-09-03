"""Add durable expense processing queue and WhatsApp outbox."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260901_0002"
down_revision: str | None = "20260901_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    processed_status = postgresql.ENUM(
        "PENDING",
        "PROCESSING",
        "PROCESSED",
        "NEEDS_CLARIFICATION",
        "NOT_EXPENSE",
        "FAILED",
        name="processed_message_status",
        create_type=False,
    )
    outbound_status = postgresql.ENUM(
        "PENDING",
        "CLAIMED",
        "SENDING",
        "SENT",
        "UNKNOWN",
        "FAILED",
        name="outbound_message_status",
        create_type=False,
    )
    outbound_kind = postgresql.ENUM(
        "EXPENSE_CONFIRMATION",
        "CLARIFICATION",
        name="outbound_message_kind",
        create_type=False,
    )
    processed_status.create(op.get_bind())
    outbound_status.create(op.get_bind())
    outbound_kind.create(op.get_bind())

    op.add_column("processed_messages", sa.Column("accepted_text", sa.Text(), nullable=True))
    op.add_column("processed_messages", sa.Column("message_timestamp", sa.DateTime(timezone=True)))
    op.add_column(
        "processed_messages",
        sa.Column("status", processed_status, nullable=True),
    )
    op.add_column(
        "processed_messages",
        sa.Column("attempt_count", sa.Integer(), server_default="0", nullable=False),
    )
    op.add_column(
        "processed_messages",
        sa.Column("available_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.add_column("processed_messages", sa.Column("locked_at", sa.DateTime(timezone=True)))
    op.add_column("processed_messages", sa.Column("error_code", sa.String(64)))
    op.execute(
        "UPDATE processed_messages "
        "SET accepted_text = '', message_timestamp = created_at, status = 'PROCESSED'"
    )
    op.alter_column("processed_messages", "accepted_text", nullable=False)
    op.alter_column("processed_messages", "message_timestamp", nullable=False)
    op.alter_column("processed_messages", "status", nullable=False)
    op.alter_column("processed_messages", "available_at", nullable=False)
    op.create_index("ix_processed_messages_status", "processed_messages", ["status"])
    op.create_index("ix_processed_messages_available_at", "processed_messages", ["available_at"])
    op.create_index(
        "ix_processed_messages_claim",
        "processed_messages",
        ["status", "available_at", "created_at"],
    )

    op.add_column("expenses", sa.Column("processed_message_id", sa.Uuid()))
    op.create_foreign_key(
        "fk_expenses_processed_message_id_processed_messages",
        "expenses",
        "processed_messages",
        ["processed_message_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_unique_constraint(
        "uq_expenses_processed_message_id", "expenses", ["processed_message_id"]
    )

    op.create_table(
        "outbound_messages",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("expense_id", sa.Uuid(), sa.ForeignKey("expenses.id", ondelete="SET NULL")),
        sa.Column("destination", sa.String(32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("kind", outbound_kind, nullable=False),
        sa.Column("dedup_key", sa.String(255), nullable=False),
        sa.Column("status", outbound_status, server_default="PENDING", nullable=False),
        sa.Column("attempt_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "available_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("locked_at", sa.DateTime(timezone=True)),
        sa.Column("claimed_at", sa.DateTime(timezone=True)),
        sa.Column("sending_at", sa.DateTime(timezone=True)),
        sa.Column("claim_token", sa.Uuid()),
        sa.Column("provider_message_id", sa.String(255)),
        sa.Column("error_code", sa.String(64)),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("sent_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("dedup_key", name="uq_outbound_messages_dedup_key"),
    )
    op.create_index("ix_outbound_messages_status", "outbound_messages", ["status"])
    op.create_index("ix_outbound_messages_available_at", "outbound_messages", ["available_at"])
    op.create_index(
        "ix_outbound_messages_claim",
        "outbound_messages",
        ["status", "available_at", "created_at"],
    )


def downgrade() -> None:
    op.drop_table("outbound_messages")
    op.drop_constraint("uq_expenses_processed_message_id", "expenses", type_="unique")
    op.drop_constraint(
        "fk_expenses_processed_message_id_processed_messages", "expenses", type_="foreignkey"
    )
    op.drop_column("expenses", "processed_message_id")
    op.drop_index("ix_processed_messages_claim", table_name="processed_messages")
    op.drop_index("ix_processed_messages_available_at", table_name="processed_messages")
    op.drop_index("ix_processed_messages_status", table_name="processed_messages")
    for column in (
        "error_code",
        "locked_at",
        "available_at",
        "attempt_count",
        "status",
        "message_timestamp",
        "accepted_text",
    ):
        op.drop_column("processed_messages", column)
    sa.Enum(name="outbound_message_kind").drop(op.get_bind())
    sa.Enum(name="outbound_message_status").drop(op.get_bind())
    sa.Enum(name="processed_message_status").drop(op.get_bind())
