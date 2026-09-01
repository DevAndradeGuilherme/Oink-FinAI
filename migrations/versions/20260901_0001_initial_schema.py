"""Create initial financial control schema and categories."""

from collections.abc import Sequence
from uuid import UUID

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260901_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

CATEGORIES = (
    ("Alimentação", "alimentacao"),
    ("Transporte", "transporte"),
    ("Moradia", "moradia"),
    ("Saúde", "saude"),
    ("Educação", "educacao"),
    ("Lazer", "lazer"),
    ("Compras", "compras"),
    ("Assinaturas", "assinaturas"),
    ("Contas", "contas"),
    ("Impostos", "impostos"),
    ("Trabalho", "trabalho"),
    ("Viagem", "viagem"),
    ("Outros", "outros"),
)


def timestamp_columns() -> tuple[sa.Column, sa.Column]:
    return (
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )


def upgrade() -> None:
    conversation_status = sa.Enum(
        "IDLE", "EDITING_EXPENSE", "REMOVING_EXPENSE", name="conversation_status"
    )
    history_action = sa.Enum("CREATED", "UPDATED", "DELETED", name="expense_history_action")
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("phone_number", sa.String(32), nullable=False),
        sa.Column("display_name", sa.String(120)),
        sa.Column("timezone", sa.String(64), nullable=False, server_default="America/Sao_Paulo"),
        *timestamp_columns(),
        sa.UniqueConstraint("phone_number", name="uq_users_phone_number"),
    )
    op.create_index("ix_users_phone_number", "users", ["phone_number"])

    op.create_table(
        "categories",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("name", sa.String(80), nullable=False),
        sa.Column("slug", sa.String(80), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        *timestamp_columns(),
        sa.UniqueConstraint("name", name="uq_categories_name"),
        sa.UniqueConstraint("slug", name="uq_categories_slug"),
    )

    op.create_table(
        "expenses",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("category_id", sa.Uuid(), sa.ForeignKey("categories.id"), nullable=False),
        sa.Column("amount", sa.Numeric(14, 2), nullable=False),
        sa.Column("description", sa.String(255), nullable=False),
        sa.Column("expense_date", sa.Date(), nullable=False),
        sa.Column("merchant", sa.String(160)),
        sa.Column("payment_method", sa.String(80)),
        sa.Column("deleted_at", sa.DateTime(timezone=True)),
        *timestamp_columns(),
        sa.CheckConstraint("amount > 0", name="amount_positive"),
    )
    op.create_index("ix_expenses_user_id", "expenses", ["user_id"])
    op.create_index("ix_expenses_category_id", "expenses", ["category_id"])
    op.create_index("ix_expenses_deleted_at", "expenses", ["deleted_at"])

    op.create_table(
        "processed_messages",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("provider", sa.String(40), nullable=False),
        sa.Column("instance_id", sa.String(120), nullable=False),
        sa.Column("external_message_id", sa.String(255), nullable=False),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="SET NULL")),
        *timestamp_columns(),
        sa.UniqueConstraint(
            "provider",
            "instance_id",
            "external_message_id",
            name="uq_processed_message_external_identity",
        ),
    )

    op.create_table(
        "conversation_states",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("status", conversation_status, nullable=False, server_default="IDLE"),
        sa.Column(
            "active_expense_id", sa.Uuid(), sa.ForeignKey("expenses.id", ondelete="SET NULL")
        ),
        *timestamp_columns(),
        sa.UniqueConstraint("user_id", name="uq_conversation_states_user_id"),
    )
    op.create_index("ix_conversation_states_user_id", "conversation_states", ["user_id"])

    op.create_table(
        "expense_history",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "expense_id",
            sa.Uuid(),
            sa.ForeignKey("expenses.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("action", history_action, nullable=False),
        sa.Column("changes", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        *timestamp_columns(),
    )
    op.create_index("ix_expense_history_expense_id", "expense_history", ["expense_id"])

    categories = sa.table(
        "categories",
        sa.column("id", sa.Uuid()),
        sa.column("name", sa.String()),
        sa.column("slug", sa.String()),
    )
    op.bulk_insert(
        categories,
        [
            {"id": UUID(int=index), "name": name, "slug": slug}
            for index, (name, slug) in enumerate(CATEGORIES, start=1)
        ],
    )


def downgrade() -> None:
    op.drop_table("expense_history")
    op.drop_table("conversation_states")
    op.drop_table("processed_messages")
    op.drop_table("expenses")
    op.drop_table("categories")
    op.drop_table("users")
    sa.Enum(name="expense_history_action").drop(op.get_bind())
    sa.Enum(name="conversation_status").drop(op.get_bind())
