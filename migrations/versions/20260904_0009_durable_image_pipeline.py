"""Add durable image analysis checkpoint and expense source."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260904_0009"
down_revision: str | None = "20260903_0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("processed_messages", sa.Column("media_caption", sa.String(2000)))
    op.add_column("processed_messages", sa.Column("image_analysis", postgresql.JSONB()))
    op.add_column("processed_messages", sa.Column("image_analyzed_at", sa.DateTime(timezone=True)))
    op.drop_constraint("processed_message_source_type_valid", "processed_messages", type_="check")
    op.create_check_constraint(
        "processed_message_source_type_valid",
        "processed_messages",
        "source_type IN ('TEXT', 'AUDIO', 'IMAGE')",
    )
    op.create_check_constraint(
        "processed_message_media_caption_length",
        "processed_messages",
        "media_caption IS NULL OR length(media_caption) <= 2000",
    )
    op.drop_constraint("expense_source_type_valid", "expenses", type_="check")
    op.create_check_constraint(
        "expense_source_type_valid",
        "expenses",
        "source_type IN ('TEXT', 'AUDIO', 'IMAGE')",
    )


def downgrade() -> None:
    """Forward-only: durable image checkpoints and provenance must not be discarded."""
