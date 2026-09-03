"""Add durable audio transcription state and expense source."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260903_0008"
down_revision: str | None = "20260902_0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "processed_messages",
        sa.Column("source_type", sa.String(10), server_default="TEXT", nullable=False),
    )
    op.add_column("processed_messages", sa.Column("media_remote_jid", sa.String(255)))
    op.add_column("processed_messages", sa.Column("media_mime_type", sa.String(200)))
    op.add_column("processed_messages", sa.Column("media_duration_seconds", sa.Integer()))
    op.add_column("processed_messages", sa.Column("media_is_voice_note", sa.Boolean()))
    op.add_column("processed_messages", sa.Column("transcribed_at", sa.DateTime(timezone=True)))
    op.create_check_constraint(
        "processed_message_source_type_valid",
        "processed_messages",
        "source_type IN ('TEXT', 'AUDIO')",
    )
    op.create_check_constraint(
        "media_duration_non_negative",
        "processed_messages",
        "media_duration_seconds IS NULL OR media_duration_seconds >= 0",
    )
    op.create_check_constraint(
        "processed_message_audio_transcript_length",
        "processed_messages",
        "source_type <> 'AUDIO' OR length(accepted_text) <= 10000",
    )

    op.add_column(
        "expenses",
        sa.Column("source_type", sa.String(10), server_default="TEXT", nullable=False),
    )
    op.create_check_constraint(
        "expense_source_type_valid", "expenses", "source_type IN ('TEXT', 'AUDIO')"
    )


def downgrade() -> None:
    """Forward-only: audio metadata and transcript provenance must not be discarded."""
