from datetime import datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from oink_finai.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from oink_finai.domain.enums import UsageOperation, UsageReservationState


class UsageLedger(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Content-free durable admission and usage record."""

    __tablename__ = "usage_ledger"
    __table_args__ = (
        UniqueConstraint(
            "processed_message_id",
            "operation",
            "durable_attempt",
            name="uq_usage_ledger_message_operation_attempt",
        ),
        Index("ix_usage_ledger_window_operation", "window_day_start", "operation", "state"),
        Index("ix_usage_ledger_user_window", "user_id", "window_day_start", "state"),
        Index("ix_usage_ledger_reconciliation", "state", "transmitted_at", "created_at"),
        CheckConstraint("durable_attempt >= 0", name="usage_ledger_attempt_non_negative"),
        CheckConstraint(
            "input_tokens IS NULL OR input_tokens >= 0",
            name="usage_ledger_input_tokens_non_negative",
        ),
        CheckConstraint(
            "output_tokens IS NULL OR output_tokens >= 0",
            name="usage_ledger_output_tokens_non_negative",
        ),
        CheckConstraint(
            "audio_seconds IS NULL OR audio_seconds >= 0",
            name="usage_ledger_audio_seconds_non_negative",
        ),
    )

    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="RESTRICT"))
    processed_message_id: Mapped[UUID] = mapped_column(
        ForeignKey("processed_messages.id", ondelete="RESTRICT")
    )
    operation: Mapped[UsageOperation] = mapped_column(Enum(UsageOperation, name="usage_operation"))
    model: Mapped[str | None] = mapped_column(String(128))
    durable_attempt: Mapped[int] = mapped_column(Integer)
    window_minute_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    window_day_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    state: Mapped[UsageReservationState] = mapped_column(
        Enum(UsageReservationState, name="usage_reservation_state"), index=True
    )
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    audio_seconds: Mapped[Decimal | None] = mapped_column(Numeric(12, 3))
    transmitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    released_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
