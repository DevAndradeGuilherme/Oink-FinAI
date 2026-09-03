from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import DateTime, Enum, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from oink_finai.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from oink_finai.domain.enums import ProcessedMessageStatus


class ProcessedMessage(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "processed_messages"
    __table_args__ = (
        UniqueConstraint(
            "provider",
            "instance_id",
            "external_message_id",
            name="uq_processed_message_external_identity",
        ),
        Index("ix_processed_messages_claim", "status", "available_at", "created_at"),
        Index("ix_processed_messages_retry", "status", "next_attempt_at", "created_at"),
    )

    provider: Mapped[str] = mapped_column(String(40))
    instance_id: Mapped[str] = mapped_column(String(120))
    external_message_id: Mapped[str] = mapped_column(String(255))
    user_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    accepted_text: Mapped[str] = mapped_column(Text, default="")
    message_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    status: Mapped[ProcessedMessageStatus] = mapped_column(
        Enum(ProcessedMessageStatus, name="processed_message_status"),
        default=ProcessedMessageStatus.PENDING,
        index=True,
    )
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
    )
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(64))
    processing_attempts: Mapped[int] = mapped_column(Integer, default=0)
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error_code: Mapped[str | None] = mapped_column(String(64))

    user = relationship("User", back_populates="processed_messages")
    expense = relationship("Expense", back_populates="processed_message", uselist=False)
