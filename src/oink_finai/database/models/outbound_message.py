from datetime import datetime
from uuid import UUID

from sqlalchemy import JSON, DateTime, Enum, ForeignKey, Index, Integer, String, Text, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from oink_finai.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from oink_finai.domain.enums import OutboundMessageKind, OutboundMessageStatus


class OutboundMessage(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "outbound_messages"
    __table_args__ = (Index("ix_outbound_messages_claim", "status", "available_at", "created_at"),)

    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    expense_id: Mapped[UUID | None] = mapped_column(ForeignKey("expenses.id", ondelete="SET NULL"))
    destination: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    content_type: Mapped[str] = mapped_column(String(20), default="TEXT")
    actions: Mapped[list[dict[str, str]] | None] = mapped_column(
        JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql")
    )
    fallback_content: Mapped[str | None] = mapped_column(Text)
    kind: Mapped[OutboundMessageKind] = mapped_column(
        Enum(OutboundMessageKind, name="outbound_message_kind")
    )
    dedup_key: Mapped[str] = mapped_column(String(255), unique=True)
    status: Mapped[OutboundMessageStatus] = mapped_column(
        Enum(OutboundMessageStatus, name="outbound_message_status"),
        default=OutboundMessageStatus.PENDING,
        index=True,
    )
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    sending_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    claim_token: Mapped[UUID | None] = mapped_column(Uuid)
    provider_message_id: Mapped[str | None] = mapped_column(String(255))
    error_code: Mapped[str | None] = mapped_column(String(64))
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    user = relationship("User", back_populates="outbound_messages")
    expense = relationship("Expense", back_populates="outbound_messages")
