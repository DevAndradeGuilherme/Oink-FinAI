from datetime import datetime
from uuid import UUID

from sqlalchemy import JSON, DateTime, Enum, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from oink_finai.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from oink_finai.domain.enums import ConversationStatus


class ConversationState(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "conversation_states"
    __table_args__ = (UniqueConstraint("user_id", name="uq_conversation_states_user_id"),)

    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    status: Mapped[ConversationStatus] = mapped_column(
        Enum(ConversationStatus, name="conversation_status"), default=ConversationStatus.IDLE
    )
    active_expense_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("expenses.id", ondelete="SET NULL")
    )
    context: Mapped[dict[str, object] | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql")
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    user = relationship("User", back_populates="conversation_state")
    active_expense = relationship("Expense")
