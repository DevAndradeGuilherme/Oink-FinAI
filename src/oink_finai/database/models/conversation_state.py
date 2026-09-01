from uuid import UUID

from sqlalchemy import Enum, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from oink_finai.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from oink_finai.domain.enums import ConversationStatus


class ConversationState(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "conversation_states"

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), unique=True, index=True
    )
    status: Mapped[ConversationStatus] = mapped_column(
        Enum(ConversationStatus, name="conversation_status"), default=ConversationStatus.IDLE
    )
    active_expense_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("expenses.id", ondelete="SET NULL")
    )

    user = relationship("User", back_populates="conversation_state")
    active_expense = relationship("Expense")
