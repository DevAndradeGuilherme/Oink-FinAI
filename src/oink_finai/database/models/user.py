from typing import TYPE_CHECKING

from sqlalchemy import String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from oink_finai.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from oink_finai.database.models.conversation_state import ConversationState
    from oink_finai.database.models.expense import Expense


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("phone_number", name="uq_users_phone_number"),)

    phone_number: Mapped[str] = mapped_column(String(32), index=True)
    display_name: Mapped[str | None] = mapped_column(String(120))
    timezone: Mapped[str] = mapped_column(String(64), default="America/Sao_Paulo")

    expenses: Mapped[list["Expense"]] = relationship(back_populates="user")
    conversation_state: Mapped["ConversationState | None"] = relationship(
        back_populates="user", uselist=False
    )
