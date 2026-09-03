from uuid import UUID

from sqlalchemy import JSON, Enum, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from oink_finai.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from oink_finai.domain.enums import ExpenseHistoryAction


class ExpenseHistory(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "expense_history"

    expense_id: Mapped[UUID] = mapped_column(
        ForeignKey("expenses.id", ondelete="CASCADE"), index=True
    )
    action: Mapped[ExpenseHistoryAction] = mapped_column(
        Enum(ExpenseHistoryAction, name="expense_history_action")
    )
    changes: Mapped[dict[str, object]] = mapped_column(JSON().with_variant(JSONB, "postgresql"))

    expense = relationship("Expense", back_populates="history")
