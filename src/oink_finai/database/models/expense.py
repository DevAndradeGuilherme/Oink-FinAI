from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import CheckConstraint, Date, DateTime, ForeignKey, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from oink_finai.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from oink_finai.database.models.category import Category
    from oink_finai.database.models.expense_history import ExpenseHistory
    from oink_finai.database.models.user import User


class Expense(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "expenses"
    __table_args__ = (CheckConstraint("amount > 0", name="amount_positive"),)

    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    category_id: Mapped[UUID] = mapped_column(ForeignKey("categories.id"), index=True)
    amount: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    description: Mapped[str] = mapped_column(String(255))
    expense_date: Mapped[date] = mapped_column(Date)
    merchant: Mapped[str | None] = mapped_column(String(160))
    payment_method: Mapped[str | None] = mapped_column(String(80))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)

    user: Mapped["User"] = relationship(back_populates="expenses")
    category: Mapped["Category"] = relationship(back_populates="expenses")
    history: Mapped[list["ExpenseHistory"]] = relationship(back_populates="expense")

    @validates("amount")
    def validate_amount(self, _key: str, value: Decimal) -> Decimal:
        if not isinstance(value, Decimal):
            raise TypeError("amount must be Decimal")
        if value <= 0:
            raise ValueError("amount must be greater than zero")
        return value.quantize(Decimal("0.01"))
