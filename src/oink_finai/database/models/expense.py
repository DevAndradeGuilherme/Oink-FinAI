from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import CheckConstraint, Date, DateTime, ForeignKey, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from oink_finai.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from oink_finai.domain.enums import MessageSourceType
from oink_finai.domain.expense_limits import (
    EXPENSE_AMOUNT_MAX,
    EXPENSE_AMOUNT_PRECISION,
    EXPENSE_AMOUNT_SCALE,
    EXPENSE_DESCRIPTION_MAX_LENGTH,
    EXPENSE_MERCHANT_MAX_LENGTH,
    EXPENSE_PAYMENT_METHOD_MAX_LENGTH,
)

if TYPE_CHECKING:
    from oink_finai.database.models.category import Category
    from oink_finai.database.models.expense_history import ExpenseHistory
    from oink_finai.database.models.processed_message import ProcessedMessage
    from oink_finai.database.models.user import User


class Expense(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "expenses"
    __table_args__ = (
        CheckConstraint("amount > 0", name="amount_positive"),
        CheckConstraint("source_type IN ('TEXT', 'AUDIO')", name="expense_source_type_valid"),
    )

    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    processed_message_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("processed_messages.id", ondelete="RESTRICT"), unique=True
    )
    category_id: Mapped[UUID] = mapped_column(ForeignKey("categories.id"), index=True)
    amount: Mapped[Decimal] = mapped_column(
        Numeric(EXPENSE_AMOUNT_PRECISION, EXPENSE_AMOUNT_SCALE), nullable=False
    )
    description: Mapped[str] = mapped_column(String(EXPENSE_DESCRIPTION_MAX_LENGTH))
    expense_date: Mapped[date] = mapped_column(Date)
    merchant: Mapped[str | None] = mapped_column(String(EXPENSE_MERCHANT_MAX_LENGTH))
    payment_method: Mapped[str | None] = mapped_column(String(EXPENSE_PAYMENT_METHOD_MAX_LENGTH))
    source_type: Mapped[MessageSourceType] = mapped_column(
        String(10), default=MessageSourceType.TEXT
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)

    user: Mapped["User"] = relationship(back_populates="expenses")
    category: Mapped["Category"] = relationship(back_populates="expenses")
    history: Mapped[list["ExpenseHistory"]] = relationship(back_populates="expense")
    processed_message: Mapped["ProcessedMessage | None"] = relationship(back_populates="expense")
    outbound_messages = relationship("OutboundMessage", back_populates="expense")

    @validates("amount")
    def validate_amount(self, _key: str, value: Decimal) -> Decimal:
        if not isinstance(value, Decimal):
            raise TypeError("amount must be Decimal")
        if not value.is_finite() or value <= 0:
            raise ValueError("amount must be greater than zero")
        if value > EXPENSE_AMOUNT_MAX:
            raise ValueError("amount is outside supported range")
        if value.as_tuple().exponent < -EXPENSE_AMOUNT_SCALE:
            raise ValueError("amount has too many decimal places")
        return value
