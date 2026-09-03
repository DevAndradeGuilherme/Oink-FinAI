from typing import TYPE_CHECKING

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from oink_finai.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from oink_finai.database.models.expense import Expense


class Category(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "categories"

    name: Mapped[str] = mapped_column(String(80), unique=True)
    slug: Mapped[str] = mapped_column(String(80), unique=True)
    is_active: Mapped[bool] = mapped_column(default=True)

    expenses: Mapped[list["Expense"]] = relationship(back_populates="category")
