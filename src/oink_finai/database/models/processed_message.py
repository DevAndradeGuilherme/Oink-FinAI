from uuid import UUID

from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from oink_finai.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class ProcessedMessage(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "processed_messages"
    __table_args__ = (
        UniqueConstraint(
            "provider",
            "instance_id",
            "external_message_id",
            name="uq_processed_message_external_identity",
        ),
    )

    provider: Mapped[str] = mapped_column(String(40))
    instance_id: Mapped[str] = mapped_column(String(120))
    external_message_id: Mapped[str] = mapped_column(String(255))
    user_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
