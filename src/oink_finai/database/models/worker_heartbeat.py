from datetime import datetime
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, Enum, Index, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from oink_finai.database.base import Base
from oink_finai.domain.enums import WorkerHeartbeatStatus


class WorkerHeartbeat(Base):
    """Opaque, content-free liveness record for one worker process."""

    __tablename__ = "worker_heartbeats"
    __table_args__ = (
        Index("ix_worker_heartbeats_status_last_seen", "status", "last_seen_at"),
        CheckConstraint("last_seen_at >= started_at", name="worker_heartbeat_seen_after_start"),
    )

    worker_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[WorkerHeartbeatStatus] = mapped_column(
        Enum(WorkerHeartbeatStatus, name="worker_heartbeat_status")
    )
    release: Mapped[str | None] = mapped_column(String(128))
