"""Add content-free durable worker heartbeats."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260910_0014"
down_revision: str | None = "20260909_0013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    heartbeat_status = postgresql.ENUM(
        "RUNNING",
        "STOPPING",
        "STOPPED",
        name="worker_heartbeat_status",
        create_type=False,
    )
    heartbeat_status.create(op.get_bind())
    op.create_table(
        "worker_heartbeats",
        sa.Column("worker_id", sa.Uuid(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", heartbeat_status, nullable=False),
        sa.Column("release", sa.String(length=128), nullable=True),
        sa.CheckConstraint(
            "last_seen_at >= started_at",
            name="ck_worker_heartbeats_worker_heartbeat_seen_after_start",
        ),
        sa.PrimaryKeyConstraint("worker_id", name="pk_worker_heartbeats"),
    )
    op.create_index(
        "ix_worker_heartbeats_status_last_seen",
        "worker_heartbeats",
        ["status", "last_seen_at"],
    )


def downgrade() -> None:
    """Forward-only: worker liveness history is intentionally preserved."""
