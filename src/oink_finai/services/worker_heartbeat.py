import asyncio
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from oink_finai.database.models import WorkerHeartbeat
from oink_finai.domain.enums import WorkerHeartbeatStatus


class WorkerHealthState(StrEnum):
    HEALTHY = "HEALTHY"
    STALE = "STALE"
    ABSENT = "ABSENT"
    STOPPED = "STOPPED"


class WorkerHeartbeatService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        stale_seconds: float,
        database_timeout_seconds: float,
        retention_days: int,
    ) -> None:
        self._session_factory = session_factory
        self._stale = timedelta(seconds=stale_seconds)
        self._database_timeout_seconds = database_timeout_seconds
        self._retention = timedelta(days=retention_days)

    async def start(self, worker_id: UUID, release: str | None = None) -> None:
        async with asyncio.timeout(self._database_timeout_seconds):
            async with self._session_factory() as session, session.begin():
                now = await self._database_now(session)
                await session.execute(
                    delete(WorkerHeartbeat).where(
                        WorkerHeartbeat.last_seen_at < now - self._retention
                    )
                )
                heartbeat = await session.get(WorkerHeartbeat, worker_id)
                if heartbeat is None:
                    session.add(
                        WorkerHeartbeat(
                            worker_id=worker_id,
                            started_at=now,
                            last_seen_at=now,
                            status=WorkerHeartbeatStatus.RUNNING,
                            release=release,
                        )
                    )
                else:
                    heartbeat.started_at = now
                    heartbeat.last_seen_at = now
                    heartbeat.status = WorkerHeartbeatStatus.RUNNING
                    heartbeat.release = release

    async def beat(self, worker_id: UUID) -> bool:
        async with asyncio.timeout(self._database_timeout_seconds):
            async with self._session_factory() as session, session.begin():
                now = await self._database_now(session)
                result = await session.execute(
                    update(WorkerHeartbeat)
                    .where(
                        WorkerHeartbeat.worker_id == worker_id,
                        WorkerHeartbeat.status == WorkerHeartbeatStatus.RUNNING,
                    )
                    .values(last_seen_at=now)
                )
                return bool(result.rowcount)

    async def stopping(self, worker_id: UUID) -> bool:
        return await self._set_status(worker_id, WorkerHeartbeatStatus.STOPPING)

    async def stopped(self, worker_id: UUID) -> bool:
        return await self._set_status(worker_id, WorkerHeartbeatStatus.STOPPED)

    async def check_worker(self, worker_id: UUID) -> WorkerHealthState:
        async with asyncio.timeout(self._database_timeout_seconds):
            async with self._session_factory() as session:
                now = await self._database_now(session)
                heartbeat = await session.get(WorkerHeartbeat, worker_id)
        if heartbeat is None:
            return WorkerHealthState.ABSENT
        if heartbeat.status is WorkerHeartbeatStatus.STOPPED:
            return WorkerHealthState.STOPPED
        if (
            heartbeat.status is WorkerHeartbeatStatus.RUNNING
            and self._as_utc(heartbeat.last_seen_at) >= now - self._stale
        ):
            return WorkerHealthState.HEALTHY
        return WorkerHealthState.STALE

    async def check_aggregate(self) -> WorkerHealthState:
        async with asyncio.timeout(self._database_timeout_seconds):
            async with self._session_factory() as session:
                now = await self._database_now(session)
                rows = list(await session.scalars(select(WorkerHeartbeat)))
        if not rows:
            return WorkerHealthState.ABSENT
        if any(
            row.status is WorkerHeartbeatStatus.RUNNING
            and self._as_utc(row.last_seen_at) >= now - self._stale
            for row in rows
        ):
            return WorkerHealthState.HEALTHY
        if all(row.status is WorkerHeartbeatStatus.STOPPED for row in rows):
            return WorkerHealthState.STOPPED
        return WorkerHealthState.STALE

    async def _set_status(self, worker_id: UUID, status: WorkerHeartbeatStatus) -> bool:
        async with asyncio.timeout(self._database_timeout_seconds):
            async with self._session_factory() as session, session.begin():
                now = await self._database_now(session)
                result = await session.execute(
                    update(WorkerHeartbeat)
                    .where(WorkerHeartbeat.worker_id == worker_id)
                    .values(status=status, last_seen_at=now)
                )
                return bool(result.rowcount)

    @classmethod
    async def _database_now(cls, session: AsyncSession) -> datetime:
        value = await session.scalar(select(func.current_timestamp()))
        if not isinstance(value, datetime):
            raise RuntimeError("database clock is unavailable")
        return cls._as_utc(value)

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
