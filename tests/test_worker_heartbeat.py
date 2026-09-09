import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest_asyncio
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from oink_finai.database.base import Base
from oink_finai.database.models import WorkerHeartbeat
from oink_finai.services.worker_heartbeat import WorkerHealthState, WorkerHeartbeatService
from oink_finai.worker import _maintain_heartbeat


@pytest_asyncio.fixture
async def factory(tmp_path: Path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'heartbeat.db').as_posix()}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    result = async_sessionmaker(engine, expire_on_commit=False)
    yield result
    await engine.dispose()


def service(factory: async_sessionmaker[AsyncSession]) -> WorkerHeartbeatService:
    return WorkerHeartbeatService(
        factory, stale_seconds=45, database_timeout_seconds=1, retention_days=7
    )


async def test_heartbeat_starts_renews_and_contains_only_opaque_fields(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    worker_id = uuid4()
    heartbeat = service(factory)
    await heartbeat.start(worker_id, "release-opaque")
    async with factory() as session, session.begin():
        row = await session.get(WorkerHeartbeat, worker_id)
        assert row is not None
        row.started_at = datetime.now(UTC) - timedelta(minutes=3)
        row.last_seen_at = datetime.now(UTC) - timedelta(minutes=2)
    assert await heartbeat.beat(worker_id)
    assert await heartbeat.check_worker(worker_id) is WorkerHealthState.HEALTHY
    assert set(WorkerHeartbeat.__table__.columns.keys()) == {
        "worker_id",
        "started_at",
        "last_seen_at",
        "status",
        "release",
    }
    assert "release-opaque" not in repr(row)


async def test_stale_crash_stopped_and_absent_are_distinct(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    worker_id = uuid4()
    heartbeat = service(factory)
    assert await heartbeat.check_worker(worker_id) is WorkerHealthState.ABSENT
    await heartbeat.start(worker_id)
    async with factory() as session, session.begin():
        row = await session.get(WorkerHeartbeat, worker_id)
        assert row is not None
        row.started_at = datetime.now(UTC) - timedelta(minutes=3)
        row.last_seen_at = datetime.now(UTC) - timedelta(minutes=2)
    assert await heartbeat.check_worker(worker_id) is WorkerHealthState.STALE
    assert await heartbeat.stopped(worker_id)
    assert await heartbeat.check_worker(worker_id) is WorkerHealthState.STOPPED


async def test_two_workers_and_aggregate_health(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    first, second = uuid4(), uuid4()
    heartbeat = service(factory)
    await heartbeat.start(first)
    await heartbeat.start(second)
    assert first != second
    assert await heartbeat.check_aggregate() is WorkerHealthState.HEALTHY
    await heartbeat.stopped(first)
    assert await heartbeat.check_aggregate() is WorkerHealthState.HEALTHY
    await heartbeat.stopped(second)
    assert await heartbeat.check_aggregate() is WorkerHealthState.STOPPED
    async with factory() as session:
        assert await session.scalar(select(func.count()).select_from(WorkerHeartbeat)) == 2


async def test_stopping_is_not_healthy_and_does_not_revive_on_beat(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    worker_id = uuid4()
    heartbeat = service(factory)
    await heartbeat.start(worker_id)
    assert await heartbeat.stopping(worker_id)
    assert await heartbeat.check_worker(worker_id) is WorkerHealthState.STALE
    assert not await heartbeat.beat(worker_id)


async def test_start_removes_only_records_older_than_retention(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    expired, recent, current = uuid4(), uuid4(), uuid4()
    heartbeat = service(factory)
    await heartbeat.start(expired)
    await heartbeat.start(recent)
    async with factory() as session, session.begin():
        expired_row = await session.get(WorkerHeartbeat, expired)
        recent_row = await session.get(WorkerHeartbeat, recent)
        assert expired_row is not None and recent_row is not None
        expired_row.started_at = datetime.now(UTC) - timedelta(days=9)
        expired_row.last_seen_at = datetime.now(UTC) - timedelta(days=8)
    await heartbeat.start(current)
    async with factory() as session:
        assert await session.get(WorkerHeartbeat, expired) is None
        assert await session.get(WorkerHeartbeat, recent) is not None
        assert await session.get(WorkerHeartbeat, current) is not None


async def test_periodic_task_writes_by_interval_and_marks_stopping() -> None:
    stop = asyncio.Event()

    class FakeHeartbeat:
        def __init__(self) -> None:
            self.beats = 0
            self.stopping_calls = 0
            self.two_beats = asyncio.Event()

        async def beat(self, _worker_id):
            self.beats += 1
            if self.beats >= 2:
                self.two_beats.set()
                stop.set()

        async def stopping(self, _worker_id):
            self.stopping_calls += 1

    fake = FakeHeartbeat()
    task = asyncio.create_task(_maintain_heartbeat(fake, uuid4(), stop, 0.01))  # type: ignore[arg-type]
    await asyncio.wait_for(fake.two_beats.wait(), timeout=0.5)
    await task
    assert fake.beats == 2
    assert fake.stopping_calls == 1
