from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest_asyncio
from sqlalchemy import event, func, select, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from oink_finai.config.settings import Settings
from oink_finai.database.base import Base
from oink_finai.database.models import (
    OutboundMessage,
    ProcessedMessage,
    UsageLedger,
    User,
    WorkerHeartbeat,
)
from oink_finai.domain.enums import (
    MessageSourceType,
    OutboundMessageKind,
    OutboundMessageStatus,
    ProcessedMessageStatus,
    UsageOperation,
    UsageReservationState,
    WorkerHeartbeatStatus,
)
from oink_finai.operational_check import OperationalChecker, OperationalStatus
from oink_finai.services.readiness import EXPECTED_ALEMBIC_HEAD


@pytest_asyncio.fixture
async def database(
    tmp_path: Path,
) -> AsyncIterator[tuple[AsyncEngine, async_sessionmaker[AsyncSession]]]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'operations.db').as_posix()}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        await connection.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32))"))
        await connection.execute(
            text("INSERT INTO alembic_version VALUES (:version)"),
            {"version": EXPECTED_ALEMBIC_HEAD},
        )
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield engine, factory
    await engine.dispose()


def settings(**overrides) -> Settings:
    values = {
        "operational_queue_warning_count": 2,
        "operational_queue_critical_count": 4,
        "operational_queue_warning_age_seconds": 60,
        "operational_queue_critical_age_seconds": 300,
        "operational_outbox_warning_count": 2,
        "operational_outbox_critical_count": 4,
        "operational_outbox_warning_age_seconds": 60,
        "operational_outbox_critical_age_seconds": 300,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


async def seed_healthy_worker(factory: async_sessionmaker[AsyncSession]) -> None:
    now = datetime.now(UTC)
    async with factory() as session, session.begin():
        session.add(
            WorkerHeartbeat(
                worker_id=uuid4(),
                started_at=now,
                last_seen_at=now,
                status=WorkerHeartbeatStatus.RUNNING,
                release=None,
            )
        )


async def test_healthy_diagnostic_is_deterministic_read_only_and_exit_zero(database) -> None:
    engine, factory = database
    await seed_healthy_worker(factory)
    statements: list[str] = []

    def capture_statement(_connection, _cursor, statement, *_args) -> None:
        statements.append(statement.strip())

    event.listen(engine.sync_engine, "before_cursor_execute", capture_statement)
    before: dict[str, int] = {}
    async with factory() as session:
        for model in (ProcessedMessage, OutboundMessage, UsageLedger, WorkerHeartbeat):
            before[model.__tablename__] = int(
                await session.scalar(select(func.count()).select_from(model)) or 0
            )
    statements.clear()
    report = await OperationalChecker(factory, settings()).check()
    first = report.render()
    second = report.render()
    event.remove(engine.sync_engine, "before_cursor_execute", capture_statement)

    assert report.status is OperationalStatus.OK and report.exit_code == 0
    assert first == second
    assert report.checks["schema"] == {"status": "ok"}
    assert report.checks["api_heartbeat"]["applicable"] is False
    assert json_load(first)["status"] == "ok"
    assert statements and all(
        statement.upper().startswith(("SELECT", "WITH")) for statement in statements
    )
    async with factory() as session:
        for model in (ProcessedMessage, OutboundMessage, UsageLedger, WorkerHeartbeat):
            assert (
                int(await session.scalar(select(func.count()).select_from(model)) or 0)
                == before[model.__tablename__]
            )


async def test_stale_worker_is_critical(database) -> None:
    _engine, factory = database
    old = datetime.now(UTC) - timedelta(minutes=5)
    async with factory() as session, session.begin():
        session.add(
            WorkerHeartbeat(
                worker_id=uuid4(),
                started_at=old,
                last_seen_at=old,
                status=WorkerHeartbeatStatus.RUNNING,
                release=None,
            )
        )
    report = await OperationalChecker(factory, settings()).check()

    assert report.status is OperationalStatus.CRITICAL and report.exit_code == 2
    assert report.checks["worker_heartbeat"] == {
        "status": "critical",
        "state": "stale",
        "count": 0,
    }


async def test_schema_divergence_is_critical_without_querying_newer_tables(database) -> None:
    _engine, factory = database
    async with factory() as session, session.begin():
        await session.execute(text("UPDATE alembic_version SET version_num='20260909_0013'"))
    report = await OperationalChecker(factory, settings()).check()

    assert report.status is OperationalStatus.CRITICAL
    assert report.checks["database"]["status"] == "ok"
    assert report.checks["schema"]["status"] == "critical"


async def test_old_queues_warn_and_expired_lock_is_critical(database) -> None:
    _engine, factory = database
    await seed_healthy_worker(factory)
    now = datetime.now(UTC)
    async with factory() as session, session.begin():
        user = User(phone_number="synthetic-queue", timezone="UTC")
        session.add(user)
        await session.flush()
        session.add_all(
            [
                ProcessedMessage(
                    provider="synthetic",
                    instance_id="synthetic",
                    external_message_id="old-pending",
                    user_id=user.id,
                    accepted_text="synthetic",
                    source_type=MessageSourceType.TEXT,
                    message_timestamp=now,
                    status=ProcessedMessageStatus.PENDING,
                    available_at=now - timedelta(minutes=2),
                    processing_attempts=1,
                    next_attempt_at=now - timedelta(minutes=2),
                    created_at=now - timedelta(minutes=2),
                ),
                ProcessedMessage(
                    provider="synthetic",
                    instance_id="synthetic",
                    external_message_id="expired-lock",
                    user_id=user.id,
                    accepted_text="synthetic",
                    source_type=MessageSourceType.TEXT,
                    message_timestamp=now,
                    status=ProcessedMessageStatus.PROCESSING,
                    available_at=now,
                    locked_at=now - timedelta(minutes=10),
                    created_at=now,
                ),
            ]
        )
    report = await OperationalChecker(factory, settings()).check()

    assert report.checks["processing_queue"]["status"] == "warning"
    assert report.checks["processing_retries_overdue"]["status"] == "warning"
    assert report.checks["processing_locks_expired"]["status"] == "critical"
    assert report.status is OperationalStatus.CRITICAL


async def test_unknown_failed_blocked_and_usage_are_reported_without_changes(database) -> None:
    _engine, factory = database
    await seed_healthy_worker(factory)
    now = datetime.now(UTC)
    async with factory() as session, session.begin():
        user = User(phone_number="synthetic-diagnostics", timezone="UTC")
        session.add(user)
        await session.flush()
        message = ProcessedMessage(
            provider="synthetic",
            instance_id="synthetic",
            external_message_id="diagnostic-message",
            user_id=user.id,
            accepted_text="synthetic",
            source_type=MessageSourceType.TEXT,
            message_timestamp=now,
            status=ProcessedMessageStatus.PROCESSED,
            available_at=now,
        )
        session.add(message)
        await session.flush()
        unknown = OutboundMessage(
            user_id=user.id,
            processed_message_id=message.id,
            destination="synthetic",
            content="synthetic",
            content_type="TEXT",
            kind=OutboundMessageKind.QUERY_RESULT,
            dedup_key="diagnostic-unknown",
            status=OutboundMessageStatus.UNKNOWN,
            available_at=now,
            sequence_no=1,
            sequence_count=2,
        )
        failed = OutboundMessage(
            user_id=user.id,
            processed_message_id=None,
            destination="synthetic",
            content="synthetic",
            content_type="TEXT",
            kind=OutboundMessageKind.ACTION_ERROR,
            dedup_key="diagnostic-failed",
            status=OutboundMessageStatus.FAILED,
            available_at=now,
        )
        blocked = OutboundMessage(
            user_id=user.id,
            processed_message_id=message.id,
            destination="synthetic",
            content="synthetic",
            content_type="TEXT",
            kind=OutboundMessageKind.QUERY_RESULT,
            dedup_key="diagnostic-blocked",
            status=OutboundMessageStatus.PENDING,
            available_at=now,
            sequence_no=2,
            sequence_count=2,
        )
        session.add_all([unknown, failed, blocked])
        session.add_all(
            [
                reserved := UsageLedger(
                    user_id=user.id,
                    processed_message_id=message.id,
                    operation=UsageOperation.TEXT_INTERPRETATION,
                    model="synthetic-model",
                    durable_attempt=1,
                    window_minute_start=now.replace(second=0, microsecond=0),
                    window_day_start=now.replace(hour=0, minute=0, second=0, microsecond=0),
                    state=UsageReservationState.RESERVED,
                    created_at=now - timedelta(minutes=10),
                ),
                ambiguous := UsageLedger(
                    user_id=user.id,
                    processed_message_id=message.id,
                    operation=UsageOperation.QUERY_INTERPRETATION,
                    model="synthetic-model",
                    durable_attempt=2,
                    window_minute_start=now.replace(second=0, microsecond=0),
                    window_day_start=now.replace(hour=0, minute=0, second=0, microsecond=0),
                    state=UsageReservationState.AMBIGUOUS,
                ),
            ]
        )
    report = await OperationalChecker(factory, settings()).check()

    assert report.status is OperationalStatus.WARNING and report.exit_code == 1
    assert report.checks["outbox_unknown"]["count"] == 1
    assert report.checks["outbox_failed"]["count"] == 1
    assert report.checks["query_sequences_blocked"]["count"] == 1
    assert report.checks["usage_reserved_stale"]["count"] == 1
    assert report.checks["usage_ambiguous"]["count"] == 1
    async with factory() as session:
        saved_unknown = await session.get(OutboundMessage, unknown.id)
        saved_failed = await session.get(OutboundMessage, failed.id)
        saved_reserved = await session.get(UsageLedger, reserved.id)
        saved_ambiguous = await session.get(UsageLedger, ambiguous.id)
        assert saved_unknown is not None and saved_unknown.status is OutboundMessageStatus.UNKNOWN
        assert saved_failed is not None and saved_failed.status is OutboundMessageStatus.FAILED
        assert saved_reserved is not None and saved_reserved.state is UsageReservationState.RESERVED
        assert (
            saved_ambiguous is not None and saved_ambiguous.state is UsageReservationState.AMBIGUOUS
        )


def json_load(value: str) -> dict[str, object]:
    import json

    return json.loads(value)
