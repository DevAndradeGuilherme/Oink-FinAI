import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from sqlalchemy import exists, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import aliased

from oink_finai.config.settings import Settings, get_settings
from oink_finai.database.models import (
    OutboundMessage,
    ProcessedMessage,
    UsageLedger,
    WorkerHeartbeat,
)
from oink_finai.domain.enums import (
    OutboundMessageKind,
    OutboundMessageStatus,
    ProcessedMessageStatus,
    UsageReservationState,
    WorkerHeartbeatStatus,
)
from oink_finai.services.readiness import EXPECTED_ALEMBIC_HEAD


class OperationalStatus(StrEnum):
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"


_STATUS_RANK = {
    OperationalStatus.OK: 0,
    OperationalStatus.WARNING: 1,
    OperationalStatus.CRITICAL: 2,
}


@dataclass(frozen=True)
class OperationalReport:
    status: OperationalStatus
    checks: dict[str, dict[str, object]]

    @property
    def exit_code(self) -> int:
        return _STATUS_RANK[self.status]

    def render(self) -> str:
        return json.dumps(
            {"status": self.status.value, "checks": self.checks},
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )


class OperationalChecker:
    """Read-only, content-free operational diagnosis based on PostgreSQL state."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        settings: Settings,
        *,
        expected_head: str = EXPECTED_ALEMBIC_HEAD,
    ) -> None:
        self._session_factory = session_factory
        self._settings = settings
        self._expected_head = expected_head

    async def check(self) -> OperationalReport:
        checks: dict[str, dict[str, object]] = {}
        try:
            async with asyncio.timeout(self._settings.operational_check_database_timeout_seconds):
                async with self._session_factory() as session:
                    now = self._as_utc(await session.scalar(select(func.current_timestamp())))
                    checks["database"] = {"status": OperationalStatus.OK.value}
                    versions = list(
                        await session.scalars(text("SELECT version_num FROM alembic_version"))
                    )
                    if versions != [self._expected_head]:
                        checks["schema"] = {"status": OperationalStatus.CRITICAL.value}
                        checks["api_heartbeat"] = {
                            "status": OperationalStatus.OK.value,
                            "applicable": False,
                        }
                        return self._report(checks)
                    checks["schema"] = {"status": OperationalStatus.OK.value}
                    checks["api_heartbeat"] = {
                        "status": OperationalStatus.OK.value,
                        "applicable": False,
                    }
                    await self._worker_check(session, now, checks)
                    await self._queue_checks(session, now, checks)
                    await self._outbox_checks(session, now, checks)
                    await self._usage_checks(session, now, checks)
        except Exception:
            return OperationalReport(
                OperationalStatus.CRITICAL,
                {"database": {"status": OperationalStatus.CRITICAL.value}},
            )
        return self._report(checks)

    async def _worker_check(
        self,
        session: AsyncSession,
        now: datetime,
        checks: dict[str, dict[str, object]],
    ) -> None:
        rows = list(await session.scalars(select(WorkerHeartbeat)))
        recent_cutoff = now - timedelta(seconds=self._settings.worker_heartbeat_stale_seconds)
        recent = [
            row
            for row in rows
            if row.status is WorkerHeartbeatStatus.RUNNING
            and self._as_utc(row.last_seen_at) >= recent_cutoff
        ]
        if recent:
            state = "healthy"
            status = OperationalStatus.OK
        elif not rows:
            state = "absent"
            status = OperationalStatus.CRITICAL
        elif all(row.status is WorkerHeartbeatStatus.STOPPED for row in rows):
            state = "stopped"
            status = OperationalStatus.CRITICAL
        else:
            state = "stale"
            status = OperationalStatus.CRITICAL
        checks["worker_heartbeat"] = {
            "status": status.value,
            "state": state,
            "count": len(recent),
        }

    async def _queue_checks(
        self,
        session: AsyncSession,
        now: datetime,
        checks: dict[str, dict[str, object]],
    ) -> None:
        queue = await self._count_oldest(
            session,
            ProcessedMessage.created_at,
            ProcessedMessage.status.in_(
                [ProcessedMessageStatus.PENDING, ProcessedMessageStatus.PROCESSING]
            ),
        )
        checks["processing_queue"] = self._threshold_check(
            queue,
            now,
            warning_count=self._settings.operational_queue_warning_count,
            critical_count=self._settings.operational_queue_critical_count,
            warning_age=self._settings.operational_queue_warning_age_seconds,
            critical_age=self._settings.operational_queue_critical_age_seconds,
        )
        retries = await self._count_oldest(
            session,
            ProcessedMessage.next_attempt_at,
            ProcessedMessage.status == ProcessedMessageStatus.PENDING,
            ProcessedMessage.processing_attempts > 0,
            ProcessedMessage.next_attempt_at.is_not(None),
            ProcessedMessage.next_attempt_at <= now,
        )
        checks["processing_retries_overdue"] = self._attention_check(
            retries,
            now,
            critical_count=self._settings.operational_queue_critical_count,
            critical_age=self._settings.operational_queue_critical_age_seconds,
        )
        expired = await self._count_oldest(
            session,
            ProcessedMessage.locked_at,
            ProcessedMessage.status == ProcessedMessageStatus.PROCESSING,
            ProcessedMessage.locked_at.is_not(None),
            ProcessedMessage.locked_at
            < now - timedelta(seconds=self._settings.worker_processing_lock_timeout_seconds),
        )
        checks["processing_locks_expired"] = self._attention_check(
            expired, now, always_critical=True
        )

    async def _outbox_checks(
        self,
        session: AsyncSession,
        now: datetime,
        checks: dict[str, dict[str, object]],
    ) -> None:
        queue = await self._count_oldest(
            session,
            OutboundMessage.created_at,
            OutboundMessage.status.in_(
                [OutboundMessageStatus.PENDING, OutboundMessageStatus.SENDING]
            ),
        )
        checks["outbox_queue"] = self._threshold_check(
            queue,
            now,
            warning_count=self._settings.operational_outbox_warning_count,
            critical_count=self._settings.operational_outbox_critical_count,
            warning_age=self._settings.operational_outbox_warning_age_seconds,
            critical_age=self._settings.operational_outbox_critical_age_seconds,
        )
        for name, status in (
            ("outbox_unknown", OutboundMessageStatus.UNKNOWN),
            ("outbox_failed", OutboundMessageStatus.FAILED),
        ):
            metric = await self._count_oldest(
                session, OutboundMessage.created_at, OutboundMessage.status == status
            )
            checks[name] = self._attention_check(metric, now)

        current = aliased(OutboundMessage)
        earlier = aliased(OutboundMessage)
        blocked = await self._count_oldest(
            session,
            current.created_at,
            current.kind == OutboundMessageKind.QUERY_RESULT,
            current.sequence_no.is_not(None),
            current.status != OutboundMessageStatus.SENT,
            exists(
                select(earlier.id).where(
                    earlier.processed_message_id == current.processed_message_id,
                    earlier.kind == current.kind,
                    earlier.sequence_no < current.sequence_no,
                    earlier.status.in_(
                        [OutboundMessageStatus.UNKNOWN, OutboundMessageStatus.FAILED]
                    ),
                )
            ),
        )
        checks["query_sequences_blocked"] = self._attention_check(blocked, now)

    async def _usage_checks(
        self,
        session: AsyncSession,
        now: datetime,
        checks: dict[str, dict[str, object]],
    ) -> None:
        stale_reserved = await self._count_oldest(
            session,
            UsageLedger.created_at,
            UsageLedger.state == UsageReservationState.RESERVED,
            UsageLedger.created_at
            < now - timedelta(seconds=self._settings.openai_reservation_stale_seconds),
        )
        checks["usage_reserved_stale"] = self._attention_check(stale_reserved, now)
        ambiguous = await self._count_oldest(
            session,
            UsageLedger.created_at,
            UsageLedger.state == UsageReservationState.AMBIGUOUS,
        )
        checks["usage_ambiguous"] = self._attention_check(ambiguous, now)

    @staticmethod
    async def _count_oldest(
        session: AsyncSession, timestamp_column, *conditions
    ) -> tuple[int, datetime | None]:
        row = (
            await session.execute(
                select(func.count(), func.min(timestamp_column)).where(*conditions)
            )
        ).one()
        return int(row[0]), row[1]

    @classmethod
    def _threshold_check(
        cls,
        metric: tuple[int, datetime | None],
        now: datetime,
        *,
        warning_count: int,
        critical_count: int,
        warning_age: float,
        critical_age: float,
    ) -> dict[str, object]:
        count, oldest = metric
        age = cls._age_seconds(now, oldest)
        if count >= critical_count or age >= critical_age:
            status = OperationalStatus.CRITICAL
        elif count >= warning_count or age >= warning_age:
            status = OperationalStatus.WARNING
        else:
            status = OperationalStatus.OK
        return cls._metric_payload(status, count, age)

    @classmethod
    def _attention_check(
        cls,
        metric: tuple[int, datetime | None],
        now: datetime,
        *,
        critical_count: int | None = None,
        critical_age: float | None = None,
        always_critical: bool = False,
    ) -> dict[str, object]:
        count, oldest = metric
        age = cls._age_seconds(now, oldest)
        critical = count > 0 and (
            always_critical
            or (critical_count is not None and count >= critical_count)
            or (critical_age is not None and age >= critical_age)
        )
        status = (
            OperationalStatus.CRITICAL
            if critical
            else OperationalStatus.WARNING
            if count
            else OperationalStatus.OK
        )
        return cls._metric_payload(status, count, age)

    @staticmethod
    def _metric_payload(
        status: OperationalStatus, count: int, oldest_age_seconds: int
    ) -> dict[str, object]:
        payload: dict[str, object] = {"status": status.value, "count": count}
        if count:
            payload["oldest_age_seconds"] = oldest_age_seconds
        return payload

    @classmethod
    def _age_seconds(cls, now: datetime, value: datetime | None) -> int:
        if value is None:
            return 0
        return max(0, int((now - cls._as_utc(value)).total_seconds()))

    @staticmethod
    def _as_utc(value: datetime | None) -> datetime:
        if not isinstance(value, datetime):
            raise RuntimeError("database clock is unavailable")
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

    @staticmethod
    def _report(checks: dict[str, dict[str, object]]) -> OperationalReport:
        status = max(
            (OperationalStatus(check["status"]) for check in checks.values()),
            key=_STATUS_RANK.__getitem__,
            default=OperationalStatus.CRITICAL,
        )
        return OperationalReport(status, checks)


async def run() -> OperationalReport:
    try:
        settings = get_settings()
        engine = create_async_engine(settings.database_url_value, pool_pre_ping=True)
    except Exception:
        return OperationalReport(
            OperationalStatus.CRITICAL,
            {"database": {"status": OperationalStatus.CRITICAL.value}},
        )
    try:
        factory = async_sessionmaker(engine, expire_on_commit=False)
        return await OperationalChecker(factory, settings).check()
    finally:
        try:
            await engine.dispose()
        except Exception:
            pass


def main() -> None:
    try:
        report = asyncio.run(run())
    except Exception:
        report = OperationalReport(
            OperationalStatus.CRITICAL,
            {"database": {"status": OperationalStatus.CRITICAL.value}},
        )
    sys.stdout.write(report.render() + "\n")
    raise SystemExit(report.exit_code)


if __name__ == "__main__":
    main()
