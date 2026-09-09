import asyncio
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from uuid import UUID

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from oink_finai.config.settings import Settings
from oink_finai.database.models import ProcessedMessage, UsageLedger
from oink_finai.domain.enums import UsageOperation, UsageReservationState
from oink_finai.services.openai_usage import OpenAIUsageMetrics

_ADMISSION_LOCK_KEY = 6_021_884_105_117_903_913
_local_admission_lock = asyncio.Lock()
_OPENAI_OPERATIONS = (
    UsageOperation.TEXT_INTERPRETATION,
    UsageOperation.QUERY_INTERPRETATION,
    UsageOperation.IMAGE_ANALYSIS,
    UsageOperation.AUDIO_TRANSCRIPTION,
)


class AdmissionDenial(StrEnum):
    USER_MINUTE = "USER_MINUTE"
    USER_DAY = "USER_DAY"
    GLOBAL_DAY = "GLOBAL_DAY"
    OPERATION_USER_DAY = "OPERATION_USER_DAY"
    OPERATION_GLOBAL_DAY = "OPERATION_GLOBAL_DAY"
    GLOBAL_CONCURRENCY = "GLOBAL_CONCURRENCY"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    allowed: bool
    ledger_id: UUID | None = None
    denial: AdmissionDenial | None = None
    retry_at: datetime | None = None


class UsageControl:
    """Serializes admission in PostgreSQL and persists content-free accounting."""

    def __init__(
        self,
        settings: Settings,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        *,
        clock=lambda: datetime.now(UTC),
    ) -> None:
        self._settings = settings
        self._session_factory = session_factory
        self._clock = clock

    async def admit_inbound(
        self,
        session: AsyncSession,
        *,
        message_id: UUID,
        user_id: UUID,
    ) -> AdmissionResult:
        now = self._now()
        async with _local_admission_lock:
            await self._acquire_database_lock(session)
            existing = await self._existing(session, message_id, UsageOperation.INBOUND_MESSAGE, 0)
            if existing is not None:
                return AdmissionResult(True, existing.id)
            minute_start, day_start = self._windows(now)
            active_states = self._counted_states()
            user_minute = await session.scalar(
                select(func.count(UsageLedger.id)).where(
                    UsageLedger.operation == UsageOperation.INBOUND_MESSAGE,
                    UsageLedger.user_id == user_id,
                    UsageLedger.window_minute_start == minute_start,
                    UsageLedger.state.in_(active_states),
                )
            )
            user_day = await self._count(
                session,
                day_start=day_start,
                operation=UsageOperation.INBOUND_MESSAGE,
                user_id=user_id,
            )
            global_day = await self._count(
                session, day_start=day_start, operation=UsageOperation.INBOUND_MESSAGE
            )
            if int(user_minute or 0) >= self._settings.inbound_user_per_minute_limit:
                return AdmissionResult(
                    False,
                    denial=AdmissionDenial.USER_MINUTE,
                    retry_at=minute_start + timedelta(minutes=1),
                )
            if user_day >= self._settings.inbound_user_per_day_limit:
                return AdmissionResult(
                    False, denial=AdmissionDenial.USER_DAY, retry_at=self._next_day(day_start)
                )
            if global_day >= self._settings.inbound_global_per_day_limit:
                return AdmissionResult(
                    False, denial=AdmissionDenial.GLOBAL_DAY, retry_at=self._next_day(day_start)
                )
            ledger = UsageLedger(
                user_id=user_id,
                processed_message_id=message_id,
                operation=UsageOperation.INBOUND_MESSAGE,
                model=None,
                durable_attempt=0,
                window_minute_start=minute_start,
                window_day_start=day_start,
                state=UsageReservationState.COMPLETED,
                completed_at=now,
            )
            session.add(ledger)
            await session.flush()
            return AdmissionResult(True, ledger.id)

    async def reserve_openai(
        self,
        *,
        message_id: UUID,
        operation: UsageOperation,
        model: str,
    ) -> AdmissionResult:
        if self._session_factory is None:
            raise RuntimeError("usage control session factory is not configured")
        if operation not in _OPENAI_OPERATIONS:
            raise ValueError("operation is not an OpenAI operation")
        if not isinstance(model, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", model
        ):
            raise ValueError("model is invalid")
        async with _local_admission_lock, self._session_factory() as session, session.begin():
            await self._acquire_database_lock(session)
            now = self._now()
            await self._reconcile_stale_locked(session, now)
            message = await session.get(ProcessedMessage, message_id)
            if message is None or message.user_id is None:
                return AdmissionResult(False, denial=AdmissionDenial.IDEMPOTENCY_CONFLICT)
            attempt = message.processing_attempts
            existing = await self._existing(session, message_id, operation, attempt)
            if existing is not None:
                return AdmissionResult(False, existing.id, AdmissionDenial.IDEMPOTENCY_CONFLICT)
            minute_start, day_start = self._windows(now)
            user_total = await self._openai_total(session, day_start, message.user_id)
            global_total = await self._openai_total(session, day_start)
            user_operation = await self._count(
                session, day_start=day_start, operation=operation, user_id=message.user_id
            )
            global_operation = await self._count(session, day_start=day_start, operation=operation)
            user_limit, global_limit = self._operation_limits(operation)
            next_day = self._next_day(day_start)
            if user_total >= self._settings.openai_user_per_day_limit:
                return AdmissionResult(False, denial=AdmissionDenial.USER_DAY, retry_at=next_day)
            if global_total >= self._settings.openai_global_per_day_limit:
                return AdmissionResult(False, denial=AdmissionDenial.GLOBAL_DAY, retry_at=next_day)
            if user_operation >= user_limit:
                return AdmissionResult(
                    False, denial=AdmissionDenial.OPERATION_USER_DAY, retry_at=next_day
                )
            if global_operation >= global_limit:
                return AdmissionResult(
                    False, denial=AdmissionDenial.OPERATION_GLOBAL_DAY, retry_at=next_day
                )
            active = await session.scalar(
                select(func.count(UsageLedger.id)).where(
                    UsageLedger.operation.in_(_OPENAI_OPERATIONS),
                    UsageLedger.state == UsageReservationState.RESERVED,
                )
            )
            if int(active or 0) >= self._settings.openai_global_concurrency_limit:
                return AdmissionResult(
                    False,
                    denial=AdmissionDenial.GLOBAL_CONCURRENCY,
                    retry_at=now
                    + timedelta(seconds=self._settings.openai_concurrency_retry_seconds),
                )
            ledger = UsageLedger(
                user_id=message.user_id,
                processed_message_id=message.id,
                operation=operation,
                model=model,
                durable_attempt=attempt,
                window_minute_start=minute_start,
                window_day_start=day_start,
                state=UsageReservationState.RESERVED,
            )
            session.add(ledger)
            await session.flush()
            return AdmissionResult(True, ledger.id)

    async def mark_transmitted(self, ledger_id: UUID) -> None:
        await self._transition(ledger_id, transmitted=True)

    async def complete(self, ledger_id: UUID, metrics: OpenAIUsageMetrics | None = None) -> None:
        await self._transition(ledger_id, completed=True, metrics=metrics)

    async def settle_failure(
        self, ledger_id: UUID, metrics: OpenAIUsageMetrics | None = None
    ) -> UsageReservationState | None:
        if self._session_factory is None:
            return None
        async with self._session_factory() as session, session.begin():
            ledger = await session.scalar(
                select(UsageLedger).where(UsageLedger.id == ledger_id).with_for_update()
            )
            if ledger is None or ledger.state is not UsageReservationState.RESERVED:
                return ledger.state if ledger is not None else None
            now = self._now()
            if metrics is not None:
                ledger.input_tokens = metrics.input_tokens
                ledger.output_tokens = metrics.output_tokens
                ledger.audio_seconds = metrics.audio_seconds
            if ledger.transmitted_at is None:
                ledger.state = UsageReservationState.RELEASED
                ledger.released_at = now
            else:
                ledger.state = UsageReservationState.AMBIGUOUS
                ledger.completed_at = now
            return ledger.state

    async def reconcile_stale(self) -> int:
        if self._session_factory is None:
            return 0
        async with _local_admission_lock, self._session_factory() as session, session.begin():
            await self._acquire_database_lock(session)
            return await self._reconcile_stale_locked(session, self._now())

    async def _transition(
        self,
        ledger_id: UUID,
        *,
        transmitted: bool = False,
        completed: bool = False,
        metrics: OpenAIUsageMetrics | None = None,
    ) -> None:
        if self._session_factory is None:
            return
        async with self._session_factory() as session, session.begin():
            ledger = await session.scalar(
                select(UsageLedger).where(UsageLedger.id == ledger_id).with_for_update()
            )
            if ledger is None or ledger.state is not UsageReservationState.RESERVED:
                return
            now = self._now()
            if transmitted and ledger.transmitted_at is None:
                ledger.transmitted_at = now
            if completed:
                ledger.transmitted_at = ledger.transmitted_at or now
                ledger.completed_at = now
                ledger.state = UsageReservationState.COMPLETED
                if metrics is not None:
                    ledger.input_tokens = metrics.input_tokens
                    ledger.output_tokens = metrics.output_tokens
                    ledger.audio_seconds = metrics.audio_seconds

    async def _reconcile_stale_locked(self, session: AsyncSession, now: datetime) -> int:
        cutoff = now - timedelta(seconds=self._settings.openai_reservation_stale_seconds)
        ledgers = list(
            await session.scalars(
                select(UsageLedger)
                .where(
                    UsageLedger.state == UsageReservationState.RESERVED,
                    UsageLedger.created_at < cutoff,
                )
                .with_for_update(skip_locked=True)
            )
        )
        for ledger in ledgers:
            if ledger.transmitted_at is None:
                ledger.state = UsageReservationState.RELEASED
                ledger.released_at = now
            else:
                ledger.state = UsageReservationState.AMBIGUOUS
                ledger.completed_at = now
        return len(ledgers)

    async def _openai_total(
        self, session: AsyncSession, day_start: datetime, user_id: UUID | None = None
    ) -> int:
        statement = select(func.count(UsageLedger.id)).where(
            UsageLedger.operation.in_(_OPENAI_OPERATIONS),
            UsageLedger.window_day_start == day_start,
            UsageLedger.state.in_(self._counted_states()),
        )
        if user_id is not None:
            statement = statement.where(UsageLedger.user_id == user_id)
        return int(await session.scalar(statement) or 0)

    async def _count(
        self,
        session: AsyncSession,
        *,
        day_start: datetime,
        operation: UsageOperation,
        user_id: UUID | None = None,
    ) -> int:
        statement = select(func.count(UsageLedger.id)).where(
            UsageLedger.operation == operation,
            UsageLedger.window_day_start == day_start,
            UsageLedger.state.in_(self._counted_states()),
        )
        if user_id is not None:
            statement = statement.where(UsageLedger.user_id == user_id)
        return int(await session.scalar(statement) or 0)

    async def _existing(
        self,
        session: AsyncSession,
        message_id: UUID,
        operation: UsageOperation,
        attempt: int,
    ) -> UsageLedger | None:
        return await session.scalar(
            select(UsageLedger).where(
                UsageLedger.processed_message_id == message_id,
                UsageLedger.operation == operation,
                UsageLedger.durable_attempt == attempt,
            )
        )

    async def _acquire_database_lock(self, session: AsyncSession) -> None:
        bind = session.get_bind()
        if bind is not None and bind.dialect.name == "postgresql":
            await session.execute(
                text("SELECT pg_advisory_xact_lock(:lock_key)"),
                {"lock_key": _ADMISSION_LOCK_KEY},
            )

    def _operation_limits(self, operation: UsageOperation) -> tuple[int, int]:
        prefix = {
            UsageOperation.TEXT_INTERPRETATION: "text",
            UsageOperation.QUERY_INTERPRETATION: "query",
            UsageOperation.IMAGE_ANALYSIS: "image",
            UsageOperation.AUDIO_TRANSCRIPTION: "audio",
        }[operation]
        return (
            getattr(self._settings, f"openai_{prefix}_user_per_day_limit"),
            getattr(self._settings, f"openai_{prefix}_global_per_day_limit"),
        )

    def _now(self) -> datetime:
        value = self._clock()
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

    @staticmethod
    def _windows(now: datetime) -> tuple[datetime, datetime]:
        return now.replace(second=0, microsecond=0), now.replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    @staticmethod
    def _next_day(day_start: datetime) -> datetime:
        return day_start + timedelta(days=1)

    @staticmethod
    def _counted_states() -> tuple[UsageReservationState, ...]:
        return (
            UsageReservationState.RESERVED,
            UsageReservationState.COMPLETED,
            UsageReservationState.AMBIGUOUS,
        )
