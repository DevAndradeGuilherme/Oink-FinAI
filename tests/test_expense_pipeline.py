import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest
import pytest_asyncio
from sqlalchemy import func, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import DataError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from oink_finai.database.base import Base
from oink_finai.database.models import (
    Category,
    ConversationState,
    Expense,
    OutboundMessage,
    ProcessedMessage,
    User,
)
from oink_finai.domain.enums import (
    ExpenseCategory,
    ExpenseIntent,
    OutboundMessageKind,
    OutboundMessageStatus,
    PaymentMethod,
    ProcessedMessageStatus,
)
from oink_finai.providers.whatsapp import EvolutionProviderError, WhatsAppProvider
from oink_finai.schemas.expense_interpretation import ExpenseInterpretation
from oink_finai.services.expense_interpreter import ExpenseInterpreter
from oink_finai.services.expense_processing import ExpenseProcessingService
from oink_finai.services.gemini_errors import (
    GeminiErrorMetadata,
    GeminiRateLimitError,
    GeminiRequestError,
    GeminiTimeoutError,
    GeminiUnavailableError,
)
from oink_finai.services.outbox_delivery import OutboundMessageClaim, OutboxDeliveryService


@pytest_asyncio.fixture
async def factory(tmp_path: Path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    database_path = (tmp_path / "pipeline.db").as_posix()
    engine = create_async_engine(f"sqlite+aiosqlite:///{database_path}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    yield session_factory
    await engine.dispose()


def interpretation(intent: ExpenseIntent = ExpenseIntent.CREATE_EXPENSE) -> ExpenseInterpretation:
    create = intent is ExpenseIntent.CREATE_EXPENSE
    return ExpenseInterpretation(
        intent=intent,
        amount=Decimal("42.50") if create else None,
        amount_evidence="42,50" if create else None,
        description="Mercado" if create else None,
        merchant="Mercado" if create else None,
        category=ExpenseCategory.FOOD,
        payment_method=PaymentMethod.PIX if create else None,
        expense_date=None,
        confidence=0.99,
        missing_fields=[],
        reasoning_summary="valid",
    )


class FakeInterpreter(ExpenseInterpreter):
    def __init__(self, results: list[ExpenseInterpretation | Exception]) -> None:
        self.results = results
        self.calls = 0

    async def interpret(
        self, message: str, *, reference_timestamp: datetime
    ) -> ExpenseInterpretation:
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        if isinstance(result, Exception):
            raise result
        return result


class MutableClock:
    def __init__(self, current: datetime) -> None:
        self.current = current

    def __call__(self) -> datetime:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += timedelta(seconds=seconds)


async def seed(
    factory: async_sessionmaker[AsyncSession],
    *,
    status: ProcessedMessageStatus = ProcessedMessageStatus.PENDING,
    timezone: str = "America/Sao_Paulo",
    available_at: datetime | None = None,
    next_attempt_at: datetime | None = None,
    processing_attempts: int = 0,
) -> ProcessedMessage:
    async with factory() as session, session.begin():
        user = User(phone_number="5511999999999", timezone=timezone)
        category = Category(name=ExpenseCategory.FOOD.value, slug="alimentacao")
        session.add_all([user, category])
        await session.flush()
        session.add(ConversationState(user_id=user.id))
        message = ProcessedMessage(
            provider="evolution",
            instance_id="finance-instance",
            external_message_id="message-1",
            user_id=user.id,
            accepted_text="Mercado 42,50 no Pix",
            message_timestamp=datetime(2026, 9, 2, 2, 30, tzinfo=UTC),
            status=status,
            available_at=available_at or datetime.now(UTC),
            locked_at=datetime.now(UTC) if status == ProcessedMessageStatus.PROCESSING else None,
            next_attempt_at=next_attempt_at,
            processing_attempts=processing_attempts,
        )
        session.add(message)
        await session.flush()
        return message


def service(
    factory: async_sessionmaker[AsyncSession], interpreter: FakeInterpreter
) -> ExpenseProcessingService:
    return ExpenseProcessingService(
        factory,
        lambda _timezone: interpreter,
        retry_base_seconds=0.001,
        retry_max_seconds=0.001,
        jitter=lambda: 0,
    )


async def test_create_expense_is_idempotent_and_uses_user_timezone(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    message = await seed(factory)
    processor = service(factory, FakeInterpreter([interpretation()]))
    claimed = await processor.claim(10)

    await asyncio.gather(processor.process(message.id), processor.process(message.id))

    async with factory() as session:
        expenses = list(await session.scalars(select(Expense)))
        outbox = list(await session.scalars(select(OutboundMessage)))
        assert claimed == [message.id]
        assert len(expenses) == 1
        assert len(outbox) == 1
        assert expenses[0].amount == Decimal("42.50")
        assert expenses[0].expense_date.isoformat() == "2026-09-01"
        assert expenses[0].processed_message_id == message.id
        assert outbox[0].expense_id == expenses[0].id
        saved = await session.get(ProcessedMessage, message.id)
        assert saved is not None
        assert saved.error_code is None
        assert saved.last_error_code is None
        assert saved.processing_attempts == 1
        assert outbox[0].content == (
            "✅ Gasto registrado\n\n"
            " Valor: R$ 42,50\n"
            " Descrição: Mercado\n"
            f" Categoria: {ExpenseCategory.FOOD.value}\n"
            " Data: 01/09/2026\n"
            " Pagamento: Pix"
        )


@pytest.mark.parametrize(
    ("changes", "error_code"),
    [
        ({"amount": Decimal("1000000000000.00")}, "AMOUNT_OUT_OF_RANGE"),
        ({"amount": Decimal("1.001")}, "AMOUNT_SCALE_EXCEEDED"),
        ({"merchant": "m" * 161}, "MERCHANT_TOO_LONG"),
    ],
)
async def test_database_limit_rejection_is_terminal_and_not_retried(
    factory: async_sessionmaker[AsyncSession],
    changes: dict[str, object],
    error_code: str,
) -> None:
    message = await seed(factory)
    result = interpretation().model_copy(update=changes)
    interpreter = FakeInterpreter([result])
    processor = service(factory, interpreter)
    await processor.claim(1)

    await processor.process(message.id)
    await processor.process(message.id)

    async with factory() as session:
        saved = await session.get(ProcessedMessage, message.id)
        assert saved is not None
        assert saved.status == ProcessedMessageStatus.FAILED
        assert saved.error_code == error_code
        assert await session.scalar(select(func.count()).select_from(Expense)) == 0
        assert await session.scalar(select(func.count()).select_from(OutboundMessage)) == 0
    assert interpreter.calls == 1
    assert await processor.recover_stale(datetime.now(UTC) + timedelta(hours=1)) == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"amount": Decimal("999999999999.99")},
        {"merchant": "m" * 160},
    ],
)
async def test_database_limit_boundary_is_accepted(
    factory: async_sessionmaker[AsyncSession], changes: dict[str, object]
) -> None:
    message = await seed(factory)
    processor = service(factory, FakeInterpreter([interpretation().model_copy(update=changes)]))
    await processor.claim(1)

    await processor.process(message.id)

    async with factory() as session:
        saved = await session.get(ProcessedMessage, message.id)
        assert saved is not None and saved.status == ProcessedMessageStatus.PROCESSED
        assert await session.scalar(select(func.count()).select_from(Expense)) == 1
        assert await session.scalar(select(func.count()).select_from(OutboundMessage)) == 1


class PersistenceFailureProcessor(ExpenseProcessingService):
    error: Exception

    async def _create_expense(self, *args, **kwargs) -> None:
        raise self.error


def persistence_failure_service(
    factory: async_sessionmaker[AsyncSession],
    interpreter: FakeInterpreter,
    error: Exception,
) -> PersistenceFailureProcessor:
    processor = PersistenceFailureProcessor(factory, lambda _timezone: interpreter)
    processor.error = error
    return processor


async def test_data_error_rolls_back_and_marks_terminal_failure(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    message = await seed(factory)
    interpreter = FakeInterpreter([interpretation()])
    processor = persistence_failure_service(
        factory, interpreter, DataError("statement", {}, Exception("driver detail"))
    )
    await processor.claim(1)

    await processor.process(message.id)

    async with factory() as session:
        saved = await session.get(ProcessedMessage, message.id)
        assert saved is not None
        assert saved.status == ProcessedMessageStatus.FAILED
        assert saved.error_code == "PERSISTENCE_DATA_ERROR"
        assert await session.scalar(select(func.count()).select_from(Expense)) == 0
        assert await session.scalar(select(func.count()).select_from(OutboundMessage)) == 0
    assert interpreter.calls == 1


async def test_transient_database_error_is_not_classified_as_terminal_data_error(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    message = await seed(factory)
    processor = persistence_failure_service(
        factory,
        FakeInterpreter([interpretation()]),
        OperationalError("statement", {}, Exception("temporary")),
    )
    await processor.claim(1)

    with pytest.raises(OperationalError):
        await processor.process(message.id)

    async with factory() as session:
        saved = await session.get(ProcessedMessage, message.id)
        assert saved is not None
        assert saved.status == ProcessedMessageStatus.PROCESSING
        assert saved.error_code is None


@pytest.mark.parametrize(
    ("intent", "expected_status", "outbox_count"),
    [
        (ExpenseIntent.UNCLEAR, ProcessedMessageStatus.NEEDS_CLARIFICATION, 1),
        (ExpenseIntent.NOT_EXPENSE, ProcessedMessageStatus.NOT_EXPENSE, 0),
    ],
)
async def test_non_create_results_do_not_create_expense(
    factory: async_sessionmaker[AsyncSession],
    intent: ExpenseIntent,
    expected_status: ProcessedMessageStatus,
    outbox_count: int,
) -> None:
    message = await seed(factory)
    processor = service(factory, FakeInterpreter([interpretation(intent)]))
    await processor.claim(1)
    await processor.process(message.id)

    async with factory() as session:
        assert await session.scalar(select(func.count()).select_from(Expense)) == 0
        assert (
            await session.scalar(select(func.count()).select_from(OutboundMessage)) == outbox_count
        )
        saved = await session.get(ProcessedMessage, message.id)
        assert saved is not None and saved.status == expected_status


async def test_permanent_gemini_error_is_not_retried(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    message = await seed(factory)
    interpreter = FakeInterpreter([GeminiRequestError("sanitized")])
    processor = service(factory, interpreter)
    await processor.claim(1)
    await processor.process(message.id)

    async with factory() as session:
        saved = await session.get(ProcessedMessage, message.id)
        assert interpreter.calls == 1
        assert saved is not None and saved.status == ProcessedMessageStatus.FAILED
        assert saved.error_code == "GEMINI_REQUEST"
        assert saved.last_error_code == "GEMINI_REQUEST"


async def test_503_schedules_durable_retry_without_nested_attempt(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    message = await seed(factory)
    transient = GeminiUnavailableError(
        "sanitized",
        metadata=GeminiErrorMetadata(
            exception_class="ServerError", category="transient", duration_ms=1, http_status=503
        ),
    )
    interpreter = FakeInterpreter([transient, interpretation()])
    processor = service(factory, interpreter)
    await processor.claim(1)
    await processor.process(message.id)

    async with factory() as session:
        saved = await session.get(ProcessedMessage, message.id)
        assert saved is not None
        assert saved.status == ProcessedMessageStatus.PENDING
        assert saved.processing_attempts == 1
        assert saved.next_attempt_at is not None
        assert processor._now().tzinfo is not None
        assert saved.last_error_code == "GEMINI_UNAVAILABLE"
    assert interpreter.calls == 1


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (GeminiTimeoutError("sanitized"), "GEMINI_TIMEOUT"),
        (GeminiRateLimitError("sanitized"), "GEMINI_RATE_LIMIT"),
    ],
)
async def test_retryable_error_schedules_retry(
    factory: async_sessionmaker[AsyncSession],
    error: Exception,
    code: str,
) -> None:
    message = await seed(factory)
    interpreter = FakeInterpreter([error])
    processor = service(factory, interpreter)
    await processor.claim(1)

    await processor.process(message.id)

    async with factory() as session:
        saved = await session.get(ProcessedMessage, message.id)
        assert saved is not None
        assert saved.status == ProcessedMessageStatus.PENDING
        assert saved.last_error_code == code
        assert saved.next_attempt_at is not None
    assert interpreter.calls == 1


async def test_future_retry_becomes_eligible_at_scheduled_time(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    now = datetime(2026, 9, 2, 12, tzinfo=UTC)
    clock = MutableClock(now)
    message = await seed(factory)
    async with factory() as session, session.begin():
        saved = await session.get(ProcessedMessage, message.id)
        assert saved is not None
        saved.next_attempt_at = now + timedelta(seconds=30)
    processor = ExpenseProcessingService(factory, lambda _: FakeInterpreter([]), clock=clock)

    assert await processor.claim(1) == []
    clock.advance(30)
    assert await processor.claim(1) == [message.id]


@pytest.mark.parametrize(
    ("available_offset", "retry_offset", "is_eligible"),
    [
        (30, None, False),
        (-30, None, True),
        (-30, 30, False),
        (-30, -1, True),
    ],
)
async def test_claim_requires_initial_and_retry_availability(
    factory: async_sessionmaker[AsyncSession],
    available_offset: int,
    retry_offset: int | None,
    is_eligible: bool,
) -> None:
    now = datetime(2026, 9, 2, 12, tzinfo=UTC)
    message = await seed(
        factory,
        available_at=now + timedelta(seconds=available_offset),
        next_attempt_at=(
            now + timedelta(seconds=retry_offset) if retry_offset is not None else None
        ),
    )
    processor = ExpenseProcessingService(
        factory, lambda _: FakeInterpreter([]), clock=MutableClock(now)
    )

    claimed = await processor.claim(1)

    assert claimed == ([message.id] if is_eligible else [])


@pytest.mark.parametrize(("attempts", "is_claimed"), [(3, True), (4, False)])
async def test_claim_never_starts_attempt_beyond_limit(
    factory: async_sessionmaker[AsyncSession], attempts: int, is_claimed: bool
) -> None:
    message = await seed(factory, processing_attempts=attempts)
    processor = ExpenseProcessingService(factory, lambda _: FakeInterpreter([]), max_attempts=4)

    claimed = await processor.claim(1)

    assert claimed == ([message.id] if is_claimed else [])
    async with factory() as session:
        saved = await session.get(ProcessedMessage, message.id)
        assert saved is not None
        assert saved.processing_attempts == (4 if is_claimed else attempts)


def test_backoff_grows_caps_and_jitter_stays_in_range() -> None:
    factory = None
    low = ExpenseProcessingService(  # type: ignore[arg-type]
        factory,
        lambda _: FakeInterpreter([]),
        retry_base_seconds=30,
        retry_max_seconds=100,
        jitter=lambda: 0,
    )
    high = ExpenseProcessingService(  # type: ignore[arg-type]
        factory,
        lambda _: FakeInterpreter([]),
        retry_base_seconds=30,
        retry_max_seconds=100,
        jitter=lambda: 1,
    )

    assert [low._retry_delay(attempt).total_seconds() for attempt in range(1, 5)] == [
        15,
        30,
        50,
        50,
    ]
    assert [high._retry_delay(attempt).total_seconds() for attempt in range(1, 5)] == [
        30,
        60,
        100,
        100,
    ]


async def test_later_success_creates_one_expense_and_confirmation(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    now = datetime(2026, 9, 2, 12, tzinfo=UTC)
    clock = MutableClock(now)
    transient = GeminiUnavailableError(
        "sanitized",
        metadata=GeminiErrorMetadata(
            exception_class="ServerError", category="transient", duration_ms=1, http_status=503
        ),
    )
    interpreter = FakeInterpreter([transient, interpretation()])
    processor = ExpenseProcessingService(
        factory,
        lambda _: interpreter,
        max_attempts=4,
        retry_base_seconds=30,
        retry_max_seconds=300,
        jitter=lambda: 1,
        clock=clock,
    )
    message = await seed(factory)
    initial_available_at = message.available_at

    await processor.claim(1)
    await processor.process(message.id)
    clock.advance(30)
    await processor.claim(1)
    await processor.process(message.id)

    async with factory() as session:
        saved = await session.get(ProcessedMessage, message.id)
        outbox = list(await session.scalars(select(OutboundMessage)))
        assert saved is not None and saved.status == ProcessedMessageStatus.PROCESSED
        assert saved.error_code is None
        assert saved.last_error_code == "GEMINI_UNAVAILABLE"
        assert saved.next_attempt_at is None
        assert saved.locked_at is None
        assert saved.available_at.replace(tzinfo=UTC) == initial_available_at
        assert saved.processing_attempts == 2
        assert await session.scalar(select(func.count()).select_from(Expense)) == 1
        assert len(outbox) == 1
        assert outbox[0].kind == OutboundMessageKind.EXPENSE_CONFIRMATION
    assert interpreter.calls == 2


@pytest.mark.parametrize(
    ("error", "result_intent", "expected_status", "expected_error_code"),
    [
        (
            GeminiTimeoutError("sanitized"),
            ExpenseIntent.UNCLEAR,
            ProcessedMessageStatus.NEEDS_CLARIFICATION,
            "GEMINI_TIMEOUT",
        ),
        (
            GeminiRateLimitError("sanitized"),
            ExpenseIntent.NOT_EXPENSE,
            ProcessedMessageStatus.NOT_EXPENSE,
            "GEMINI_RATE_LIMIT",
        ),
    ],
)
async def test_non_expense_success_after_retry_clears_only_active_error(
    factory: async_sessionmaker[AsyncSession],
    error: Exception,
    result_intent: ExpenseIntent,
    expected_status: ProcessedMessageStatus,
    expected_error_code: str,
) -> None:
    now = datetime(2026, 9, 2, 12, tzinfo=UTC)
    clock = MutableClock(now)
    interpreter = FakeInterpreter([error, interpretation(result_intent)])
    processor = ExpenseProcessingService(
        factory,
        lambda _: interpreter,
        retry_base_seconds=30,
        retry_max_seconds=300,
        jitter=lambda: 1,
        clock=clock,
    )
    message = await seed(factory)

    await processor.claim(1)
    await processor.process(message.id)
    clock.advance(30)
    await processor.claim(1)
    await processor.process(message.id)

    async with factory() as session:
        saved = await session.get(ProcessedMessage, message.id)
        assert saved is not None and saved.status == expected_status
        assert saved.error_code is None
        assert saved.last_error_code == expected_error_code
        assert saved.next_attempt_at is None
        assert saved.locked_at is None
        assert saved.processing_attempts == 2
        assert await session.scalar(select(func.count()).select_from(Expense)) == 0


async def test_retry_exhaustion_creates_one_failure_notification(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    now = datetime(2026, 9, 2, 12, tzinfo=UTC)
    clock = MutableClock(now)
    transient = GeminiTimeoutError("sanitized")
    interpreter = FakeInterpreter([transient])
    processor = ExpenseProcessingService(
        factory,
        lambda _: interpreter,
        max_attempts=2,
        retry_base_seconds=30,
        retry_max_seconds=300,
        jitter=lambda: 1,
        clock=clock,
    )
    message = await seed(factory)

    await processor.claim(1)
    await processor.process(message.id)
    clock.advance(30)
    await processor.claim(1)
    await processor.process(message.id)
    await processor.process(message.id)

    async with factory() as session:
        saved = await session.get(ProcessedMessage, message.id)
        outbox = list(await session.scalars(select(OutboundMessage)))
        assert saved is not None and saved.status == ProcessedMessageStatus.FAILED
        assert saved.next_attempt_at is None
        assert saved.locked_at is None
        assert len(outbox) == 1
        assert outbox[0].expense_id is None
        assert outbox[0].kind == OutboundMessageKind.PROCESSING_FAILURE
        assert "GEMINI" not in outbox[0].content
        assert await session.scalar(select(func.count()).select_from(Expense)) == 0
    assert interpreter.calls == 2
    assert await processor.recover_stale(clock.current + timedelta(days=1)) == 0

    async with factory() as session, session.begin():
        notification = await session.scalar(select(OutboundMessage))
        assert notification is not None
        notification.available_at = datetime.now(UTC) - timedelta(seconds=1)
    provider = FakeProvider()
    delivery = OutboxDeliveryService(factory, provider)
    outbound_claim = (await delivery.claim(1))[0]
    await delivery.send(outbound_claim)
    async with factory() as session:
        notification = await session.scalar(select(OutboundMessage))
        assert notification is not None
        assert notification.status == OutboundMessageStatus.SENT
    assert provider.calls == 1


async def test_old_failed_message_remains_untouched(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    message = await seed(factory, status=ProcessedMessageStatus.FAILED)
    processor = service(factory, FakeInterpreter([interpretation()]))

    assert await processor.recover_stale(datetime.now(UTC) + timedelta(days=1)) == 0
    assert await processor.claim(1) == []
    async with factory() as session:
        saved = await session.get(ProcessedMessage, message.id)
        assert saved is not None
        assert saved.processing_attempts == 0
        assert saved.status == ProcessedMessageStatus.FAILED


async def test_two_processing_workers_do_not_claim_same_attempt(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    message = await seed(factory)
    first = service(factory, FakeInterpreter([interpretation()]))
    second = service(factory, FakeInterpreter([interpretation()]))

    first_claims = await first.claim(1)
    second_claims = await second.claim(1)

    assert first_claims == [message.id]
    assert second_claims == []
    async with factory() as session:
        saved = await session.get(ProcessedMessage, message.id)
        assert saved is not None and saved.processing_attempts == 1


async def test_restart_recovers_stale_processing_only(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    now = datetime(2026, 9, 2, 12, tzinfo=UTC)
    stale = await seed(
        factory,
        status=ProcessedMessageStatus.PROCESSING,
        available_at=now - timedelta(hours=1),
        processing_attempts=1,
    )
    async with factory() as session, session.begin():
        saved = await session.get(ProcessedMessage, stale.id)
        assert saved is not None
        saved.locked_at = now - timedelta(hours=1)
    processor = ExpenseProcessingService(
        factory,
        lambda _: FakeInterpreter([interpretation()]),
        retry_base_seconds=30,
        retry_max_seconds=30,
        jitter=lambda: 1,
        clock=MutableClock(now),
    )

    assert await processor.recover_stale(now - timedelta(minutes=5)) == 1
    assert await processor.claim(1) == []
    async with factory() as session:
        saved = await session.get(ProcessedMessage, stale.id)
        assert saved is not None
        assert saved.status == ProcessedMessageStatus.PENDING
        assert saved.processing_attempts == 1
        assert saved.available_at.replace(tzinfo=UTC) == (now - timedelta(hours=1))
        assert saved.next_attempt_at is not None
        assert saved.error_code == "PROCESSING_INTERRUPTED"


@pytest.mark.parametrize("attempts", [4, 5])
async def test_stale_at_or_above_limit_fails_once(
    factory: async_sessionmaker[AsyncSession], attempts: int
) -> None:
    now = datetime(2026, 9, 2, 12, tzinfo=UTC)
    stale = await seed(
        factory,
        status=ProcessedMessageStatus.PROCESSING,
        processing_attempts=attempts,
    )
    async with factory() as session, session.begin():
        saved = await session.get(ProcessedMessage, stale.id)
        assert saved is not None
        saved.locked_at = now - timedelta(hours=1)
    processor = ExpenseProcessingService(
        factory, lambda _: FakeInterpreter([]), max_attempts=4, clock=MutableClock(now)
    )

    assert await processor.recover_stale(now - timedelta(minutes=5)) == 1
    assert await processor.recover_stale(now - timedelta(minutes=5)) == 0

    async with factory() as session:
        saved = await session.get(ProcessedMessage, stale.id)
        assert saved is not None
        assert saved.status == ProcessedMessageStatus.FAILED
        assert saved.processing_attempts == attempts
        assert saved.error_code == "PROCESSING_ATTEMPTS_EXHAUSTED"
        assert saved.next_attempt_at is None
        assert saved.locked_at is None
        assert await session.scalar(select(func.count()).select_from(OutboundMessage)) == 1


async def test_claim_reconciles_exhausted_pending_once(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    message = await seed(factory, processing_attempts=4)
    processor = ExpenseProcessingService(factory, lambda _: FakeInterpreter([]), max_attempts=4)

    assert await processor.claim(1) == []
    assert await processor.claim(1) == []

    async with factory() as session:
        saved = await session.get(ProcessedMessage, message.id)
        assert saved is not None
        assert saved.status == ProcessedMessageStatus.FAILED
        assert saved.error_code == "PROCESSING_ATTEMPTS_EXHAUSTED"
        assert await session.scalar(select(func.count()).select_from(OutboundMessage)) == 1


async def test_historical_processed_message_is_not_claimed(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    historical = await seed(factory, status=ProcessedMessageStatus.PROCESSED)
    async with factory() as session, session.begin():
        saved = await session.get(ProcessedMessage, historical.id)
        assert saved is not None
        saved.accepted_text = ""
    processor = service(factory, FakeInterpreter([interpretation()]))

    assert await processor.claim(1) == []


def test_claim_query_uses_skip_locked() -> None:
    statement = select(ProcessedMessage).with_for_update(skip_locked=True)
    compiled = str(statement.compile(dialect=postgresql.dialect()))

    assert "FOR UPDATE SKIP LOCKED" in compiled


def test_processing_lock_query_only_locks_processed_messages() -> None:
    statement = ExpenseProcessingService._locked_message_statement(UUID(int=0))
    compiled = str(statement.compile(dialect=postgresql.dialect()))

    assert "FOR UPDATE OF processed_messages" in compiled
    assert "users" not in compiled


def test_outbox_claim_query_uses_skip_locked() -> None:
    statement = select(OutboundMessage).with_for_update(skip_locked=True)
    compiled = str(statement.compile(dialect=postgresql.dialect()))

    assert "FOR UPDATE SKIP LOCKED" in compiled


class FakeProvider(WhatsAppProvider):
    def __init__(self, result: str | Exception = "provider-1") -> None:
        self.result = result
        self.calls = 0

    async def parse_webhook(self, payload: dict[str, object]):
        raise NotImplementedError

    async def send_text(self, phone_number: str, text: str) -> str | None:
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


async def seed_outbox(
    factory: async_sessionmaker[AsyncSession],
    status: OutboundMessageStatus = OutboundMessageStatus.PENDING,
    *,
    suffix: str = "",
) -> OutboundMessage:
    async with factory() as session, session.begin():
        user = User(phone_number=f"5511888888{suffix or '88'}")
        session.add(user)
        await session.flush()
        message = OutboundMessage(
            user_id=user.id,
            destination=user.phone_number,
            content="safe confirmation",
            kind=OutboundMessageKind.CLARIFICATION,
            dedup_key=f"unique-key{suffix}",
            status=status,
            available_at=datetime.now(UTC),
            claimed_at=datetime.now(UTC) if status == OutboundMessageStatus.CLAIMED else None,
            sending_at=datetime.now(UTC) if status == OutboundMessageStatus.SENDING else None,
        )
        session.add(message)
        await session.flush()
        return message


async def test_outbox_success(factory: async_sessionmaker[AsyncSession]) -> None:
    message = await seed_outbox(factory)
    provider = FakeProvider()
    delivery = OutboxDeliveryService(factory, provider)
    claim = (await delivery.claim(1))[0]
    await delivery.send(claim)
    await delivery.send(claim)

    async with factory() as session:
        saved = await session.get(OutboundMessage, message.id)
        assert saved is not None and saved.status == OutboundMessageStatus.SENT
        assert saved.provider_message_id == "provider-1"
        assert provider.calls == 1


async def test_pretransmission_failure_returns_to_pending(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    message = await seed_outbox(factory)
    provider = FakeProvider(EvolutionProviderError("sanitized", outcome_unknown=False))
    delivery = OutboxDeliveryService(factory, provider, max_attempts=2)
    await delivery.send((await delivery.claim(1))[0])

    async with factory() as session:
        saved = await session.get(OutboundMessage, message.id)
        assert saved is not None and saved.status == OutboundMessageStatus.PENDING
        assert saved.attempt_count == 1


async def test_ambiguous_timeout_becomes_unknown(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    message = await seed_outbox(factory)
    provider = FakeProvider(EvolutionProviderError("sanitized", outcome_unknown=True))
    delivery = OutboxDeliveryService(factory, provider)
    await delivery.send((await delivery.claim(1))[0])

    async with factory() as session:
        saved = await session.get(OutboundMessage, message.id)
        assert saved is not None and saved.status == OutboundMessageStatus.UNKNOWN
        assert await delivery.claim(1) == []


async def test_abandoned_sending_is_never_reclaimed(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    await seed_outbox(factory, OutboundMessageStatus.SENDING)
    provider = FakeProvider()
    delivery = OutboxDeliveryService(factory, provider)

    assert await delivery.claim(10) == []
    assert provider.calls == 0


async def test_stale_claim_returns_to_pending_and_can_be_claimed(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    message = await seed_outbox(factory, OutboundMessageStatus.CLAIMED)
    async with factory() as session, session.begin():
        saved = await session.get(OutboundMessage, message.id)
        assert saved is not None
        saved.claimed_at = datetime.now(UTC) - timedelta(hours=1)
        saved.claim_token = UUID("00000000-0000-0000-0000-000000000001")
    delivery = OutboxDeliveryService(factory, FakeProvider())

    assert await delivery.recover_stale(datetime.now(UTC) - timedelta(minutes=5)) == (1, 0)
    claims = await delivery.claim(1)
    assert [claim.message_id for claim in claims] == [message.id]


async def test_stale_sending_becomes_unknown_and_is_not_reclaimed(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    message = await seed_outbox(factory, OutboundMessageStatus.SENDING)
    async with factory() as session, session.begin():
        saved = await session.get(OutboundMessage, message.id)
        assert saved is not None
        saved.sending_at = datetime.now(UTC) - timedelta(hours=1)
    delivery = OutboxDeliveryService(factory, FakeProvider())

    assert await delivery.recover_stale(datetime.now(UTC) - timedelta(minutes=5)) == (0, 1)
    assert await delivery.claim(1) == []
    async with factory() as session:
        saved = await session.get(OutboundMessage, message.id)
        assert saved is not None and saved.status == OutboundMessageStatus.UNKNOWN


async def test_recent_sending_is_not_recovered(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    message = await seed_outbox(factory, OutboundMessageStatus.SENDING)
    delivery = OutboxDeliveryService(factory, FakeProvider())

    assert await delivery.recover_stale(datetime.now(UTC) - timedelta(minutes=5)) == (0, 0)
    async with factory() as session:
        saved = await session.get(OutboundMessage, message.id)
        assert saved is not None and saved.status == OutboundMessageStatus.SENDING


@pytest.mark.parametrize(
    "status",
    [
        OutboundMessageStatus.SENT,
        OutboundMessageStatus.UNKNOWN,
        OutboundMessageStatus.FAILED,
    ],
)
async def test_terminal_outbox_status_is_never_recovered(
    factory: async_sessionmaker[AsyncSession], status: OutboundMessageStatus
) -> None:
    message = await seed_outbox(factory, status)
    delivery = OutboxDeliveryService(factory, FakeProvider())

    assert await delivery.recover_stale(datetime.now(UTC) + timedelta(hours=1)) == (0, 0)
    assert await delivery.claim(1) == []
    async with factory() as session:
        saved = await session.get(OutboundMessage, message.id)
        assert saved is not None and saved.status == status


async def test_two_workers_do_not_claim_same_message(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    message = await seed_outbox(factory)
    first = OutboxDeliveryService(factory, FakeProvider())
    second = OutboxDeliveryService(factory, FakeProvider())

    first_claims = await first.claim(1)
    second_claims = await second.claim(1)

    assert [claim.message_id for claim in first_claims] == [message.id]
    assert second_claims == []


async def test_worker_does_not_send_after_losing_claim(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    await seed_outbox(factory)
    provider = FakeProvider()
    delivery = OutboxDeliveryService(factory, provider)
    claim = (await delivery.claim(1))[0]

    await delivery.send(OutboundMessageClaim(claim.message_id, UUID(int=0)))

    assert provider.calls == 0


async def test_recovery_is_idempotent(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    await seed_outbox(factory, OutboundMessageStatus.CLAIMED)
    async with factory() as session, session.begin():
        message = await session.scalar(select(OutboundMessage))
        assert message is not None
        message.claimed_at = datetime.now(UTC) - timedelta(hours=1)
    delivery = OutboxDeliveryService(factory, FakeProvider())
    cutoff = datetime.now(UTC) - timedelta(minutes=5)

    assert await delivery.recover_stale(cutoff) == (1, 0)
    assert await delivery.recover_stale(cutoff) == (0, 0)
