import asyncio
import os
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy import delete, func, select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from oink_finai.api.routes.evolution_webhook import _clarification_origin
from oink_finai.database.models import (
    Category,
    ConversationState,
    Expense,
    ExpenseHistory,
    OutboundMessage,
    ProcessedMessage,
    User,
)
from oink_finai.domain.enums import (
    ConversationStatus,
    ExpenseCategory,
    ExpenseClarificationField,
    ExpenseHistoryAction,
    ExpenseIntent,
    MessageSourceType,
    OutboundMessageKind,
    OutboundMessageStatus,
    ProcessedMessageStatus,
)
from oink_finai.providers.whatsapp import WhatsAppProvider
from oink_finai.schemas.expense_clarification import ExpenseClarificationContext
from oink_finai.schemas.expense_interpretation import ExpenseInterpretation
from oink_finai.services.expense_interpreter import ExpenseInterpreter
from oink_finai.services.expense_processing import ExpenseProcessingService
from oink_finai.services.interpretation_errors import (
    AIErrorMetadata,
    InterpretationInvalidResponseError,
    InterpretationRequestError,
    InterpretationTimeoutError,
    InterpretationUnavailableError,
)
from oink_finai.services.outbox_delivery import OutboxDeliveryService

pytestmark = pytest.mark.skipif(
    "OINK_TEST_POSTGRES_URL" not in os.environ,
    reason="OINK_TEST_POSTGRES_URL not configured",
)


class NoCallProvider(WhatsAppProvider):
    async def parse_webhook(self, payload: dict[str, object]):
        raise AssertionError("provider must not be called")

    async def send_text(self, phone_number: str, text: str) -> str | None:
        raise AssertionError("provider must not be called")


class ErrorInterpreter(ExpenseInterpreter):
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def interpret(self, message: str, *, reference_timestamp: datetime):
        raise self.error


class NotExpenseInterpreter(ExpenseInterpreter):
    async def interpret(self, message: str, *, reference_timestamp: datetime):
        return ExpenseInterpretation(
            intent=ExpenseIntent.NOT_EXPENSE,
            amount=None,
            amount_evidence=None,
            description=None,
            merchant=None,
            category=ExpenseCategory.OTHER,
            payment_method=None,
            expense_date=None,
            confidence=1,
            missing_fields=[],
            reasoning_summary="not an expense",
        )


class CompleteExpenseInterpreter(ExpenseInterpreter):
    async def interpret(self, message: str, *, reference_timestamp: datetime):
        return ExpenseInterpretation(
            intent=ExpenseIntent.CREATE_EXPENSE,
            amount=Decimal("17.50"),
            amount_evidence="17,50",
            description="Synthetic",
            merchant=None,
            category=ExpenseCategory.FOOD,
            payment_method=None,
            expense_date=None,
            confidence=1,
            missing_fields=[],
            reasoning_summary="synthetic",
        )


async def seed_processing_message(
    factory: async_sessionmaker,
    *,
    processing_attempts: int = 1,
) -> tuple[User, ProcessedMessage]:
    unique = uuid4().hex
    async with factory() as session, session.begin():
        user = User(phone_number=f"pg-{unique[:20]}")
        session.add(user)
        await session.flush()
        message = ProcessedMessage(
            provider="postgres-test",
            instance_id=unique,
            external_message_id=unique,
            user_id=user.id,
            accepted_text="synthetic",
            message_timestamp=datetime.now(UTC),
            status=ProcessedMessageStatus.PROCESSING,
            available_at=datetime.now(UTC),
            locked_at=datetime.now(UTC),
            processing_attempts=processing_attempts,
        )
        session.add(message)
        await session.flush()
        return user, message


async def cleanup_processing_message(factory: async_sessionmaker, user_id) -> None:
    async with factory() as session, session.begin():
        await session.execute(delete(OutboundMessage).where(OutboundMessage.user_id == user_id))
        await session.execute(delete(ProcessedMessage).where(ProcessedMessage.user_id == user_id))
        await session.execute(delete(User).where(User.id == user_id))


async def test_postgres_rejects_numeric_14_2_overflow() -> None:
    engine = create_async_engine(os.environ["OINK_TEST_POSTGRES_URL"])
    with pytest.raises(DBAPIError) as captured:
        async with engine.begin() as connection:
            await connection.execute(text("SELECT CAST(1000000000000.00 AS NUMERIC(14, 2))"))
    assert getattr(captured.value.orig, "sqlstate", None) == "22003"
    assert ExpenseProcessingService._is_data_exception(captured.value)
    await engine.dispose()


async def test_postgres_claim_and_recovery_transitions() -> None:
    engine = create_async_engine(os.environ["OINK_TEST_POSTGRES_URL"])
    factory = async_sessionmaker(engine, expire_on_commit=False)
    unique = uuid4().hex[:20]
    async with factory() as session, session.begin():
        user = User(phone_number=f"test-{unique}")
        session.add(user)
        await session.flush()
        message = OutboundMessage(
            user_id=user.id,
            destination="test-destination",
            content="synthetic",
            kind=OutboundMessageKind.CLARIFICATION,
            dedup_key=f"postgres-test-{unique}",
            status=OutboundMessageStatus.PENDING,
            available_at=datetime.now(UTC),
        )
        session.add(message)
        await session.flush()
        message_id = message.id

    delivery = OutboxDeliveryService(factory, NoCallProvider())
    claim = (await delivery.claim(1))[0]
    async with factory() as session, session.begin():
        message = await session.get(OutboundMessage, message_id)
        assert message is not None
        assert message.status == OutboundMessageStatus.CLAIMED
        message.claimed_at = datetime.now(UTC) - timedelta(hours=1)

    assert await delivery.recover_stale(datetime.now(UTC) - timedelta(minutes=5)) == (1, 0)
    claim = (await delivery.claim(1))[0]
    assert await delivery._start_sending(claim) is not None
    async with factory() as session, session.begin():
        message = await session.get(OutboundMessage, message_id)
        assert message is not None
        message.sending_at = datetime.now(UTC) - timedelta(hours=1)

    assert await delivery.recover_stale(datetime.now(UTC) - timedelta(minutes=5)) == (0, 1)
    async with factory() as session:
        message = await session.get(OutboundMessage, message_id)
        assert message is not None
        assert message.status == OutboundMessageStatus.UNKNOWN
    await engine.dispose()


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_code"),
    [
        (
            InterpretationUnavailableError(
                "sanitized",
                metadata=AIErrorMetadata(
                    exception_class="ServerError",
                    category="transient",
                    duration_ms=1,
                    http_status=503,
                ),
            ),
            ProcessedMessageStatus.PENDING,
            "GEMINI_UNAVAILABLE",
        ),
        (
            InterpretationTimeoutError("sanitized"),
            ProcessedMessageStatus.PENDING,
            "GEMINI_TIMEOUT",
        ),
        (
            InterpretationRequestError("sanitized"),
            ProcessedMessageStatus.FAILED,
            "GEMINI_REQUEST",
        ),
    ],
)
async def test_postgres_processing_errors_lock_only_message_and_leave_transaction_usable(
    error: Exception,
    expected_status: ProcessedMessageStatus,
    expected_code: str,
) -> None:
    engine = create_async_engine(os.environ["OINK_TEST_POSTGRES_URL"])
    factory = async_sessionmaker(engine, expire_on_commit=False)
    user, message = await seed_processing_message(factory)
    processor = ExpenseProcessingService(
        factory,
        lambda _: ErrorInterpreter(error),
        retry_base_seconds=3600,
        retry_max_seconds=3600,
        jitter=lambda: 1,
    )

    try:
        async with factory() as session, session.begin():
            locked = await session.scalar(
                ExpenseProcessingService._locked_message_statement(message.id)
            )
            assert locked is not None
            assert await session.scalar(text("SELECT 1")) == 1

        await processor.process(message.id)

        async with factory() as session:
            saved = await session.get(ProcessedMessage, message.id)
            assert saved is not None
            assert saved.status == expected_status
            assert saved.error_code == expected_code
            assert saved.last_error_code == expected_code
            assert saved.available_at.tzinfo is not None
            if expected_status == ProcessedMessageStatus.PENDING:
                assert saved.next_attempt_at is not None
                assert saved.next_attempt_at.tzinfo is not None
            assert await session.scalar(select(func.count()).select_from(ProcessedMessage)) >= 1
    finally:
        await cleanup_processing_message(factory, user.id)
        await engine.dispose()


async def test_postgres_concurrent_retry_is_applied_once() -> None:
    engine = create_async_engine(os.environ["OINK_TEST_POSTGRES_URL"])
    factory = async_sessionmaker(engine, expire_on_commit=False)
    user, message = await seed_processing_message(factory)
    processor = ExpenseProcessingService(
        factory,
        lambda _: ErrorInterpreter(InterpretationTimeoutError("unused")),
        max_attempts=3,
        retry_base_seconds=3600,
        retry_max_seconds=3600,
        jitter=lambda: 1,
    )

    try:
        await asyncio.gather(
            processor._retry_or_fail(message.id, "GEMINI_TIMEOUT"),
            processor._retry_or_fail(message.id, "GEMINI_TIMEOUT"),
        )

        async with factory() as session:
            saved = await session.get(ProcessedMessage, message.id)
            assert saved is not None
            assert saved.status == ProcessedMessageStatus.PENDING
            assert saved.attempt_count == 1
            assert saved.next_attempt_at is not None
            assert (
                await session.scalar(
                    select(func.count())
                    .select_from(OutboundMessage)
                    .where(OutboundMessage.user_id == user.id)
                )
                == 0
            )
    finally:
        await cleanup_processing_message(factory, user.id)
        await engine.dispose()


async def test_postgres_concurrent_retry_exhaustion_creates_one_notification() -> None:
    engine = create_async_engine(os.environ["OINK_TEST_POSTGRES_URL"])
    factory = async_sessionmaker(engine, expire_on_commit=False)
    user, message = await seed_processing_message(factory, processing_attempts=2)
    processor = ExpenseProcessingService(
        factory,
        lambda _: ErrorInterpreter(InterpretationTimeoutError("unused")),
        max_attempts=2,
    )

    try:
        await asyncio.gather(
            processor._retry_or_fail(message.id, "GEMINI_TIMEOUT"),
            processor._retry_or_fail(message.id, "GEMINI_TIMEOUT"),
        )

        async with factory() as session:
            saved = await session.get(ProcessedMessage, message.id)
            assert saved is not None
            assert saved.status == ProcessedMessageStatus.FAILED
            assert saved.attempt_count == 1
            assert saved.next_attempt_at is None
            notifications = list(
                await session.scalars(
                    select(OutboundMessage).where(OutboundMessage.user_id == user.id)
                )
            )
            assert len(notifications) == 1
            assert notifications[0].kind == OutboundMessageKind.PROCESSING_FAILURE
    finally:
        await cleanup_processing_message(factory, user.id)
        await engine.dispose()


async def test_postgres_two_workers_do_not_start_attempt_beyond_limit() -> None:
    engine = create_async_engine(os.environ["OINK_TEST_POSTGRES_URL"])
    factory = async_sessionmaker(engine, expire_on_commit=False)
    user, message = await seed_processing_message(factory, processing_attempts=2)
    available_at = datetime(2099, 1, 1, tzinfo=UTC)
    async with factory() as session, session.begin():
        saved = await session.get(ProcessedMessage, message.id)
        assert saved is not None
        saved.status = ProcessedMessageStatus.PENDING
        saved.locked_at = None
        saved.available_at = available_at
        saved.created_at = datetime(2000, 1, 1, tzinfo=UTC)

    def clock() -> datetime:
        return datetime(2100, 1, 1, tzinfo=UTC)

    first = ExpenseProcessingService(
        factory, lambda _: ErrorInterpreter(Exception()), max_attempts=3, clock=clock
    )
    second = ExpenseProcessingService(
        factory, lambda _: ErrorInterpreter(Exception()), max_attempts=3, clock=clock
    )

    try:
        claims = await asyncio.gather(first.claim(1), second.claim(1))

        assert sum(message.id in worker_claims for worker_claims in claims) == 1
        async with factory() as session:
            saved = await session.get(ProcessedMessage, message.id)
            assert saved is not None
            assert saved.status == ProcessedMessageStatus.PROCESSING
            assert saved.processing_attempts == 3
            assert saved.available_at.tzinfo is not None
            assert saved.locked_at is not None and saved.locked_at.tzinfo is not None
    finally:
        await cleanup_processing_message(factory, user.id)
    await engine.dispose()


async def test_postgres_concurrent_clarification_answers_create_one_expense() -> None:
    engine = create_async_engine(os.environ["OINK_TEST_POSTGRES_URL"])
    factory = async_sessionmaker(engine, expire_on_commit=False)
    unique = uuid4().hex
    now = datetime.now(UTC)
    async with factory() as session, session.begin():
        category = await session.scalar(
            select(Category).where(Category.name == ExpenseCategory.FOOD.value)
        )
        if category is None:
            category = Category(name=ExpenseCategory.FOOD.value, slug=f"food-{unique[:12]}")
            session.add(category)
        user = User(phone_number=f"clarification-{unique[:16]}")
        session.add(user)
        await session.flush()
        origin = ProcessedMessage(
            provider="postgres-test",
            instance_id=unique,
            external_message_id=f"origin-{unique}",
            user_id=user.id,
            accepted_text="synthetic incomplete",
            source_type=MessageSourceType.TEXT,
            message_timestamp=now,
            status=ProcessedMessageStatus.NEEDS_CLARIFICATION,
            available_at=now,
            processing_attempts=1,
        )
        session.add(origin)
        await session.flush()
        context = ExpenseClarificationContext(
            origin_message_id=origin.id,
            source_type=MessageSourceType.TEXT,
            reference_timestamp=now,
            requested_field=ExpenseClarificationField.AMOUNT,
            remaining_fields=(ExpenseClarificationField.AMOUNT,),
            description="Synthetic",
            category=ExpenseCategory.FOOD,
        )
        session.add(
            ConversationState(
                user_id=user.id,
                status=ConversationStatus.WAITING_EXPENSE_CLARIFICATION,
                context=context.payload(),
                expires_at=now + timedelta(minutes=5),
            )
        )
        responses = [
            ProcessedMessage(
                provider="postgres-test",
                instance_id=unique,
                external_message_id=f"response-{index}-{unique}",
                user_id=user.id,
                clarification_origin_message_id=origin.id,
                accepted_text="17,50",
                source_type=MessageSourceType.TEXT,
                message_timestamp=now,
                status=ProcessedMessageStatus.PROCESSING,
                available_at=now,
                locked_at=now,
                processing_attempts=1,
            )
            for index in range(2)
        ]
        session.add_all(responses)
        await session.flush()
        user_id = user.id
        origin_id = origin.id
        response_ids = [response.id for response in responses]

    processors = [
        ExpenseProcessingService(
            factory,
            lambda _timezone: CompleteExpenseInterpreter(),
            clock=lambda: now,
        )
        for _ in response_ids
    ]
    await asyncio.gather(
        *(
            processor.process(message_id)
            for processor, message_id in zip(processors, response_ids, strict=True)
        )
    )

    async with factory() as session:
        expenses = await session.scalar(
            select(func.count())
            .select_from(Expense)
            .where(Expense.processed_message_id == origin_id)
        )
        confirmations = await session.scalar(
            select(func.count())
            .select_from(OutboundMessage)
            .where(
                OutboundMessage.user_id == user_id,
                OutboundMessage.kind == OutboundMessageKind.EXPENSE_CONFIRMATION,
            )
        )
        state = await session.scalar(
            select(ConversationState).where(ConversationState.user_id == user_id)
        )
        statuses = list(
            await session.scalars(
                select(ProcessedMessage.status).where(ProcessedMessage.id.in_(response_ids))
            )
        )
        assert expenses == 1 and confirmations == 1
        assert state is not None and state.status is ConversationStatus.IDLE
        assert state.context is None and state.expires_at is None
        assert sorted(status.value for status in statuses) == ["NOT_EXPENSE", "PROCESSED"]

    async with factory() as session, session.begin():
        await session.execute(delete(OutboundMessage).where(OutboundMessage.user_id == user_id))
        await session.execute(delete(Expense).where(Expense.processed_message_id == origin_id))
        await session.execute(delete(ConversationState).where(ConversationState.user_id == user_id))
        await session.execute(delete(ProcessedMessage).where(ProcessedMessage.user_id == user_id))
        await session.execute(delete(User).where(User.id == user_id))
    await engine.dispose()


@pytest.mark.parametrize("outcome", ["schema", "timeout", "503", "valid", "webhook"])
async def test_postgres_concurrent_schema_failure_reasks_once_and_preserves_draft(
    outcome: str,
) -> None:
    engine = create_async_engine(os.environ["OINK_TEST_POSTGRES_URL"])
    factory = async_sessionmaker(engine, expire_on_commit=False)
    unique = uuid4().hex
    now = datetime.now(UTC)
    async with factory() as session, session.begin():
        user = User(phone_number=f"clarification-error-{unique[:12]}")
        session.add(user)
        await session.flush()
        origin = ProcessedMessage(
            provider="postgres-test",
            instance_id=unique,
            external_message_id=f"origin-error-{unique}",
            user_id=user.id,
            accepted_text="Gastei 80 reais",
            source_type=MessageSourceType.TEXT,
            message_timestamp=now,
            status=ProcessedMessageStatus.NEEDS_CLARIFICATION,
            available_at=now,
            processing_attempts=1,
        )
        session.add(origin)
        await session.flush()
        context = ExpenseClarificationContext(
            origin_message_id=origin.id,
            source_type=MessageSourceType.TEXT,
            reference_timestamp=now,
            requested_field=ExpenseClarificationField.DESCRIPTION,
            remaining_fields=(ExpenseClarificationField.DESCRIPTION,),
            amount=Decimal("80.00"),
            category=ExpenseCategory.OTHER,
        )
        state = ConversationState(
            user_id=user.id,
            status=ConversationStatus.WAITING_EXPENSE_CLARIFICATION,
            context=context.payload(),
            expires_at=now + timedelta(minutes=5),
        )
        response = ProcessedMessage(
            provider="postgres-test",
            instance_id=unique,
            external_message_id=f"response-error-{unique}",
            user_id=user.id,
            clarification_origin_message_id=None,
            accepted_text="Foi no mercado",
            source_type=MessageSourceType.TEXT,
            message_timestamp=now,
            status=ProcessedMessageStatus.PROCESSING,
            available_at=now,
            locked_at=now,
            processing_attempts=1,
        )
        session.add_all([state, response])
        await session.flush()
        user_id = user.id
        response_id = response.id

    clock_now = now
    entered = 0
    ready = asyncio.Event()

    class ConcurrentInterpreter(ExpenseInterpreter):
        async def interpret(self, message: str, *, reference_timestamp: datetime):
            nonlocal entered, clock_now
            entered += 1
            if entered == 2:
                ready.set()
            await asyncio.wait_for(ready.wait(), timeout=5)
            if outcome != "schema":
                clock_now = now + timedelta(minutes=6)
            if outcome == "webhook":
                async with factory() as session, session.begin():
                    locked = await session.scalar(
                        select(ConversationState)
                        .where(ConversationState.user_id == user_id)
                        .with_for_update()
                    )
                    assert _clarification_origin(locked, clock_now) == origin.id
                    assert locked.status is ConversationStatus.WAITING_EXPENSE_CLARIFICATION
            if outcome == "valid":
                return await CompleteExpenseInterpreter().interpret(
                    message, reference_timestamp=reference_timestamp
                )
            error = {
                "timeout": InterpretationTimeoutError,
                "503": InterpretationUnavailableError,
            }.get(outcome, InterpretationInvalidResponseError)
            raise error("synthetic")

    processor = ExpenseProcessingService(
        factory,
        lambda _timezone: ConcurrentInterpreter(),
        clock=lambda: clock_now,
    )
    await asyncio.gather(processor.process(response_id), processor.process(response_id))

    async with factory() as session:
        saved = await session.get(ProcessedMessage, response_id)
        saved_state = await session.scalar(
            select(ConversationState).where(ConversationState.user_id == user_id)
        )
        notifications = list(
            await session.scalars(select(OutboundMessage).where(OutboundMessage.user_id == user_id))
        )
        expected_status = (
            ProcessedMessageStatus.NEEDS_CLARIFICATION
            if outcome == "valid"
            else ProcessedMessageStatus.FAILED
        )
        assert saved is not None and saved.status is expected_status
        if outcome in {"schema", "webhook"}:
            assert saved.error_code == "GEMINI_SCHEMA_INVALID"
        assert saved.attempt_count == 1
        assert saved.clarification_origin_message_id == origin.id
        assert saved_state is not None
        assert saved_state.status is ConversationStatus.WAITING_EXPENSE_CLARIFICATION
        assert saved_state.context == context.payload()
        assert len(notifications) == 1
        assert notifications[0].kind is OutboundMessageKind.CLARIFICATION
        assert (
            await session.scalar(
                select(func.count()).select_from(Expense).where(Expense.user_id == user_id)
            )
            == 0
        )

    async with factory() as session, session.begin():
        await session.execute(delete(OutboundMessage).where(OutboundMessage.user_id == user_id))
        await session.execute(delete(ConversationState).where(ConversationState.user_id == user_id))
        await session.execute(delete(ProcessedMessage).where(ProcessedMessage.user_id == user_id))
        await session.execute(delete(User).where(User.id == user_id))
    await engine.dispose()


async def test_postgres_concurrent_delete_confirmations_delete_once() -> None:
    engine = create_async_engine(os.environ["OINK_TEST_POSTGRES_URL"])
    factory = async_sessionmaker(engine, expire_on_commit=False)
    unique = uuid4().hex
    now = datetime.now(UTC)
    async with factory() as session, session.begin():
        user = User(phone_number=f"delete-{unique[:20]}")
        category = Category(name=f"Delete test {unique}", slug=f"delete-{unique}")
        session.add_all([user, category])
        await session.flush()
        expense = Expense(
            user_id=user.id,
            category_id=category.id,
            amount=Decimal("10.00"),
            description="Concurrent delete",
            expense_date=date.today(),
        )
        session.add(expense)
        await session.flush()
        session.add(
            ConversationState(
                user_id=user.id,
                status=ConversationStatus.WAITING_EXPENSE_DELETE_CONFIRM,
                active_expense_id=expense.id,
                context={"action": "DELETE", "expense_id": str(expense.id)},
                expires_at=now + timedelta(minutes=10),
            )
        )
        messages = [
            ProcessedMessage(
                provider="postgres-delete-test",
                instance_id=unique,
                external_message_id=f"{unique}-{index}",
                user_id=user.id,
                accepted_text=f"confirmar-remocao {expense.id}",
                message_timestamp=now,
                status=ProcessedMessageStatus.PROCESSING,
                available_at=now,
                locked_at=now,
                processing_attempts=1,
            )
            for index in range(2)
        ]
        session.add_all(messages)
        await session.flush()
        user_id, expense_id = user.id, expense.id
        message_ids = [message.id for message in messages]

    processor = ExpenseProcessingService(
        factory, lambda _: ErrorInterpreter(AssertionError("interpreter called"))
    )
    try:
        await asyncio.gather(*(processor.process(message_id) for message_id in message_ids))
        async with factory() as session:
            saved = await session.get(Expense, expense_id)
            history_count = await session.scalar(
                select(func.count())
                .select_from(ExpenseHistory)
                .where(
                    ExpenseHistory.expense_id == expense_id,
                    ExpenseHistory.action == ExpenseHistoryAction.DELETE,
                )
            )
            confirmation_count = await session.scalar(
                select(func.count())
                .select_from(OutboundMessage)
                .where(
                    OutboundMessage.expense_id == expense_id,
                    OutboundMessage.kind == OutboundMessageKind.EXPENSE_DELETED,
                )
            )
            assert saved is not None and saved.deleted_at is not None
            assert history_count == 1
            assert confirmation_count == 1
            state_values = (
                await session.execute(
                    text(
                        "SELECT context IS NULL, context::text, active_expense_id IS NULL, "
                        "expires_at IS NULL FROM conversation_states WHERE user_id = :user_id"
                    ),
                    {"user_id": user_id},
                )
            ).one()
            assert state_values == (True, None, True, True)
    finally:
        async with factory() as session, session.begin():
            await session.execute(delete(OutboundMessage).where(OutboundMessage.user_id == user_id))
            await session.execute(
                delete(ExpenseHistory).where(ExpenseHistory.expense_id == expense_id)
            )
            await session.execute(
                delete(ConversationState).where(ConversationState.user_id == user_id)
            )
            await session.execute(delete(Expense).where(Expense.id == expense_id))
            await session.execute(
                delete(ProcessedMessage).where(ProcessedMessage.user_id == user_id)
            )
            await session.execute(delete(User).where(User.id == user_id))
            await session.execute(delete(Category).where(Category.slug == f"delete-{unique}"))
        await engine.dispose()


@pytest.mark.parametrize("incoming_text", ["cancelar", "mensagem normal sem gasto"])
async def test_postgres_idle_resets_store_sql_nulls(incoming_text: str) -> None:
    engine = create_async_engine(os.environ["OINK_TEST_POSTGRES_URL"])
    factory = async_sessionmaker(engine, expire_on_commit=False)
    unique = uuid4().hex
    now = datetime.now(UTC)
    async with factory() as session, session.begin():
        user = User(phone_number=f"reset-{unique[:20]}")
        category = Category(name=f"Reset test {unique}", slug=f"reset-{unique}")
        session.add_all([user, category])
        await session.flush()
        expense = Expense(
            user_id=user.id,
            category_id=category.id,
            amount=Decimal("10.00"),
            description="Pending deletion",
            expense_date=date.today(),
        )
        session.add(expense)
        await session.flush()
        state = ConversationState(
            user_id=user.id,
            status=ConversationStatus.WAITING_EXPENSE_DELETE_CONFIRM,
            active_expense_id=expense.id,
            context={"action": "DELETE", "expense_id": str(expense.id)},
            expires_at=now + timedelta(minutes=10),
        )
        message = ProcessedMessage(
            provider="postgres-reset-test",
            instance_id=unique,
            external_message_id=unique,
            user_id=user.id,
            accepted_text=incoming_text,
            message_timestamp=now,
            status=ProcessedMessageStatus.PROCESSING,
            available_at=now,
            locked_at=now,
            processing_attempts=1,
        )
        session.add_all([state, message])
        await session.flush()
        user_id, expense_id, message_id = user.id, expense.id, message.id

    async with factory() as session:
        stored_context = await session.scalar(
            text("SELECT context->>'action' FROM conversation_states WHERE user_id = :user_id"),
            {"user_id": user_id},
        )
        assert stored_context == "DELETE"

    processor = ExpenseProcessingService(factory, lambda _: NotExpenseInterpreter())
    try:
        await processor.process(message_id)
        async with factory() as session:
            state_values = (
                await session.execute(
                    text(
                        "SELECT status::text, context IS NULL, context::text, "
                        "active_expense_id IS NULL, expires_at IS NULL "
                        "FROM conversation_states WHERE user_id = :user_id"
                    ),
                    {"user_id": user_id},
                )
            ).one()
            assert state_values == ("IDLE", True, None, True, True)
    finally:
        async with factory() as session, session.begin():
            await session.execute(delete(OutboundMessage).where(OutboundMessage.user_id == user_id))
            await session.execute(
                delete(ConversationState).where(ConversationState.user_id == user_id)
            )
            await session.execute(delete(Expense).where(Expense.id == expense_id))
            await session.execute(
                delete(ProcessedMessage).where(ProcessedMessage.user_id == user_id)
            )
            await session.execute(delete(User).where(User.id == user_id))
            await session.execute(delete(Category).where(Category.slug == f"reset-{unique}"))
        await engine.dispose()
