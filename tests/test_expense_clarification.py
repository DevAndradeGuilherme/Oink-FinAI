from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import UUID, uuid4

import pytest_asyncio
from sqlalchemy import func, select
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
    ConversationStatus,
    ExpenseCategory,
    ExpenseClarificationField,
    ExpenseIntent,
    MessageSourceType,
    OutboundMessageKind,
    PaymentMethod,
    ProcessedMessageStatus,
)
from oink_finai.schemas.expense_clarification import ExpenseClarificationContext
from oink_finai.schemas.expense_interpretation import ExpenseInterpretation
from oink_finai.schemas.image_analysis import ImageAnalysisWarning, ImageDocumentType
from oink_finai.schemas.image_checkpoint import ImageAnalysisCheckpoint
from oink_finai.services.expense_interpreter import ExpenseInterpreter
from oink_finai.services.expense_processing import (
    CLARIFICATION_QUESTIONS,
    ExpenseProcessingService,
)


@pytest_asyncio.fixture
async def factory(tmp_path: Path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    database_path = (tmp_path / "clarification.db").as_posix()
    engine = create_async_engine(f"sqlite+aiosqlite:///{database_path}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    yield session_factory
    await engine.dispose()


class QueueInterpreter(ExpenseInterpreter):
    def __init__(self, results: list[ExpenseInterpretation]) -> None:
        self.results = results
        self.calls: list[tuple[str, datetime]] = []

    async def interpret(
        self, message: str, *, reference_timestamp: datetime
    ) -> ExpenseInterpretation:
        self.calls.append((message, reference_timestamp))
        return self.results[len(self.calls) - 1]


class MutableClock:
    def __init__(self) -> None:
        self.current = datetime(2026, 9, 5, 12, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += timedelta(seconds=seconds)


def unclear(
    *,
    description: str | None = "Mercado",
    missing: list[str] | None = None,
    confidence: float = 0.9,
) -> ExpenseInterpretation:
    return ExpenseInterpretation(
        intent=ExpenseIntent.UNCLEAR,
        amount=None,
        amount_evidence=None,
        description=description,
        merchant="Mercado" if description else None,
        category=ExpenseCategory.FOOD,
        payment_method=PaymentMethod.PIX,
        expense_date=None,
        confidence=confidence,
        missing_fields=missing or ["amount"],
        reasoning_summary="synthetic unclear result",
    )


def complete(
    *,
    amount: str = "42.50",
    description: str = "Mercado",
    expense_date=None,
    confidence: float = 0.99,
) -> ExpenseInterpretation:
    return ExpenseInterpretation(
        intent=ExpenseIntent.CREATE_EXPENSE,
        amount=Decimal(amount),
        amount_evidence=amount,
        description=description,
        merchant="Mercado",
        category=ExpenseCategory.FOOD,
        payment_method=PaymentMethod.PIX,
        expense_date=expense_date,
        confidence=confidence,
        missing_fields=[],
        reasoning_summary="synthetic complete result",
    )


def not_expense() -> ExpenseInterpretation:
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
        reasoning_summary="synthetic not expense",
    )


async def seed_user(factory: async_sessionmaker[AsyncSession]) -> UUID:
    async with factory() as session, session.begin():
        user = User(phone_number=f"synthetic-{uuid4().hex[:20]}")
        session.add_all([user, Category(name=ExpenseCategory.FOOD.value, slug="food")])
        await session.flush()
        session.add(ConversationState(user_id=user.id))
        return user.id


async def add_message(
    factory: async_sessionmaker[AsyncSession],
    user_id: UUID,
    text: str,
    *,
    source_type: MessageSourceType = MessageSourceType.TEXT,
    checkpoint: ImageAnalysisCheckpoint | None = None,
    clarification_origin_id: UUID | None = None,
) -> UUID:
    async with factory() as session, session.begin():
        message = ProcessedMessage(
            provider="synthetic",
            instance_id="synthetic-instance",
            external_message_id=uuid4().hex,
            user_id=user_id,
            clarification_origin_message_id=clarification_origin_id,
            accepted_text=text,
            source_type=source_type,
            transcribed_at=(
                datetime(2026, 9, 5, 12, tzinfo=UTC)
                if source_type is MessageSourceType.AUDIO
                else None
            ),
            image_analysis=checkpoint.payload() if checkpoint else None,
            image_analyzed_at=(datetime(2026, 9, 5, 12, tzinfo=UTC) if checkpoint else None),
            message_timestamp=datetime(2026, 9, 5, 12, tzinfo=UTC),
            status=ProcessedMessageStatus.PENDING,
            available_at=datetime(2026, 9, 5, 12, tzinfo=UTC),
        )
        session.add(message)
        await session.flush()
        return message.id


def processor(
    factory: async_sessionmaker[AsyncSession],
    interpreter: QueueInterpreter,
    clock: MutableClock,
) -> ExpenseProcessingService:
    return ExpenseProcessingService(
        factory,
        lambda _timezone: interpreter,
        clock=clock,
        clarification_ttl_seconds=60,
        clarification_min_confidence=0.75,
        retry_base_seconds=0,
        retry_max_seconds=0,
        jitter=lambda: 0,
    )


async def run_message(service: ExpenseProcessingService, message_id: UUID) -> None:
    assert message_id in await service.claim(10)
    await service.process(message_id)


async def assert_waiting_for(
    factory: async_sessionmaker[AsyncSession],
    message_id: UUID,
    field: ExpenseClarificationField,
) -> ExpenseClarificationContext:
    async with factory() as session:
        message = await session.get(ProcessedMessage, message_id)
        state = await session.scalar(select(ConversationState))
        outbox = await session.scalar(
            select(OutboundMessage).where(
                OutboundMessage.kind == OutboundMessageKind.CLARIFICATION,
                OutboundMessage.dedup_key.like(f"processed-message:{message_id}:%"),
            )
        )
        assert message is not None and message.status is ProcessedMessageStatus.NEEDS_CLARIFICATION
        assert state is not None
        assert state.status is ConversationStatus.WAITING_EXPENSE_CLARIFICATION
        assert state.active_expense_id is None and state.expires_at is not None
        context = ExpenseClarificationContext.model_validate(state.context)
        assert context.requested_field is field
        assert outbox is not None and outbox.content == CLARIFICATION_QUESTIONS[field]
        assert await session.scalar(select(func.count()).select_from(Expense)) == 0
        return context


async def test_incomplete_text_creates_minimal_durable_state_without_expense(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    user_id = await seed_user(factory)
    message_id = await add_message(factory, user_id, "synthetic incomplete expense")
    interpreter = QueueInterpreter([unclear()])
    service = processor(factory, interpreter, MutableClock())

    await run_message(service, message_id)

    context = await assert_waiting_for(factory, message_id, ExpenseClarificationField.AMOUNT)
    assert context.origin_message_id == message_id
    assert context.source_type is MessageSourceType.TEXT
    forbidden = {"audio", "image", "payload", "prompt", "response", "transcript", "reasoning"}
    assert forbidden.isdisjoint(context.payload())


async def test_incomplete_audio_uses_transcript_checkpoint_then_waits(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    user_id = await seed_user(factory)
    message_id = await add_message(
        factory,
        user_id,
        "synthetic incomplete transcript",
        source_type=MessageSourceType.AUDIO,
    )
    service = processor(factory, QueueInterpreter([unclear()]), MutableClock())

    await run_message(service, message_id)

    context = await assert_waiting_for(factory, message_id, ExpenseClarificationField.AMOUNT)
    assert context.source_type is MessageSourceType.AUDIO


async def test_multiple_explicit_text_amounts_override_model_choice(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    user_id = await seed_user(factory)
    message_id = await add_message(factory, user_id, "paguei R$ 10,00 e R$ 12,00")
    service = processor(factory, QueueInterpreter([complete(amount="12.00")]), MutableClock())

    await run_message(service, message_id)

    context = await assert_waiting_for(factory, message_id, ExpenseClarificationField.AMOUNT)
    assert context.amount is None


async def test_ambiguous_date_is_requested_before_description_or_merchant(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    user_id = await seed_user(factory)
    message_id = await add_message(factory, user_id, "synthetic ambiguous date")
    result = complete().model_copy(
        update={"missing_fields": ["merchant", "description", "expense_date"]}
    )
    service = processor(factory, QueueInterpreter([result]), MutableClock())

    await run_message(service, message_id)

    context = await assert_waiting_for(factory, message_id, ExpenseClarificationField.EXPENSE_DATE)
    assert context.remaining_fields == (
        ExpenseClarificationField.EXPENSE_DATE,
        ExpenseClarificationField.DESCRIPTION,
        ExpenseClarificationField.MERCHANT,
    )


async def test_image_with_multiple_amounts_and_dates_asks_one_field_at_a_time(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    checkpoint = ImageAnalysisCheckpoint(
        version=1,
        document_type=ImageDocumentType.RECEIPT,
        is_financial_document=True,
        is_legible=True,
        confidence=0.99,
        warnings=[ImageAnalysisWarning.MULTIPLE_AMOUNTS, ImageAnalysisWarning.MULTIPLE_DATES],
        amount_candidates=[
            {"value": "10.00", "evidence": "R$ 10,00", "label": "SUBTOTAL"},
            {"value": "12.00", "evidence": "R$ 12,00", "label": "TOTAL"},
        ],
        date_candidates=[
            {"value": "2026-09-04", "evidence": "04/09/2026", "label": "EMISSAO"},
            {"value": "2026-09-05", "evidence": "05/09/2026", "label": "PAGAMENTO"},
        ],
        merchant_candidates=[{"value": "Mercado", "evidence": "Mercado"}],
        payment_method_candidates=[{"value": "Pix", "evidence": "Pix"}],
    )
    user_id = await seed_user(factory)
    message_id = await add_message(
        factory, user_id, "", source_type=MessageSourceType.IMAGE, checkpoint=checkpoint
    )
    service = processor(factory, QueueInterpreter([complete(amount="10.00")]), MutableClock())

    await run_message(service, message_id)

    context = await assert_waiting_for(factory, message_id, ExpenseClarificationField.AMOUNT)
    assert context.remaining_fields[:2] == (
        ExpenseClarificationField.AMOUNT,
        ExpenseClarificationField.EXPENSE_DATE,
    )
    assert context.amount is None and context.expense_date is None


async def test_valid_answer_creates_one_expense_and_one_confirmation(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    user_id = await seed_user(factory)
    origin_id = await add_message(factory, user_id, "synthetic incomplete expense")
    interpreter = QueueInterpreter([unclear(), complete()])
    service = processor(factory, interpreter, MutableClock())
    await run_message(service, origin_id)
    response_id = await add_message(factory, user_id, "42,50", clarification_origin_id=origin_id)

    await run_message(service, response_id)

    async with factory() as session:
        state = await session.scalar(select(ConversationState))
        expense = await session.scalar(select(Expense))
        origins = await session.scalars(
            select(ProcessedMessage).where(ProcessedMessage.id.in_([origin_id, response_id]))
        )
        kinds = list(await session.scalars(select(OutboundMessage.kind)))
        assert state is not None and state.status is ConversationStatus.IDLE
        assert state.context is None and state.expires_at is None
        assert expense is not None and expense.processed_message_id == origin_id
        assert expense.source_type == MessageSourceType.TEXT
        assert {message.status for message in origins} == {ProcessedMessageStatus.PROCESSED}
        assert kinds.count(OutboundMessageKind.EXPENSE_CONFIRMATION) == 1
        assert kinds.count(OutboundMessageKind.CLARIFICATION) == 1
        assert await session.scalar(select(func.count()).select_from(Expense)) == 1
    contextual_input = interpreter.calls[1][0]
    assert "synthetic incomplete expense" not in contextual_input
    assert "reasoning_summary" not in contextual_input
    assert '"requested_field":"amount"' in contextual_input


async def test_invalid_answer_reasks_without_duplicate_state_or_retry_outbox(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    user_id = await seed_user(factory)
    origin_id = await add_message(factory, user_id, "synthetic incomplete expense")
    interpreter = QueueInterpreter([unclear(), unclear()])
    service = processor(factory, interpreter, MutableClock())
    await run_message(service, origin_id)
    response_id = await add_message(
        factory,
        user_id,
        "synthetic invalid answer",
        clarification_origin_id=origin_id,
    )

    await run_message(service, response_id)
    await service.process(response_id)

    async with factory() as session:
        assert await session.scalar(select(func.count()).select_from(ConversationState)) == 1
        assert await session.scalar(select(func.count()).select_from(Expense)) == 0
        assert (
            await session.scalar(
                select(func.count())
                .select_from(OutboundMessage)
                .where(OutboundMessage.kind == OutboundMessageKind.CLARIFICATION)
            )
            == 2
        )


async def test_expired_state_returns_to_idle_without_expense(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    user_id = await seed_user(factory)
    origin_id = await add_message(factory, user_id, "synthetic incomplete expense")
    interpreter = QueueInterpreter([unclear()])
    clock = MutableClock()
    service = processor(factory, interpreter, clock)
    await run_message(service, origin_id)
    clock.advance(61)
    response_id = await add_message(factory, user_id, "42,50", clarification_origin_id=origin_id)

    await run_message(service, response_id)

    async with factory() as session:
        state = await session.scalar(select(ConversationState))
        response = await session.get(ProcessedMessage, response_id)
        assert state is not None and state.status is ConversationStatus.IDLE
        assert state.context is None and state.expires_at is None
        assert response is not None and response.status is ProcessedMessageStatus.NOT_EXPENSE
        assert await session.scalar(select(func.count()).select_from(Expense)) == 0
    assert len(interpreter.calls) == 1


async def test_answer_from_another_user_cannot_consume_clarification(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    owner_id = await seed_user(factory)
    origin_id = await add_message(factory, owner_id, "synthetic incomplete expense")
    interpreter = QueueInterpreter([unclear(), not_expense()])
    service = processor(factory, interpreter, MutableClock())
    await run_message(service, origin_id)
    async with factory() as session, session.begin():
        other = User(phone_number=f"synthetic-{uuid4().hex[:20]}")
        session.add(other)
        await session.flush()
        session.add(ConversationState(user_id=other.id))
        other_id = other.id
    response_id = await add_message(factory, other_id, "42,50")

    await run_message(service, response_id)

    async with factory() as session:
        owner_state = await session.scalar(
            select(ConversationState).where(ConversationState.user_id == owner_id)
        )
        response = await session.get(ProcessedMessage, response_id)
        assert owner_state is not None
        assert owner_state.status is ConversationStatus.WAITING_EXPENSE_CLARIFICATION
        assert response is not None and response.status is ProcessedMessageStatus.NOT_EXPENSE
        assert await session.scalar(select(func.count()).select_from(Expense)) == 0


async def test_complete_new_message_replaces_clarification_as_new_intent(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    user_id = await seed_user(factory)
    origin_id = await add_message(factory, user_id, "synthetic incomplete expense")
    interpreter = QueueInterpreter([unclear(), complete(amount="30.00", description="Uber")])
    service = processor(factory, interpreter, MutableClock())
    await run_message(service, origin_id)
    new_message_id = await add_message(
        factory,
        user_id,
        "gastei R$ 30 no Uber",
        clarification_origin_id=origin_id,
    )

    await run_message(service, new_message_id)

    async with factory() as session:
        state = await session.scalar(select(ConversationState))
        expense = await session.scalar(select(Expense))
        assert state is not None and state.status is ConversationStatus.IDLE
        assert expense is not None and expense.processed_message_id == new_message_id
        assert await session.scalar(select(func.count()).select_from(Expense)) == 1


async def test_multiple_missing_fields_are_resolved_sequentially(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    user_id = await seed_user(factory)
    origin_id = await add_message(factory, user_id, "synthetic incomplete expense")
    interpreter = QueueInterpreter(
        [
            unclear(description=None, missing=["amount", "description"]),
            complete(description="temporary model value"),
            complete(description="Almoço"),
        ]
    )
    service = processor(factory, interpreter, MutableClock())
    await run_message(service, origin_id)
    amount_response = await add_message(
        factory, user_id, "42,50", clarification_origin_id=origin_id
    )
    await run_message(service, amount_response)

    context = await assert_waiting_for(
        factory, amount_response, ExpenseClarificationField.DESCRIPTION
    )
    assert context.amount == Decimal("42.50") and context.description is None

    description_response = await add_message(
        factory, user_id, "Almoço", clarification_origin_id=origin_id
    )
    await run_message(service, description_response)

    async with factory() as session:
        expense = await session.scalar(select(Expense))
        assert expense is not None and expense.description == "Almoço"
        assert await session.scalar(select(func.count()).select_from(Expense)) == 1


async def test_low_confidence_requires_explicit_intent_confirmation(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    user_id = await seed_user(factory)
    origin_id = await add_message(factory, user_id, "synthetic low confidence expense")
    interpreter = QueueInterpreter([complete(confidence=0.5), complete()])
    service = processor(factory, interpreter, MutableClock())

    await run_message(service, origin_id)

    await assert_waiting_for(factory, origin_id, ExpenseClarificationField.INTENT)
    response_id = await add_message(factory, user_id, "sim", clarification_origin_id=origin_id)
    await run_message(service, response_id)

    async with factory() as session:
        assert await session.scalar(select(func.count()).select_from(Expense)) == 1
        state = await session.scalar(select(ConversationState))
        assert state is not None and state.status is ConversationStatus.IDLE
