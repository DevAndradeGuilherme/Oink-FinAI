import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from oink_finai.api.routes.evolution_webhook import _clarification_origin
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
    CLARIFICATION_RETRY_PREFIX,
    ExpenseProcessingService,
)
from oink_finai.services.gemini_errors import (
    GeminiModelUnavailableError,
    GeminiSchemaError,
    GeminiTimeoutError,
    GeminiUnavailableError,
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
    def __init__(self, results: list[ExpenseInterpretation | Exception]) -> None:
        self.results = results
        self.calls: list[tuple[str, datetime]] = []

    async def interpret(
        self, message: str, *, reference_timestamp: datetime
    ) -> ExpenseInterpretation:
        self.calls.append((message, reference_timestamp))
        result = self.results[len(self.calls) - 1]
        if isinstance(result, Exception):
            raise result
        return result


class ExpiringSchemaInterpreter(ExpenseInterpreter):
    def __init__(
        self,
        draft: ExpenseInterpretation,
        clock: "MutableClock",
    ) -> None:
        self.draft = draft
        self.clock = clock
        self.calls = 0

    async def interpret(
        self, message: str, *, reference_timestamp: datetime
    ) -> ExpenseInterpretation:
        self.calls += 1
        if self.calls == 1:
            return self.draft
        self.clock.advance(61)
        raise GeminiSchemaError("sanitized invalid provider output")


class ExpiringValidInterpreter(ExpenseInterpreter):
    def __init__(
        self,
        draft: ExpenseInterpretation,
        answer: ExpenseInterpretation,
        clock: "MutableClock",
    ) -> None:
        self.draft = draft
        self.answer = answer
        self.clock = clock
        self.calls = 0

    async def interpret(
        self, message: str, *, reference_timestamp: datetime
    ) -> ExpenseInterpretation:
        self.calls += 1
        if self.calls == 1:
            return self.draft
        self.clock.advance(61)
        return self.answer


class DraftThenSchemaErrorInterpreter(ExpenseInterpreter):
    def __init__(self, draft: ExpenseInterpretation) -> None:
        self.draft = draft
        self.calls = 0

    async def interpret(
        self, message: str, *, reference_timestamp: datetime
    ) -> ExpenseInterpretation:
        self.calls += 1
        if self.calls == 1:
            return self.draft
        raise GeminiSchemaError("sanitized invalid provider output")


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
        session.add_all(
            [
                user,
                Category(name=ExpenseCategory.FOOD.value, slug="food"),
                Category(name=ExpenseCategory.OTHER.value, slug="other"),
            ]
        )
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
    interpreter: ExpenseInterpreter,
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


async def test_explicit_pending_fields_are_resolved_one_at_a_time(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    user_id = await seed_user(factory)
    origin_id = await add_message(factory, user_id, "Gastei 80 reais")
    draft = ExpenseInterpretation(
        intent=ExpenseIntent.CREATE_EXPENSE,
        amount=Decimal("80.00"),
        amount_evidence="80 reais",
        description=None,
        merchant=None,
        category=ExpenseCategory.OTHER,
        payment_method=None,
        expense_date=date(2026, 9, 5),
        confidence=0.7,
        missing_fields=["description", "payment_method", "intent"],
        reasoning_summary="synthetic incomplete result",
    )
    description_answer = ExpenseInterpretation(
        intent=ExpenseIntent.CREATE_EXPENSE,
        amount=None,
        amount_evidence=None,
        description="Mercado",
        merchant="Mercado",
        category=ExpenseCategory.OTHER,
        payment_method=None,
        expense_date=None,
        confidence=0.99,
        missing_fields=[],
        reasoning_summary="synthetic clarification result",
    )
    payment_answer = description_answer.model_copy(
        update={"description": None, "merchant": None, "payment_method": PaymentMethod.PIX}
    )
    intent_answer = description_answer.model_copy(update={"description": None, "merchant": None})
    interpreter = QueueInterpreter([draft, description_answer, payment_answer, intent_answer])
    service = processor(factory, interpreter, MutableClock())
    await run_message(service, origin_id)
    description_response_id = await add_message(
        factory,
        user_id,
        "Foi no mercado",
        clarification_origin_id=origin_id,
    )

    await run_message(service, description_response_id)
    context = await assert_waiting_for(
        factory, description_response_id, ExpenseClarificationField.PAYMENT_METHOD
    )
    assert context.description == "Mercado"
    assert context.remaining_fields == (
        ExpenseClarificationField.PAYMENT_METHOD,
        ExpenseClarificationField.INTENT,
    )

    payment_response_id = await add_message(
        factory,
        user_id,
        "Pix",
        clarification_origin_id=origin_id,
    )
    await run_message(service, payment_response_id)
    context = await assert_waiting_for(
        factory, payment_response_id, ExpenseClarificationField.INTENT
    )
    assert context.description == "Mercado"
    assert context.payment_method is PaymentMethod.PIX
    assert context.remaining_fields == (ExpenseClarificationField.INTENT,)

    intent_response_id = await add_message(
        factory,
        user_id,
        "Sim",
        clarification_origin_id=origin_id,
    )
    await run_message(service, intent_response_id)
    await service.process(intent_response_id)

    async with factory() as session:
        expense = await session.scalar(select(Expense))
        state = await session.scalar(select(ConversationState))
        messages = list(
            await session.scalars(
                select(ProcessedMessage).where(
                    ProcessedMessage.id.in_(
                        [
                            origin_id,
                            description_response_id,
                            payment_response_id,
                            intent_response_id,
                        ]
                    )
                )
            )
        )
        outbox = list(await session.scalars(select(OutboundMessage)))
        assert expense is not None
        assert expense.processed_message_id == origin_id
        assert expense.amount == Decimal("80.00")
        assert expense.description == "Mercado"
        assert expense.expense_date == date(2026, 9, 5)
        category = await session.get(Category, expense.category_id)
        assert category is not None and category.name == ExpenseCategory.OTHER.value
        assert state is not None and state.status is ConversationStatus.IDLE
        assert state.context is None and state.active_expense_id is None
        assert state.expires_at is None
        status_by_id = {message.id: message.status for message in messages}
        assert status_by_id[origin_id] is ProcessedMessageStatus.PROCESSED
        assert status_by_id[description_response_id] is ProcessedMessageStatus.NEEDS_CLARIFICATION
        assert status_by_id[payment_response_id] is ProcessedMessageStatus.NEEDS_CLARIFICATION
        assert status_by_id[intent_response_id] is ProcessedMessageStatus.PROCESSED
        assert [message.kind for message in outbox].count(
            OutboundMessageKind.EXPENSE_CONFIRMATION
        ) == 1
        assert [message.kind for message in outbox].count(OutboundMessageKind.CLARIFICATION) == 3


async def test_optional_payment_method_absent_does_not_block_completion(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    user_id = await seed_user(factory)
    origin_id = await add_message(factory, user_id, "Gastei 80 reais")
    draft = complete(amount="80.00", description="placeholder").model_copy(
        update={
            "description": None,
            "merchant": None,
            "payment_method": None,
            "missing_fields": ["description"],
        }
    )
    answer = complete(amount="80.00", description="Mercado").model_copy(
        update={"amount": None, "amount_evidence": None, "payment_method": None}
    )
    service = processor(factory, QueueInterpreter([draft, answer]), MutableClock())
    await run_message(service, origin_id)
    response_id = await add_message(
        factory, user_id, "Foi no mercado", clarification_origin_id=origin_id
    )

    await run_message(service, response_id)

    async with factory() as session:
        expense = await session.scalar(select(Expense))
        assert expense is not None
        assert expense.payment_method is None
        assert await session.scalar(select(func.count()).select_from(Expense)) == 1


async def test_value_explicitly_marked_missing_remains_pending(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    user_id = await seed_user(factory)
    origin_id = await add_message(factory, user_id, "Gastei 80 reais")
    draft = complete(amount="80.00", description="placeholder").model_copy(
        update={"description": None, "missing_fields": ["description"]}
    )
    ambiguous_answer = complete(description="Mercado").model_copy(
        update={
            "amount": None,
            "amount_evidence": None,
            "missing_fields": ["description"],
        }
    )
    service = processor(factory, QueueInterpreter([draft, ambiguous_answer]), MutableClock())
    await run_message(service, origin_id)
    response_id = await add_message(
        factory, user_id, "Foi no mercado", clarification_origin_id=origin_id
    )

    await run_message(service, response_id)

    context = await assert_waiting_for(factory, response_id, ExpenseClarificationField.DESCRIPTION)
    assert context.description is None
    async with factory() as session:
        assert await session.scalar(select(func.count()).select_from(Expense)) == 0


@pytest.mark.parametrize("intent", [ExpenseIntent.CREATE_EXPENSE, ExpenseIntent.UNCLEAR])
async def test_all_new_ambiguities_survive_unresolved_answer_and_resolve_in_stages(
    factory: async_sessionmaker[AsyncSession], intent: ExpenseIntent
) -> None:
    user_id = await seed_user(factory)
    origin = await add_message(factory, user_id, "Gastei 80 reais")
    draft = complete(amount="80.00").model_copy(
        update={"description": None, "payment_method": None, "missing_fields": ["description"]}
    )
    answer = complete().model_copy(update={"amount": None, "amount_evidence": None})
    ambiguous = answer.model_copy(
        update={"intent": intent, "missing_fields": ["description", "payment_method"]}
    )
    service = processor(
        factory, QueueInterpreter([draft, ambiguous, answer, answer]), MutableClock()
    )
    await run_message(service, origin)
    first = await add_message(
        factory, user_id, "synthetic ambiguous", clarification_origin_id=origin
    )
    await run_message(service, first)
    context = await assert_waiting_for(factory, first, ExpenseClarificationField.DESCRIPTION)
    assert context.remaining_fields == (
        ExpenseClarificationField.DESCRIPTION,
        ExpenseClarificationField.PAYMENT_METHOD,
    )
    second = await add_message(factory, user_id, "Mercado", clarification_origin_id=origin)
    await run_message(service, second)
    context = await assert_waiting_for(factory, second, ExpenseClarificationField.PAYMENT_METHOD)
    assert context.payment_method is None
    third = await add_message(factory, user_id, "Pix", clarification_origin_id=origin)
    await run_message(service, third)
    await service.process(third)
    async with factory() as session:
        assert await session.scalar(select(func.count()).select_from(Expense)) == 1
        assert (
            await session.scalar(
                select(func.count())
                .select_from(OutboundMessage)
                .where(OutboundMessage.kind == OutboundMessageKind.EXPENSE_CONFIRMATION)
            )
            == 1
        )
        state = await session.scalar(select(ConversationState))
        assert state.status is ConversationStatus.IDLE and state.context is None


@pytest.mark.parametrize("outcome", ["valid", "schema", "timeout", "503"])
@pytest.mark.parametrize("webhook_arrives", [False, True])
async def test_late_reply_transition_matrix(
    factory: async_sessionmaker[AsyncSession], outcome: str, webhook_arrives: bool
) -> None:
    user = await seed_user(factory)
    origin = await add_message(factory, user, "Gastei 80 reais")
    draft = complete(amount="80.00").model_copy(
        update={"description": None, "missing_fields": ["description"]}
    )
    clock = MutableClock()
    reply_id = None

    class Interpreter(ExpenseInterpreter):
        async def interpret(self, message: str, *, reference_timestamp: datetime):
            if reply_id is None:
                return draft
            async with factory() as session:
                saved = await session.get(ProcessedMessage, reply_id)
                state = await session.scalar(select(ConversationState))
                context = ExpenseClarificationContext.model_validate(state.context)
                assert saved.clarification_origin_message_id == origin
                assert reply_id in context.reply_bindings
            clock.advance(61)
            if webhook_arrives:
                async with factory() as session, session.begin():
                    state = await session.scalar(select(ConversationState).with_for_update())
                    assert _clarification_origin(state, clock()) == origin
                    assert state.status is ConversationStatus.WAITING_EXPENSE_CLARIFICATION
                    assert state.context is not None
            errors = {
                "schema": GeminiSchemaError("synthetic"),
                "timeout": GeminiTimeoutError("synthetic"),
                "503": GeminiUnavailableError("synthetic"),
            }
            if outcome in errors:
                raise errors[outcome]
            return complete().model_copy(update={"amount": None, "amount_evidence": None})

    service = processor(factory, Interpreter(), clock)
    await run_message(service, origin)
    reply_id = await add_message(factory, user, "Mercado")
    await run_message(service, reply_id)
    await service.process(reply_id)
    async with factory() as session:
        state = await session.scalar(select(ConversationState))
        saved = await session.get(ProcessedMessage, reply_id)
        context = ExpenseClarificationContext.model_validate(state.context)
        assert state.status is ConversationStatus.WAITING_EXPENSE_CLARIFICATION
        assert context.amount == Decimal("80.00") and context.description is None
        assert not context.reply_bindings
        assert state.expires_at.replace(tzinfo=UTC) > clock()
        assert saved.next_attempt_at is None and saved.locked_at is None
        assert saved.status in {
            ProcessedMessageStatus.FAILED,
            ProcessedMessageStatus.NEEDS_CLARIFICATION,
        }
        assert await session.scalar(select(func.count()).select_from(Expense)) == 0
        replies = list(
            await session.scalars(
                select(OutboundMessage).where(
                    OutboundMessage.dedup_key.like(f"processed-message:{reply_id}:%")
                )
            )
        )
        assert len(replies) == 1
        assert replies[0].content == (
            f"{CLARIFICATION_RETRY_PREFIX} {CLARIFICATION_QUESTIONS[context.requested_field]}"
        )


@pytest.mark.parametrize("retry_delayed", [False, True])
async def test_durable_retry_keeps_binding_and_cannot_silently_expire(
    factory: async_sessionmaker[AsyncSession], retry_delayed: bool
) -> None:
    user = await seed_user(factory)
    origin = await add_message(factory, user, "synthetic origin")
    clock = MutableClock()
    interpreter = QueueInterpreter([unclear(), GeminiTimeoutError("synthetic"), complete()])
    service = processor(factory, interpreter, clock)
    await run_message(service, origin)
    reply = await add_message(factory, user, "42,50")
    await run_message(service, reply)
    async with factory() as session:
        saved = await session.get(ProcessedMessage, reply)
        state = await session.scalar(select(ConversationState))
        context = ExpenseClarificationContext.model_validate(state.context)
        assert saved.status is ProcessedMessageStatus.PENDING
        assert reply in context.reply_bindings
    if retry_delayed:
        clock.advance(61)
    # Recreate worker: the binding and retry policy must survive process loss.
    service = processor(factory, interpreter, clock)
    await run_message(service, reply)
    async with factory() as session:
        state = await session.scalar(select(ConversationState))
        expense_count = await session.scalar(select(func.count()).select_from(Expense))
        replies = list(
            await session.scalars(
                select(OutboundMessage).where(
                    OutboundMessage.dedup_key.like(f"processed-message:{reply}:%")
                )
            )
        )
        assert len(replies) == 1
        if retry_delayed:
            assert expense_count == 0
            assert state.status is ConversationStatus.WAITING_EXPENSE_CLARIFICATION
            assert replies[0].kind is OutboundMessageKind.CLARIFICATION
        else:
            assert expense_count == 1 and state.status is ConversationStatus.IDLE
            assert replies[0].kind is OutboundMessageKind.EXPENSE_CONFIRMATION


@pytest.mark.parametrize("intervention", ["cancel", "new_intent", "other_user"])
async def test_inflight_reply_respects_explicit_supersession_and_user_isolation(
    factory: async_sessionmaker[AsyncSession], intervention: str
) -> None:
    user = await seed_user(factory)
    origin = await add_message(factory, user, "synthetic origin")
    clock = MutableClock()
    reply = None

    class Interpreter(ExpenseInterpreter):
        async def interpret(self, message: str, *, reference_timestamp: datetime):
            if reply is None:
                return unclear()
            target = user
            if intervention == "other_user":
                async with factory() as session, session.begin():
                    other = User(phone_number=f"other-{uuid4().hex[:20]}")
                    session.add(other)
                    await session.flush()
                    target = other.id
                    session.add(ConversationState(user_id=target))
            command = "cancelar" if intervention == "cancel" else "gastei 30 reais no Uber"
            incoming = await add_message(factory, target, command)
            nested = processor(factory, QueueInterpreter([complete(amount="30.00")]), clock)
            await run_message(nested, incoming)
            return complete()

    service = processor(factory, Interpreter(), clock)
    await run_message(service, origin)
    reply = await add_message(factory, user, "42,50")
    await run_message(service, reply)
    async with factory() as session:
        state = await session.scalar(
            select(ConversationState).where(ConversationState.user_id == user)
        )
        saved = await session.get(ProcessedMessage, reply)
        assert state.status is ConversationStatus.IDLE and state.context is None
        own_expenses = await session.scalar(
            select(func.count()).select_from(Expense).where(Expense.user_id == user)
        )
        assert own_expenses == (0 if intervention == "cancel" else 1)
        assert saved.status is (
            ProcessedMessageStatus.PROCESSED
            if intervention == "other_user"
            else ProcessedMessageStatus.NOT_EXPENSE
        )


async def test_received_reply_cannot_be_reinterpreted_for_a_new_draft_revision(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    user = await seed_user(factory)
    origin = await add_message(factory, user, "synthetic origin")
    clock = MutableClock()
    interpreter = QueueInterpreter(
        [unclear(description=None, missing=["amount", "description"]), complete()]
    )
    service = processor(factory, interpreter, clock)
    await run_message(service, origin)
    first = await add_message(factory, user, "42,50")
    second = await add_message(factory, user, "42,50")
    async with factory() as session, session.begin():
        state = await session.scalar(select(ConversationState))
        for reply in (first, second):
            message = await session.get(ProcessedMessage, reply)
            message.clarification_origin_message_id = _clarification_origin(
                state, clock(), reply_id=reply
            )
    assert await service.claim(10) == [first, second]
    await service.process(first)
    await service.process(second)
    async with factory() as session:
        state = await session.scalar(select(ConversationState))
        context = ExpenseClarificationContext.model_validate(state.context)
        assert context.requested_field is ExpenseClarificationField.DESCRIPTION
        assert context.amount == Decimal("42.50") and context.description is None
        assert not context.reply_bindings
        assert len(interpreter.calls) == 2
        assert await session.scalar(select(func.count()).select_from(Expense)) == 0


async def test_ambiguous_intent_denial_does_not_cancel_draft(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    user = await seed_user(factory)
    origin = await add_message(factory, user, "synthetic origin")
    service = processor(
        factory,
        QueueInterpreter(
            [
                complete(confidence=0.5),
                not_expense().model_copy(update={"missing_fields": ["intent"]}),
            ]
        ),
        MutableClock(),
    )
    await run_message(service, origin)
    reply = await add_message(factory, user, "synthetic ambiguous intent")
    await run_message(service, reply)
    await assert_waiting_for(factory, reply, ExpenseClarificationField.INTENT)


async def test_bound_audio_retry_preserves_download_reference(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    user = await seed_user(factory)
    origin = await add_message(factory, user, "synthetic origin")
    clock = MutableClock()
    service = processor(factory, QueueInterpreter([unclear()]), clock)
    await run_message(service, origin)
    reply = await add_message(factory, user, "", source_type=MessageSourceType.AUDIO)
    async with factory() as session, session.begin():
        state = await session.scalar(select(ConversationState))
        message = await session.get(ProcessedMessage, reply)
        message.transcribed_at = None
        message.media_remote_jid = "synthetic-media-reference"
        message.clarification_origin_message_id = _clarification_origin(
            state, clock(), reply_id=reply
        )
    assert await service.claim(10) == [reply]
    await service._retry_or_fail(reply, "TRANSCRIPTION_TIMEOUT")
    async with factory() as session:
        message = await session.get(ProcessedMessage, reply)
        assert message.status is ProcessedMessageStatus.PENDING
        assert message.media_remote_jid == "synthetic-media-reference"


@pytest.mark.parametrize("failure", ["audio", "exhausted"])
async def test_terminal_recovery_releases_binding_without_losing_draft(
    factory: async_sessionmaker[AsyncSession], failure: str
) -> None:
    user = await seed_user(factory)
    origin = await add_message(factory, user, "synthetic origin")
    clock = MutableClock()
    service = processor(factory, QueueInterpreter([unclear()]), clock)
    await run_message(service, origin)
    reply = await add_message(factory, user, "synthetic reply")
    async with factory() as session, session.begin():
        state = await session.scalar(select(ConversationState))
        message = await session.get(ProcessedMessage, reply)
        message.clarification_origin_message_id = _clarification_origin(
            state, clock(), reply_id=reply
        )
    assert await service.claim(10) == [reply]
    if failure == "audio":
        await service._mark_audio_failed(reply, "TRANSCRIPTION_INVALID_RESPONSE", "synthetic")
    else:
        async with factory() as session, session.begin():
            message = await session.get(ProcessedMessage, reply)
            message.processing_attempts = 3
        clock.advance(61)
        assert await service.recover_stale(clock()) == 1
    async with factory() as session:
        state = await session.scalar(select(ConversationState))
        context = ExpenseClarificationContext.model_validate(state.context)
        assert not context.reply_bindings
        assert state.status is ConversationStatus.WAITING_EXPENSE_CLARIFICATION
        assert await session.scalar(select(func.count()).select_from(Expense)) == 0
        assert (
            await session.scalar(
                select(func.count())
                .select_from(OutboundMessage)
                .where(OutboundMessage.dedup_key.like(f"processed-message:{reply}:%"))
            )
            == 1
        )


@pytest.mark.parametrize(
    ("provider_error", "expected_code"),
    [
        (GeminiSchemaError("sanitized invalid provider output"), "GEMINI_SCHEMA_INVALID"),
        (GeminiModelUnavailableError("sanitized unavailable model"), "GEMINI_MODEL_UNAVAILABLE"),
    ],
)
async def test_provider_failure_preserves_draft_and_reasks_once(
    factory: async_sessionmaker[AsyncSession],
    provider_error: Exception,
    expected_code: str,
) -> None:
    user_id = await seed_user(factory)
    origin_id = await add_message(factory, user_id, "Gastei 80 reais")
    draft = ExpenseInterpretation(
        intent=ExpenseIntent.CREATE_EXPENSE,
        amount=Decimal("80.00"),
        amount_evidence="80 reais",
        description=None,
        merchant=None,
        category=ExpenseCategory.OTHER,
        payment_method=None,
        expense_date=date(2026, 9, 5),
        confidence=0.9,
        missing_fields=["description"],
        reasoning_summary="synthetic incomplete result",
    )
    clock = MutableClock()
    interpreter = QueueInterpreter([draft, provider_error])
    service = processor(factory, interpreter, clock)
    await run_message(service, origin_id)
    async with factory() as session:
        state_before = await session.scalar(select(ConversationState))
        assert state_before is not None
        context_before = state_before.context
        expiry_before = state_before.expires_at
    clock.advance(10)
    response_id = await add_message(
        factory,
        user_id,
        "Foi no mercado",
        clarification_origin_id=origin_id,
    )

    await run_message(service, response_id)
    await asyncio.gather(service.process(response_id), service.process(response_id))

    async with factory() as session:
        response = await session.get(ProcessedMessage, response_id)
        origin = await session.get(ProcessedMessage, origin_id)
        state = await session.scalar(select(ConversationState))
        outbox = list(
            await session.scalars(
                select(OutboundMessage).where(
                    OutboundMessage.dedup_key.like(f"processed-message:{response_id}:%")
                )
            )
        )
        assert response is not None and response.status is ProcessedMessageStatus.FAILED
        assert response.error_code == expected_code
        assert response.next_attempt_at is None and response.locked_at is None
        assert origin is not None
        assert origin.status is ProcessedMessageStatus.NEEDS_CLARIFICATION
        assert state is not None
        assert state.status is ConversationStatus.WAITING_EXPENSE_CLARIFICATION
        assert state.context == context_before
        assert state.expires_at is not None and expiry_before is not None
        assert state.expires_at > expiry_before
        assert len(outbox) == 1
        assert outbox[0].kind is OutboundMessageKind.CLARIFICATION
        assert outbox[0].content == (
            f"{CLARIFICATION_RETRY_PREFIX} "
            f"{CLARIFICATION_QUESTIONS[ExpenseClarificationField.DESCRIPTION]}"
        )
        assert await session.scalar(select(func.count()).select_from(Expense)) == 0
        assert (
            await session.scalar(
                select(func.count())
                .select_from(OutboundMessage)
                .where(OutboundMessage.kind == OutboundMessageKind.EXPENSE_CONFIRMATION)
            )
            == 0
        )


async def test_schema_failure_after_state_expiry_renews_and_reasks_once(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    user_id = await seed_user(factory)
    origin_id = await add_message(factory, user_id, "Gastei 80 reais")
    draft = ExpenseInterpretation(
        intent=ExpenseIntent.CREATE_EXPENSE,
        amount=Decimal("80.00"),
        amount_evidence="80 reais",
        description=None,
        merchant=None,
        category=ExpenseCategory.OTHER,
        payment_method=None,
        expense_date=date(2026, 9, 5),
        confidence=0.9,
        missing_fields=["description"],
        reasoning_summary="synthetic incomplete result",
    )
    clock = MutableClock()
    interpreter = ExpiringSchemaInterpreter(draft, clock)
    service = processor(factory, interpreter, clock)
    await run_message(service, origin_id)
    async with factory() as session:
        state_before = await session.scalar(select(ConversationState))
        assert state_before is not None
        context_before = state_before.context
        expiry_before = state_before.expires_at

    response_id = await add_message(
        factory,
        user_id,
        "Foi no mercado",
        clarification_origin_id=origin_id,
    )
    await run_message(service, response_id)
    await asyncio.gather(service.process(response_id), service.process(response_id))

    async with factory() as session:
        response = await session.get(ProcessedMessage, response_id)
        state = await session.scalar(select(ConversationState))
        outbox = list(
            await session.scalars(
                select(OutboundMessage).where(
                    OutboundMessage.dedup_key.like(f"processed-message:{response_id}:%")
                )
            )
        )
        current_time = clock().replace(tzinfo=None)
        assert expiry_before is not None and expiry_before < current_time
        assert response is not None and response.status is ProcessedMessageStatus.FAILED
        assert response.error_code == "GEMINI_SCHEMA_INVALID"
        assert response.next_attempt_at is None and response.locked_at is None
        assert state is not None
        assert state.status is ConversationStatus.WAITING_EXPENSE_CLARIFICATION
        assert state.context == context_before
        assert state.expires_at is not None and state.expires_at > current_time
        assert len(outbox) == 1
        assert outbox[0].content == (
            f"{CLARIFICATION_RETRY_PREFIX} "
            f"{CLARIFICATION_QUESTIONS[ExpenseClarificationField.DESCRIPTION]}"
        )
        assert "Foi no mercado" not in outbox[0].content
        assert await session.scalar(select(func.count()).select_from(Expense)) == 0


async def test_valid_answer_after_state_expiry_renews_and_reasks_once(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    user_id = await seed_user(factory)
    origin_id = await add_message(factory, user_id, "Gastei 80 reais")
    draft = complete(amount="80.00", description="placeholder").model_copy(
        update={"description": None, "missing_fields": ["description"]}
    )
    answer = complete(description="Mercado").model_copy(
        update={"amount": None, "amount_evidence": None}
    )
    clock = MutableClock()
    service = processor(
        factory,
        ExpiringValidInterpreter(draft, answer, clock),
        clock,
    )
    await run_message(service, origin_id)
    async with factory() as session:
        state_before = await session.scalar(select(ConversationState))
        assert state_before is not None
        context_before = state_before.context
        expiry_before = state_before.expires_at
    response_id = await add_message(
        factory, user_id, "Foi no mercado", clarification_origin_id=origin_id
    )

    await run_message(service, response_id)
    await asyncio.gather(service.process(response_id), service.process(response_id))

    async with factory() as session:
        response = await session.get(ProcessedMessage, response_id)
        state = await session.scalar(select(ConversationState))
        replies = list(
            await session.scalars(
                select(OutboundMessage).where(
                    OutboundMessage.dedup_key.like(f"processed-message:{response_id}:%")
                )
            )
        )
        current_time = clock().replace(tzinfo=None)
        assert expiry_before is not None and expiry_before < current_time
        assert response is not None
        assert response.status is ProcessedMessageStatus.NEEDS_CLARIFICATION
        assert response.error_code is None
        assert state is not None
        assert state.status is ConversationStatus.WAITING_EXPENSE_CLARIFICATION
        assert state.context == context_before
        assert state.expires_at is not None and state.expires_at > current_time
        assert len(replies) == 1
        assert replies[0].content == (
            f"{CLARIFICATION_RETRY_PREFIX} "
            f"{CLARIFICATION_QUESTIONS[ExpenseClarificationField.DESCRIPTION]}"
        )
        assert await session.scalar(select(func.count()).select_from(Expense)) == 0


async def test_same_batch_unlinked_reply_is_bound_and_schema_failure_reasks_once(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    user_id = await seed_user(factory)
    origin_id = await add_message(factory, user_id, "Gastei 80 reais")
    response_id = await add_message(factory, user_id, "Foi no mercado")
    draft = complete(amount="80.00", description="placeholder").model_copy(
        update={"description": None, "missing_fields": ["description"]}
    )
    interpreter = DraftThenSchemaErrorInterpreter(draft)
    service = processor(factory, interpreter, MutableClock())

    claimed = await service.claim(10)
    assert claimed == [origin_id, response_id]
    await service.process(origin_id)
    await service.process(response_id)

    async with factory() as session:
        response = await session.get(ProcessedMessage, response_id)
        state = await session.scalar(select(ConversationState))
        replies = list(
            await session.scalars(
                select(OutboundMessage).where(
                    OutboundMessage.dedup_key.like(f"processed-message:{response_id}:%")
                )
            )
        )
        assert response is not None
        assert response.clarification_origin_message_id == origin_id
        assert response.status is ProcessedMessageStatus.FAILED
        assert response.error_code == "GEMINI_SCHEMA_INVALID"
        assert state is not None
        assert state.status is ConversationStatus.WAITING_EXPENSE_CLARIFICATION
        context = ExpenseClarificationContext.model_validate(state.context)
        assert context.origin_message_id == origin_id
        assert len(replies) == 1
        assert replies[0].kind is OutboundMessageKind.CLARIFICATION
        assert await session.scalar(select(func.count()).select_from(Expense)) == 0


@pytest.mark.parametrize(
    "origin_source_type",
    [MessageSourceType.TEXT, MessageSourceType.AUDIO, MessageSourceType.IMAGE],
)
async def test_clarification_completion_preserves_origin_source_type(
    factory: async_sessionmaker[AsyncSession],
    origin_source_type: MessageSourceType,
) -> None:
    user_id = await seed_user(factory)
    origin_id = await add_message(
        factory,
        user_id,
        "validated origin checkpoint",
        source_type=origin_source_type,
    )
    clock = MutableClock()
    context = ExpenseClarificationContext(
        origin_message_id=origin_id,
        source_type=origin_source_type,
        reference_timestamp=clock(),
        requested_field=ExpenseClarificationField.DESCRIPTION,
        remaining_fields=(ExpenseClarificationField.DESCRIPTION,),
        amount=Decimal("80.00"),
        category=ExpenseCategory.FOOD,
        expense_date=date(2026, 9, 5),
    )
    async with factory() as session, session.begin():
        origin = await session.get(ProcessedMessage, origin_id)
        state = await session.scalar(
            select(ConversationState).where(ConversationState.user_id == user_id)
        )
        assert origin is not None and state is not None
        origin.status = ProcessedMessageStatus.NEEDS_CLARIFICATION
        state.status = ConversationStatus.WAITING_EXPENSE_CLARIFICATION
        state.context = context.payload()
        state.expires_at = clock() + timedelta(minutes=5)

    answer = ExpenseInterpretation(
        intent=ExpenseIntent.CREATE_EXPENSE,
        amount=None,
        amount_evidence=None,
        description="Mercado",
        merchant="Mercado",
        category=ExpenseCategory.FOOD,
        payment_method=None,
        expense_date=None,
        confidence=0.99,
        missing_fields=[],
        reasoning_summary="synthetic clarification result",
    )
    response_id = await add_message(
        factory,
        user_id,
        "Foi no mercado",
        clarification_origin_id=origin_id,
    )
    service = processor(factory, QueueInterpreter([answer]), clock)

    await run_message(service, response_id)

    async with factory() as session:
        expense = await session.scalar(select(Expense))
        response = await session.get(ProcessedMessage, response_id)
        assert expense is not None and expense.source_type == origin_source_type
        assert response is not None and response.source_type == MessageSourceType.TEXT
        assert response.transcribed_at is None and response.image_analyzed_at is None
        assert await session.scalar(select(func.count()).select_from(Expense)) == 1
        assert (
            await session.scalar(
                select(func.count())
                .select_from(OutboundMessage)
                .where(OutboundMessage.kind == OutboundMessageKind.EXPENSE_CONFIRMATION)
            )
            == 1
        )


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
        state = await session.scalar(select(ConversationState))
        response = await session.get(ProcessedMessage, response_id)
        assert state is not None
        assert state.status is ConversationStatus.WAITING_EXPENSE_CLARIFICATION
        context = ExpenseClarificationContext.model_validate(state.context)
        assert context.origin_message_id == origin_id
        assert response is not None
        assert response.status is ProcessedMessageStatus.NEEDS_CLARIFICATION
        assert await session.scalar(select(func.count()).select_from(Expense)) == 0
        clarification_messages = list(
            await session.scalars(
                select(OutboundMessage).where(
                    OutboundMessage.kind == OutboundMessageKind.CLARIFICATION
                )
            )
        )
        assert len(clarification_messages) == 2
        assert (
            clarification_messages[-1].content == CLARIFICATION_QUESTIONS[context.requested_field]
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
