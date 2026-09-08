import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest
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
    ExpenseIntent,
    MessageSourceType,
    OutboundMessageKind,
    PaymentMethod,
    ProcessedMessageStatus,
)
from oink_finai.schemas.expense_interpretation import ExpenseInterpretation
from oink_finai.services.expense_interpreter import ExpenseInterpreter
from oink_finai.services.expense_processing import (
    INCOMPLETE_EXPENSE_TEMPLATE,
    ExpenseProcessingService,
)

NOW = datetime(2026, 9, 8, 15, tzinfo=UTC)


@pytest_asyncio.fixture
async def factory(tmp_path: Path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'incomplete.db').as_posix()}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    yield session_factory
    await engine.dispose()


class FakeInterpreter(ExpenseInterpreter):
    def __init__(self, results: list[ExpenseInterpretation]) -> None:
        self.results = results
        self.calls = 0
        self.messages: list[str] = []

    async def interpret(
        self, message: str, *, reference_timestamp: datetime
    ) -> ExpenseInterpretation:
        self.messages.append(message)
        result = self.results[self.calls]
        self.calls += 1
        return result


def interpretation(**changes: object) -> ExpenseInterpretation:
    values: dict[str, object] = {
        "intent": ExpenseIntent.CREATE_EXPENSE,
        "amount": Decimal("32.90"),
        "amount_evidence": "R$ 32,90",
        "description": "Gasolina",
        "merchant": None,
        "category": ExpenseCategory.TRANSPORT,
        "payment_method": PaymentMethod.PIX,
        "expense_date": None,
        "confidence": 0.95,
        "missing_fields": [],
        "reasoning_summary": "valid",
    }
    values.update(changes)
    return ExpenseInterpretation(**values)


async def seed_user(factory: async_sessionmaker[AsyncSession]) -> User:
    async with factory() as session, session.begin():
        user = User(phone_number="5511999999999", timezone="America/Sao_Paulo")
        session.add(user)
        session.add_all(
            [
                Category(name=ExpenseCategory.TRANSPORT.value, slug="transporte"),
                Category(name=ExpenseCategory.OTHER.value, slug="outros"),
            ]
        )
        await session.flush()
        return user


async def add_message(
    factory: async_sessionmaker[AsyncSession],
    user_id: UUID,
    text: str,
    *,
    suffix: str,
    clarification_origin_message_id: UUID | None = None,
) -> ProcessedMessage:
    async with factory() as session, session.begin():
        message = ProcessedMessage(
            provider="synthetic",
            instance_id="synthetic",
            external_message_id=f"message-{suffix}",
            user_id=user_id,
            clarification_origin_message_id=clarification_origin_message_id,
            accepted_text=text,
            source_type=MessageSourceType.TEXT,
            message_timestamp=NOW,
            status=ProcessedMessageStatus.PENDING,
            available_at=NOW,
        )
        session.add(message)
        await session.flush()
        return message


def service(
    factory: async_sessionmaker[AsyncSession], interpreter: FakeInterpreter
) -> ExpenseProcessingService:
    return ExpenseProcessingService(factory, lambda _timezone: interpreter, clock=lambda: NOW)


async def process_one(processor: ExpenseProcessingService, message: ProcessedMessage) -> None:
    assert await processor.claim(1) == [message.id]
    await processor.process(message.id)


async def test_real_message_creates_transport_expense_without_clarification(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    user = await seed_user(factory)
    message = await add_message(
        factory,
        user.id,
        "Gastei R$ 32,90 de gasolina hoje no Pix",
        suffix="real",
    )
    interpreter = FakeInterpreter(
        [interpretation(description=None, missing_fields=["description", "merchant"])]
    )
    await process_one(service(factory, interpreter), message)

    async with factory() as session:
        expense = await session.scalar(select(Expense))
        outbound = await session.scalar(select(OutboundMessage))
        saved = await session.get(ProcessedMessage, message.id)
        assert expense is not None
        assert expense.amount == Decimal("32.90")
        assert expense.description.casefold() == "gasolina"
        category = await session.get(Category, expense.category_id)
        assert category is not None and category.name == ExpenseCategory.TRANSPORT.value
        assert expense.payment_method == PaymentMethod.PIX.value
        assert expense.expense_date.isoformat() == "2026-09-08"
        assert expense.processed_message_id == message.id
        assert saved is not None and saved.status is ProcessedMessageStatus.PROCESSED
        assert outbound is not None and outbound.kind is OutboundMessageKind.EXPENSE_CONFIRMATION
        assert await session.scalar(select(func.count()).select_from(ConversationState)) == 0
        assert interpreter.calls == 1


@pytest.mark.parametrize(
    ("changes", "missing", "rendered"),
    [
        (
            {"amount": None, "amount_evidence": None, "missing_fields": ["amount"]},
            ["amount"],
            "valor",
        ),
        (
            {"description": None, "missing_fields": ["description"]},
            ["description"],
            "descrição",
        ),
        (
            {
                "amount": None,
                "amount_evidence": None,
                "description": None,
                "missing_fields": ["description", "amount"],
            },
            ["amount", "description"],
            "valor e descrição",
        ),
    ],
)
async def test_missing_required_fields_get_one_complete_resubmission_response(
    factory: async_sessionmaker[AsyncSession],
    changes: dict[str, object],
    missing: list[str],
    rendered: str,
) -> None:
    user = await seed_user(factory)
    message = await add_message(factory, user.id, "incomplete", suffix="missing")
    result = interpretation(**changes)
    interpreter = FakeInterpreter([result])
    processor = service(factory, interpreter)
    assert processor._missing_required_fields(result) == missing
    await process_one(processor, message)

    async with factory() as session:
        saved = await session.get(ProcessedMessage, message.id)
        outbound = await session.scalar(select(OutboundMessage))
        assert saved is not None and saved.status is ProcessedMessageStatus.PROCESSED
        assert await session.scalar(select(func.count()).select_from(Expense)) == 0
        assert await session.scalar(select(func.count()).select_from(ConversationState)) == 0
        assert outbound is not None and outbound.kind is OutboundMessageKind.INCOMPLETE_EXPENSE
        assert outbound.content == INCOMPLETE_EXPENSE_TEMPLATE.format(missing_fields=rendered)
        assert outbound.content.count("Exemplo: Gastei R$ 32,90 com gasolina.") == 1
        assert message.clarification_origin_message_id is None
        assert interpreter.calls == 1


@pytest.mark.parametrize("optional_field", ["merchant", "payment_method"])
async def test_missing_optional_field_never_blocks_creation(
    factory: async_sessionmaker[AsyncSession], optional_field: str
) -> None:
    user = await seed_user(factory)
    message = await add_message(factory, user.id, "complete", suffix=optional_field)
    changes = {optional_field: None, "missing_fields": [optional_field]}
    await process_one(service(factory, FakeInterpreter([interpretation(**changes)])), message)

    async with factory() as session:
        expense = await session.scalar(select(Expense))
        outbound = await session.scalar(select(OutboundMessage))
        assert expense is not None
        assert outbound is not None and outbound.kind is OutboundMessageKind.EXPENSE_CONFIRMATION
        assert await session.scalar(select(func.count()).select_from(ConversationState)) == 0


async def test_missing_date_uses_user_local_today(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    user = await seed_user(factory)
    message = await add_message(factory, user.id, "complete", suffix="date")
    await process_one(
        service(factory, FakeInterpreter([interpretation(expense_date=None)])), message
    )

    async with factory() as session:
        expense = await session.scalar(select(Expense))
        assert expense is not None and expense.expense_date.isoformat() == "2026-09-08"


async def test_missing_category_uses_other(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    user = await seed_user(factory)
    message = await add_message(factory, user.id, "complete", suffix="category")
    await process_one(
        service(
            factory,
            FakeInterpreter([interpretation(category=None, missing_fields=["category"])]),
        ),
        message,
    )

    async with factory() as session:
        expense = await session.scalar(select(Expense))
        assert expense is not None
        category = await session.get(Category, expense.category_id)
        assert category is not None and category.name == ExpenseCategory.OTHER.value


async def test_next_message_is_independent_and_does_not_inherit_previous_data(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    user = await seed_user(factory)
    first = await add_message(factory, user.id, "R$ 32,90", suffix="first")
    second = await add_message(factory, user.id, "gasolina", suffix="second")
    interpreter = FakeInterpreter(
        [
            interpretation(description=None, missing_fields=["description"]),
            interpretation(
                intent=ExpenseIntent.NOT_EXPENSE,
                amount=None,
                amount_evidence=None,
                description=None,
                category=ExpenseCategory.OTHER,
                payment_method=None,
                missing_fields=[],
            ),
        ]
    )
    processor = service(factory, interpreter)

    assert await processor.claim(2) == [first.id, second.id]
    await processor.process(first.id)
    await processor.process(second.id)

    async with factory() as session:
        saved_first = await session.get(ProcessedMessage, first.id)
        saved_second = await session.get(ProcessedMessage, second.id)
        assert saved_first is not None and saved_first.status is ProcessedMessageStatus.PROCESSED
        assert (
            saved_second is not None and saved_second.status is ProcessedMessageStatus.NOT_EXPENSE
        )
        assert saved_second.clarification_origin_message_id is None
        assert await session.scalar(select(func.count()).select_from(Expense)) == 0
        assert await session.scalar(select(func.count()).select_from(ConversationState)) == 0
        assert interpreter.messages == ["R$ 32,90", "gasolina"]


async def test_duplicate_concurrent_processing_creates_one_expense_and_confirmation(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    user = await seed_user(factory)
    message = await add_message(factory, user.id, "complete", suffix="concurrent")
    interpreter = FakeInterpreter([interpretation(), interpretation()])
    processor = service(factory, interpreter)
    assert await processor.claim(1) == [message.id]

    await asyncio.gather(processor.process(message.id), processor.process(message.id))

    async with factory() as session:
        assert await session.scalar(select(func.count()).select_from(Expense)) == 1
        assert await session.scalar(select(func.count()).select_from(OutboundMessage)) == 1
        assert await session.scalar(select(func.count()).select_from(ConversationState)) == 0


async def test_historical_waiting_state_does_not_contaminate_new_message(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    user = await seed_user(factory)
    origin = await add_message(factory, user.id, "historical", suffix="origin")
    async with factory() as session, session.begin():
        origin_saved = await session.get(ProcessedMessage, origin.id)
        assert origin_saved is not None
        origin_saved.status = ProcessedMessageStatus.NEEDS_CLARIFICATION
        session.add(
            ConversationState(
                user_id=user.id,
                status=ConversationStatus.WAITING_EXPENSE_CLARIFICATION,
                context={"historical": True},
                expires_at=NOW,
            )
        )
    message = await add_message(
        factory,
        user.id,
        "Gastei R$ 32,90 de gasolina hoje no Pix",
        suffix="new",
        clarification_origin_message_id=origin.id,
    )
    await process_one(service(factory, FakeInterpreter([interpretation()])), message)

    async with factory() as session:
        expense = await session.scalar(select(Expense))
        state = await session.scalar(select(ConversationState))
        assert expense is not None and expense.processed_message_id == message.id
        assert state is not None and state.status is ConversationStatus.IDLE
        assert state.context is None and state.expires_at is None
