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

from oink_finai.database.base import Base
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
    ExpenseHistoryAction,
    ExpenseIntent,
    OutboundMessageKind,
    ProcessedMessageStatus,
)
from oink_finai.schemas.expense_interpretation import ExpenseInterpretation
from oink_finai.services.expense_commands import (
    ExpenseCommand,
    ExpenseCommandType,
    encode_expense_action,
    expense_command_text,
    parse_expense_action,
    parse_expense_command,
)
from oink_finai.services.expense_interpreter import ExpenseInterpreter
from oink_finai.services.expense_processing import (
    ExpenseProcessingService,
    format_expense_confirmation,
)


@pytest_asyncio.fixture
async def factory(tmp_path: Path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'commands.db').as_posix()}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    yield session_factory
    await engine.dispose()


class RecordingInterpreter(ExpenseInterpreter):
    def __init__(self, result: ExpenseInterpretation | None = None) -> None:
        self.calls = 0
        self.result = result

    async def interpret(
        self, message: str, *, reference_timestamp: datetime
    ) -> ExpenseInterpretation:
        self.calls += 1
        if self.result is None:
            raise AssertionError("command called interpreter")
        return self.result


class Clock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


async def seed_expense(
    factory: async_sessionmaker[AsyncSession], *, phone: str = "5511999999999"
) -> tuple[User, Expense]:
    async with factory() as session, session.begin():
        user = User(phone_number=phone, timezone="America/Sao_Paulo")
        category = await session.scalar(select(Category).where(Category.slug == "alimentacao"))
        if category is None:
            category = Category(name=ExpenseCategory.FOOD.value, slug="alimentacao")
            session.add(category)
        session.add(user)
        await session.flush()
        session.add(ConversationState(user_id=user.id))
        expense = Expense(
            user_id=user.id,
            category_id=category.id,
            amount=Decimal("27.40"),
            description="Compra de remédios",
            expense_date=date(2026, 9, 2),
            merchant="Farmácia Central",
            payment_method="Débito",
        )
        session.add(expense)
        await session.flush()
        return user, expense


async def add_message(
    factory: async_sessionmaker[AsyncSession], user_id: UUID, text: str
) -> ProcessedMessage:
    async with factory() as session, session.begin():
        message = ProcessedMessage(
            provider="evolution",
            instance_id="business",
            external_message_id=str(uuid4()),
            user_id=user_id,
            accepted_text=text,
            message_timestamp=datetime(2026, 9, 2, 12, tzinfo=UTC),
            status=ProcessedMessageStatus.PENDING,
            available_at=datetime(2026, 9, 2, 12, tzinfo=UTC),
        )
        session.add(message)
        await session.flush()
        return message


def processor(
    factory: async_sessionmaker[AsyncSession],
    interpreter: RecordingInterpreter,
    clock: Clock | None = None,
) -> ExpenseProcessingService:
    return ExpenseProcessingService(
        factory,
        lambda _: interpreter,
        clock=clock or Clock(datetime(2026, 9, 2, 12, tzinfo=UTC)),
    )


@pytest.mark.parametrize(
    ("text", "command_type"),
    [
        (f"remover {uuid4()}", ExpenseCommandType.REMOVE),
        (f"REMOVER {uuid4()}", ExpenseCommandType.REMOVE),
        (f"confirmar-remocao {uuid4()}", ExpenseCommandType.CONFIRM_REMOVE),
        (f"editar {uuid4()}", ExpenseCommandType.EDIT),
        ("cancelar", ExpenseCommandType.CANCEL),
        ("cancelar agora", ExpenseCommandType.INVALID),
        ("remover 12345678", ExpenseCommandType.INVALID),
        (f"remover {uuid4()} agora", ExpenseCommandType.INVALID),
        ("remover inválido", ExpenseCommandType.INVALID),
    ],
)
def test_parser_reserved_commands(text: str, command_type: ExpenseCommandType) -> None:
    command = parse_expense_command(text)
    assert command is not None and command.type is command_type


def test_parser_leaves_normal_messages_for_interpreter() -> None:
    assert parse_expense_command("remédios 27,40 no débito") is None


@pytest.mark.parametrize(
    "command_type",
    [
        ExpenseCommandType.EDIT,
        ExpenseCommandType.REMOVE,
        ExpenseCommandType.CONFIRM_REMOVE,
    ],
)
def test_interactive_actions_round_trip_with_full_uuid(command_type: ExpenseCommandType) -> None:
    expense_id = uuid4()
    command = ExpenseCommand(command_type, expense_id)
    action_id = encode_expense_action(command)

    assert str(expense_id) in action_id
    assert parse_expense_action(action_id) == command
    assert parse_expense_command(expense_command_text(command)) == command


def test_cancel_interactive_action_round_trip() -> None:
    command = ExpenseCommand(ExpenseCommandType.CANCEL)
    assert parse_expense_action(encode_expense_action(command)) == command


@pytest.mark.parametrize(
    "action_id",
    ["oink:v1:r:short", "oink:v1:r:not-a-uuid", "oink:v1:unknown", "other:v1:c"],
)
def test_invalid_interactive_action_is_rejected(action_id: str) -> None:
    assert parse_expense_action(action_id) is None


def test_expense_confirmation_exact_text_and_brazilian_formatting() -> None:
    expense = Expense(
        id=UUID("12345678-1234-5678-9abc-123456789abc"),
        user_id=uuid4(),
        category_id=uuid4(),
        amount=Decimal("1234.56"),
        description="Mercado do bairro",
        expense_date=date(2026, 9, 2),
    )

    assert format_expense_confirmation(expense, "Alimentação") == (
        "✅ Novo Gasto Registrado!\n\n"
        "📝 Descrição: Mercado do bairro\n"
        "🛍️ Categoria: Alimentação\n"
        "💵 Valor: R$ 1.234,56\n\n"
        "📅 Data: 02/09/2026\n"
        "⚙️ ID: 12345678"
    )


async def test_confirmation_contains_visual_short_id_without_commands(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    _, expense = await seed_expense(factory)
    text = format_expense_confirmation(expense, ExpenseCategory.FOOD.value)
    assert str(expense.id) not in text
    assert f"⚙️ ID: {str(expense.id)[:8]}" in text
    assert "!oink" not in text
    assert "editar" not in text.lower()
    assert "remover" not in text.lower()
    assert "botões" not in text.lower()


async def test_remove_requests_confirmation_without_deleting_or_calling_interpreter(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    user, expense = await seed_expense(factory)
    message = await add_message(factory, user.id, f"remover {expense.id}")
    interpreter = RecordingInterpreter()
    service = processor(factory, interpreter)
    assert await service.claim(1) == [message.id]
    await service.process(message.id)

    async with factory() as session:
        saved_expense = await session.get(Expense, expense.id)
        state = await session.scalar(
            select(ConversationState).where(ConversationState.user_id == user.id)
        )
        outbox = list(await session.scalars(select(OutboundMessage)))
        assert saved_expense is not None and saved_expense.deleted_at is None
        assert state is not None
        assert state.status is ConversationStatus.WAITING_EXPENSE_DELETE_CONFIRM
        assert state.active_expense_id == expense.id
        assert state.expires_at is not None
        assert len(outbox) == 1
        assert outbox[0].kind is OutboundMessageKind.DELETE_CONFIRMATION_REQUEST
        assert outbox[0].content_type == "BUTTONS"
        assert outbox[0].actions is not None
        assert [action["label"] for action in outbox[0].actions] == [
            "️ Confirmar exclusão",
            "Cancelar",
        ]
        assert str(expense.id) in outbox[0].actions[0]["id"]
        assert outbox[0].fallback_content is not None
        assert f"confirmar-remocao {expense.id}" in outbox[0].fallback_content
        assert interpreter.calls == 0


async def request_then_confirm(
    factory: async_sessionmaker[AsyncSession], user: User, expense: Expense
) -> tuple[ExpenseProcessingService, RecordingInterpreter]:
    interpreter = RecordingInterpreter()
    service = processor(factory, interpreter)
    request = await add_message(factory, user.id, f"remover {expense.id}")
    await service.claim(1)
    await service.process(request.id)
    confirm = await add_message(factory, user.id, f"confirmar-remocao {expense.id}")
    await service.claim(1)
    await service.process(confirm.id)
    return service, interpreter


async def test_confirmation_soft_deletes_and_writes_history_once(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    user, expense = await seed_expense(factory)
    _, interpreter = await request_then_confirm(factory, user, expense)

    async with factory() as session:
        saved = await session.get(Expense, expense.id)
        state = await session.scalar(
            select(ConversationState).where(ConversationState.user_id == user.id)
        )
        histories = list(await session.scalars(select(ExpenseHistory)))
        deleted = list(
            await session.scalars(
                select(OutboundMessage).where(
                    OutboundMessage.kind == OutboundMessageKind.EXPENSE_DELETED
                )
            )
        )
        assert saved is not None and saved.deleted_at is not None
        assert len(histories) == 1
        assert histories[0].action is ExpenseHistoryAction.DELETE
        assert histories[0].changes["old_data"]["deleted_at"] is None
        assert histories[0].changes["new_data"]["deleted_at"] is not None
        assert len(deleted) == 1
        assert state is not None and state.status is ConversationStatus.IDLE
        assert state.active_expense_id is None and state.context is None
        assert state.expires_at is None
        assert interpreter.calls == 0


async def test_cancel_clears_pending_state_without_deleting(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    user, expense = await seed_expense(factory)
    interpreter = RecordingInterpreter()
    service = processor(factory, interpreter)
    request = await add_message(factory, user.id, f"remover {expense.id}")
    await service.claim(1)
    await service.process(request.id)
    cancel = await add_message(factory, user.id, "cancelar")
    await service.claim(1)
    await service.process(cancel.id)

    async with factory() as session:
        saved = await session.get(Expense, expense.id)
        state = await session.scalar(select(ConversationState))
        outbox = await session.scalar(
            select(OutboundMessage).where(
                OutboundMessage.kind == OutboundMessageKind.ACTION_CANCELLED
            )
        )
        assert saved is not None and saved.deleted_at is None
        assert state is not None and state.status is ConversationStatus.IDLE
        assert outbox is not None and outbox.content == "Operação cancelada."
        assert interpreter.calls == 0


async def test_expired_confirmation_does_not_delete(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    user, expense = await seed_expense(factory)
    clock = Clock(datetime(2026, 9, 2, 12, tzinfo=UTC))
    interpreter = RecordingInterpreter()
    service = processor(factory, interpreter, clock)
    request = await add_message(factory, user.id, f"remover {expense.id}")
    await service.claim(1)
    await service.process(request.id)
    clock.value += timedelta(minutes=11)
    confirm = await add_message(factory, user.id, f"confirmar-remocao {expense.id}")
    await service.claim(1)
    await service.process(confirm.id)

    async with factory() as session:
        saved = await session.get(Expense, expense.id)
        state = await session.scalar(select(ConversationState))
        assert saved is not None and saved.deleted_at is None
        assert state is not None and state.status is ConversationStatus.IDLE


async def test_normal_message_cancels_pending_then_creates_new_expense(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    user, expense = await seed_expense(factory)
    result = ExpenseInterpretation(
        intent=ExpenseIntent.CREATE_EXPENSE,
        amount=Decimal("10.00"),
        amount_evidence="10",
        description="Café",
        merchant=None,
        category=ExpenseCategory.FOOD,
        payment_method=None,
        expense_date=date(2026, 9, 2),
        confidence=1,
        missing_fields=[],
        reasoning_summary="valid",
    )
    interpreter = RecordingInterpreter(result)
    service = processor(factory, interpreter)
    request = await add_message(factory, user.id, f"remover {expense.id}")
    await service.claim(1)
    await service.process(request.id)
    normal = await add_message(factory, user.id, "café 10 reais")
    await service.claim(1)
    await service.process(normal.id)

    async with factory() as session:
        state = await session.scalar(select(ConversationState))
        count = await session.scalar(select(func.count()).select_from(Expense))
        original = await session.get(Expense, expense.id)
        assert state is not None and state.status is ConversationStatus.IDLE
        assert count == 2
        assert original is not None and original.deleted_at is None
        assert interpreter.calls == 1


async def test_other_user_and_repeated_confirmation_cannot_delete_twice(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    owner, expense = await seed_expense(factory)
    other, _ = await seed_expense(factory, phone="5511888888888")
    interpreter = RecordingInterpreter()
    service = processor(factory, interpreter)
    unauthorized = await add_message(factory, other.id, f"remover {expense.id}")
    await service.claim(1)
    await service.process(unauthorized.id)
    await request_then_confirm(factory, owner, expense)
    repeated = await add_message(factory, owner.id, f"confirmar-remocao {expense.id}")
    await service.claim(1)
    await asyncio.gather(service.process(repeated.id), service.process(repeated.id))

    async with factory() as session:
        history_count = await session.scalar(select(func.count()).select_from(ExpenseHistory))
        deleted_count = await session.scalar(
            select(func.count())
            .select_from(OutboundMessage)
            .where(OutboundMessage.kind == OutboundMessageKind.EXPENSE_DELETED)
        )
        unauthorized_outbox = await session.scalar(
            select(OutboundMessage).where(OutboundMessage.user_id == other.id)
        )
        assert history_count == 1
        assert deleted_count == 1
        assert unauthorized_outbox is not None
        assert unauthorized_outbox.content == "Gasto não encontrado ou indisponível."


@pytest.mark.parametrize(
    "text",
    ["remover 12345678", "cancelar agora", f"editar {uuid4()}", f"remover {uuid4()} extra"],
)
async def test_reserved_or_edit_commands_never_call_interpreter(
    factory: async_sessionmaker[AsyncSession], text: str
) -> None:
    user, _ = await seed_expense(factory)
    message = await add_message(factory, user.id, text)
    interpreter = RecordingInterpreter()
    service = processor(factory, interpreter)
    await service.claim(1)
    await service.process(message.id)

    async with factory() as session:
        saved = await session.get(ProcessedMessage, message.id)
        outbox = await session.scalar(
            select(OutboundMessage).where(OutboundMessage.user_id == user.id)
        )
        assert saved is not None and saved.status is ProcessedMessageStatus.PROCESSED
        assert outbox is not None and outbox.kind is OutboundMessageKind.ACTION_ERROR
        assert interpreter.calls == 0


async def test_already_deleted_expense_is_unavailable(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    user, expense = await seed_expense(factory)
    async with factory() as session, session.begin():
        saved = await session.get(Expense, expense.id)
        assert saved is not None
        saved.deleted_at = datetime(2026, 9, 2, 11, tzinfo=UTC)
    message = await add_message(factory, user.id, f"remover {expense.id}")
    interpreter = RecordingInterpreter()
    service = processor(factory, interpreter)
    await service.claim(1)
    await service.process(message.id)

    async with factory() as session:
        outbox = await session.scalar(select(OutboundMessage))
        history_count = await session.scalar(select(func.count()).select_from(ExpenseHistory))
        assert outbox is not None
        assert outbox.content == "Gasto não encontrado ou indisponível."
        assert history_count == 0
