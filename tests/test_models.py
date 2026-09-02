from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from oink_finai.database.models import Category, ConversationState, Expense, ProcessedMessage, User
from oink_finai.domain.enums import ConversationStatus


async def persist_expense_dependencies(session: AsyncSession) -> tuple[User, Category]:
    user = User(phone_number="5511999990000")
    category = Category(name="Alimentação", slug="alimentacao")
    session.add_all([user, category])
    await session.flush()
    return user, category


@pytest.mark.asyncio
async def test_expense_requires_decimal_and_rejects_excess_scale(session: AsyncSession) -> None:
    user, category = await persist_expense_dependencies(session)
    with pytest.raises(ValueError, match="decimal places"):
        Expense(
            user_id=user.id,
            category_id=category.id,
            amount=Decimal("12.345"),
            description="Almoço",
            expense_date=date(2026, 9, 1),
        )
    expense = Expense(
        user_id=user.id,
        category_id=category.id,
        amount=Decimal("12.34"),
        description="Almoço",
        expense_date=date(2026, 9, 1),
    )

    with pytest.raises(TypeError, match="Decimal"):
        expense.amount = 12.34  # type: ignore[assignment]

    with pytest.raises(ValueError, match="greater than zero"):
        expense.amount = Decimal("0")


@pytest.mark.asyncio
async def test_expense_supports_soft_delete(session: AsyncSession) -> None:
    user, category = await persist_expense_dependencies(session)
    expense = Expense(
        user_id=user.id,
        category_id=category.id,
        amount=Decimal("25.90"),
        description="Transporte",
        expense_date=date(2026, 9, 1),
    )
    session.add(expense)
    await session.flush()

    expense.deleted_at = datetime.now(UTC)
    await session.commit()

    assert expense.deleted_at is not None


@pytest.mark.asyncio
async def test_processed_message_external_identity_is_unique(session: AsyncSession) -> None:
    identity = {
        "provider": "evolution",
        "instance_id": "primary",
        "external_message_id": "message-123",
    }
    session.add(ProcessedMessage(**identity))
    await session.commit()
    session.add(ProcessedMessage(**identity))

    with pytest.raises(IntegrityError):
        await session.commit()


@pytest.mark.asyncio
async def test_conversation_state_defaults_to_idle(session: AsyncSession) -> None:
    user = User(phone_number="5511999990001")
    session.add(user)
    await session.flush()
    state = ConversationState(user_id=user.id)
    session.add(state)
    await session.flush()

    assert state.status is ConversationStatus.IDLE
    assert state.active_expense_id is None
    assert state.context is None
    assert state.expires_at is None
