from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from oink_finai.config.settings import Settings
from oink_finai.database.base import Base
from oink_finai.database.models import ProcessedMessage, UsageLedger, User
from oink_finai.domain.enums import (
    ExpenseCategory,
    ExpenseIntent,
    ProcessedMessageStatus,
    UsageOperation,
    UsageReservationState,
)
from oink_finai.schemas.expense_interpretation import ExpenseInterpretation
from oink_finai.services.expense_interpreter import ExpenseInterpreter
from oink_finai.services.expense_processing import ExpenseProcessingService
from oink_finai.services.interpretation_errors import InterpretationTimeoutError
from oink_finai.services.openai_usage import OpenAIUsageMetrics, mark_openai_request_transmitted
from oink_finai.services.usage_control import AdmissionDenial, UsageControl


class MutableClock:
    def __init__(self, current: datetime) -> None:
        self.current = current

    def __call__(self) -> datetime:
        return self.current


class MeteredInterpreter(ExpenseInterpreter):
    def __init__(self, outcomes: list[ExpenseInterpretation | Exception]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    async def interpret(self, message: str, *, reference_timestamp: datetime):
        await mark_openai_request_transmitted()
        outcome = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


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
        reasoning_summary="synthetic",
    )


@pytest_asyncio.fixture
async def factory(tmp_path: Path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'usage.db').as_posix()}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    yield session_factory
    await engine.dispose()


def settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "inbound_user_per_minute_limit": 2,
        "inbound_user_per_day_limit": 3,
        "inbound_global_per_day_limit": 5,
        "openai_user_per_day_limit": 3,
        "openai_global_per_day_limit": 5,
        "openai_global_concurrency_limit": 2,
        "openai_text_user_per_day_limit": 3,
        "openai_text_global_per_day_limit": 5,
        "openai_query_user_per_day_limit": 3,
        "openai_query_global_per_day_limit": 5,
        "openai_image_user_per_day_limit": 3,
        "openai_image_global_per_day_limit": 5,
        "openai_audio_user_per_day_limit": 3,
        "openai_audio_global_per_day_limit": 5,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


async def seed_message(
    factory: async_sessionmaker[AsyncSession],
    *,
    user: User | None = None,
    attempt: int = 1,
    status: ProcessedMessageStatus = ProcessedMessageStatus.PROCESSING,
) -> tuple[User, ProcessedMessage]:
    async with factory() as session, session.begin():
        if user is None:
            user = User(phone_number=f"synthetic-{uuid4().hex[:20]}", timezone="UTC")
            session.add(user)
            await session.flush()
        else:
            user = await session.merge(user)
        message = ProcessedMessage(
            provider="synthetic",
            instance_id="usage-tests",
            external_message_id=uuid4().hex,
            user_id=user.id,
            accepted_text="synthetic",
            message_timestamp=datetime.now(UTC),
            status=status,
            available_at=datetime.now(UTC),
            processing_attempts=attempt,
        )
        session.add(message)
        await session.flush()
        return user, message


async def admit_inbound(
    factory: async_sessionmaker[AsyncSession],
    control: UsageControl,
    user: User | None = None,
):
    user, message = await seed_message(factory, user=user, attempt=0)
    async with factory() as session, session.begin():
        return (
            user,
            message,
            await control.admit_inbound(session, message_id=message.id, user_id=user.id),
        )


async def test_inbound_exact_limit_users_global_and_utc_rollover(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    clock = MutableClock(datetime(2026, 9, 9, 23, 59, 30, tzinfo=UTC))
    control = UsageControl(settings(), clock=clock)
    user_a, _, first = await admit_inbound(factory, control)
    _, _, exact = await admit_inbound(factory, control, user_a)
    _, _, blocked = await admit_inbound(factory, control, user_a)
    assert first.allowed and exact.allowed
    assert blocked.denial is AdmissionDenial.USER_MINUTE
    assert blocked.retry_at == datetime(2026, 9, 10, 0, 0, tzinfo=UTC)

    user_b, _, independent = await admit_inbound(factory, control)
    assert independent.allowed and user_b.id != user_a.id
    clock.current = datetime(2026, 9, 10, 0, 0, tzinfo=UTC)
    _, _, after_rollover = await admit_inbound(factory, control, user_a)
    assert after_rollover.allowed


async def test_inbound_global_limit_blocks_next_operation(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    control = UsageControl(
        settings(
            inbound_user_per_minute_limit=5,
            inbound_user_per_day_limit=2,
            inbound_global_per_day_limit=2,
        )
    )
    assert (await admit_inbound(factory, control))[2].allowed
    assert (await admit_inbound(factory, control))[2].allowed
    blocked = (await admit_inbound(factory, control))[2]
    assert blocked.denial is AdmissionDenial.GLOBAL_DAY


@pytest.mark.parametrize(
    "operation",
    [
        UsageOperation.TEXT_INTERPRETATION,
        UsageOperation.QUERY_INTERPRETATION,
        UsageOperation.IMAGE_ANALYSIS,
        UsageOperation.AUDIO_TRANSCRIPTION,
    ],
)
async def test_openai_operation_reservation_completion_and_metrics(
    factory: async_sessionmaker[AsyncSession], operation: UsageOperation
) -> None:
    _, message = await seed_message(factory)
    control = UsageControl(settings(), factory)
    reserved = await control.reserve_openai(
        message_id=message.id, operation=operation, model="synthetic-model"
    )
    assert reserved.allowed and reserved.ledger_id is not None
    await control.mark_transmitted(reserved.ledger_id)
    await control.complete(reserved.ledger_id, OpenAIUsageMetrics(11, 7))

    async with factory() as session:
        ledger = await session.get(UsageLedger, reserved.ledger_id)
        assert ledger is not None
        assert ledger.state is UsageReservationState.COMPLETED
        assert (ledger.input_tokens, ledger.output_tokens) == (11, 7)
        assert ledger.model == "synthetic-model"


async def test_openai_idempotency_concurrency_release_and_ambiguous_failure(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    user, first_message = await seed_message(factory)
    _, second_message = await seed_message(factory, user=user)
    _, third_message = await seed_message(factory)
    control = UsageControl(settings(openai_global_concurrency_limit=1), factory)

    first = await control.reserve_openai(
        message_id=first_message.id,
        operation=UsageOperation.TEXT_INTERPRETATION,
        model="synthetic-model",
    )
    duplicate = await control.reserve_openai(
        message_id=first_message.id,
        operation=UsageOperation.TEXT_INTERPRETATION,
        model="synthetic-model",
    )
    concurrent = await control.reserve_openai(
        message_id=second_message.id,
        operation=UsageOperation.TEXT_INTERPRETATION,
        model="synthetic-model",
    )
    assert first.allowed
    assert duplicate.denial is AdmissionDenial.IDEMPOTENCY_CONFLICT
    assert concurrent.denial is AdmissionDenial.GLOBAL_CONCURRENCY

    assert first.ledger_id is not None
    assert await control.settle_failure(first.ledger_id) is UsageReservationState.RELEASED
    second = await control.reserve_openai(
        message_id=second_message.id,
        operation=UsageOperation.TEXT_INTERPRETATION,
        model="synthetic-model",
    )
    assert second.allowed and second.ledger_id is not None
    await control.mark_transmitted(second.ledger_id)
    assert (
        await control.settle_failure(second.ledger_id, OpenAIUsageMetrics(13, 5))
        is UsageReservationState.AMBIGUOUS
    )

    third = await control.reserve_openai(
        message_id=third_message.id,
        operation=UsageOperation.TEXT_INTERPRETATION,
        model="synthetic-model",
    )
    assert third.allowed
    async with factory() as session:
        ambiguous = await session.get(UsageLedger, second.ledger_id)
        assert ambiguous is not None
        assert (ambiguous.input_tokens, ambiguous.output_tokens) == (13, 5)


async def test_ledger_repr_and_columns_contain_no_content_fields(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    _, message = await seed_message(factory)
    control = UsageControl(settings(), factory)
    reservation = await control.reserve_openai(
        message_id=message.id,
        operation=UsageOperation.TEXT_INTERPRETATION,
        model="safe-model",
    )
    assert reservation.ledger_id is not None
    async with factory() as session:
        ledger = await session.get(UsageLedger, reservation.ledger_id)
        assert ledger is not None
        forbidden = {"phone", "jid", "transcript", "prompt", "response", "content"}
        assert not any(
            token in column.name for column in UsageLedger.__table__.columns for token in forbidden
        )
        representation = repr(ledger)
        assert "synthetic-" not in representation
        assert "accepted_text" not in representation
        assert await session.scalar(select(func.count()).select_from(UsageLedger)) == 1


async def test_durable_retry_gets_new_reservation_and_checkpoint_avoids_another(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    _, message = await seed_message(factory, attempt=0, status=ProcessedMessageStatus.PENDING)
    interpreter = MeteredInterpreter([InterpretationTimeoutError(), not_expense()])
    control = UsageControl(settings(), factory)
    processor = ExpenseProcessingService(
        factory,
        lambda _timezone: interpreter,
        max_attempts=3,
        retry_base_seconds=0.001,
        retry_max_seconds=0.001,
        jitter=lambda: 0,
        usage_control=control,
    )

    assert await processor.claim(1) == [message.id]
    await processor.process(message.id)
    async with factory() as session, session.begin():
        stored = await session.get(ProcessedMessage, message.id)
        assert stored is not None
        stored.next_attempt_at = datetime.now(UTC) - timedelta(seconds=1)
    assert await processor.claim(1) == [message.id]
    await processor.process(message.id)

    async with factory() as session:
        ledgers = list(
            await session.scalars(
                select(UsageLedger)
                .where(UsageLedger.processed_message_id == message.id)
                .order_by(UsageLedger.durable_attempt)
            )
        )
        assert [item.state for item in ledgers] == [
            UsageReservationState.AMBIGUOUS,
            UsageReservationState.COMPLETED,
        ]
        stored = await session.get(ProcessedMessage, message.id)
        assert stored is not None
        stored.status = ProcessedMessageStatus.PROCESSING
        stored.processing_attempts += 1
        await session.commit()

    await processor.process(message.id)
    async with factory() as session:
        assert (
            await session.scalar(
                select(func.count())
                .select_from(UsageLedger)
                .where(UsageLedger.processed_message_id == message.id)
            )
            == 2
        )
    assert interpreter.calls == 2
