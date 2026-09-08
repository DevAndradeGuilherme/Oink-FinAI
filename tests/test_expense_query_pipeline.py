import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime
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
    ExpenseCategory,
    ExpenseIntent,
    MessageSourceType,
    OutboundMessageKind,
    OutboundMessageStatus,
    ProcessedMessageStatus,
)
from oink_finai.domain.expense_query import (
    ExpenseQueryIntent,
    ExpenseQueryMetric,
    QueryUnclearReason,
)
from oink_finai.providers.whatsapp import EvolutionProviderError, WhatsAppProvider
from oink_finai.schemas.expense_interpretation import ExpenseInterpretation
from oink_finai.schemas.expense_query import (
    ExpenseQueryFilters,
    ExpenseQueryPeriod,
    ExpenseQueryPlan,
)
from oink_finai.schemas.expense_query_checkpoint import ExpenseClassificationCheckpoint
from oink_finai.schemas.expense_query_messages import ExpenseQueryFormattedMessages
from oink_finai.schemas.expense_query_result import (
    ExpenseAggregateResult,
    ExpenseQueryMetadata,
)
from oink_finai.services.expense_interpreter import ExpenseInterpreter
from oink_finai.services.expense_processing import ExpenseProcessingService
from oink_finai.services.expense_query_executor import ExpenseQueryExecutor
from oink_finai.services.expense_query_interpreter import ExpenseQueryInterpreter
from oink_finai.services.expense_query_result_formatter import ExpenseQueryResultFormatter
from oink_finai.services.interpretation_errors import (
    InterpretationRateLimitError,
    InterpretationTimeoutError,
    InterpretationUnavailableError,
)
from oink_finai.services.outbox_delivery import OutboxDeliveryService
from oink_finai.services.pipeline_timing import PipelineTiming


@pytest_asyncio.fixture
async def query_factory(tmp_path: Path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'queries.db').as_posix()}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


def query_classification() -> ExpenseInterpretation:
    return ExpenseInterpretation(
        intent=ExpenseIntent.QUERY,
        amount=None,
        amount_evidence=None,
        description=None,
        merchant=None,
        category=None,
        payment_method=None,
        expense_date=None,
        confidence=1,
        missing_fields=[],
        reasoning_summary="query",
    )


def expense_classification() -> ExpenseInterpretation:
    return ExpenseInterpretation(
        intent=ExpenseIntent.CREATE_EXPENSE,
        amount=Decimal("42.50"),
        amount_evidence="42,50",
        description="Mercado",
        merchant=None,
        category=ExpenseCategory.FOOD,
        payment_method=None,
        expense_date=date(2026, 9, 8),
        confidence=1,
        missing_fields=[],
        reasoning_summary="expense",
    )


def empty_filters() -> ExpenseQueryFilters:
    return ExpenseQueryFilters(
        category=None,
        merchant=None,
        payment_method=None,
        source_type=None,
        min_amount=None,
        max_amount=None,
    )


def total_plan(intent: ExpenseQueryIntent = ExpenseQueryIntent.AGGREGATE) -> ExpenseQueryPlan:
    inactive = intent in {ExpenseQueryIntent.NOT_QUERY, ExpenseQueryIntent.QUERY_UNCLEAR}
    return ExpenseQueryPlan(
        intent=intent,
        metric=None if inactive else ExpenseQueryMetric.TOTAL,
        group_by=None,
        period=(
            None
            if inactive
            else ExpenseQueryPeriod(start_date=date(2026, 9, 1), end_date=date(2026, 9, 30))
        ),
        comparison_period=None,
        filters=empty_filters(),
        sort_by=None,
        sort_direction=None,
        limit=0 if inactive else 1,
        offset=0,
        unclear_reason=(
            None
            if intent is not ExpenseQueryIntent.QUERY_UNCLEAR
            else QueryUnclearReason.AMBIGUOUS_PERIOD
        ),
        timezone="America/Sao_Paulo",
        reference_date=date(2026, 9, 8),
    )


def aggregate_result() -> ExpenseAggregateResult:
    plan = total_plan()
    return ExpenseAggregateResult(
        metric=ExpenseQueryMetric.TOTAL,
        value=Decimal("42.50"),
        record_count=1,
        metadata=ExpenseQueryMetadata(
            period=plan.period,
            comparison_period=None,
            filters=plan.filters,
            sort_by=None,
            sort_direction=None,
            limit=1,
            offset=0,
            timezone=plan.timezone,
            reference_date=plan.reference_date,
        ),
    )


class FakeClassifier(ExpenseInterpreter):
    def __init__(self, results: list[ExpenseInterpretation | Exception]) -> None:
        self.results = results
        self.calls = 0
        self.messages: list[str] = []

    async def interpret(self, message: str, *, reference_timestamp: datetime):
        self.messages.append(message)
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        if isinstance(result, Exception):
            raise result
        return result


class FakeQueryInterpreter(ExpenseQueryInterpreter):
    def __init__(self, results: list[ExpenseQueryPlan | Exception]) -> None:
        self.results = results
        self.calls = 0

    async def interpret(self, message: str, *, reference_timestamp: datetime):
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        if isinstance(result, Exception):
            raise result
        return result


class FakeExecutor(ExpenseQueryExecutor):
    def __init__(self, results: list[ExpenseAggregateResult | Exception]) -> None:
        self.results = results
        self.calls = 0
        self.user_ids: list[UUID] = []

    async def execute(self, *, user_id: UUID, plan: ExpenseQueryPlan):
        self.user_ids.append(user_id)
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        if isinstance(result, Exception):
            raise result
        return result


class FakeFormatter(ExpenseQueryResultFormatter):
    def __init__(self, pages: tuple[str, ...] = ("resultado seguro",)) -> None:
        self.pages = pages
        self.calls = 0

    def format(self, result):
        self.calls += 1
        return ExpenseQueryFormattedMessages(
            messages=self.pages,
            truncated=False,
            total_items=len(self.pages),
            displayed_items=len(self.pages),
        )


class CrashOnceFormatter(ExpenseQueryResultFormatter):
    def __init__(self) -> None:
        self.calls = 0

    def format(self, result):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("simulated crash")
        return ExpenseQueryFormattedMessages(
            messages=("resultado seguro",),
            truncated=False,
            total_items=1,
            displayed_items=1,
        )


class SequenceProvider(WhatsAppProvider):
    def __init__(self, results: list[str | Exception]) -> None:
        self.results = results
        self.calls = 0
        self.contents: list[str] = []

    async def parse_webhook(self, payload):
        raise AssertionError

    async def send_text(self, phone_number: str, text: str):
        self.contents.append(text)
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        if isinstance(result, Exception):
            raise result
        return result


async def seed_query(
    factory: async_sessionmaker[AsyncSession],
    *,
    source_type: MessageSourceType = MessageSourceType.TEXT,
    text: str = "Quanto gastei este mês?",
) -> ProcessedMessage:
    async with factory() as session, session.begin():
        user = User(phone_number="5511999999999", timezone="America/Sao_Paulo")
        session.add(user)
        await session.flush()
        message = ProcessedMessage(
            provider="evolution",
            instance_id="query-instance",
            external_message_id="query-message",
            user_id=user.id,
            accepted_text=text,
            source_type=source_type,
            transcribed_at=(
                datetime(2026, 9, 8, tzinfo=UTC) if source_type is MessageSourceType.AUDIO else None
            ),
            message_timestamp=datetime(2026, 9, 8, 12, tzinfo=UTC),
            status=ProcessedMessageStatus.PENDING,
            available_at=datetime(2026, 9, 8, 12, tzinfo=UTC),
        )
        session.add(message)
        await session.flush()
        return message


def query_service(
    factory,
    classifier: FakeClassifier,
    query_interpreter: FakeQueryInterpreter,
    executor: FakeExecutor,
    formatter: ExpenseQueryResultFormatter | None = None,
    *,
    max_attempts: int = 3,
    timing: PipelineTiming | None = None,
) -> ExpenseProcessingService:
    return ExpenseProcessingService(
        factory,
        lambda _timezone: classifier,
        query_interpreter_factory=lambda _timezone: query_interpreter,
        query_executor=executor,
        query_formatter=formatter or FakeFormatter(),
        max_attempts=max_attempts,
        retry_base_seconds=0,
        retry_max_seconds=0,
        jitter=lambda: 0,
        clock=lambda: datetime(2026, 9, 8, 12, tzinfo=UTC),
        timing=timing,
    )


async def test_text_query_uses_checkpoints_and_transactional_outbox(query_factory) -> None:
    message = await seed_query(query_factory)
    classifier = FakeClassifier([query_classification()])
    query_interpreter = FakeQueryInterpreter([total_plan()])
    executor = FakeExecutor([aggregate_result()])
    service = query_service(query_factory, classifier, query_interpreter, executor)

    assert await service.claim(1) == [message.id]
    await service.process(message.id)

    async with query_factory() as session:
        saved = await session.get(ProcessedMessage, message.id)
        outbox = list(await session.scalars(select(OutboundMessage)))
        assert saved is not None and saved.status is ProcessedMessageStatus.PROCESSED
        assert saved.classification_intent is ExpenseIntent.QUERY
        assert saved.classification_checkpoint is not None and saved.classified_at is not None
        assert saved.query_plan_checkpoint is not None and saved.query_plan_created_at is not None
        assert saved.query_executed_at is not None and saved.query_page_count == 1
        assert saved.query_result_checkpoint is None
        assert saved.locked_at is None and saved.next_attempt_at is None
        assert len(outbox) == 1
        assert outbox[0].kind is OutboundMessageKind.QUERY_RESULT
        assert (outbox[0].sequence_no, outbox[0].sequence_count) == (1, 1)
        assert outbox[0].processed_message_id == message.id
        assert await session.scalar(select(func.count()).select_from(Expense)) == 0
    assert classifier.calls == query_interpreter.calls == executor.calls == 1
    assert executor.user_ids == [message.user_id]


async def test_audio_transcript_can_be_query_without_media_reprocessing(query_factory) -> None:
    message = await seed_query(query_factory, source_type=MessageSourceType.AUDIO)
    classifier = FakeClassifier([query_classification()])
    query_interpreter = FakeQueryInterpreter([total_plan()])
    executor = FakeExecutor([aggregate_result()])
    service = query_service(query_factory, classifier, query_interpreter, executor)

    await service.claim(1)
    await service.process(message.id)
    assert classifier.messages == ["Quanto gastei este mês?"]
    assert executor.calls == 1


async def test_expense_keeps_single_interpretation_and_old_path(query_factory) -> None:
    async with query_factory() as session, session.begin():
        session.add(Category(name=ExpenseCategory.FOOD.value, slug="food"))
    message = await seed_query(query_factory, text="Mercado 42,50")
    classifier = FakeClassifier([expense_classification()])
    query_interpreter = FakeQueryInterpreter([AssertionError("query interpreter called")])
    executor = FakeExecutor([AssertionError("query executor called")])
    service = query_service(query_factory, classifier, query_interpreter, executor)

    await service.claim(1)
    await service.process(message.id)
    async with query_factory() as session:
        assert await session.scalar(select(func.count()).select_from(Expense)) == 1
    assert classifier.calls == 1
    assert query_interpreter.calls == executor.calls == 0


async def test_query_unclear_creates_one_independent_guidance(query_factory) -> None:
    message = await seed_query(query_factory)
    classifier = FakeClassifier([query_classification()])
    query_interpreter = FakeQueryInterpreter([total_plan(ExpenseQueryIntent.QUERY_UNCLEAR)])
    executor = FakeExecutor([AssertionError("executor called")])
    service = query_service(query_factory, classifier, query_interpreter, executor)

    await service.claim(1)
    await service.process(message.id)
    async with query_factory() as session:
        outbox = list(await session.scalars(select(OutboundMessage)))
        states = list(await session.scalars(select(ConversationState)))
        assert len(outbox) == 1 and outbox[0].kind is OutboundMessageKind.QUERY_GUIDANCE
        assert states == []
    assert executor.calls == 0


@pytest.mark.parametrize(
    "failure",
    [
        InterpretationTimeoutError(),
        InterpretationRateLimitError(),
        InterpretationUnavailableError(),
    ],
)
async def test_query_interpretation_transient_failures_retry_durably(
    query_factory, failure: Exception
) -> None:
    message = await seed_query(query_factory)
    classifier = FakeClassifier([query_classification()])
    query_interpreter = FakeQueryInterpreter([failure, total_plan()])
    executor = FakeExecutor([aggregate_result()])
    service = query_service(query_factory, classifier, query_interpreter, executor)

    await service.claim(1)
    await service.process(message.id)
    assert await service.claim(1) == [message.id]
    await service.process(message.id)

    async with query_factory() as session:
        saved = await session.get(ProcessedMessage, message.id)
        assert saved is not None and saved.status is ProcessedMessageStatus.PROCESSED
        assert saved.last_error_code in {
            "QUERY_TIMEOUT",
            "QUERY_RATE_LIMIT",
            "QUERY_UNAVAILABLE",
        }
    assert classifier.calls == 1
    assert query_interpreter.calls == 2


async def test_plan_checkpoint_prevents_second_openai_call_after_database_retry(
    query_factory,
) -> None:
    message = await seed_query(query_factory)
    classifier = FakeClassifier([query_classification()])
    query_interpreter = FakeQueryInterpreter([total_plan()])
    executor = FakeExecutor([ConnectionError(), aggregate_result()])
    service = query_service(query_factory, classifier, query_interpreter, executor)

    await service.claim(1)
    await service.process(message.id)
    async with query_factory() as session:
        saved = await session.get(ProcessedMessage, message.id)
        assert saved is not None and saved.query_plan_created_at is not None
        assert saved.status is ProcessedMessageStatus.PENDING
    await service.claim(1)
    await service.process(message.id)

    assert classifier.calls == query_interpreter.calls == 1
    assert executor.calls == 2


async def test_execution_checkpoint_prevents_second_sql_after_crash(query_factory) -> None:
    message = await seed_query(query_factory)
    classifier = FakeClassifier([query_classification()])
    query_interpreter = FakeQueryInterpreter([total_plan()])
    executor = FakeExecutor([aggregate_result()])
    formatter = CrashOnceFormatter()
    service = query_service(
        query_factory,
        classifier,
        query_interpreter,
        executor,
        formatter,
    )

    await service.claim(1)
    with pytest.raises(RuntimeError, match="simulated crash"):
        await service.process(message.id)
    await service.recover_stale(datetime(2026, 9, 9, tzinfo=UTC))
    assert await service.claim(1) == [message.id]
    await service.process(message.id)

    assert classifier.calls == query_interpreter.calls == executor.calls == 1
    assert formatter.calls == 2
    async with query_factory() as session:
        saved = await session.get(ProcessedMessage, message.id)
        assert saved is not None and saved.status is ProcessedMessageStatus.PROCESSED
        assert saved.query_executed_at is not None
        assert saved.query_result_checkpoint is None
        assert await session.scalar(select(func.count()).select_from(OutboundMessage)) == 1


async def test_invalid_checkpoint_never_reaches_executor(query_factory) -> None:
    message = await seed_query(query_factory)
    classifier = FakeClassifier([query_classification()])
    query_interpreter = FakeQueryInterpreter([total_plan()])
    executor = FakeExecutor([aggregate_result()])
    service = query_service(query_factory, classifier, query_interpreter, executor)
    await service.claim(1)
    async with query_factory() as session, session.begin():
        saved = await session.get(ProcessedMessage, message.id)
        assert saved is not None
        saved.classification_intent = ExpenseIntent.QUERY
        saved.classification_checkpoint = {"version": 99}
        saved.classified_at = datetime(2026, 9, 8, tzinfo=UTC)

    await service.process(message.id)
    assert executor.calls == 0
    async with query_factory() as session:
        outbox = await session.scalar(select(OutboundMessage))
        assert outbox is not None and outbox.kind is OutboundMessageKind.QUERY_GUIDANCE


async def test_invalid_plan_checkpoint_never_executes_sql(query_factory) -> None:
    message = await seed_query(query_factory)
    executor = FakeExecutor([aggregate_result()])
    service = query_service(
        query_factory,
        FakeClassifier([AssertionError("classifier called")]),
        FakeQueryInterpreter([AssertionError("query interpreter called")]),
        executor,
    )
    await service.claim(1)
    classification = ExpenseClassificationCheckpoint.from_interpretation(query_classification())
    async with query_factory() as session, session.begin():
        saved = await session.get(ProcessedMessage, message.id)
        assert saved is not None
        saved.classification_intent = ExpenseIntent.QUERY
        saved.classification_checkpoint = classification.payload()
        saved.classified_at = datetime(2026, 9, 8, tzinfo=UTC)
        saved.query_plan_checkpoint = {"version": 1, "plan": {"user_id": str(message.user_id)}}
        saved.query_plan_created_at = datetime(2026, 9, 8, tzinfo=UTC)

    await service.process(message.id)
    assert executor.calls == 0
    async with query_factory() as session:
        outbox = await session.scalar(select(OutboundMessage))
        assert outbox is not None and outbox.kind is OutboundMessageKind.QUERY_GUIDANCE


async def test_reprocessing_terminal_query_never_duplicates_pages(query_factory) -> None:
    message = await seed_query(query_factory)
    service = query_service(
        query_factory,
        FakeClassifier([query_classification()]),
        FakeQueryInterpreter([total_plan()]),
        FakeExecutor([aggregate_result()]),
        FakeFormatter(("página 1", "página 2", "página 3")),
    )
    await service.claim(1)
    await service.process(message.id)
    await asyncio.gather(service.process(message.id), service.process(message.id))

    async with query_factory() as session:
        rows = list(
            await session.scalars(select(OutboundMessage).order_by(OutboundMessage.sequence_no))
        )
        assert [row.content for row in rows] == ["página 1", "página 2", "página 3"]
        assert [row.sequence_no for row in rows] == [1, 2, 3]


async def test_outbox_pages_are_claimed_and_retried_in_order(query_factory) -> None:
    message = await seed_query(query_factory)
    service = query_service(
        query_factory,
        FakeClassifier([query_classification()]),
        FakeQueryInterpreter([total_plan()]),
        FakeExecutor([aggregate_result()]),
        FakeFormatter(("página 1", "página 2", "página 3")),
    )
    await service.claim(1)
    await service.process(message.id)
    provider = SequenceProvider(
        [
            "provider-1",
            EvolutionProviderError("safe", outcome_unknown=False),
            "provider-2",
            "provider-3",
        ]
    )
    delivery = OutboxDeliveryService(query_factory, provider, max_attempts=3, retry_base_seconds=0)

    first = await delivery.claim(10)
    assert len(first) == 1
    await delivery.send(first[0])
    second = await delivery.claim(10)
    assert len(second) == 1
    await delivery.send(second[0])
    retry = await delivery.claim(10)
    assert len(retry) == 1 and retry[0].message_id == second[0].message_id
    await delivery.send(retry[0])
    third = await delivery.claim(10)
    assert len(third) == 1
    await delivery.send(third[0])
    assert provider.contents == ["página 1", "página 2", "página 2", "página 3"]


async def test_unknown_page_blocks_every_later_page(query_factory) -> None:
    message = await seed_query(query_factory)
    service = query_service(
        query_factory,
        FakeClassifier([query_classification()]),
        FakeQueryInterpreter([total_plan()]),
        FakeExecutor([aggregate_result()]),
        FakeFormatter(("página 1", "página 2", "página 3")),
    )
    await service.claim(1)
    await service.process(message.id)
    provider = SequenceProvider(
        ["provider-1", EvolutionProviderError("safe", outcome_unknown=True)]
    )
    delivery = OutboxDeliveryService(query_factory, provider)
    await delivery.send((await delivery.claim(10))[0])
    await delivery.send((await delivery.claim(10))[0])
    assert await delivery.claim(10) == []
    async with query_factory() as session:
        statuses = list(
            await session.scalars(
                select(OutboundMessage.status).order_by(OutboundMessage.sequence_no)
            )
        )
        assert statuses == [
            OutboundMessageStatus.SENT,
            OutboundMessageStatus.UNKNOWN,
            OutboundMessageStatus.PENDING,
        ]


async def test_terminal_page_failure_blocks_every_later_page(query_factory) -> None:
    message = await seed_query(query_factory)
    service = query_service(
        query_factory,
        FakeClassifier([query_classification()]),
        FakeQueryInterpreter([total_plan()]),
        FakeExecutor([aggregate_result()]),
        FakeFormatter(("pÃ¡gina 1", "pÃ¡gina 2", "pÃ¡gina 3")),
    )
    await service.claim(1)
    await service.process(message.id)
    provider = SequenceProvider(
        ["provider-1", EvolutionProviderError("safe", outcome_unknown=False)]
    )
    delivery = OutboxDeliveryService(query_factory, provider, max_attempts=1)
    await delivery.send((await delivery.claim(10))[0])
    await delivery.send((await delivery.claim(10))[0])

    assert await delivery.claim(10) == []
    async with query_factory() as session:
        statuses = list(
            await session.scalars(
                select(OutboundMessage.status).order_by(OutboundMessage.sequence_no)
            )
        )
        assert statuses == [
            OutboundMessageStatus.SENT,
            OutboundMessageStatus.FAILED,
            OutboundMessageStatus.PENDING,
        ]


async def test_query_timing_contains_only_sanitized_metadata(
    query_factory, caplog: pytest.LogCaptureFixture
) -> None:
    secret = "Mercado Secreto R$ 987,65 em 01/09/2026"
    message = await seed_query(query_factory, text=secret)
    service = query_service(
        query_factory,
        FakeClassifier([query_classification()]),
        FakeQueryInterpreter([total_plan()]),
        FakeExecutor([aggregate_result()]),
        timing=PipelineTiming(True),
    )
    with caplog.at_level(logging.INFO):
        await service.claim(1)
        await service.process(message.id)

    query_records = [
        record
        for record in caplog.records
        if record.name == "oink_finai.pipeline_timing" and record.event.startswith("query_")
    ]
    assert {
        "query_interpretation_started",
        "query_interpretation_completed",
        "query_plan_checkpoint_started",
        "query_plan_checkpoint_completed",
        "query_execution_started",
        "query_execution_completed",
        "query_formatting_started",
        "query_formatting_completed",
        "query_outbox_created",
        "query_processing_completed",
    } <= {record.event for record in query_records}
    assert secret not in " ".join(str(record.__dict__) for record in query_records)
    assert all(not hasattr(record, "merchant") for record in query_records)


async def test_second_worker_cannot_claim_already_claimed_query(query_factory) -> None:
    message = await seed_query(query_factory)
    first = query_service(
        query_factory,
        FakeClassifier([query_classification()]),
        FakeQueryInterpreter([total_plan()]),
        FakeExecutor([aggregate_result()]),
    )
    second = query_service(
        query_factory,
        FakeClassifier([query_classification()]),
        FakeQueryInterpreter([total_plan()]),
        FakeExecutor([aggregate_result()]),
    )
    claims = [await first.claim(1), await second.claim(1)]
    assert sum(message.id in claimed for claimed in claims) == 1
