import asyncio
import os
from datetime import UTC, date, datetime
from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from oink_finai.database.models import Category, Expense, OutboundMessage, ProcessedMessage, User
from oink_finai.domain.enums import (
    ExpenseCategory,
    ExpenseIntent,
    MessageSourceType,
    OutboundMessageKind,
    ProcessedMessageStatus,
)
from oink_finai.domain.expense_query import ExpenseQueryIntent, ExpenseQueryMetric
from oink_finai.repositories import SQLAlchemyExpenseQueryExecutor
from oink_finai.schemas.expense_interpretation import ExpenseInterpretation
from oink_finai.schemas.expense_query import (
    ExpenseQueryFilters,
    ExpenseQueryPeriod,
    ExpenseQueryPlan,
)
from oink_finai.services.expense_interpreter import ExpenseInterpreter
from oink_finai.services.expense_processing import ExpenseProcessingService
from oink_finai.services.expense_query_interpreter import ExpenseQueryInterpreter
from oink_finai.services.whatsapp_expense_query_result_formatter import (
    WhatsAppExpenseQueryResultFormatter,
)

pytestmark = pytest.mark.skipif(
    "OINK_TEST_POSTGRES_URL" not in os.environ,
    reason="OINK_TEST_POSTGRES_URL not configured",
)


class QueryClassifier(ExpenseInterpreter):
    def __init__(self) -> None:
        self.calls = 0

    async def interpret(self, message: str, *, reference_timestamp: datetime):
        self.calls += 1
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


class TotalQueryInterpreter(ExpenseQueryInterpreter):
    def __init__(self) -> None:
        self.calls = 0

    async def interpret(self, message: str, *, reference_timestamp: datetime):
        self.calls += 1
        return ExpenseQueryPlan(
            intent=ExpenseQueryIntent.AGGREGATE,
            metric=ExpenseQueryMetric.TOTAL,
            group_by=None,
            period=ExpenseQueryPeriod(start_date=date(2026, 9, 1), end_date=date(2026, 9, 30)),
            comparison_period=None,
            filters=ExpenseQueryFilters(
                category=None,
                merchant=None,
                payment_method=None,
                source_type=None,
                min_amount=None,
                max_amount=None,
            ),
            sort_by=None,
            sort_direction=None,
            limit=1,
            offset=0,
            unclear_reason=None,
            timezone="America/Sao_Paulo",
            reference_date=date(2026, 9, 8),
        )


async def seed_pipeline(factory):
    unique = uuid4().hex
    async with factory() as session, session.begin():
        category = await session.scalar(
            select(Category).where(Category.name == ExpenseCategory.FOOD.value)
        )
        assert category is not None
        user = User(phone_number=f"query-pg-{unique[:18]}")
        other = User(phone_number=f"query-pg-other-{unique[:12]}")
        session.add_all([user, other])
        await session.flush()
        session.add_all(
            [
                Expense(
                    user_id=user.id,
                    category_id=category.id,
                    amount=Decimal("10.10"),
                    description="visible",
                    expense_date=date(2026, 9, 8),
                ),
                Expense(
                    user_id=user.id,
                    category_id=category.id,
                    amount=Decimal("99.99"),
                    description="deleted",
                    expense_date=date(2026, 9, 8),
                    deleted_at=datetime(2026, 9, 8, tzinfo=UTC),
                ),
                Expense(
                    user_id=other.id,
                    category_id=category.id,
                    amount=Decimal("777.77"),
                    description="other user",
                    expense_date=date(2026, 9, 8),
                ),
            ]
        )
        message = ProcessedMessage(
            provider="postgres-query-test",
            instance_id=unique,
            external_message_id=unique,
            user_id=user.id,
            accepted_text="Quanto gastei este mês?",
            source_type=MessageSourceType.TEXT,
            message_timestamp=datetime(2026, 9, 8, 12, tzinfo=UTC),
            status=ProcessedMessageStatus.PENDING,
            available_at=datetime.now(UTC),
        )
        session.add(message)
        await session.flush()
        return user.id, other.id, message.id


async def cleanup(factory, user_ids, message_id) -> None:
    async with factory() as session, session.begin():
        await session.execute(
            delete(OutboundMessage).where(OutboundMessage.processed_message_id == message_id)
        )
        await session.execute(delete(ProcessedMessage).where(ProcessedMessage.id == message_id))
        await session.execute(delete(User).where(User.id.in_(user_ids)))


async def test_postgres_query_pipeline_isolates_user_and_soft_deleted() -> None:
    engine = create_async_engine(os.environ["OINK_TEST_POSTGRES_URL"])
    factory = async_sessionmaker(engine, expire_on_commit=False)
    user_id, other_id, message_id = await seed_pipeline(factory)
    classifier = QueryClassifier()
    query_interpreter = TotalQueryInterpreter()
    service = ExpenseProcessingService(
        factory,
        lambda _timezone: classifier,
        query_interpreter_factory=lambda _timezone: query_interpreter,
        query_executor=SQLAlchemyExpenseQueryExecutor(factory),
        query_formatter=WhatsAppExpenseQueryResultFormatter(),
    )
    try:
        await service.claim(1)
        before = None
        async with factory() as session:
            before = await session.scalar(select(func.count()).select_from(Expense))
        await service.process(message_id)
        async with factory() as session:
            outbox = await session.scalar(
                select(OutboundMessage).where(OutboundMessage.processed_message_id == message_id)
            )
            after = await session.scalar(select(func.count()).select_from(Expense))
            assert outbox is not None and outbox.kind is OutboundMessageKind.QUERY_RESULT
            assert "R$ 10,10" in outbox.content
            assert "99,99" not in outbox.content and "777,77" not in outbox.content
            assert before == after
        assert classifier.calls == query_interpreter.calls == 1
    finally:
        await cleanup(factory, [user_id, other_id], message_id)
        await engine.dispose()


async def test_postgres_two_workers_claim_one_query() -> None:
    engine = create_async_engine(os.environ["OINK_TEST_POSTGRES_URL"])
    factory = async_sessionmaker(engine, expire_on_commit=False)
    user_id, other_id, message_id = await seed_pipeline(factory)
    first = ExpenseProcessingService(factory, lambda _timezone: QueryClassifier())
    second = ExpenseProcessingService(factory, lambda _timezone: QueryClassifier())
    try:
        claims = await asyncio.gather(first.claim(1), second.claim(1))
        assert sum(message_id in worker_claims for worker_claims in claims) == 1
    finally:
        await cleanup(factory, [user_id, other_id], message_id)
        await engine.dispose()
