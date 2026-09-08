import asyncio
import os
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from oink_finai.database.models import Category, Expense, User
from oink_finai.domain.enums import ExpenseCategory, MessageSourceType, PaymentMethod
from oink_finai.domain.expense_query import (
    ExpenseQueryGroup,
    ExpenseQueryIntent,
    ExpenseQueryMetric,
    ExpenseQuerySortField,
    SortDirection,
)
from oink_finai.repositories.sqlalchemy_expense_query_executor import (
    SQLAlchemyExpenseQueryExecutor,
)
from oink_finai.schemas.expense_query import (
    ExpenseQueryFilters,
    ExpenseQueryPeriod,
    ExpenseQueryPlan,
)
from oink_finai.schemas.expense_query_result import (
    ExpenseAggregateResult,
    ExpenseComparisonResult,
    ExpenseGroupResult,
    ExpenseListResult,
)
from oink_finai.services.expense_query_executor import (
    InvalidExpenseQueryPlanError,
    UnsupportedExpenseQueryIntentError,
)

pytestmark = pytest.mark.skipif(
    "OINK_TEST_POSTGRES_URL" not in os.environ,
    reason="OINK_TEST_POSTGRES_URL not configured",
)

SEPTEMBER = ExpenseQueryPeriod(start_date=date(2026, 9, 1), end_date=date(2026, 9, 30))
AUGUST = ExpenseQueryPeriod(start_date=date(2026, 8, 1), end_date=date(2026, 8, 31))
JUNE = ExpenseQueryPeriod(start_date=date(2026, 6, 1), end_date=date(2026, 6, 30))
JULY = ExpenseQueryPeriod(start_date=date(2026, 7, 1), end_date=date(2026, 7, 31))


def empty_filters(**overrides: object) -> ExpenseQueryFilters:
    values = {
        "category": None,
        "merchant": None,
        "payment_method": None,
        "source_type": None,
        "min_amount": None,
        "max_amount": None,
    }
    values.update(overrides)
    return ExpenseQueryFilters(**values)


def query_plan(
    intent: ExpenseQueryIntent,
    *,
    metric: ExpenseQueryMetric | None = None,
    group_by: ExpenseQueryGroup | None = None,
    period: ExpenseQueryPeriod | None = SEPTEMBER,
    comparison_period: ExpenseQueryPeriod | None = None,
    filters: ExpenseQueryFilters | None = None,
    sort_by: ExpenseQuerySortField | None = None,
    direction: SortDirection | None = None,
    limit: int = 1,
    offset: int = 0,
) -> ExpenseQueryPlan:
    return ExpenseQueryPlan(
        intent=intent,
        metric=metric,
        group_by=group_by,
        period=period,
        comparison_period=comparison_period,
        filters=filters or empty_filters(),
        sort_by=sort_by,
        sort_direction=direction,
        limit=limit,
        offset=offset,
        unclear_reason=None,
        timezone="America/Sao_Paulo",
        reference_date=date(2026, 9, 8),
    )


def list_plan(
    *,
    period: ExpenseQueryPeriod = SEPTEMBER,
    filters: ExpenseQueryFilters | None = None,
    sort_by: ExpenseQuerySortField = ExpenseQuerySortField.DATE,
    direction: SortDirection = SortDirection.ASC,
    limit: int = 100,
    offset: int = 0,
) -> ExpenseQueryPlan:
    return query_plan(
        ExpenseQueryIntent.LIST,
        period=period,
        filters=filters,
        sort_by=sort_by,
        direction=direction,
        limit=limit,
        offset=offset,
    )


def aggregate_plan(
    metric: ExpenseQueryMetric,
    *,
    period: ExpenseQueryPeriod = SEPTEMBER,
    filters: ExpenseQueryFilters | None = None,
) -> ExpenseQueryPlan:
    return query_plan(
        ExpenseQueryIntent.AGGREGATE,
        metric=metric,
        period=period,
        filters=filters,
    )


class QueryContext:
    def __init__(
        self,
        factory: async_sessionmaker[AsyncSession],
        user_id: UUID,
        other_user_id: UUID,
    ) -> None:
        self.factory = factory
        self.executor = SQLAlchemyExpenseQueryExecutor(factory)
        self.user_id = user_id
        self.other_user_id = other_user_id


@pytest_asyncio.fixture
async def query_context() -> AsyncIterator[QueryContext]:
    engine = create_async_engine(os.environ["OINK_TEST_POSTGRES_URL"])
    factory = async_sessionmaker(engine, expire_on_commit=False)
    unique = uuid4().hex
    async with factory() as session, session.begin():
        categories = {
            name: category
            for category in await session.scalars(
                select(Category).where(
                    Category.name.in_(
                        [
                            ExpenseCategory.FOOD.value,
                            ExpenseCategory.TRANSPORT.value,
                            ExpenseCategory.HEALTH.value,
                        ]
                    )
                )
            )
            for name in [category.name]
        }
        user = User(phone_number=f"query-{unique[:20]}")
        other_user = User(phone_number=f"query-other-{unique[:14]}")
        session.add_all([user, other_user])
        await session.flush()

        def expense(
            identifier: int,
            amount: str,
            expense_date: date,
            category: ExpenseCategory,
            merchant: str,
            payment_method: PaymentMethod = PaymentMethod.PIX,
            source_type: MessageSourceType = MessageSourceType.TEXT,
            *,
            user_id: UUID = user.id,
            deleted: bool = False,
        ) -> Expense:
            return Expense(
                id=UUID(int=identifier),
                user_id=user_id,
                category_id=categories[category.value].id,
                amount=Decimal(amount),
                description=f"Expense {identifier}",
                expense_date=expense_date,
                merchant=merchant,
                payment_method=payment_method.value,
                source_type=source_type,
                deleted_at=datetime(2026, 9, 8, tzinfo=UTC) if deleted else None,
            )

        rows = [
            expense(101, "10.10", date(2026, 9, 1), ExpenseCategory.FOOD, "Market"),
            expense(
                102,
                "20.20",
                date(2026, 9, 8),
                ExpenseCategory.TRANSPORT,
                "Ride",
                PaymentMethod.CREDIT,
                MessageSourceType.AUDIO,
            ),
            expense(
                103,
                "30.30",
                date(2026, 9, 30),
                ExpenseCategory.FOOD,
                "Market",
                PaymentMethod.PIX,
                MessageSourceType.IMAGE,
            ),
            expense(104, "40.40", date(2026, 8, 31), ExpenseCategory.HEALTH, "Clinic"),
            expense(105, "50.50", date(2026, 10, 1), ExpenseCategory.FOOD, "Market"),
            expense(
                106,
                "99.99",
                date(2026, 9, 8),
                ExpenseCategory.FOOD,
                "Deleted",
                deleted=True,
            ),
            expense(
                107,
                "777.77",
                date(2026, 9, 8),
                ExpenseCategory.FOOD,
                "Other user",
                user_id=other_user.id,
            ),
        ]
        special_merchants = ["100% market", "under_score", "Bob's shop", "東京; DROP TABLE users"]
        rows.extend(
            expense(200 + index, "7.77", date(2026, 7, 10), ExpenseCategory.FOOD, merchant)
            for index, merchant in enumerate(special_merchants)
        )
        session.add_all(rows)
        await session.flush()
        context = QueryContext(factory, user.id, other_user.id)

    try:
        yield context
    finally:
        async with factory() as session, session.begin():
            await session.execute(delete(User).where(User.id.in_([user.id, other_user.id])))
        await engine.dispose()


async def execute(context: QueryContext, plan: ExpenseQueryPlan):
    return await context.executor.execute(user_id=context.user_id, plan=plan)


async def test_list_is_user_scoped_and_excludes_soft_deleted(query_context: QueryContext) -> None:
    result = await execute(query_context, list_plan())

    assert isinstance(result, ExpenseListResult)
    assert [item.amount for item in result.items] == [
        Decimal("10.10"),
        Decimal("20.20"),
        Decimal("30.30"),
    ]
    assert result.metadata.period == SEPTEMBER
    assert all(item.id not in {UUID(int=106), UUID(int=107)} for item in result.items)


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        (ExpenseQueryMetric.TOTAL, Decimal("60.60")),
        (ExpenseQueryMetric.COUNT, 3),
        (ExpenseQueryMetric.AVERAGE, Decimal("20.20")),
        (ExpenseQueryMetric.MINIMUM, Decimal("10.10")),
        (ExpenseQueryMetric.MAXIMUM, Decimal("30.30")),
    ],
)
async def test_each_aggregate_metric(
    query_context: QueryContext, metric: ExpenseQueryMetric, expected: Decimal | int
) -> None:
    result = await execute(query_context, aggregate_plan(metric))

    assert isinstance(result, ExpenseAggregateResult)
    assert result.value == expected
    assert result.record_count == 3
    if metric is not ExpenseQueryMetric.COUNT:
        assert isinstance(result.value, Decimal)


@pytest.mark.parametrize(
    ("group", "expected_count"),
    [
        (ExpenseQueryGroup.DAY, 3),
        (ExpenseQueryGroup.WEEK, 3),
        (ExpenseQueryGroup.MONTH, 1),
        (ExpenseQueryGroup.CATEGORY, 2),
        (ExpenseQueryGroup.MERCHANT, 2),
        (ExpenseQueryGroup.PAYMENT_METHOD, 2),
        (ExpenseQueryGroup.SOURCE_TYPE, 3),
    ],
)
async def test_each_grouping(
    query_context: QueryContext, group: ExpenseQueryGroup, expected_count: int
) -> None:
    plan = query_plan(
        ExpenseQueryIntent.GROUP,
        metric=ExpenseQueryMetric.TOTAL,
        group_by=group,
        sort_by=ExpenseQuerySortField.GROUP_KEY,
        direction=SortDirection.ASC,
        limit=100,
    )
    result = await execute(query_context, plan)

    assert isinstance(result, ExpenseGroupResult)
    assert len(result.items) == expected_count
    assert sum((item.value for item in result.items), Decimal("0")) == Decimal("60.60")
    assert result.kind == ("CATEGORY_BREAKDOWN" if group is ExpenseQueryGroup.CATEGORY else "GROUP")


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        (empty_filters(category=ExpenseCategory.FOOD), [Decimal("10.10"), Decimal("30.30")]),
        (empty_filters(merchant="Ride"), [Decimal("20.20")]),
        (empty_filters(payment_method=PaymentMethod.CREDIT), [Decimal("20.20")]),
        (empty_filters(source_type=MessageSourceType.IMAGE), [Decimal("30.30")]),
        (empty_filters(min_amount=Decimal("20.20")), [Decimal("20.20"), Decimal("30.30")]),
        (empty_filters(max_amount=Decimal("20.20")), [Decimal("10.10"), Decimal("20.20")]),
    ],
)
async def test_each_filter(
    query_context: QueryContext,
    filters: ExpenseQueryFilters,
    expected: list[Decimal],
) -> None:
    result = await execute(query_context, list_plan(filters=filters))
    assert isinstance(result, ExpenseListResult)
    assert [item.amount for item in result.items] == expected
    assert result.metadata.filters == filters


async def test_all_filters_are_combined(query_context: QueryContext) -> None:
    filters = empty_filters(
        category=ExpenseCategory.FOOD,
        merchant="Market",
        payment_method=PaymentMethod.PIX,
        source_type=MessageSourceType.TEXT,
        min_amount=Decimal("10.10"),
        max_amount=Decimal("10.10"),
    )
    result = await execute(query_context, list_plan(filters=filters))
    assert isinstance(result, ExpenseListResult)
    assert [item.id for item in result.items] == [UUID(int=101)]


async def test_period_boundaries_are_inclusive_and_decimal_exact(
    query_context: QueryContext,
) -> None:
    result = await execute(query_context, aggregate_plan(ExpenseQueryMetric.TOTAL))
    assert isinstance(result, ExpenseAggregateResult)
    assert result.value == Decimal("60.60")
    assert isinstance(result.value, Decimal)


async def test_empty_category_and_average_return_consistent_zero(
    query_context: QueryContext,
) -> None:
    filters = empty_filters(category=ExpenseCategory.EDUCATION)
    listed = await execute(query_context, list_plan(filters=filters))
    average = await execute(
        query_context, aggregate_plan(ExpenseQueryMetric.AVERAGE, filters=filters)
    )
    assert isinstance(listed, ExpenseListResult) and listed.items == ()
    assert isinstance(average, ExpenseAggregateResult)
    assert average.value == Decimal("0") and average.record_count == 0
    assert average.value.is_finite()


@pytest.mark.parametrize(
    "merchant",
    ["100% market", "under_score", "Bob's shop", "東京; DROP TABLE users"],
)
async def test_merchant_special_characters_remain_exact_parameters(
    query_context: QueryContext, merchant: str
) -> None:
    result = await execute(
        query_context,
        list_plan(period=JULY, filters=empty_filters(merchant=merchant)),
    )
    assert isinstance(result, ExpenseListResult)
    assert len(result.items) == 1
    assert result.items[0].merchant == merchant


async def test_ranking_uses_fixed_amount_order(query_context: QueryContext) -> None:
    plan = query_plan(
        ExpenseQueryIntent.RANK,
        sort_by=ExpenseQuerySortField.AMOUNT,
        direction=SortDirection.DESC,
        limit=2,
    )
    result = await execute(query_context, plan)
    assert isinstance(result, ExpenseListResult)
    assert result.kind == "TOP_EXPENSES"
    assert [item.amount for item in result.items] == [Decimal("30.30"), Decimal("20.20")]


async def test_pagination_is_deterministic_with_uuid_tiebreaker(
    query_context: QueryContext,
) -> None:
    first = await execute(
        query_context,
        list_plan(
            period=JULY,
            sort_by=ExpenseQuerySortField.AMOUNT,
            direction=SortDirection.ASC,
            limit=2,
        ),
    )
    second = await execute(
        query_context,
        list_plan(
            period=JULY,
            sort_by=ExpenseQuerySortField.AMOUNT,
            direction=SortDirection.ASC,
            limit=2,
            offset=2,
        ),
    )
    assert isinstance(first, ExpenseListResult) and isinstance(second, ExpenseListResult)
    assert [item.id for item in first.items + second.items] == [
        UUID(int=200),
        UUID(int=201),
        UUID(int=202),
        UUID(int=203),
    ]


async def test_comparison_calculates_change_and_handles_zero_baseline(
    query_context: QueryContext,
) -> None:
    plan = query_plan(
        ExpenseQueryIntent.COMPARE,
        metric=ExpenseQueryMetric.TOTAL,
        comparison_period=AUGUST,
    )
    result = await execute(query_context, plan)
    assert isinstance(result, ExpenseComparisonResult)
    assert result.value == Decimal("60.60")
    assert result.comparison_value == Decimal("40.40")
    assert result.absolute_change == Decimal("20.20")
    assert result.percentage_change == Decimal("50.0")

    zero_plan = query_plan(
        ExpenseQueryIntent.COMPARE,
        metric=ExpenseQueryMetric.TOTAL,
        comparison_period=JUNE,
    )
    zero = await execute(query_context, zero_plan)
    assert isinstance(zero, ExpenseComparisonResult)
    assert zero.comparison_value == Decimal("0")
    assert zero.percentage_change is None


async def test_read_only_execution_is_concurrent(query_context: QueryContext) -> None:
    plans = [
        aggregate_plan(ExpenseQueryMetric.TOTAL),
        aggregate_plan(ExpenseQueryMetric.COUNT),
        list_plan(limit=2),
    ]
    results = await asyncio.gather(*(execute(query_context, plan) for plan in plans))
    assert len(results) == 3


async def test_non_query_plan_never_opens_session(query_context: QueryContext) -> None:
    plan = ExpenseQueryPlan(
        intent=ExpenseQueryIntent.NOT_QUERY,
        metric=None,
        group_by=None,
        period=None,
        comparison_period=None,
        filters=empty_filters(),
        sort_by=None,
        sort_direction=None,
        limit=0,
        offset=0,
        unclear_reason=None,
        timezone="America/Sao_Paulo",
        reference_date=date(2026, 9, 8),
    )
    with pytest.raises(UnsupportedExpenseQueryIntentError):
        await execute(query_context, plan)


async def test_constructed_invalid_plan_is_rejected_before_session() -> None:
    class NoSessionFactory:
        def __call__(self):
            raise AssertionError("session must not be opened")

    invalid_values = dict(list_plan().__dict__)
    invalid_values["limit"] = 10_000
    invalid = ExpenseQueryPlan.model_construct(**invalid_values)
    executor = SQLAlchemyExpenseQueryExecutor(NoSessionFactory())  # type: ignore[arg-type]
    with pytest.raises(InvalidExpenseQueryPlanError):
        await executor.execute(user_id=uuid4(), plan=invalid)
