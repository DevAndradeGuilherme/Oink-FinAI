from datetime import date
from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import Date, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql.elements import ColumnElement

from oink_finai.database.models import Category, Expense
from oink_finai.domain.enums import ExpenseCategory, MessageSourceType, PaymentMethod
from oink_finai.domain.expense_query import (
    ExpenseQueryGroup,
    ExpenseQueryIntent,
    ExpenseQueryMetric,
    ExpenseQuerySortField,
    SortDirection,
)
from oink_finai.schemas.expense_query import (
    ExpenseQueryFilters,
    ExpenseQueryPeriod,
    ExpenseQueryPlan,
)
from oink_finai.schemas.expense_query_result import (
    ExpenseAggregateResult,
    ExpenseComparisonResult,
    ExpenseGroupItem,
    ExpenseGroupResult,
    ExpenseListItem,
    ExpenseListResult,
    ExpenseQueryMetadata,
    ExpenseQueryResult,
)
from oink_finai.services.expense_query_executor import (
    ExpenseQueryExecutor,
    InvalidExpenseQueryPlanError,
    UnsupportedExpenseQueryIntentError,
)

MetricExpression = ColumnElement[Any]


class SQLAlchemyExpenseQueryExecutor(ExpenseQueryExecutor):
    """Compile closed, validated plans into user-scoped SQLAlchemy expressions."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def execute(self, *, user_id: UUID, plan: ExpenseQueryPlan) -> ExpenseQueryResult:
        validated = self._revalidate(user_id, plan)
        if validated.intent in {
            ExpenseQueryIntent.NOT_QUERY,
            ExpenseQueryIntent.QUERY_UNCLEAR,
        }:
            raise UnsupportedExpenseQueryIntentError("plan does not contain an executable query")

        async with self._session_factory() as session:
            if validated.intent in {ExpenseQueryIntent.LIST, ExpenseQueryIntent.RANK}:
                return await self._execute_list(session, user_id, validated)
            if validated.intent is ExpenseQueryIntent.AGGREGATE:
                return await self._execute_aggregate(session, user_id, validated)
            if validated.intent is ExpenseQueryIntent.GROUP:
                return await self._execute_group(session, user_id, validated)
            if validated.intent is ExpenseQueryIntent.COMPARE:
                return await self._execute_comparison(session, user_id, validated)
        raise UnsupportedExpenseQueryIntentError("query intent is not supported")

    @staticmethod
    def _revalidate(user_id: UUID, plan: ExpenseQueryPlan) -> ExpenseQueryPlan:
        if not isinstance(user_id, UUID) or not isinstance(plan, ExpenseQueryPlan):
            raise InvalidExpenseQueryPlanError("trusted user id or query plan is invalid")
        try:
            return ExpenseQueryPlan.model_validate(plan.model_dump())
        except (ValidationError, TypeError, ValueError) as exc:
            raise InvalidExpenseQueryPlanError("query plan failed execution validation") from exc

    @staticmethod
    def _metadata(plan: ExpenseQueryPlan) -> ExpenseQueryMetadata:
        return ExpenseQueryMetadata(
            period=plan.period,
            comparison_period=plan.comparison_period,
            filters=plan.filters,
            sort_by=plan.sort_by,
            sort_direction=plan.sort_direction,
            limit=plan.limit,
            offset=plan.offset,
            timezone=plan.timezone,
            reference_date=plan.reference_date,
        )

    @staticmethod
    def _predicates(
        user_id: UUID,
        period: ExpenseQueryPeriod | None,
        filters: ExpenseQueryFilters,
    ) -> tuple[ColumnElement[bool], ...]:
        predicates: list[ColumnElement[bool]] = [
            Expense.user_id == user_id,
            Expense.deleted_at.is_(None),
        ]
        if period is not None:
            predicates.extend(
                [
                    Expense.expense_date >= period.start_date,
                    Expense.expense_date <= period.end_date,
                ]
            )
        if filters.category is not None:
            predicates.append(Category.name == filters.category.value)
        if filters.merchant is not None:
            predicates.append(Expense.merchant == filters.merchant)
        if filters.payment_method is not None:
            predicates.append(Expense.payment_method == filters.payment_method.value)
        if filters.source_type is not None:
            predicates.append(Expense.source_type == filters.source_type.value)
        if filters.min_amount is not None:
            predicates.append(Expense.amount >= filters.min_amount)
        if filters.max_amount is not None:
            predicates.append(Expense.amount <= filters.max_amount)
        return tuple(predicates)

    @staticmethod
    def _direction(expression: ColumnElement[Any], direction: SortDirection):
        return expression.asc() if direction is SortDirection.ASC else expression.desc()

    async def _execute_list(
        self, session: AsyncSession, user_id: UUID, plan: ExpenseQueryPlan
    ) -> ExpenseListResult:
        sort_columns = {
            ExpenseQuerySortField.DATE: Expense.expense_date,
            ExpenseQuerySortField.AMOUNT: Expense.amount,
        }
        sort_column = sort_columns[plan.sort_by]
        statement = (
            select(
                Expense.id,
                Expense.amount,
                Expense.description,
                Expense.expense_date,
                Category.name.label("category"),
                Expense.merchant,
                Expense.payment_method,
                Expense.source_type,
            )
            .select_from(Expense)
            .join(Category, Expense.category_id == Category.id)
            .where(*self._predicates(user_id, plan.period, plan.filters))
            .order_by(self._direction(sort_column, plan.sort_direction), Expense.id.asc())
            .limit(plan.limit)
            .offset(plan.offset)
        )
        rows = (await session.execute(statement)).all()
        items = tuple(
            ExpenseListItem(
                id=row.id,
                amount=self._money(row.amount),
                description=row.description,
                expense_date=row.expense_date,
                category=ExpenseCategory(row.category),
                merchant=row.merchant,
                payment_method=(
                    PaymentMethod(row.payment_method) if row.payment_method is not None else None
                ),
                source_type=MessageSourceType(row.source_type),
            )
            for row in rows
        )
        kind = "TOP_EXPENSES" if plan.intent is ExpenseQueryIntent.RANK else "LIST"
        return ExpenseListResult(kind=kind, items=items, metadata=self._metadata(plan))

    async def _execute_aggregate(
        self, session: AsyncSession, user_id: UUID, plan: ExpenseQueryPlan
    ) -> ExpenseAggregateResult:
        value, count = await self._aggregate_values(
            session, user_id, plan.period, plan.filters, plan.metric
        )
        return ExpenseAggregateResult(
            metric=plan.metric,
            value=value,
            record_count=count,
            metadata=self._metadata(plan),
        )

    async def _aggregate_values(
        self,
        session: AsyncSession,
        user_id: UUID,
        period: ExpenseQueryPeriod | None,
        filters: ExpenseQueryFilters,
        metric: ExpenseQueryMetric,
    ) -> tuple[Decimal | int, int]:
        expression = self._metric_expression(metric)
        statement = (
            select(expression.label("value"), func.count(Expense.id).label("record_count"))
            .select_from(Expense)
            .join(Category, Expense.category_id == Category.id)
            .where(*self._predicates(user_id, period, filters))
        )
        row = (await session.execute(statement)).one()
        return self._metric_value(metric, row.value), int(row.record_count)

    async def _execute_group(
        self, session: AsyncSession, user_id: UUID, plan: ExpenseQueryPlan
    ) -> ExpenseGroupResult:
        group_expression = self._group_expression(plan.group_by)
        metric_expression = self._metric_expression(plan.metric)
        order_expression = (
            metric_expression if plan.sort_by is ExpenseQuerySortField.METRIC else group_expression
        )
        order = self._direction(order_expression, plan.sort_direction).nulls_last()
        statement = (
            select(
                group_expression.label("group_key"),
                metric_expression.label("value"),
                func.count(Expense.id).label("record_count"),
            )
            .select_from(Expense)
            .join(Category, Expense.category_id == Category.id)
            .where(*self._predicates(user_id, plan.period, plan.filters))
            .group_by(group_expression)
            .order_by(order, group_expression.asc().nulls_last())
            .limit(plan.limit)
            .offset(plan.offset)
        )
        rows = (await session.execute(statement)).all()
        items = tuple(
            ExpenseGroupItem(
                key=self._group_key(row.group_key),
                value=self._metric_value(plan.metric, row.value),
                record_count=int(row.record_count),
            )
            for row in rows
        )
        kind = "CATEGORY_BREAKDOWN" if plan.group_by is ExpenseQueryGroup.CATEGORY else "GROUP"
        return ExpenseGroupResult(
            kind=kind,
            metric=plan.metric,
            group_by=plan.group_by,
            items=items,
            metadata=self._metadata(plan),
        )

    async def _execute_comparison(
        self, session: AsyncSession, user_id: UUID, plan: ExpenseQueryPlan
    ) -> ExpenseComparisonResult:
        value, count = await self._aggregate_values(
            session, user_id, plan.period, plan.filters, plan.metric
        )
        comparison, comparison_count = await self._aggregate_values(
            session, user_id, plan.comparison_period, plan.filters, plan.metric
        )
        absolute = value - comparison
        baseline = Decimal(comparison)
        percentage = None if baseline == 0 else (Decimal(value) - baseline) / baseline * 100
        return ExpenseComparisonResult(
            metric=plan.metric,
            value=value,
            comparison_value=comparison,
            absolute_change=absolute,
            percentage_change=percentage,
            record_count=count,
            comparison_record_count=comparison_count,
            metadata=self._metadata(plan),
        )

    @staticmethod
    def _metric_expression(metric: ExpenseQueryMetric) -> MetricExpression:
        expressions = {
            ExpenseQueryMetric.TOTAL: func.sum(Expense.amount),
            ExpenseQueryMetric.COUNT: func.count(Expense.id),
            ExpenseQueryMetric.AVERAGE: func.avg(Expense.amount),
            ExpenseQueryMetric.MINIMUM: func.min(Expense.amount),
            ExpenseQueryMetric.MAXIMUM: func.max(Expense.amount),
        }
        return expressions[metric]

    @staticmethod
    def _group_expression(group: ExpenseQueryGroup) -> ColumnElement[Any]:
        expressions = {
            ExpenseQueryGroup.DAY: Expense.expense_date,
            ExpenseQueryGroup.WEEK: cast(func.date_trunc("week", Expense.expense_date), Date),
            ExpenseQueryGroup.MONTH: cast(func.date_trunc("month", Expense.expense_date), Date),
            ExpenseQueryGroup.CATEGORY: Category.name,
            ExpenseQueryGroup.MERCHANT: Expense.merchant,
            ExpenseQueryGroup.PAYMENT_METHOD: Expense.payment_method,
            ExpenseQueryGroup.SOURCE_TYPE: Expense.source_type,
        }
        return expressions[group]

    @classmethod
    def _metric_value(cls, metric: ExpenseQueryMetric, value: Any) -> Decimal | int:
        if metric is ExpenseQueryMetric.COUNT:
            return int(value or 0)
        return cls._money(value)

    @staticmethod
    def _money(value: Any) -> Decimal:
        if value is None:
            return Decimal("0")
        if not isinstance(value, Decimal):
            raise TypeError("database returned a non-Decimal monetary value")
        if not value.is_finite():
            raise ValueError("database returned a non-finite monetary value")
        return value

    @staticmethod
    def _group_key(value: Any) -> date | str | None:
        if value is None or isinstance(value, (date, str)):
            return value
        raise TypeError("database returned an invalid group key")
