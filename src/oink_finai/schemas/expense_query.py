import re
from datetime import date
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel, ConfigDict, model_validator

from oink_finai.domain.enums import ExpenseCategory, MessageSourceType, PaymentMethod
from oink_finai.domain.expense_limits import EXPENSE_AMOUNT_MAX
from oink_finai.domain.expense_query import (
    MAX_GROUP_LIMIT,
    MAX_LIST_LIMIT,
    MAX_QUERY_OFFSET,
    MAX_QUERY_RANGE_DAYS,
    MAX_RANK_LIMIT,
    ExpenseQueryGroup,
    ExpenseQueryIntent,
    ExpenseQueryMetric,
    ExpenseQuerySortField,
    QueryUnclearReason,
    SortDirection,
)

_ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DECIMAL_PATTERN = re.compile(r"^(?:0|[1-9]\d*)(?:\.\d{1,2})?$")


class DateRangeTransport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_date: str
    end_date: str


class OpenAIExpenseQueryTransport(BaseModel):
    """Strict provider DTO. It contains no user identity or executable field names."""

    model_config = ConfigDict(extra="forbid")

    intent: ExpenseQueryIntent
    metric: ExpenseQueryMetric | None
    group_by: ExpenseQueryGroup | None
    period: DateRangeTransport | None
    comparison_period: DateRangeTransport | None
    category: ExpenseCategory | None
    merchant: str | None
    payment_method: PaymentMethod | None
    source_type: MessageSourceType | None
    min_amount: str | None
    max_amount: str | None
    sort_by: ExpenseQuerySortField | None
    sort_direction: SortDirection | None
    limit: int | None
    offset: int | None
    unclear_reason: QueryUnclearReason | None


class ExpenseQueryPeriod(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    start_date: date
    end_date: date

    @model_validator(mode="after")
    def validate_range(self) -> "ExpenseQueryPeriod":
        if self.start_date > self.end_date:
            raise ValueError("query period is inverted")
        days = (self.end_date - self.start_date).days + 1
        if days > MAX_QUERY_RANGE_DAYS:
            raise ValueError("query period exceeds the allowed range")
        return self


class ExpenseQueryFilters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    category: ExpenseCategory | None
    merchant: str | None
    payment_method: PaymentMethod | None
    source_type: MessageSourceType | None
    min_amount: Decimal | None
    max_amount: Decimal | None

    @model_validator(mode="after")
    def validate_filters(self) -> "ExpenseQueryFilters":
        if self.merchant is not None:
            normalized = self.merchant.strip()
            if not normalized or len(normalized) > 160:
                raise ValueError("merchant filter is invalid")
            object.__setattr__(self, "merchant", normalized)
        for amount in (self.min_amount, self.max_amount):
            if amount is not None and (
                not amount.is_finite() or amount < 0 or amount > EXPENSE_AMOUNT_MAX
            ):
                raise ValueError("amount filter is invalid")
        if (
            self.min_amount is not None
            and self.max_amount is not None
            and self.min_amount > self.max_amount
        ):
            raise ValueError("amount range is inverted")
        return self


class ExpenseQueryPlan(BaseModel):
    """Provider-independent plan consumed by a future, user-scoped SQL compiler."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    intent: ExpenseQueryIntent
    metric: ExpenseQueryMetric | None
    group_by: ExpenseQueryGroup | None
    period: ExpenseQueryPeriod | None
    comparison_period: ExpenseQueryPeriod | None
    filters: ExpenseQueryFilters
    sort_by: ExpenseQuerySortField | None
    sort_direction: SortDirection | None
    limit: int
    offset: int
    unclear_reason: QueryUnclearReason | None
    timezone: str
    reference_date: date

    @model_validator(mode="after")
    def validate_plan(self) -> "ExpenseQueryPlan":
        if not 0 <= self.offset <= MAX_QUERY_OFFSET:
            raise ValueError("query offset is outside the allowed range")

        if self.intent in {ExpenseQueryIntent.NOT_QUERY, ExpenseQueryIntent.QUERY_UNCLEAR}:
            self._validate_non_query()
            return self

        if self.unclear_reason is not None:
            raise ValueError("query plan cannot contain an unclear reason")
        if self.intent is ExpenseQueryIntent.LIST:
            self._validate_list()
        elif self.intent is ExpenseQueryIntent.AGGREGATE:
            self._validate_aggregate()
        elif self.intent is ExpenseQueryIntent.GROUP:
            self._validate_group()
        elif self.intent is ExpenseQueryIntent.RANK:
            self._validate_rank()
        elif self.intent is ExpenseQueryIntent.COMPARE:
            self._validate_compare()
        return self

    def _validate_non_query(self) -> None:
        if self.intent is ExpenseQueryIntent.QUERY_UNCLEAR:
            if self.unclear_reason is None:
                raise ValueError("QUERY_UNCLEAR requires a reason")
        elif self.unclear_reason is not None:
            raise ValueError("NOT_QUERY cannot contain an unclear reason")
        empty_filters = all(
            value is None
            for value in (
                self.filters.category,
                self.filters.merchant,
                self.filters.payment_method,
                self.filters.source_type,
                self.filters.min_amount,
                self.filters.max_amount,
            )
        )
        if (
            any(
                value is not None
                for value in (
                    self.metric,
                    self.group_by,
                    self.period,
                    self.comparison_period,
                    self.sort_by,
                    self.sort_direction,
                )
            )
            or not empty_filters
            or self.limit != 0
            or self.offset != 0
        ):
            raise ValueError("non-query result cannot contain a query plan")

    def _validate_list(self) -> None:
        if (
            self.metric is not None
            or self.group_by is not None
            or self.comparison_period is not None
        ):
            raise ValueError("LIST contains incompatible aggregation fields")
        if not 1 <= self.limit <= MAX_LIST_LIMIT:
            raise ValueError("LIST limit is outside the allowed range")
        if self.sort_by not in {ExpenseQuerySortField.DATE, ExpenseQuerySortField.AMOUNT}:
            raise ValueError("LIST sort field is invalid")
        if self.sort_direction is None:
            raise ValueError("LIST requires a sort direction")

    def _validate_aggregate(self) -> None:
        if self.metric is None or self.group_by is not None or self.comparison_period is not None:
            raise ValueError("AGGREGATE fields are invalid")
        if self.sort_by is not None or self.sort_direction is not None:
            raise ValueError("AGGREGATE cannot be sorted")
        if self.limit != 1 or self.offset != 0:
            raise ValueError("AGGREGATE pagination is invalid")

    def _validate_group(self) -> None:
        if self.metric is None or self.group_by is None or self.comparison_period is not None:
            raise ValueError("GROUP fields are invalid")
        if not 1 <= self.limit <= MAX_GROUP_LIMIT:
            raise ValueError("GROUP limit is outside the allowed range")
        if self.sort_by not in {ExpenseQuerySortField.METRIC, ExpenseQuerySortField.GROUP_KEY}:
            raise ValueError("GROUP sort field is invalid")
        if self.sort_direction is None:
            raise ValueError("GROUP requires a sort direction")

    def _validate_rank(self) -> None:
        if (
            self.metric is not None
            or self.group_by is not None
            or self.comparison_period is not None
        ):
            raise ValueError("RANK fields are invalid")
        if not 1 <= self.limit <= MAX_RANK_LIMIT or self.offset != 0:
            raise ValueError("RANK pagination is invalid")
        if self.sort_by is not ExpenseQuerySortField.AMOUNT:
            raise ValueError("RANK must sort by amount")
        if self.sort_direction is not SortDirection.DESC:
            raise ValueError("RANK must use descending order")

    def _validate_compare(self) -> None:
        if self.metric is None or self.group_by is not None:
            raise ValueError("COMPARE fields are invalid")
        if self.period is None or self.comparison_period is None:
            raise ValueError("COMPARE requires two periods")
        if self.sort_by is not None or self.sort_direction is not None:
            raise ValueError("COMPARE cannot be sorted")
        if self.limit != 1 or self.offset != 0:
            raise ValueError("COMPARE pagination is invalid")


def parse_iso_date(value: str) -> date:
    if not _ISO_DATE_PATTERN.fullmatch(value):
        raise ValueError("date must use ISO YYYY-MM-DD format")
    return date.fromisoformat(value)


def parse_decimal(value: str | None) -> Decimal | None:
    if value is None:
        return None
    if not _DECIMAL_PATTERN.fullmatch(value):
        raise ValueError("amount must be a normalized decimal string")
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("amount is invalid") from exc


def convert_period(value: DateRangeTransport | None) -> ExpenseQueryPeriod | None:
    if value is None:
        return None
    return ExpenseQueryPeriod(
        start_date=parse_iso_date(value.start_date),
        end_date=parse_iso_date(value.end_date),
    )
