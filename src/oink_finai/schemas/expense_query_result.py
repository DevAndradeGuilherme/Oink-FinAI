from datetime import date
from decimal import Decimal
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from oink_finai.domain.enums import ExpenseCategory, MessageSourceType, PaymentMethod
from oink_finai.domain.expense_query import (
    ExpenseQueryGroup,
    ExpenseQueryMetric,
    ExpenseQuerySortField,
    SortDirection,
)
from oink_finai.schemas.expense_query import (
    ExpenseQueryFilters,
    ExpenseQueryPeriod,
)


class ExpenseQueryMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    period: ExpenseQueryPeriod | None
    comparison_period: ExpenseQueryPeriod | None
    filters: ExpenseQueryFilters
    sort_by: ExpenseQuerySortField | None
    sort_direction: SortDirection | None
    limit: int
    offset: int
    timezone: str
    reference_date: date


class ExpenseListItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: UUID
    amount: Decimal
    description: str
    expense_date: date
    category: ExpenseCategory
    merchant: str | None
    payment_method: PaymentMethod | None
    source_type: MessageSourceType


class ExpenseListResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Literal["LIST", "TOP_EXPENSES"]
    items: tuple[ExpenseListItem, ...]
    metadata: ExpenseQueryMetadata


class ExpenseAggregateResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Literal["AGGREGATE"] = "AGGREGATE"
    metric: ExpenseQueryMetric
    value: Decimal | int
    record_count: int
    metadata: ExpenseQueryMetadata


class ExpenseGroupItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    key: date | str | None
    value: Decimal | int
    record_count: int


class ExpenseGroupResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Literal["GROUP", "CATEGORY_BREAKDOWN"]
    metric: ExpenseQueryMetric
    group_by: ExpenseQueryGroup
    items: tuple[ExpenseGroupItem, ...]
    metadata: ExpenseQueryMetadata


class ExpenseComparisonResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Literal["COMPARE"] = "COMPARE"
    metric: ExpenseQueryMetric
    value: Decimal | int
    comparison_value: Decimal | int
    absolute_change: Decimal | int
    percentage_change: Decimal | None
    record_count: int
    comparison_record_count: int
    metadata: ExpenseQueryMetadata


ExpenseQueryResult = (
    ExpenseListResult | ExpenseAggregateResult | ExpenseGroupResult | ExpenseComparisonResult
)
