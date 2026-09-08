from enum import StrEnum


class ExpenseQueryIntent(StrEnum):
    LIST = "LIST"
    AGGREGATE = "AGGREGATE"
    GROUP = "GROUP"
    RANK = "RANK"
    COMPARE = "COMPARE"
    NOT_QUERY = "NOT_QUERY"
    QUERY_UNCLEAR = "QUERY_UNCLEAR"

    # Domain aliases used by the execution layer. Provider output stays unchanged.
    TOP_EXPENSES = "RANK"
    CATEGORY_BREAKDOWN = "GROUP"


class ExpenseQueryMetric(StrEnum):
    TOTAL = "TOTAL"
    COUNT = "COUNT"
    AVERAGE = "AVERAGE"
    MINIMUM = "MINIMUM"
    MAXIMUM = "MAXIMUM"


class ExpenseQueryGroup(StrEnum):
    DAY = "DAY"
    WEEK = "WEEK"
    MONTH = "MONTH"
    CATEGORY = "CATEGORY"
    MERCHANT = "MERCHANT"
    PAYMENT_METHOD = "PAYMENT_METHOD"
    SOURCE_TYPE = "SOURCE_TYPE"


class ExpenseQuerySortField(StrEnum):
    DATE = "DATE"
    AMOUNT = "AMOUNT"
    METRIC = "METRIC"
    GROUP_KEY = "GROUP_KEY"


class SortDirection(StrEnum):
    ASC = "ASC"
    DESC = "DESC"


class QueryUnclearReason(StrEnum):
    AMBIGUOUS_REQUEST = "AMBIGUOUS_REQUEST"
    AMBIGUOUS_PERIOD = "AMBIGUOUS_PERIOD"
    AMBIGUOUS_FILTER = "AMBIGUOUS_FILTER"
    MISSING_SCOPE = "MISSING_SCOPE"


MAX_QUERY_RANGE_DAYS = 366
MAX_LIST_LIMIT = 100
MAX_GROUP_LIMIT = 100
MAX_RANK_LIMIT = 50
MAX_QUERY_OFFSET = 10_000
