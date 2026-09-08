from abc import ABC, abstractmethod
from uuid import UUID

from oink_finai.schemas.expense_query import ExpenseQueryPlan
from oink_finai.schemas.expense_query_result import ExpenseQueryResult


class ExpenseQueryExecutionError(Exception):
    """Base error for plans that cannot be safely executed."""


class InvalidExpenseQueryPlanError(ExpenseQueryExecutionError):
    pass


class UnsupportedExpenseQueryIntentError(ExpenseQueryExecutionError):
    pass


class ExpenseQueryExecutor(ABC):
    @abstractmethod
    async def execute(self, *, user_id: UUID, plan: ExpenseQueryPlan) -> ExpenseQueryResult:
        """Execute one validated plan inside the trusted user scope."""
