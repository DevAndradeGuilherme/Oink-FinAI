from abc import ABC, abstractmethod
from datetime import datetime

from oink_finai.schemas.expense_query import ExpenseQueryPlan


class ExpenseQueryInterpreter(ABC):
    @abstractmethod
    async def interpret(self, message: str, *, reference_timestamp: datetime) -> ExpenseQueryPlan:
        """Build a safe query plan without persistence or external side effects."""
