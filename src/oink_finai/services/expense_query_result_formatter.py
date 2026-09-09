from abc import ABC, abstractmethod

from oink_finai.schemas.expense_query_messages import (
    ExpenseQueryFormattedMessages,
    ExpenseQueryFormattingContext,
)
from oink_finai.schemas.expense_query_result import ExpenseQueryResult


class ExpenseQueryResultFormatter(ABC):
    @abstractmethod
    def format(
        self,
        result: ExpenseQueryResult,
        *,
        context: ExpenseQueryFormattingContext,
    ) -> ExpenseQueryFormattedMessages:
        """Turn an already-scoped result into transport-ready messages."""
