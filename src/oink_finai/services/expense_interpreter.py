from abc import ABC, abstractmethod
from datetime import datetime

from oink_finai.schemas.expense_interpretation import ExpenseInterpretation


class ExpenseInterpreter(ABC):
    @abstractmethod
    async def interpret(
        self, message: str, *, reference_timestamp: datetime
    ) -> ExpenseInterpretation:
        """Interpret one message without causing persistence or messaging side effects."""
