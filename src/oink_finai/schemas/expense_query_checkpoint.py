import json
from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict

from oink_finai.domain.enums import ExpenseCategory, ExpenseIntent, PaymentMethod
from oink_finai.schemas.expense_interpretation import ExpenseInterpretation
from oink_finai.schemas.expense_query import ExpenseQueryPlan
from oink_finai.schemas.expense_query_result import ExpenseQueryResult


class ExpenseClassificationCheckpoint(BaseModel):
    """Minimal validated classification needed to resume processing."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal[1] = 1
    intent: ExpenseIntent
    amount: str | None
    description: str | None
    merchant: str | None
    category: ExpenseCategory | None
    payment_method: PaymentMethod | None
    expense_date: date | None
    missing_fields: tuple[Literal["amount", "description"], ...]

    @classmethod
    def from_interpretation(
        cls, interpretation: ExpenseInterpretation
    ) -> "ExpenseClassificationCheckpoint":
        return cls(
            intent=interpretation.intent,
            amount=str(interpretation.amount) if interpretation.amount is not None else None,
            description=interpretation.description,
            merchant=interpretation.merchant,
            category=interpretation.category,
            payment_method=interpretation.payment_method,
            expense_date=interpretation.expense_date,
            missing_fields=tuple(interpretation.missing_fields),
        )

    def to_interpretation(self) -> ExpenseInterpretation:
        return ExpenseInterpretation(
            intent=self.intent,
            amount=Decimal(self.amount) if self.amount is not None else None,
            amount_evidence=None,
            description=self.description,
            merchant=self.merchant,
            category=self.category,
            payment_method=self.payment_method,
            expense_date=self.expense_date,
            confidence=1,
            missing_fields=list(self.missing_fields),
            reasoning_summary="checkpoint",
        )

    def payload(self) -> dict[str, object]:
        return self.model_dump(mode="json")

    @classmethod
    def from_payload(cls, payload: object) -> "ExpenseClassificationCheckpoint":
        return cls.model_validate_json(json.dumps(payload))


class ExpenseQueryPlanCheckpoint(BaseModel):
    """Versioned plan checkpoint; identity and executable SQL are structurally impossible."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal[1] = 1
    plan: ExpenseQueryPlan

    def payload(self) -> dict[str, object]:
        return self.model_dump(mode="json")

    @classmethod
    def from_payload(cls, payload: object) -> "ExpenseQueryPlanCheckpoint":
        return cls.model_validate_json(json.dumps(payload))


class ExpenseQueryResultCheckpoint(BaseModel):
    """Temporary typed result used only until final outbox pages exist."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal[1] = 1
    result: ExpenseQueryResult

    def payload(self) -> dict[str, object]:
        return self.model_dump(mode="json")

    @classmethod
    def from_payload(cls, payload: object) -> "ExpenseQueryResultCheckpoint":
        return cls.model_validate_json(json.dumps(payload))
