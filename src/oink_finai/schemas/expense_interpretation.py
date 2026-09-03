from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from oink_finai.domain.enums import ExpenseCategory, ExpenseIntent, PaymentMethod


class GeminiExpenseTransport(BaseModel):
    """Transport DTO parsed before deterministic domain validation."""

    model_config = ConfigDict(extra="forbid")

    intent: ExpenseIntent
    amount: str | None
    amount_evidence: str | None
    description: str | None
    merchant: str | None
    category: ExpenseCategory
    payment_method: PaymentMethod | None
    expense_date: date | None
    confidence: float = Field(ge=0, le=1)
    missing_fields: list[str]
    reasoning_summary: str


class ExpenseInterpretation(BaseModel):
    """Validated application result. Monetary values use Decimal."""

    model_config = ConfigDict(extra="forbid")

    intent: ExpenseIntent
    amount: Decimal | None
    amount_evidence: str | None
    description: str | None
    merchant: str | None
    category: ExpenseCategory
    payment_method: PaymentMethod | None
    expense_date: date | None
    confidence: float = Field(ge=0, le=1)
    missing_fields: list[str]
    reasoning_summary: str


GEMINI_EXPENSE_TRANSPORT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": [value.value for value in ExpenseIntent]},
        "amount": {"type": ["string", "null"]},
        "amount_evidence": {"type": ["string", "null"]},
        "description": {"type": ["string", "null"]},
        "merchant": {"type": ["string", "null"]},
        "category": {"type": "string", "enum": [value.value for value in ExpenseCategory]},
        "payment_method": {
            "type": ["string", "null"],
            "enum": [value.value for value in PaymentMethod] + [None],
        },
        "expense_date": {"type": ["string", "null"]},
        "confidence": {"type": "number"},
        "missing_fields": {"type": "array", "items": {"type": "string"}},
        "reasoning_summary": {"type": "string"},
    },
    "required": [
        "intent",
        "amount",
        "amount_evidence",
        "description",
        "merchant",
        "category",
        "payment_method",
        "expense_date",
        "confidence",
        "missing_fields",
        "reasoning_summary",
    ],
    "additionalProperties": False,
}
