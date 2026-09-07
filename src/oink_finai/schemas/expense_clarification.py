import json
from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from oink_finai.domain.enums import (
    ExpenseCategory,
    ExpenseClarificationField,
    MessageSourceType,
    PaymentMethod,
)
from oink_finai.domain.expense_limits import (
    EXPENSE_AMOUNT_MAX,
    EXPENSE_DESCRIPTION_MAX_LENGTH,
    EXPENSE_MERCHANT_MAX_LENGTH,
)


class ClarificationReplyBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    revision: int = Field(ge=0)
    deadline: datetime


class ExpenseClarificationContext(BaseModel):
    """Minimal durable draft; it intentionally excludes source content and provider data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = Field(default=1, ge=1, le=1)
    origin_message_id: UUID
    source_type: MessageSourceType
    reference_timestamp: datetime
    requested_field: ExpenseClarificationField
    remaining_fields: tuple[ExpenseClarificationField, ...]
    revision: int = Field(default=0, ge=0)
    reply_bindings: dict[UUID, ClarificationReplyBinding] = Field(default_factory=dict)
    amount: Decimal | None = Field(default=None, gt=0, le=EXPENSE_AMOUNT_MAX)
    description: str | None = Field(default=None, max_length=EXPENSE_DESCRIPTION_MAX_LENGTH)
    merchant: str | None = Field(default=None, max_length=EXPENSE_MERCHANT_MAX_LENGTH)
    category: ExpenseCategory | None = None
    payment_method: PaymentMethod | None = None
    expense_date: date | None = None

    @field_validator("reference_timestamp")
    @classmethod
    def require_aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("reference timestamp must be timezone-aware")
        return value

    @field_validator("description", "merchant")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def validate_pending_fields(self) -> "ExpenseClarificationContext":
        if not self.remaining_fields or self.remaining_fields[0] is not self.requested_field:
            raise ValueError("requested field must be the first remaining field")
        if len(set(self.remaining_fields)) != len(self.remaining_fields):
            raise ValueError("remaining fields must be unique")
        return self

    def payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude_none=True)

    def same_draft(self, other: "ExpenseClarificationContext") -> bool:
        return self.model_dump(exclude={"reply_bindings"}) == other.model_dump(
            exclude={"reply_bindings"}
        )

    def interpreter_input(self, answer: str) -> str:
        known_fields = self.model_dump(
            mode="json",
            include={
                "amount",
                "description",
                "merchant",
                "category",
                "payment_method",
                "expense_date",
            },
            exclude_none=True,
        )
        payload = {
            "known_expense_fields": known_fields,
            "requested_field": self.requested_field.value,
            "user_answer": answer,
        }
        return "OINK_EXPENSE_CLARIFICATION_V1\n" + json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
