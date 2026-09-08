from datetime import date
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExpenseQueryFormattingContext(BaseModel):
    """Trusted, immutable temporal context used only while formatting query results."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    reference_date: date
    timezone: str

    @model_validator(mode="after")
    def validate_timezone(self) -> "ExpenseQueryFormattingContext":
        try:
            ZoneInfo(self.timezone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError("formatting timezone is invalid") from exc
        return self


class ExpenseQueryFormattedMessages(BaseModel):
    """Transport-ready messages. This schema never sends them."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    messages: tuple[str, ...] = Field(min_length=1)
    truncated: bool
    total_items: int = Field(ge=0)
    displayed_items: int = Field(ge=0)
