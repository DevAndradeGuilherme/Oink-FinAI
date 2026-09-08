from pydantic import BaseModel, ConfigDict, Field


class ExpenseQueryFormattedMessages(BaseModel):
    """Transport-ready messages. This schema never sends them."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    messages: tuple[str, ...] = Field(min_length=1)
    truncated: bool
    total_items: int = Field(ge=0)
    displayed_items: int = Field(ge=0)
