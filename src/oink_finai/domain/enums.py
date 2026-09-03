from enum import StrEnum


class ConversationStatus(StrEnum):
    IDLE = "IDLE"
    EDITING_EXPENSE = "EDITING_EXPENSE"
    REMOVING_EXPENSE = "REMOVING_EXPENSE"


class ExpenseHistoryAction(StrEnum):
    CREATED = "CREATED"
    UPDATED = "UPDATED"
    DELETED = "DELETED"


class MessageType(StrEnum):
    TEXT = "text"
    AUDIO = "audio"


class MediaType(StrEnum):
    AUDIO = "audio"
