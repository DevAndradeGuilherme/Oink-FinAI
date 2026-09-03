from enum import StrEnum


class ConversationStatus(StrEnum):
    IDLE = "IDLE"
    EDITING_EXPENSE = "EDITING_EXPENSE"
    REMOVING_EXPENSE = "REMOVING_EXPENSE"
    WAITING_EXPENSE_DELETE_CONFIRM = "WAITING_EXPENSE_DELETE_CONFIRM"


class ExpenseHistoryAction(StrEnum):
    CREATED = "CREATED"
    UPDATED = "UPDATED"
    DELETED = "DELETED"
    DELETE = "DELETE"


class ExpenseIntent(StrEnum):
    CREATE_EXPENSE = "CREATE_EXPENSE"
    NOT_EXPENSE = "NOT_EXPENSE"
    UNCLEAR = "UNCLEAR"


class ExpenseCategory(StrEnum):
    FOOD = "Alimentação"
    TRANSPORT = "Transporte"
    HOUSING = "Moradia"
    HEALTH = "Saúde"
    EDUCATION = "Educação"
    LEISURE = "Lazer"
    SHOPPING = "Compras"
    SUBSCRIPTIONS = "Assinaturas"
    BILLS = "Contas"
    TAXES = "Impostos"
    WORK = "Trabalho"
    TRAVEL = "Viagem"
    OTHER = "Outros"


class PaymentMethod(StrEnum):
    PIX = "Pix"
    CASH = "Dinheiro"
    CREDIT = "Crédito"
    DEBIT = "Débito"
    TRANSFER = "Transferência"
    BOLETO = "Boleto"
    OTHER = "Outro"


class ProcessedMessageStatus(StrEnum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    PROCESSED = "PROCESSED"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    NOT_EXPENSE = "NOT_EXPENSE"
    FAILED = "FAILED"


class MessageSourceType(StrEnum):
    TEXT = "TEXT"
    AUDIO = "AUDIO"


class OutboundMessageStatus(StrEnum):
    PENDING = "PENDING"
    CLAIMED = "CLAIMED"
    SENDING = "SENDING"
    SENT = "SENT"
    UNKNOWN = "UNKNOWN"
    FAILED = "FAILED"


class OutboundMessageKind(StrEnum):
    EXPENSE_CONFIRMATION = "EXPENSE_CONFIRMATION"
    CLARIFICATION = "CLARIFICATION"
    PROCESSING_FAILURE = "PROCESSING_FAILURE"
    DELETE_CONFIRMATION_REQUEST = "DELETE_CONFIRMATION_REQUEST"
    EXPENSE_DELETED = "EXPENSE_DELETED"
    ACTION_CANCELLED = "ACTION_CANCELLED"
    ACTION_ERROR = "ACTION_ERROR"
