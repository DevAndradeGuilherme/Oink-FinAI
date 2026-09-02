from enum import StrEnum


class ConversationStatus(StrEnum):
    IDLE = "IDLE"
    EDITING_EXPENSE = "EDITING_EXPENSE"
    REMOVING_EXPENSE = "REMOVING_EXPENSE"


class ExpenseHistoryAction(StrEnum):
    CREATED = "CREATED"
    UPDATED = "UPDATED"
    DELETED = "DELETED"


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
