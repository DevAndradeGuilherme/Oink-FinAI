from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID


class ExpenseCommandType(StrEnum):
    REMOVE = "REMOVE"
    CONFIRM_REMOVE = "CONFIRM_REMOVE"
    CANCEL = "CANCEL"
    EDIT = "EDIT"
    INVALID = "INVALID"


@dataclass(frozen=True)
class ExpenseCommand:
    type: ExpenseCommandType
    expense_id: UUID | None = None


COMMANDS_WITH_UUID = {
    "remover": ExpenseCommandType.REMOVE,
    "confirmar-remocao": ExpenseCommandType.CONFIRM_REMOVE,
    "editar": ExpenseCommandType.EDIT,
}


def parse_expense_command(text: str) -> ExpenseCommand | None:
    """Parse explicit commands; malformed reserved commands never reach the interpreter."""
    parts = text.split()
    if not parts:
        return None
    keyword = parts[0].casefold()
    if keyword == "cancelar":
        command_type = ExpenseCommandType.CANCEL if len(parts) == 1 else ExpenseCommandType.INVALID
        return ExpenseCommand(command_type)
    command_type = COMMANDS_WITH_UUID.get(keyword)
    if command_type is None:
        return None
    if len(parts) != 2:
        return ExpenseCommand(ExpenseCommandType.INVALID)
    try:
        expense_id = UUID(parts[1])
    except ValueError:
        return ExpenseCommand(ExpenseCommandType.INVALID)
    if str(expense_id) != parts[1].lower():
        return ExpenseCommand(ExpenseCommandType.INVALID)
    return ExpenseCommand(command_type, expense_id)
