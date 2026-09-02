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
ACTION_ID_PREFIX = "oink:v1"
ACTION_CODES = {
    ExpenseCommandType.EDIT: "e",
    ExpenseCommandType.REMOVE: "r",
    ExpenseCommandType.CONFIRM_REMOVE: "rc",
    ExpenseCommandType.CANCEL: "c",
}
ACTION_TYPES = {code: command_type for command_type, code in ACTION_CODES.items()}


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


def encode_expense_action(command: ExpenseCommand) -> str:
    code = ACTION_CODES.get(command.type)
    if code is None:
        raise ValueError("unsupported interactive action")
    if command.type is ExpenseCommandType.CANCEL:
        if command.expense_id is not None:
            raise ValueError("cancel action must not contain an expense id")
        return f"{ACTION_ID_PREFIX}:{code}"
    if command.expense_id is None:
        raise ValueError("interactive action requires an expense id")
    return f"{ACTION_ID_PREFIX}:{code}:{command.expense_id}"


def parse_expense_action(action_id: str) -> ExpenseCommand | None:
    parts = action_id.split(":")
    if len(parts) not in {3, 4} or parts[:2] != ["oink", "v1"]:
        return None
    command_type = ACTION_TYPES.get(parts[2])
    if command_type is None:
        return None
    if command_type is ExpenseCommandType.CANCEL:
        return ExpenseCommand(command_type) if len(parts) == 3 else None
    if len(parts) != 4:
        return None
    try:
        expense_id = UUID(parts[3])
    except ValueError:
        return None
    if str(expense_id) != parts[3].lower():
        return None
    return ExpenseCommand(command_type, expense_id)


def expense_command_text(command: ExpenseCommand) -> str:
    if command.type is ExpenseCommandType.CANCEL:
        return "cancelar"
    if command.expense_id is None:
        raise ValueError("command requires an expense id")
    keywords = {
        ExpenseCommandType.REMOVE: "remover",
        ExpenseCommandType.CONFIRM_REMOVE: "confirmar-remocao",
        ExpenseCommandType.EDIT: "editar",
    }
    keyword = keywords.get(command.type)
    if keyword is None:
        raise ValueError("unsupported command")
    return f"{keyword} {command.expense_id}"
