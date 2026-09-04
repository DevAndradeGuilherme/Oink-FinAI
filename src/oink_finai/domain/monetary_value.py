import re
from decimal import Decimal, InvalidOperation
from enum import StrEnum


class MonetaryValueErrorCode(StrEnum):
    EMPTY = "EMPTY"
    NON_NUMERIC = "NON_NUMERIC"
    AMBIGUOUS = "AMBIGUOUS"
    NON_POSITIVE = "NON_POSITIVE"
    SCALE_EXCEEDED = "SCALE_EXCEEDED"
    OUT_OF_RANGE = "OUT_OF_RANGE"


class MonetaryValueError(ValueError):
    def __init__(self, code: MonetaryValueErrorCode) -> None:
        self.code = code
        super().__init__(code.value)

    def __repr__(self) -> str:
        return f"MonetaryValueError(code={self.code.value!r})"


_SUPPORTED_VALUE_PATTERN = re.compile(r"(?:R\$[ \t\u00a0\u202f]*)?(?P<number>[0-9][0-9.,]*)")
_NUMBER_FRAGMENT_PATTERN = re.compile(r"[0-9]+(?:[.,][0-9]+)*")


def parse_monetary_value(value: str, *, maximum: Decimal) -> Decimal:
    """Parse one unambiguous monetary value without dropping unknown characters."""
    if not isinstance(value, str) or not value.strip():
        raise MonetaryValueError(MonetaryValueErrorCode.EMPTY)
    stripped = value.strip()
    if re.fullmatch(r"[+-]?[0-9]+(?:[.,][0-9]+)?[eE][+-]?[0-9]+", stripped):
        raise MonetaryValueError(MonetaryValueErrorCode.NON_NUMERIC)
    if "-" in stripped:
        raise MonetaryValueError(MonetaryValueErrorCode.NON_POSITIVE)

    match = _SUPPORTED_VALUE_PATTERN.fullmatch(stripped)
    if match is None:
        fragments = _NUMBER_FRAGMENT_PATTERN.findall(stripped)
        code = (
            MonetaryValueErrorCode.AMBIGUOUS
            if len(fragments) > 1
            else MonetaryValueErrorCode.NON_NUMERIC
        )
        raise MonetaryValueError(code)

    number = match.group("number")
    normalized = _normalize_number(number)
    try:
        parsed = Decimal(normalized)
    except InvalidOperation:
        raise MonetaryValueError(MonetaryValueErrorCode.NON_NUMERIC) from None
    if not parsed.is_finite():
        raise MonetaryValueError(MonetaryValueErrorCode.NON_NUMERIC)
    if parsed <= 0:
        raise MonetaryValueError(MonetaryValueErrorCode.NON_POSITIVE)
    if parsed > maximum:
        raise MonetaryValueError(MonetaryValueErrorCode.OUT_OF_RANGE)
    return parsed


def _normalize_number(number: str) -> str:
    comma_count = number.count(",")
    dot_count = number.count(".")
    if comma_count and dot_count:
        if not re.fullmatch(r"[0-9]{1,3}(?:\.[0-9]{3})+,[0-9]{1,2}", number):
            raise MonetaryValueError(MonetaryValueErrorCode.AMBIGUOUS)
        integer, fraction = number.split(",")
        return integer.replace(".", "") + "." + fraction

    if comma_count:
        if comma_count != 1:
            raise MonetaryValueError(MonetaryValueErrorCode.AMBIGUOUS)
        integer, fraction = number.split(",")
        if not integer or not fraction:
            raise MonetaryValueError(MonetaryValueErrorCode.AMBIGUOUS)
        if len(fraction) > 2:
            raise MonetaryValueError(MonetaryValueErrorCode.SCALE_EXCEEDED)
        return integer + "." + fraction

    if dot_count:
        if dot_count != 1:
            raise MonetaryValueError(MonetaryValueErrorCode.AMBIGUOUS)
        integer, fraction = number.split(".")
        if not integer or not fraction:
            raise MonetaryValueError(MonetaryValueErrorCode.AMBIGUOUS)
        if len(fraction) > 2:
            code = (
                MonetaryValueErrorCode.AMBIGUOUS
                if len(fraction) == 3 and len(integer) <= 3
                else MonetaryValueErrorCode.SCALE_EXCEEDED
            )
            raise MonetaryValueError(code)
        return number

    return number
