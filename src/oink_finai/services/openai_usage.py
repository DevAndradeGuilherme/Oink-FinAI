from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from math import isfinite


@dataclass(frozen=True, slots=True)
class OpenAIUsageMetrics:
    input_tokens: int | None = None
    output_tokens: int | None = None
    audio_seconds: Decimal | None = None


_transmission_observer: ContextVar[Callable[[], Awaitable[None]] | None] = ContextVar(
    "openai_transmission_observer", default=None
)
_usage_observer: ContextVar[Callable[[OpenAIUsageMetrics], None] | None] = ContextVar(
    "openai_usage_observer", default=None
)


@contextmanager
def observe_openai_call(
    transmission_observer: Callable[[], Awaitable[None]],
    usage_observer: Callable[[OpenAIUsageMetrics], None],
) -> Iterator[None]:
    transmission_token = _transmission_observer.set(transmission_observer)
    usage_token = _usage_observer.set(usage_observer)
    try:
        yield
    finally:
        _usage_observer.reset(usage_token)
        _transmission_observer.reset(transmission_token)


async def mark_openai_request_transmitted() -> None:
    observer = _transmission_observer.get()
    if observer is not None:
        await observer()


def record_openai_response_usage(response: object) -> None:
    observer = _usage_observer.get()
    if observer is None:
        return
    usage = getattr(response, "usage", None)
    input_tokens = _non_negative_int(
        getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", None))
    )
    output_tokens = _non_negative_int(
        getattr(usage, "output_tokens", getattr(usage, "completion_tokens", None))
    )
    audio_seconds = _non_negative_decimal(
        getattr(usage, "audio_seconds", getattr(response, "duration", None))
    )
    observer(OpenAIUsageMetrics(input_tokens, output_tokens, audio_seconds))


def _non_negative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2_147_483_647:
        return None
    return value


def _non_negative_decimal(value: object) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return None
    if isinstance(value, float) and not isfinite(value):
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return result if result.is_finite() and 0 <= result <= Decimal("999999999.999") else None
