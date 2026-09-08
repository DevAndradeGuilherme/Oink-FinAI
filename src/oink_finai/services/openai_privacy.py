import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_PRIVATE_OPERATION: ContextVar[bool] = ContextVar("openai_private_operation", default=False)
_SDK_LOGGERS = (
    "openai",
    "openai._base_client",
    "openai._response",
    "openai._legacy_response",
    "openai.audio.transcriptions",
    "openai.resources.responses",
    "httpx",
    "httpcore.connection",
    "httpcore.http11",
    "httpcore.http2",
    "httpcore.proxy",
    "httpcore.socks",
)


class _PrivateOperationFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not _PRIVATE_OPERATION.get()


_PRIVATE_FILTER = _PrivateOperationFilter()


@contextmanager
def openai_private_operation() -> Iterator[None]:
    """Suppress SDK/transport records only in the task carrying private provider data."""

    for name in _SDK_LOGGERS:
        logging.getLogger(name).addFilter(_PRIVATE_FILTER)
    token = _PRIVATE_OPERATION.set(True)
    try:
        yield
    finally:
        _PRIVATE_OPERATION.reset(token)
