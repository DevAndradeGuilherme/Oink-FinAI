from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GeminiErrorMetadata:
    exception_class: str
    category: str
    duration_ms: int
    http_status: int | None = None
    provider_code: str | None = None
    request_id_present: bool = False


class GeminiInterpreterError(Exception):
    """Base sanitized Gemini interpreter error."""

    def __init__(self, message: str, *, metadata: GeminiErrorMetadata | None = None) -> None:
        super().__init__(message)
        self.metadata = metadata


class GeminiConfigurationError(GeminiInterpreterError):
    pass


class GeminiRequestError(GeminiInterpreterError):
    pass


class GeminiAuthenticationError(GeminiInterpreterError):
    pass


class GeminiPermissionError(GeminiInterpreterError):
    pass


class GeminiModelUnavailableError(GeminiInterpreterError):
    pass


class GeminiRateLimitError(GeminiInterpreterError):
    pass


class GeminiTimeoutError(GeminiInterpreterError):
    pass


class GeminiUnavailableError(GeminiInterpreterError):
    pass


class GeminiEmptyResponseError(GeminiInterpreterError):
    pass


class GeminiSchemaError(GeminiInterpreterError):
    pass
