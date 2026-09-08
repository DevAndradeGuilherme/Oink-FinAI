from enum import StrEnum

from oink_finai.services.ai_error_metadata import AIErrorMetadata


class InterpretationErrorCode(StrEnum):
    # Stable durable codes are retained for compatibility with persisted rows and monitoring.
    CONFIGURATION = "GEMINI_CONFIGURATION"
    INVALID_REQUEST = "GEMINI_REQUEST"
    AUTHENTICATION = "GEMINI_AUTHENTICATION"
    PERMISSION = "GEMINI_PERMISSION"
    MODEL_UNAVAILABLE = "GEMINI_MODEL_UNAVAILABLE"
    RATE_LIMIT = "GEMINI_RATE_LIMIT"
    TIMEOUT = "GEMINI_TIMEOUT"
    UNAVAILABLE = "GEMINI_UNAVAILABLE"
    EMPTY_RESPONSE = "GEMINI_ERROR"
    INVALID_RESPONSE = "GEMINI_SCHEMA_INVALID"


class InterpretationError(Exception):
    """Provider-neutral, sanitized interpretation failure."""

    def __init__(
        self,
        code: InterpretationErrorCode,
        *,
        transient: bool,
        metadata: AIErrorMetadata | None = None,
    ) -> None:
        self.code = code
        self.transient = transient
        self.metadata = metadata
        super().__init__(code.value)

    def __repr__(self) -> str:
        return (
            f"InterpretationError(code={self.code.value!r}, transient={self.transient!r}, "
            f"metadata={self.metadata!r})"
        )


class InterpretationConfigurationError(InterpretationError):
    def __init__(
        self, _message: str | None = None, *, metadata: AIErrorMetadata | None = None
    ) -> None:
        super().__init__(InterpretationErrorCode.CONFIGURATION, transient=False, metadata=metadata)


class InterpretationRequestError(InterpretationError):
    def __init__(
        self, _message: str | None = None, *, metadata: AIErrorMetadata | None = None
    ) -> None:
        super().__init__(
            InterpretationErrorCode.INVALID_REQUEST, transient=False, metadata=metadata
        )


class InterpretationAuthenticationError(InterpretationError):
    def __init__(
        self, _message: str | None = None, *, metadata: AIErrorMetadata | None = None
    ) -> None:
        super().__init__(InterpretationErrorCode.AUTHENTICATION, transient=False, metadata=metadata)


class InterpretationPermissionError(InterpretationError):
    def __init__(
        self, _message: str | None = None, *, metadata: AIErrorMetadata | None = None
    ) -> None:
        super().__init__(InterpretationErrorCode.PERMISSION, transient=False, metadata=metadata)


class InterpretationModelUnavailableError(InterpretationError):
    def __init__(
        self, _message: str | None = None, *, metadata: AIErrorMetadata | None = None
    ) -> None:
        super().__init__(
            InterpretationErrorCode.MODEL_UNAVAILABLE, transient=False, metadata=metadata
        )


class InterpretationRateLimitError(InterpretationError):
    def __init__(
        self, _message: str | None = None, *, metadata: AIErrorMetadata | None = None
    ) -> None:
        super().__init__(InterpretationErrorCode.RATE_LIMIT, transient=True, metadata=metadata)


class InterpretationTimeoutError(InterpretationError):
    def __init__(
        self, _message: str | None = None, *, metadata: AIErrorMetadata | None = None
    ) -> None:
        super().__init__(InterpretationErrorCode.TIMEOUT, transient=True, metadata=metadata)


class InterpretationUnavailableError(InterpretationError):
    def __init__(
        self, _message: str | None = None, *, metadata: AIErrorMetadata | None = None
    ) -> None:
        super().__init__(InterpretationErrorCode.UNAVAILABLE, transient=True, metadata=metadata)


class InterpretationEmptyResponseError(InterpretationError):
    def __init__(
        self, _message: str | None = None, *, metadata: AIErrorMetadata | None = None
    ) -> None:
        super().__init__(InterpretationErrorCode.EMPTY_RESPONSE, transient=False, metadata=metadata)


class InterpretationInvalidResponseError(InterpretationError):
    def __init__(
        self, _message: str | None = None, *, metadata: AIErrorMetadata | None = None
    ) -> None:
        super().__init__(
            InterpretationErrorCode.INVALID_RESPONSE, transient=False, metadata=metadata
        )
