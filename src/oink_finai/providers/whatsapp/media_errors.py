from enum import StrEnum


class MediaErrorCode(StrEnum):
    CONFIGURATION = "MEDIA_CONFIGURATION_ERROR"
    AUTHENTICATION = "MEDIA_AUTHENTICATION_ERROR"
    NOT_FOUND = "MEDIA_NOT_FOUND"
    UNSUPPORTED_TYPE = "MEDIA_UNSUPPORTED_TYPE"
    TOO_LARGE = "MEDIA_TOO_LARGE"
    TOO_LONG = "MEDIA_TOO_LONG"
    INVALID_BASE64 = "MEDIA_INVALID_BASE64"
    CONTENT_MISMATCH = "MEDIA_CONTENT_MISMATCH"
    TIMEOUT = "MEDIA_TIMEOUT"
    UNAVAILABLE = "MEDIA_UNAVAILABLE"


class MediaError(Exception):
    """Sanitized media failure safe for logs and durable error classification."""

    def __init__(self, code: MediaErrorCode, *, transient: bool) -> None:
        self.code = code
        self.transient = transient
        super().__init__(code.value)

    def __repr__(self) -> str:
        return f"MediaError(code={self.code.value!r}, transient={self.transient!r})"
