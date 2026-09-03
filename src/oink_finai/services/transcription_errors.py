from enum import StrEnum

from oink_finai.services.gemini_errors import GeminiErrorMetadata


class TranscriptionErrorCode(StrEnum):
    CONFIGURATION = "TRANSCRIPTION_CONFIGURATION_ERROR"
    AUTHENTICATION = "TRANSCRIPTION_AUTHENTICATION_ERROR"
    MODEL_UNAVAILABLE = "TRANSCRIPTION_MODEL_UNAVAILABLE"
    QUOTA_EXCEEDED = "TRANSCRIPTION_QUOTA_EXCEEDED"
    TIMEOUT = "TRANSCRIPTION_TIMEOUT"
    UNAVAILABLE = "TRANSCRIPTION_UNAVAILABLE"
    INVALID_RESPONSE = "TRANSCRIPTION_INVALID_RESPONSE"
    NO_SPEECH = "TRANSCRIPTION_NO_SPEECH"
    TOO_LONG = "TRANSCRIPTION_TOO_LONG"


class TranscriptionError(Exception):
    def __init__(
        self,
        code: TranscriptionErrorCode,
        *,
        transient: bool,
        metadata: GeminiErrorMetadata | None = None,
    ) -> None:
        self.code = code
        self.transient = transient
        self.metadata = metadata
        super().__init__(code.value)

    def __repr__(self) -> str:
        return (
            f"TranscriptionError(code={self.code.value!r}, transient={self.transient!r}, "
            f"metadata={self.metadata!r})"
        )


class NoSpeechError(TranscriptionError):
    def __init__(self) -> None:
        super().__init__(TranscriptionErrorCode.NO_SPEECH, transient=False)
