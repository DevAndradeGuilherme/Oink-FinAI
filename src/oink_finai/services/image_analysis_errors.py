from enum import StrEnum

from oink_finai.services.gemini_errors import GeminiErrorMetadata


class ImageAnalysisErrorCode(StrEnum):
    TIMEOUT = "IMAGE_ANALYSIS_TIMEOUT"
    QUOTA_EXCEEDED = "IMAGE_ANALYSIS_QUOTA_EXCEEDED"
    UNAVAILABLE = "IMAGE_ANALYSIS_UNAVAILABLE"
    CONFIGURATION = "IMAGE_ANALYSIS_CONFIGURATION_ERROR"
    AUTHENTICATION = "IMAGE_ANALYSIS_AUTHENTICATION_ERROR"
    MODEL_UNAVAILABLE = "IMAGE_ANALYSIS_MODEL_UNAVAILABLE"
    INVALID_RESPONSE = "IMAGE_ANALYSIS_INVALID_RESPONSE"
    GROUNDING = "IMAGE_ANALYSIS_GROUNDING_ERROR"
    TOO_MUCH_TEXT = "IMAGE_ANALYSIS_TOO_MUCH_TEXT"
    UNSUPPORTED_INPUT = "IMAGE_ANALYSIS_UNSUPPORTED_INPUT"


class ImageAnalysisError(Exception):
    def __init__(
        self,
        code: ImageAnalysisErrorCode,
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
            f"ImageAnalysisError(code={self.code.value!r}, transient={self.transient!r}, "
            f"metadata={self.metadata!r})"
        )
