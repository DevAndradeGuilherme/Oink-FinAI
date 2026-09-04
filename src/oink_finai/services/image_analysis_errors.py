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


class GroundingFailureReason(StrEnum):
    AMOUNT_VALUE_EMPTY = "AMOUNT_VALUE_EMPTY"
    AMOUNT_VALUE_NON_NUMERIC = "AMOUNT_VALUE_NON_NUMERIC"
    AMOUNT_VALUE_AMBIGUOUS = "AMOUNT_VALUE_AMBIGUOUS"
    AMOUNT_VALUE_NON_POSITIVE = "AMOUNT_VALUE_NON_POSITIVE"
    AMOUNT_VALUE_SCALE_EXCEEDED = "AMOUNT_VALUE_SCALE_EXCEEDED"
    AMOUNT_VALUE_OUT_OF_RANGE = "AMOUNT_VALUE_OUT_OF_RANGE"
    AMOUNT_EVIDENCE_INVALID = "AMOUNT_EVIDENCE_INVALID"
    AMOUNT_EVIDENCE_NOT_FOUND = "AMOUNT_EVIDENCE_NOT_FOUND"
    AMOUNT_VALUE_MISMATCH = "AMOUNT_VALUE_MISMATCH"
    AMOUNT_PARTIAL_TOKEN = "AMOUNT_PARTIAL_TOKEN"
    AMOUNT_LABEL_INVALID = "AMOUNT_LABEL_INVALID"
    DATE_VALUE_INVALID = "DATE_VALUE_INVALID"
    DATE_EVIDENCE_INVALID = "DATE_EVIDENCE_INVALID"
    DATE_EVIDENCE_NOT_FOUND = "DATE_EVIDENCE_NOT_FOUND"
    DATE_LABEL_INVALID = "DATE_LABEL_INVALID"
    MERCHANT_VALUE_INVALID = "MERCHANT_VALUE_INVALID"
    MERCHANT_EVIDENCE_INVALID = "MERCHANT_EVIDENCE_INVALID"
    MERCHANT_EVIDENCE_NOT_FOUND = "MERCHANT_EVIDENCE_NOT_FOUND"
    PAYMENT_METHOD_VALUE_INVALID = "PAYMENT_METHOD_VALUE_INVALID"
    PAYMENT_METHOD_EVIDENCE_INVALID = "PAYMENT_METHOD_EVIDENCE_INVALID"
    PAYMENT_METHOD_EVIDENCE_NOT_FOUND = "PAYMENT_METHOD_EVIDENCE_NOT_FOUND"
    ILLEGIBLE_WITH_CANDIDATES = "ILLEGIBLE_WITH_CANDIDATES"
    NON_FINANCIAL_WITH_CANDIDATES = "NON_FINANCIAL_WITH_CANDIDATES"
    DUPLICATE_CANDIDATE = "DUPLICATE_CANDIDATE"
    CONTRADICTORY_RESULT = "CONTRADICTORY_RESULT"


class GroundingCandidateKind(StrEnum):
    AMOUNT = "AMOUNT"
    DATE = "DATE"
    MERCHANT = "MERCHANT"
    PAYMENT_METHOD = "PAYMENT_METHOD"


class ImageAnalysisError(Exception):
    def __init__(
        self,
        code: ImageAnalysisErrorCode,
        *,
        transient: bool,
        metadata: GeminiErrorMetadata | None = None,
        grounding_reason: GroundingFailureReason | None = None,
        candidate_kind: GroundingCandidateKind | None = None,
        candidate_index: int | None = None,
    ) -> None:
        self.code = code
        self.transient = transient
        self.metadata = metadata
        self.grounding_reason = grounding_reason
        self.candidate_kind = candidate_kind
        self.candidate_index = candidate_index
        super().__init__(code.value)

    def __repr__(self) -> str:
        return (
            f"ImageAnalysisError(code={self.code.value!r}, transient={self.transient!r}, "
            f"metadata={self.metadata!r}, grounding_reason={self.grounding_reason!r}, "
            f"candidate_kind={self.candidate_kind!r}, candidate_index={self.candidate_index!r})"
        )
