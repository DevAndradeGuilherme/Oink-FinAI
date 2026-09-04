from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, model_validator

from oink_finai.domain.image_analysis_limits import (
    IMAGE_ANALYSIS_CANDIDATE_VALUE_MAX_LENGTH,
    IMAGE_ANALYSIS_CAPTION_MAX_LENGTH,
    IMAGE_ANALYSIS_EVIDENCE_MAX_LENGTH,
    IMAGE_ANALYSIS_LABEL_MAX_LENGTH,
    IMAGE_ANALYSIS_MAX_AMOUNT_CANDIDATES,
    IMAGE_ANALYSIS_MAX_DATE_CANDIDATES,
    IMAGE_ANALYSIS_MAX_MERCHANT_CANDIDATES,
    IMAGE_ANALYSIS_MAX_PAYMENT_METHOD_CANDIDATES,
    IMAGE_ANALYSIS_VISIBLE_TEXT_MAX_LENGTH,
)


class ImageDocumentType(StrEnum):
    RECEIPT = "RECEIPT"
    INVOICE = "INVOICE"
    PAYMENT_RECEIPT = "PAYMENT_RECEIPT"
    BANK_TRANSFER = "BANK_TRANSFER"
    CARD_RECEIPT = "CARD_RECEIPT"
    SCREENSHOT = "SCREENSHOT"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


class ImageAnalysisWarning(StrEnum):
    BLURRED = "BLURRED"
    CROPPED = "CROPPED"
    LOW_RESOLUTION = "LOW_RESOLUTION"
    MULTIPLE_AMOUNTS = "MULTIPLE_AMOUNTS"
    MULTIPLE_DATES = "MULTIPLE_DATES"
    POSSIBLE_DUPLICATE_DOCUMENT = "POSSIBLE_DUPLICATE_DOCUMENT"
    SENSITIVE_DATA_PRESENT = "SENSITIVE_DATA_PRESENT"
    INCOMPLETE_DOCUMENT = "INCOMPLETE_DOCUMENT"
    UNSUPPORTED_CONTENT = "UNSUPPORTED_CONTENT"
    NONE = "NONE"


class GeminiAmountCandidateTransport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str = Field(max_length=IMAGE_ANALYSIS_CANDIDATE_VALUE_MAX_LENGTH)
    evidence: str = Field(max_length=IMAGE_ANALYSIS_EVIDENCE_MAX_LENGTH)
    label: str = Field(max_length=IMAGE_ANALYSIS_LABEL_MAX_LENGTH)


class GeminiDateCandidateTransport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str = Field(max_length=IMAGE_ANALYSIS_CANDIDATE_VALUE_MAX_LENGTH)
    evidence: str = Field(max_length=IMAGE_ANALYSIS_EVIDENCE_MAX_LENGTH)
    label: str = Field(max_length=IMAGE_ANALYSIS_LABEL_MAX_LENGTH)


class GeminiEvidenceCandidateTransport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str = Field(max_length=IMAGE_ANALYSIS_CANDIDATE_VALUE_MAX_LENGTH)
    evidence: str = Field(max_length=IMAGE_ANALYSIS_EVIDENCE_MAX_LENGTH)


class GeminiImageAnalysisTransport(BaseModel):
    """Gemini-only DTO. Absence uses empty arrays and visible text may be empty."""

    model_config = ConfigDict(extra="forbid")

    document_type: ImageDocumentType
    visible_text: str
    amount_candidates: list[GeminiAmountCandidateTransport] = Field(
        max_length=IMAGE_ANALYSIS_MAX_AMOUNT_CANDIDATES
    )
    date_candidates: list[GeminiDateCandidateTransport] = Field(
        max_length=IMAGE_ANALYSIS_MAX_DATE_CANDIDATES
    )
    merchant_candidates: list[GeminiEvidenceCandidateTransport] = Field(
        max_length=IMAGE_ANALYSIS_MAX_MERCHANT_CANDIDATES
    )
    payment_method_candidates: list[GeminiEvidenceCandidateTransport] = Field(
        max_length=IMAGE_ANALYSIS_MAX_PAYMENT_METHOD_CANDIDATES
    )
    is_financial_document: StrictBool
    is_legible: StrictBool
    confidence: StrictFloat = Field(ge=0, le=1)
    warnings: list[ImageAnalysisWarning] = Field(max_length=len(ImageAnalysisWarning))


class AmountCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    value: Decimal
    evidence: str = Field(max_length=IMAGE_ANALYSIS_EVIDENCE_MAX_LENGTH)
    label: str = Field(max_length=IMAGE_ANALYSIS_LABEL_MAX_LENGTH)


class DateCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    value: str = Field(max_length=IMAGE_ANALYSIS_CANDIDATE_VALUE_MAX_LENGTH)
    evidence: str = Field(max_length=IMAGE_ANALYSIS_EVIDENCE_MAX_LENGTH)
    label: str = Field(max_length=IMAGE_ANALYSIS_LABEL_MAX_LENGTH)


class EvidenceCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    value: str = Field(max_length=IMAGE_ANALYSIS_CANDIDATE_VALUE_MAX_LENGTH)
    evidence: str = Field(max_length=IMAGE_ANALYSIS_EVIDENCE_MAX_LENGTH)


class ImageAnalysis(BaseModel):
    """Provider-neutral observations. This is deliberately not an Expense."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_type: ImageDocumentType
    visible_text: str = Field(max_length=IMAGE_ANALYSIS_VISIBLE_TEXT_MAX_LENGTH)
    amount_candidates: list[AmountCandidate] = Field(
        max_length=IMAGE_ANALYSIS_MAX_AMOUNT_CANDIDATES
    )
    date_candidates: list[DateCandidate] = Field(max_length=IMAGE_ANALYSIS_MAX_DATE_CANDIDATES)
    merchant_candidates: list[EvidenceCandidate] = Field(
        max_length=IMAGE_ANALYSIS_MAX_MERCHANT_CANDIDATES
    )
    payment_method_candidates: list[EvidenceCandidate] = Field(
        max_length=IMAGE_ANALYSIS_MAX_PAYMENT_METHOD_CANDIDATES
    )
    caption: str | None = Field(default=None, max_length=IMAGE_ANALYSIS_CAPTION_MAX_LENGTH)
    is_financial_document: bool
    is_legible: bool
    confidence: float = Field(ge=0, le=1)
    warnings: list[ImageAnalysisWarning] = Field(max_length=len(ImageAnalysisWarning))

    @model_validator(mode="after")
    def validate_coherence(self) -> "ImageAnalysis":
        financial_candidates = (
            self.amount_candidates
            or self.date_candidates
            or self.merchant_candidates
            or self.payment_method_candidates
        )
        if (not self.is_financial_document or not self.is_legible) and financial_candidates:
            raise ValueError("non-financial or illegible image cannot have financial candidates")
        if ImageAnalysisWarning.NONE in self.warnings and len(self.warnings) != 1:
            raise ValueError("NONE warning cannot coexist with another warning")
        if len(set(self.warnings)) != len(self.warnings):
            raise ValueError("warnings must be unique")
        if (
            len(self.amount_candidates) > 1
            and ImageAnalysisWarning.MULTIPLE_AMOUNTS not in self.warnings
        ):
            raise ValueError("multiple amounts require MULTIPLE_AMOUNTS warning")
        if (
            len(self.date_candidates) > 1
            and ImageAnalysisWarning.MULTIPLE_DATES not in self.warnings
        ):
            raise ValueError("multiple dates require MULTIPLE_DATES warning")
        financial_types = {
            ImageDocumentType.RECEIPT,
            ImageDocumentType.INVOICE,
            ImageDocumentType.PAYMENT_RECEIPT,
            ImageDocumentType.BANK_TRANSFER,
            ImageDocumentType.CARD_RECEIPT,
        }
        if self.document_type in financial_types and not self.is_financial_document:
            raise ValueError("financial document type contradicts is_financial_document")
        return self


def _object_schema(properties: dict[str, object]) -> dict[str, object]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


_AMOUNT_CANDIDATE_SCHEMA = _object_schema(
    {"value": {"type": "string"}, "evidence": {"type": "string"}, "label": {"type": "string"}}
)
_DATE_CANDIDATE_SCHEMA = _object_schema(
    {"value": {"type": "string"}, "evidence": {"type": "string"}, "label": {"type": "string"}}
)
_EVIDENCE_CANDIDATE_SCHEMA = _object_schema(
    {"value": {"type": "string"}, "evidence": {"type": "string"}}
)

GEMINI_IMAGE_ANALYSIS_SCHEMA: dict[str, object] = _object_schema(
    {
        "document_type": {"type": "string", "enum": [item.value for item in ImageDocumentType]},
        "visible_text": {"type": "string"},
        "amount_candidates": {"type": "array", "items": _AMOUNT_CANDIDATE_SCHEMA},
        "date_candidates": {"type": "array", "items": _DATE_CANDIDATE_SCHEMA},
        "merchant_candidates": {"type": "array", "items": _EVIDENCE_CANDIDATE_SCHEMA},
        "payment_method_candidates": {
            "type": "array",
            "items": _EVIDENCE_CANDIDATE_SCHEMA,
        },
        "is_financial_document": {"type": "boolean"},
        "is_legible": {"type": "boolean"},
        "confidence": {"type": "number"},
        "warnings": {
            "type": "array",
            "items": {"type": "string", "enum": [item.value for item in ImageAnalysisWarning]},
        },
    }
)
