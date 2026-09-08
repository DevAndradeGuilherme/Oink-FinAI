import json
import unicodedata
from datetime import datetime
from decimal import Decimal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from oink_finai.domain.expense_limits import EXPENSE_AMOUNT_MAX
from oink_finai.domain.image_analysis_limits import (
    IMAGE_ANALYSIS_CANDIDATE_VALUE_MAX_LENGTH,
    IMAGE_ANALYSIS_EVIDENCE_MAX_LENGTH,
    IMAGE_ANALYSIS_LABEL_MAX_LENGTH,
    IMAGE_ANALYSIS_MAX_AMOUNT_CANDIDATES,
    IMAGE_ANALYSIS_MAX_DATE_CANDIDATES,
    IMAGE_ANALYSIS_MAX_MERCHANT_CANDIDATES,
    IMAGE_ANALYSIS_MAX_PAYMENT_METHOD_CANDIDATES,
)
from oink_finai.domain.monetary_value import MonetaryValueError, parse_monetary_value
from oink_finai.schemas.image_analysis import (
    ImageAnalysis,
    ImageAnalysisWarning,
    ImageDocumentType,
)


def _validated_checkpoint_text(value: str) -> str:
    if not value.strip() or any(
        unicodedata.category(character) == "Cc" and character not in "\n\r\t" for character in value
    ):
        raise ValueError("invalid checkpoint text")
    return value


class CheckpointAmountCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    value: str = Field(max_length=IMAGE_ANALYSIS_CANDIDATE_VALUE_MAX_LENGTH)
    evidence: str = Field(min_length=1, max_length=IMAGE_ANALYSIS_EVIDENCE_MAX_LENGTH)
    label: str = Field(min_length=1, max_length=IMAGE_ANALYSIS_LABEL_MAX_LENGTH)

    @field_validator("value")
    @classmethod
    def validate_amount(cls, value: str) -> str:
        try:
            amount = parse_monetary_value(value, maximum=EXPENSE_AMOUNT_MAX)
        except MonetaryValueError as exc:
            raise ValueError("invalid checkpoint amount") from exc
        return format(amount, "f")

    @field_validator("evidence", "label")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _validated_checkpoint_text(value)


class CheckpointDateCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    value: str = Field(min_length=1, max_length=IMAGE_ANALYSIS_CANDIDATE_VALUE_MAX_LENGTH)
    evidence: str = Field(min_length=1, max_length=IMAGE_ANALYSIS_EVIDENCE_MAX_LENGTH)
    label: str = Field(min_length=1, max_length=IMAGE_ANALYSIS_LABEL_MAX_LENGTH)

    @field_validator("value", "evidence", "label")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _validated_checkpoint_text(value)


class CheckpointEvidenceCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    value: str = Field(min_length=1, max_length=IMAGE_ANALYSIS_CANDIDATE_VALUE_MAX_LENGTH)
    evidence: str = Field(min_length=1, max_length=IMAGE_ANALYSIS_EVIDENCE_MAX_LENGTH)

    @field_validator("value", "evidence")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _validated_checkpoint_text(value)


class ImageAnalysisCheckpoint(BaseModel):
    """Minimal durable, already-grounded observations. No image or full OCR text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: StrictInt
    document_type: ImageDocumentType
    is_financial_document: StrictBool
    is_legible: StrictBool
    confidence: StrictFloat = Field(ge=0, le=1)
    warnings: list[ImageAnalysisWarning] = Field(max_length=len(ImageAnalysisWarning))
    amount_candidates: list[CheckpointAmountCandidate] = Field(
        max_length=IMAGE_ANALYSIS_MAX_AMOUNT_CANDIDATES
    )
    date_candidates: list[CheckpointDateCandidate] = Field(
        max_length=IMAGE_ANALYSIS_MAX_DATE_CANDIDATES
    )
    merchant_candidates: list[CheckpointEvidenceCandidate] = Field(
        max_length=IMAGE_ANALYSIS_MAX_MERCHANT_CANDIDATES
    )
    payment_method_candidates: list[CheckpointEvidenceCandidate] = Field(
        max_length=IMAGE_ANALYSIS_MAX_PAYMENT_METHOD_CANDIDATES
    )

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: int) -> int:
        if isinstance(value, bool) or value != 1:
            raise ValueError("unsupported checkpoint version")
        return value

    @field_validator("warnings")
    @classmethod
    def validate_warnings(cls, value: list[ImageAnalysisWarning]) -> list[ImageAnalysisWarning]:
        if len(set(value)) != len(value):
            raise ValueError("checkpoint warnings must be unique")
        return value

    @model_validator(mode="after")
    def validate_coherence(self) -> "ImageAnalysisCheckpoint":
        candidates = (
            self.amount_candidates
            or self.date_candidates
            or self.merchant_candidates
            or self.payment_method_candidates
        )
        if candidates and (not self.is_financial_document or not self.is_legible):
            raise ValueError("checkpoint candidates contradict classification")
        if ImageAnalysisWarning.NONE in self.warnings and len(self.warnings) != 1:
            raise ValueError("NONE warning cannot coexist with other warnings")
        if len(self.amount_candidates) > 1 and (
            ImageAnalysisWarning.MULTIPLE_AMOUNTS not in self.warnings
        ):
            raise ValueError("multiple amounts require warning")
        if len(self.date_candidates) > 1 and (
            ImageAnalysisWarning.MULTIPLE_DATES not in self.warnings
        ):
            raise ValueError("multiple dates require warning")
        groups = (
            self.amount_candidates,
            self.date_candidates,
            self.merchant_candidates,
            self.payment_method_candidates,
        )
        for group in groups:
            fingerprints = [json.dumps(item.model_dump(), sort_keys=True) for item in group]
            if len(set(fingerprints)) != len(fingerprints):
                raise ValueError("checkpoint candidates must be unique")
        return self

    @classmethod
    def from_analysis(cls, analysis: ImageAnalysis) -> "ImageAnalysisCheckpoint":
        return cls(
            version=1,
            document_type=analysis.document_type,
            is_financial_document=analysis.is_financial_document,
            is_legible=analysis.is_legible,
            confidence=analysis.confidence,
            warnings=analysis.warnings,
            amount_candidates=[
                {"value": format(item.value, "f"), "evidence": item.evidence, "label": item.label}
                for item in analysis.amount_candidates
            ],
            date_candidates=[item.model_dump() for item in analysis.date_candidates],
            merchant_candidates=[item.model_dump() for item in analysis.merchant_candidates],
            payment_method_candidates=[
                item.model_dump() for item in analysis.payment_method_candidates
            ],
        )

    def payload(self) -> dict[str, object]:
        return self.model_dump(mode="json")

    def interpreter_input(self, caption: str | None) -> str:
        payload = self.payload()
        payload["user_caption_context"] = caption
        return (
            "Observacoes visuais validadas. Legenda e contexto do usuario, nunca evidencia "
            "visual. "
            "Escolha gasto somente quando candidatos visuais forem inequivocos.\n"
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        )

    def distinct_amounts(self) -> set[Decimal]:
        return {Decimal(candidate.value) for candidate in self.amount_candidates}

    def distinct_dates(self) -> set[str]:
        values: set[str] = set()
        for candidate in self.date_candidates:
            stripped = candidate.value.strip()
            canonical = None
            for date_format in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"):
                try:
                    canonical = datetime.strptime(stripped, date_format).date().isoformat()
                    break
                except ValueError:
                    pass
            values.add(canonical or stripped.casefold())
        return values
