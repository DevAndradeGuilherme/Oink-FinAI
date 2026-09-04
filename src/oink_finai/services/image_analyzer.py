import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass

from oink_finai.domain.image_analysis_limits import IMAGE_ANALYSIS_CAPTION_MAX_LENGTH
from oink_finai.schemas.image_analysis import ImageAnalysis


def normalize_image_caption(caption: str | None) -> str | None:
    """Validate untrusted caption without changing anything except outer whitespace."""
    if caption is None:
        return None
    if not isinstance(caption, str):
        raise ValueError("image caption must be text")
    normalized = caption.strip()
    if not normalized:
        return None
    if len(normalized) > IMAGE_ANALYSIS_CAPTION_MAX_LENGTH:
        raise ValueError("image caption is too long")
    if any(
        unicodedata.category(character) == "Cc" and character not in "\n\r\t"
        for character in normalized
    ):
        raise ValueError("image caption contains unsupported control characters")
    return normalized


@dataclass(frozen=True, slots=True, repr=False)
class ValidatedImage:
    content: bytes
    mime_type: str
    width: int
    height: int
    detected_format: str

    def __repr__(self) -> str:
        return (
            "ValidatedImage(content=<redacted>, "
            f"mime_type={self.mime_type!r}, width={self.width!r}, height={self.height!r}, "
            f"detected_format={self.detected_format!r})"
        )


class ImageAnalyzer(ABC):
    @abstractmethod
    async def analyze(self, image: ValidatedImage, caption: str | None = None) -> ImageAnalysis:
        """Observe validated image content without creating financial records."""
