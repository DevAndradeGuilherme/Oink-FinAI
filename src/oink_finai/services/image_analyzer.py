from abc import ABC, abstractmethod
from dataclasses import dataclass

from oink_finai.schemas.image_analysis import ImageAnalysis


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
