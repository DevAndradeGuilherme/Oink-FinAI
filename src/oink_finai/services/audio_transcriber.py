from abc import ABC, abstractmethod
from dataclasses import dataclass

from oink_finai.schemas.audio import AudioTranscription


@dataclass(frozen=True, slots=True, repr=False)
class ValidatedAudio:
    content: bytes
    mime_type: str
    declared_duration_seconds: int | None = None
    is_voice_note: bool = False

    def __repr__(self) -> str:
        return (
            "ValidatedAudio(content=<redacted>, "
            f"mime_type={self.mime_type!r}, "
            f"declared_duration_seconds={self.declared_duration_seconds!r}, "
            f"is_voice_note={self.is_voice_note!r})"
        )


class AudioTranscriber(ABC):
    @abstractmethod
    async def transcribe(self, audio: ValidatedAudio) -> AudioTranscription:
        """Transcribe validated audio without interpreting its meaning."""
