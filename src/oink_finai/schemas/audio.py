from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_TRANSCRIPT_CHARACTERS = 10_000
MAX_LANGUAGE_CHARACTERS = 64


class GeminiAudioTranscriptionTransport(BaseModel):
    """Minimal Gemini transport DTO; optional language is represented by an empty string."""

    model_config = ConfigDict(extra="forbid")

    transcript: str
    has_speech: bool
    detected_language: str


class AudioTranscription(BaseModel):
    """Provider-neutral, coherently validated transcription result."""

    model_config = ConfigDict(extra="forbid")

    transcript: str = Field(max_length=MAX_TRANSCRIPT_CHARACTERS)
    has_speech: bool
    detected_language: str | None = Field(default=None, max_length=MAX_LANGUAGE_CHARACTERS)

    @model_validator(mode="after")
    def validate_coherence(self) -> "AudioTranscription":
        transcript = self.transcript.strip()
        language = self.detected_language.strip() if self.detected_language is not None else None
        if language == "":
            language = None
        if language is not None and (
            any(ord(character) < 32 or ord(character) == 127 for character in language)
        ):
            raise ValueError("detected language contains unsafe characters")
        if self.has_speech and not transcript:
            raise ValueError("speech requires a non-empty transcript")
        if not self.has_speech and transcript:
            raise ValueError("no speech requires an empty transcript")
        self.transcript = transcript
        self.detected_language = language
        return self


GEMINI_AUDIO_TRANSCRIPTION_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "transcript": {"type": "string"},
        "has_speech": {"type": "boolean"},
        "detected_language": {"type": "string"},
    },
    "required": ["transcript", "has_speech", "detected_language"],
    "additionalProperties": False,
}
