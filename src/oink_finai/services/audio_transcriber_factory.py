from oink_finai.config.settings import Settings
from oink_finai.services.audio_transcriber import AudioTranscriber
from oink_finai.services.gemini_audio_transcriber import GeminiAudioTranscriber
from oink_finai.services.transcription_errors import TranscriptionError, TranscriptionErrorCode


def create_audio_transcriber(settings: Settings) -> AudioTranscriber:
    """Select exactly one provider, without probing credentials or falling back."""
    if settings.audio_transcription_provider == "gemini":
        return GeminiAudioTranscriber(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            timeout_seconds=settings.gemini_timeout_seconds,
            max_audio_bytes=settings.media_max_bytes,
            max_duration_seconds=settings.media_max_duration_seconds,
        )
    if settings.audio_transcription_provider == "openai":
        # Keep the legacy path independent of OpenAI SDK initialization and logging setup.
        from oink_finai.services.openai_audio_transcriber import OpenAIAudioTranscriber

        return OpenAIAudioTranscriber(
            api_key=settings.openai_api_key,
            model=settings.openai_audio_transcription_model,
            timeout_seconds=settings.openai_audio_transcription_timeout_seconds,
            language=settings.openai_audio_transcription_language,
            max_audio_bytes=settings.media_max_bytes,
            max_duration_seconds=settings.media_max_duration_seconds,
        )
    raise TranscriptionError(TranscriptionErrorCode.CONFIGURATION, transient=False)
