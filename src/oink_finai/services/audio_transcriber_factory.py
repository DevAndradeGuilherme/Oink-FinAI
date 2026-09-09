from oink_finai.config.settings import Settings
from oink_finai.services.audio_transcriber import AudioTranscriber
from oink_finai.services.openai_audio_transcriber import OpenAIAudioTranscriber


def create_audio_transcriber(settings: Settings, *, client=None) -> AudioTranscriber:
    """Build the sole audio provider without probing credentials or fallback."""
    return OpenAIAudioTranscriber(
        api_key=settings.openai_api_key_value,
        model=settings.openai_audio_transcription_model,
        timeout_seconds=settings.openai_audio_transcription_timeout_seconds,
        language=settings.openai_audio_transcription_language,
        max_audio_bytes=settings.media_max_bytes,
        max_duration_seconds=settings.media_max_duration_seconds,
        client=client,
    )
