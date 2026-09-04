import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from oink_finai.config.settings import Settings
from oink_finai.services.audio_transcriber_factory import create_audio_transcriber
from oink_finai.services.gemini_audio_transcriber import GeminiAudioTranscriber
from oink_finai.services.openai_audio_transcriber import OpenAIAudioTranscriber
from oink_finai.services.transcription_errors import TranscriptionError, TranscriptionErrorCode


async def test_default_factory_preserves_gemini_configuration(monkeypatch) -> None:
    monkeypatch.delenv("AUDIO_TRANSCRIPTION_PROVIDER", raising=False)
    sdk = Mock()
    sdk.aio.aclose = AsyncMock()
    gemini_constructor = Mock(return_value=sdk)
    openai_constructor = Mock(side_effect=AssertionError("OpenAI must not initialize"))
    monkeypatch.setattr(
        "oink_finai.services.gemini_audio_transcriber.genai.Client", gemini_constructor
    )
    monkeypatch.setattr(
        "oink_finai.services.openai_audio_transcriber.AsyncOpenAI", openai_constructor
    )
    settings = Settings(
        _env_file=None,
        gemini_api_key="synthetic-gemini-key",
        gemini_model="legacy-model",
        gemini_timeout_seconds=12.5,
        media_max_bytes=1234,
        media_max_duration_seconds=15,
        openai_api_key=None,
    )

    instance = create_audio_transcriber(settings)

    assert isinstance(instance, GeminiAudioTranscriber)
    assert instance._model == "legacy-model"
    assert instance._timeout_seconds == 12.5
    assert instance._max_audio_bytes == 1234 and instance._max_duration_seconds == 15
    assert gemini_constructor.call_args.kwargs["http_options"].retry_options.attempts == 1
    openai_constructor.assert_not_called()
    await instance.aclose()
    sdk.aio.aclose.assert_awaited_once()


async def test_explicit_openai_provider_uses_only_openai_settings(monkeypatch) -> None:
    instance = create_audio_transcriber(
        Settings(
            _env_file=None,
            audio_transcription_provider="openai",
            openai_api_key="synthetic-key",
            openai_audio_transcription_model="gpt-4o-mini-transcribe",
            openai_audio_transcription_timeout_seconds=7.5,
            openai_audio_transcription_language="pt",
            media_max_bytes=1234,
            media_max_duration_seconds=15,
            gemini_api_key=None,
        )
    )
    try:
        assert isinstance(instance, OpenAIAudioTranscriber)
        assert instance._model == "gpt-4o-mini-transcribe"
        assert instance._language == "pt" and instance._timeout_seconds == 7.5
        assert instance._max_audio_bytes == 1234 and instance._max_duration_seconds == 15
        assert instance._client.max_retries == 0
        assert not instance._client.is_closed()
    finally:
        await instance.aclose()
    assert instance._client.is_closed()


@pytest.mark.parametrize("provider", ["gemini", "openai"])
def test_missing_selected_key_is_terminal_without_fallback(provider, monkeypatch) -> None:
    settings = Settings(
        _env_file=None,
        audio_transcription_provider=provider,
        gemini_api_key=None if provider == "gemini" else "synthetic-key",
        openai_api_key=None if provider == "openai" else "synthetic-key",
    )
    with pytest.raises(TranscriptionError) as caught:
        create_audio_transcriber(settings)
    assert caught.value.code is TranscriptionErrorCode.CONFIGURATION
    assert caught.value.transient is False


def test_invalid_provider_does_not_fall_back() -> None:
    settings = Settings.model_construct(audio_transcription_provider="unsupported")
    with pytest.raises(TranscriptionError) as caught:
        create_audio_transcriber(settings)
    assert caught.value.code is TranscriptionErrorCode.CONFIGURATION
    assert caught.value.transient is False


async def test_worker_uses_factory_and_closes_selected_transcriber(monkeypatch) -> None:
    from oink_finai import worker

    settings = Settings(
        _env_file=None,
        gemini_api_key="synthetic-key",
        evolution_api_key="synthetic-key",
        evolution_base_url="https://evolution.invalid",
        evolution_instance="synthetic-instance",
    )
    transcriber = Mock(aclose=AsyncMock())
    factory = Mock(return_value=transcriber)
    provider = Mock(aclose=AsyncMock())
    analyzer = Mock(aclose=AsyncMock())
    processing = Mock(
        recover_stale=AsyncMock(), claim=AsyncMock(side_effect=asyncio.CancelledError())
    )
    delivery = Mock(recover_stale=AsyncMock())
    processing_constructor = Mock(return_value=processing)
    monkeypatch.setattr(worker, "get_settings", lambda: settings)
    monkeypatch.setattr(worker, "create_audio_transcriber", factory)
    monkeypatch.setattr(worker, "EvolutionWhatsAppProvider", Mock(return_value=provider))
    monkeypatch.setattr(worker, "GeminiImageAnalyzer", Mock(return_value=analyzer))
    monkeypatch.setattr(worker, "ExpenseProcessingService", processing_constructor)
    monkeypatch.setattr(worker, "OutboxDeliveryService", Mock(return_value=delivery))
    monkeypatch.setattr(worker, "engine", Mock(dispose=AsyncMock()))
    monkeypatch.setattr(asyncio.get_running_loop(), "add_signal_handler", Mock())
    with pytest.raises(asyncio.CancelledError):
        await worker.run_worker()
    factory.assert_called_once_with(settings)
    assert processing_constructor.call_args.kwargs["audio_transcriber_factory"]() is transcriber
    transcriber.aclose.assert_awaited_once()
    analyzer.aclose.assert_awaited_once()
    provider.aclose.assert_awaited_once()
    worker.engine.dispose.assert_awaited_once()
