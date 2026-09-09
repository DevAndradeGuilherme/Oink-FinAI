import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from oink_finai.config.settings import Settings
from oink_finai.services.audio_transcriber_factory import create_audio_transcriber
from oink_finai.services.openai_audio_transcriber import OpenAIAudioTranscriber
from oink_finai.services.transcription_errors import TranscriptionError, TranscriptionErrorCode


async def test_factory_uses_only_openai_settings() -> None:
    instance = create_audio_transcriber(
        Settings(
            _env_file=None,
            openai_api_key="synthetic-key",
            openai_audio_transcription_model="gpt-transcribe",
            openai_audio_transcription_timeout_seconds=7.5,
            openai_audio_transcription_language="pt",
            media_max_bytes=1234,
            media_max_duration_seconds=15,
        )
    )
    try:
        assert isinstance(instance, OpenAIAudioTranscriber)
        assert instance._model == "gpt-transcribe"
        assert instance._language == "pt" and instance._timeout_seconds == 7.5
        assert instance._max_audio_bytes == 1234 and instance._max_duration_seconds == 15
        assert instance._client.max_retries == 0
    finally:
        await instance.aclose()


def test_missing_openai_key_is_terminal_without_fallback() -> None:
    with pytest.raises(TranscriptionError) as caught:
        create_audio_transcriber(Settings(_env_file=None, openai_api_key=None))
    assert caught.value.code is TranscriptionErrorCode.CONFIGURATION
    assert caught.value.transient is False


def test_factory_borrows_shared_client() -> None:
    client = Mock()
    client._client.follow_redirects = False
    client.with_options.return_value = SimpleNamespace(max_retries=0)
    instance = create_audio_transcriber(
        Settings(_env_file=None, openai_api_key="synthetic-key"), client=client
    )
    assert isinstance(instance, OpenAIAudioTranscriber)
    assert instance._owns_client is False
    client.with_options.assert_called_once_with(max_retries=0, timeout=90.0)


async def test_worker_shares_and_closes_one_openai_client(monkeypatch) -> None:
    from oink_finai import worker

    settings = Settings(
        _env_file=None,
        openai_api_key="synthetic-key",
        evolution_api_key="synthetic-key",
        evolution_base_url="https://evolution.invalid",
        evolution_instance="synthetic-instance",
    )
    shared_client = Mock(close=AsyncMock())
    shared_client._client.follow_redirects = False
    shared_client.with_options.return_value = shared_client
    transcriber = Mock(aclose=AsyncMock())
    audio_factory = Mock(return_value=transcriber)
    provider = Mock(aclose=AsyncMock())
    analyzer = Mock(aclose=AsyncMock())
    processing = Mock(
        recover_stale=AsyncMock(), claim=AsyncMock(side_effect=asyncio.CancelledError())
    )
    delivery = Mock(recover_stale=AsyncMock())
    processing_constructor = Mock(return_value=processing)
    monkeypatch.setattr(worker, "get_settings", lambda: settings)
    monkeypatch.setattr(worker, "create_openai_client", Mock(return_value=shared_client))
    monkeypatch.setattr(worker, "create_audio_transcriber", audio_factory)
    monkeypatch.setattr(worker, "EvolutionWhatsAppProvider", Mock(return_value=provider))
    monkeypatch.setattr(worker, "OpenAIImageAnalyzer", Mock(return_value=analyzer))
    monkeypatch.setattr(worker, "ExpenseProcessingService", processing_constructor)
    monkeypatch.setattr(worker, "OutboxDeliveryService", Mock(return_value=delivery))
    monkeypatch.setattr(worker, "engine", Mock(dispose=AsyncMock()))
    monkeypatch.setattr(asyncio.get_running_loop(), "add_signal_handler", Mock())

    with pytest.raises(asyncio.CancelledError):
        await worker.run_worker()

    audio_factory.assert_called_once_with(settings, client=shared_client)
    interpreter = processing_constructor.call_args.args[1]("America/Sao_Paulo")
    assert interpreter._client is shared_client
    assert interpreter._max_output_tokens == settings.openai_expense_max_output_tokens
    query_interpreter = processing_constructor.call_args.kwargs["query_interpreter_factory"](
        "America/Sao_Paulo"
    )
    assert query_interpreter._max_output_tokens == settings.openai_query_max_output_tokens
    worker.OpenAIImageAnalyzer.assert_called_once_with(
        api_key="synthetic-key",
        model=settings.openai_image_model,
        timeout_seconds=settings.openai_image_timeout_seconds,
        max_output_tokens=settings.openai_image_max_output_tokens,
        max_image_bytes=settings.media_max_bytes,
        client=shared_client,
    )
    transcriber.aclose.assert_awaited_once()
    analyzer.aclose.assert_awaited_once()
    provider.aclose.assert_awaited_once()
    shared_client.close.assert_awaited_once()
    worker.engine.dispose.assert_awaited_once()
