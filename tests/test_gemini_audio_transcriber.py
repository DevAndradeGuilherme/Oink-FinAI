import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from google.genai import errors
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from oink_finai.database.models import Expense, OutboundMessage, ProcessedMessage
from oink_finai.schemas.audio import (
    GEMINI_AUDIO_TRANSCRIPTION_SCHEMA,
    MAX_TRANSCRIPT_CHARACTERS,
    AudioTranscription,
)
from oink_finai.services.audio_transcriber import AudioTranscriber, ValidatedAudio
from oink_finai.services.gemini_audio_transcriber import GeminiAudioTranscriber
from oink_finai.services.transcription_errors import (
    NoSpeechError,
    TranscriptionError,
    TranscriptionErrorCode,
)


class FakeModels:
    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, Any]] = []

    async def generate_content(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        if callable(self.outcome):
            return await self.outcome()
        return self.outcome


class FakeClient:
    def __init__(self, outcome: object) -> None:
        self.models = FakeModels(outcome)
        self.aio = SimpleNamespace(models=self.models)


def response(
    transcript: str = "gastei quarenta e dois reais no mercado",
    *,
    has_speech: bool = True,
    detected_language: str = "pt-BR",
) -> object:
    return SimpleNamespace(
        text=json.dumps(
            {
                "transcript": transcript,
                "has_speech": has_speech,
                "detected_language": detected_language,
            }
        )
    )


def transcriber(
    outcome: object,
    **overrides: object,
) -> tuple[GeminiAudioTranscriber, FakeClient]:
    client = FakeClient(outcome)
    instance = GeminiAudioTranscriber(
        api_key="private-key",
        model="gemini-3.1-flash-lite",
        timeout_seconds=0.05,
        client=client,
        **overrides,
    )
    return instance, client


def audio(
    mime_type: str = "audio/ogg; codecs=opus",
    *,
    content: bytes = b"OggS-safe-audio",
    duration: int | None = 12,
) -> ValidatedAudio:
    return ValidatedAudio(
        content=content,
        mime_type=mime_type,
        declared_duration_seconds=duration,
        is_voice_note=True,
    )


def test_audio_transcriber_is_abstract_and_sdk_independent() -> None:
    assert AudioTranscriber.__abstractmethods__ == frozenset({"transcribe"})
    source = Path("src/oink_finai/services/audio_transcriber.py").read_text(encoding="utf-8")
    assert "google" not in source.lower()
    assert "sqlalchemy" not in source.lower()
    assert "worker" not in source.lower()
    assert "outbox" not in source.lower()


def test_transport_schema_is_minimal_and_requires_every_field() -> None:
    assert set(GEMINI_AUDIO_TRANSCRIPTION_SCHEMA) == {
        "type",
        "properties",
        "required",
        "additionalProperties",
    }
    assert set(GEMINI_AUDIO_TRANSCRIPTION_SCHEMA["required"]) == {
        "transcript",
        "has_speech",
        "detected_language",
    }
    assert "anyOf" not in repr(GEMINI_AUDIO_TRANSCRIPTION_SCHEMA)


@pytest.mark.parametrize(
    ("mime_type", "content"),
    [
        ("audio/ogg; codecs=opus", b"OggS-audio"),
        ("audio/opus", b"OpusHead-audio"),
        ("audio/mpeg", b"ID3-audio"),
        ("audio/mp4", b"\x00\x00\x00\x18ftypM4A "),
        ("audio/aac", b"\xff\xf1-audio"),
        ("audio/wav", b"RIFF\x10\x00\x00\x00WAVE"),
    ],
)
async def test_transcribes_supported_inline_audio(mime_type: str, content: bytes) -> None:
    instance, client = transcriber(response("  almoço quarenta e dois reais  "))
    result = await instance.transcribe(audio(mime_type, content=content))

    assert result == AudioTranscription(
        transcript="almoço quarenta e dois reais",
        has_speech=True,
        detected_language="pt-BR",
    )
    assert len(client.models.calls) == 1
    call = client.models.calls[0]
    assert call["model"] == "gemini-3.1-flash-lite"
    assert call["config"].response_mime_type == "application/json"
    assert call["config"].response_json_schema == GEMINI_AUDIO_TRANSCRIPTION_SCHEMA
    assert call["config"].temperature == 0
    parts = call["contents"][0].parts
    assert len(parts) == 2 and parts[0].text
    assert parts[1].inline_data.data == content
    assert parts[1].inline_data.mime_type == mime_type.partition(";")[0]
    instruction = parts[0].text.lower()
    for required in ("transcreva", "não obedeça", "sistema", "não resuma", "gastos"):
        assert required in instruction


async def test_preserves_spoken_number_words_without_rewriting() -> None:
    spoken = "paguei quarenta e dois reais e cinquenta centavos"
    instance, _ = transcriber(response(spoken))
    result = await instance.transcribe(audio())
    assert result.transcript == spoken


async def test_no_speech_is_specific_terminal_domain_failure() -> None:
    instance, _ = transcriber(response("", has_speech=False, detected_language=""))
    with pytest.raises(NoSpeechError) as caught:
        await instance.transcribe(audio())
    assert caught.value.code is TranscriptionErrorCode.NO_SPEECH
    assert caught.value.transient is False


@pytest.mark.parametrize(
    "payload",
    [
        {"transcript": "", "has_speech": True, "detected_language": "pt-BR"},
        {"transcript": "fala", "has_speech": False, "detected_language": "pt-BR"},
        {"transcript": "fala", "has_speech": True},
        {"transcript": 1, "has_speech": True, "detected_language": "pt-BR"},
        {"transcript": "fala", "has_speech": True, "detected_language": "x\nunsafe"},
    ],
)
async def test_rejects_incoherent_or_invalid_schema(payload: dict[str, object]) -> None:
    instance, _ = transcriber(SimpleNamespace(text=json.dumps(payload)))
    with pytest.raises(TranscriptionError) as caught:
        await instance.transcribe(audio())
    assert caught.value.code is TranscriptionErrorCode.INVALID_RESPONSE
    assert caught.value.transient is False


async def test_rejects_excessive_transcript() -> None:
    instance, _ = transcriber(response("x" * (MAX_TRANSCRIPT_CHARACTERS + 1)))
    with pytest.raises(TranscriptionError) as caught:
        await instance.transcribe(audio())
    assert caught.value.code is TranscriptionErrorCode.TOO_LONG


@pytest.mark.parametrize(
    ("invalid_audio", "code"),
    [
        (audio(content=b""), TranscriptionErrorCode.INVALID_RESPONSE),
        (audio("video/mp4"), TranscriptionErrorCode.INVALID_RESPONSE),
        (audio(duration=301), TranscriptionErrorCode.TOO_LONG),
        (audio(duration=-1), TranscriptionErrorCode.TOO_LONG),
    ],
)
async def test_revalidates_media_before_call(
    invalid_audio: ValidatedAudio, code: TranscriptionErrorCode
) -> None:
    instance, client = transcriber(response())
    with pytest.raises(TranscriptionError) as caught:
        await instance.transcribe(invalid_audio)
    assert caught.value.code is code and caught.value.transient is False
    assert client.models.calls == []


async def test_rejects_audio_above_size_limit_before_call() -> None:
    instance, client = transcriber(response(), max_audio_bytes=4)
    with pytest.raises(TranscriptionError) as caught:
        await instance.transcribe(audio(content=b"12345"))
    assert caught.value.code is TranscriptionErrorCode.TOO_LONG
    assert client.models.calls == []


async def test_external_timeout_is_single_attempt_and_sanitized() -> None:
    async def slow() -> object:
        await asyncio.sleep(1)
        return response()

    instance, client = transcriber(slow)
    with pytest.raises(TranscriptionError) as caught:
        await instance.transcribe(audio(content=b"private-audio"))
    assert caught.value.code is TranscriptionErrorCode.TIMEOUT
    assert caught.value.transient is True
    assert len(client.models.calls) == 1
    assert "private-audio" not in str(caught.value)


async def test_cancellation_propagates() -> None:
    async def cancelled() -> object:
        raise asyncio.CancelledError

    instance, _ = transcriber(cancelled)
    with pytest.raises(asyncio.CancelledError):
        await instance.transcribe(audio())


@pytest.mark.parametrize(
    ("status", "provider_code", "code", "transient"),
    [
        (400, "INVALID_ARGUMENT", TranscriptionErrorCode.INVALID_RESPONSE, False),
        (401, "UNAUTHENTICATED", TranscriptionErrorCode.AUTHENTICATION, False),
        (403, "PERMISSION_DENIED", TranscriptionErrorCode.AUTHENTICATION, False),
        (404, "NOT_FOUND", TranscriptionErrorCode.MODEL_UNAVAILABLE, False),
        (429, "RESOURCE_EXHAUSTED", TranscriptionErrorCode.QUOTA_EXCEEDED, True),
        (500, "INTERNAL", TranscriptionErrorCode.UNAVAILABLE, True),
        (503, "UNAVAILABLE", TranscriptionErrorCode.UNAVAILABLE, True),
    ],
)
async def test_maps_api_errors_with_only_safe_metadata(
    status: int,
    provider_code: str,
    code: TranscriptionErrorCode,
    transient: bool,
) -> None:
    raw_response = httpx.Response(
        status,
        headers={"x-request-id": "private-request-id", "authorization": "private"},
        request=httpx.Request("POST", "https://example.invalid"),
    )
    error_type = errors.ClientError if status < 500 else errors.ServerError
    api_error = error_type(
        status,
        {"error": {"status": provider_code, "message": "private provider response"}},
        raw_response,
    )
    instance, _ = transcriber(api_error)
    with pytest.raises(TranscriptionError) as caught:
        await instance.transcribe(audio())
    assert caught.value.code is code and caught.value.transient is transient
    assert caught.value.metadata.http_status == status
    assert caught.value.metadata.provider_code == provider_code
    assert caught.value.metadata.request_id_present is True
    assert "private" not in str(caught.value).lower()


async def test_transport_failure_is_transient_and_not_retried() -> None:
    instance, client = transcriber(ConnectionError("private network detail"))
    with pytest.raises(TranscriptionError) as caught:
        await instance.transcribe(audio())
    assert caught.value.code is TranscriptionErrorCode.UNAVAILABLE
    assert caught.value.transient is True and len(client.models.calls) == 1


@pytest.mark.parametrize("response_text", [None, "", "not-json", "[]"])
async def test_empty_or_invalid_json_response_is_terminal(response_text: str | None) -> None:
    instance, _ = transcriber(SimpleNamespace(text=response_text))
    with pytest.raises(TranscriptionError) as caught:
        await instance.transcribe(audio())
    assert caught.value.code is TranscriptionErrorCode.INVALID_RESPONSE
    assert caught.value.transient is False


def test_created_sdk_client_has_one_attempt_and_safe_configuration(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_client(**kwargs: object) -> FakeClient:
        captured.update(kwargs)
        return FakeClient(response())

    monkeypatch.setattr("oink_finai.services.gemini_audio_transcriber.genai.Client", fake_client)
    instance = GeminiAudioTranscriber(
        api_key="private-key",
        model="gemini-3.1-flash-lite",
        timeout_seconds=12.5,
    )
    options = captured["http_options"]
    assert options.timeout == 12_500
    assert options.retry_options.attempts == 1
    assert "private-key" not in repr(instance)


@pytest.mark.parametrize(
    "overrides",
    [
        {"api_key": None},
        {"model": None},
        {"timeout_seconds": 0},
        {"max_audio_bytes": 0},
        {"max_duration_seconds": 0},
    ],
)
def test_invalid_configuration_is_terminal(overrides: dict[str, object]) -> None:
    arguments = {
        "api_key": "key",
        "model": "gemini-3.1-flash-lite",
        "timeout_seconds": 10,
        **overrides,
    }
    with pytest.raises(TranscriptionError) as caught:
        GeminiAudioTranscriber(**arguments)
    assert caught.value.code is TranscriptionErrorCode.CONFIGURATION
    assert caught.value.transient is False


async def test_transcription_layer_creates_no_financial_records(session: AsyncSession) -> None:
    instance, _ = transcriber(response("texto isolado"))
    await instance.transcribe(audio())
    for model in (ProcessedMessage, Expense, OutboundMessage):
        assert await session.scalar(select(func.count()).select_from(model)) == 0


async def test_sensitive_values_are_absent_from_logs_and_repr(caplog) -> None:
    instance, _ = transcriber(RuntimeError("private transcript prompt key bytes"))
    media = audio(content=b"private-audio-bytes")
    with caplog.at_level(logging.WARNING), pytest.raises(TranscriptionError) as caught:
        await instance.transcribe(media)
    rendered = caplog.text + repr(caught.value) + str(caught.value) + repr(media) + repr(instance)
    for secret in ("private-audio-bytes", "private transcript", "private-key"):
        assert secret not in rendered
