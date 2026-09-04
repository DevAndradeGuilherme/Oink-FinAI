import asyncio
import logging
import struct
import traceback
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from email.parser import BytesParser
from email.policy import default
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from openai import AsyncOpenAI
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from oink_finai.database.models import Expense, OutboundMessage, ProcessedMessage
from oink_finai.schemas.audio import MAX_TRANSCRIPT_CHARACTERS, AudioTranscription
from oink_finai.services.audio_transcriber import AudioTranscriber, ValidatedAudio
from oink_finai.services.openai_audio_transcriber import OpenAIAudioTranscriber
from oink_finai.services.transcription_errors import TranscriptionError, TranscriptionErrorCode

PRIVATE_KEY = "sk-synthetic-private-key"
PRIVATE_TEXT = "paguei quarenta e dois reais e cinquenta centavos"
PRIVATE_DETAIL = (
    "raw-provider-body secret-header 5511999999999 5511999999999@s.whatsapp.net "
    "https://private.invalid/media?token=secret"
)


def ogg_silence() -> bytes:
    """Build a valid mono Ogg/Opus container with one 20 ms silence packet and page CRCs."""
    vendor = b"synthetic-private-audio"
    packets = (
        (2, 0, b"OpusHead" + struct.pack("<BBHIhB", 1, 1, 0, 48_000, 0, 0)),
        (0, 0, b"OpusTags" + struct.pack("<I", len(vendor)) + vendor + struct.pack("<I", 0)),
        (4, 960, b"\xf8\xff\xfe"),
    )
    pages = []
    for sequence, (flags, granule, packet) in enumerate(packets):
        page = (
            b"OggS"
            + struct.pack("<BBQIIIB", 0, flags, granule, 1, sequence, 0, 1)
            + bytes([len(packet)])
            + packet
        )
        checksum = 0
        for value in page:
            checksum ^= value << 24
            for _ in range(8):
                checksum = (checksum << 1) ^ (0x04C11DB7 if checksum & 0x80000000 else 0)
                checksum &= 0xFFFFFFFF
        pages.append(page[:22] + struct.pack("<I", checksum) + page[26:])
    return b"".join(pages)


MEDIA = ValidatedAudio(
    content=ogg_silence(),
    mime_type="audio/ogg; codecs=opus",
    declared_duration_seconds=12,
    is_voice_note=True,
)


@asynccontextmanager
async def transcriber(
    handler: Callable,
    **overrides: object,
) -> AsyncIterator[tuple[OpenAIAudioTranscriber, list[httpx.Request], AsyncOpenAI]]:
    calls: list[httpx.Request] = []

    async def capture(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        result = handler(request)
        return await result if asyncio.iscoroutine(result) else result

    async with AsyncOpenAI(
        api_key=PRIVATE_KEY,
        base_url="https://openai.invalid/v1",
        # Deliberately leave SDK retries enabled; the adapter must override them.
        max_retries=2,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(capture)),
    ) as client:
        instance = OpenAIAudioTranscriber(api_key=PRIVATE_KEY, client=client, **overrides)
        try:
            yield instance, calls, client
        finally:
            await instance.aclose()


def assert_sanitized(error: TranscriptionError, caplog: pytest.LogCaptureFixture) -> None:
    rendered = (
        repr(error)
        + str(error)
        + repr(vars(error))
        + "".join(traceback.format_exception(error))
        + caplog.text
        + repr([vars(record) for record in caplog.records])
    )
    for value in (
        PRIVATE_KEY,
        PRIVATE_TEXT,
        PRIVATE_DETAIL,
        "synthetic-private-audio",
        "secret-header",
        "5511999999999",
        "https://",
        "authorization",
        "raw-provider-body",
    ):
        assert value not in rendered
    assert error.__context__ is None
    assert error.__cause__ is None


async def test_ogg_multipart_contract_single_call_and_private_result(caplog) -> None:
    with caplog.at_level(logging.DEBUG):
        async with transcriber(
            lambda _: httpx.Response(200, json={"text": f"  {PRIVATE_TEXT}  "})
        ) as (instance, calls, client):
            result = await instance.transcribe(MEDIA)

            assert isinstance(instance, AudioTranscriber)
            assert result == AudioTranscription(
                transcript=PRIVATE_TEXT, has_speech=True, detected_language=None
            )
            assert len(calls) == 1
            assert calls[0].method == "POST"
            assert calls[0].url.path == "/v1/audio/transcriptions"
            assert calls[0].headers["authorization"] == f"Bearer {PRIVATE_KEY}"
            assert client.max_retries == 2
            assert instance._client.max_retries == 0
            multipart = BytesParser(policy=default).parsebytes(
                f"Content-Type: {calls[0].headers['content-type']}\r\n\r\n".encode()
                + calls[0].content
            )
            parts = {
                part.get_param("name", header="content-disposition"): part
                for part in multipart.iter_parts()
            }
            assert set(parts) == {"file", "model", "language", "response_format"}
            assert parts["model"].get_content().strip() == "gpt-4o-mini-transcribe"
            assert parts["language"].get_content().strip() == "pt"
            assert parts["response_format"].get_content().strip() == "json"
            assert parts["file"].get_filename() == "audio.ogg"
            assert parts["file"].get_content_type() == "audio/ogg"
            assert parts["file"].get_payload(decode=True) == MEDIA.content

    rendered = caplog.text + repr(instance) + repr(result) + str(result) + repr(MEDIA)
    for private in (PRIVATE_KEY, PRIVATE_TEXT, "synthetic-private-audio", "https://"):
        assert private not in rendered
    assert result.model_dump()["transcript"] == PRIVATE_TEXT


@pytest.mark.parametrize(
    ("status", "code", "transient"),
    [
        (400, TranscriptionErrorCode.INVALID_RESPONSE, False),
        (401, TranscriptionErrorCode.AUTHENTICATION, False),
        (403, TranscriptionErrorCode.AUTHENTICATION, False),
        (404, TranscriptionErrorCode.MODEL_UNAVAILABLE, False),
        (408, TranscriptionErrorCode.TIMEOUT, True),
        (409, TranscriptionErrorCode.INVALID_RESPONSE, False),
        (422, TranscriptionErrorCode.INVALID_RESPONSE, False),
        (429, TranscriptionErrorCode.QUOTA_EXCEEDED, True),
        (500, TranscriptionErrorCode.UNAVAILABLE, True),
        (502, TranscriptionErrorCode.UNAVAILABLE, True),
        (503, TranscriptionErrorCode.UNAVAILABLE, True),
        (504, TranscriptionErrorCode.TIMEOUT, True),
        (599, TranscriptionErrorCode.UNAVAILABLE, True),
        (307, TranscriptionErrorCode.INVALID_RESPONSE, False),
    ],
)
async def test_http_failure_is_sanitized_without_retries_or_fallback(
    status, code, transient, caplog, monkeypatch
) -> None:
    fallback = AsyncMock(side_effect=AssertionError("fallback must not run"))
    monkeypatch.setattr(
        "oink_finai.services.gemini_audio_transcriber.GeminiAudioTranscriber.transcribe", fallback
    )
    with caplog.at_level(logging.DEBUG):
        async with transcriber(
            lambda _: httpx.Response(
                status,
                json={"error": {"message": PRIVATE_DETAIL, "code": PRIVATE_KEY}},
                headers={
                    "x-request-id": PRIVATE_DETAIL,
                    "x-should-retry": "true",
                    "retry-after": "0",
                    "location": "https://private.invalid/redirect",
                },
            )
        ) as (instance, calls, _):
            with pytest.raises(TranscriptionError) as caught:
                await instance.transcribe(MEDIA)
            assert caught.value.code is code and caught.value.transient is transient
            assert len(calls) == 1
            fallback.assert_not_called()
            assert_sanitized(caught.value, caplog)


@pytest.mark.parametrize(
    ("error_type", "code"),
    [
        (httpx.ReadTimeout, TranscriptionErrorCode.TIMEOUT),
        (httpx.ConnectTimeout, TranscriptionErrorCode.TIMEOUT),
        (httpx.ConnectError, TranscriptionErrorCode.UNAVAILABLE),
        (httpx.ReadError, TranscriptionErrorCode.UNAVAILABLE),
        (httpx.RemoteProtocolError, TranscriptionErrorCode.UNAVAILABLE),
    ],
)
async def test_transport_errors_are_transient_without_retry(error_type, code, caplog) -> None:
    def fail(request):
        raise error_type(PRIVATE_DETAIL, request=request)

    with caplog.at_level(logging.DEBUG):
        async with transcriber(fail) as (instance, calls, _):
            with pytest.raises(TranscriptionError) as caught:
                await instance.transcribe(MEDIA)
            assert caught.value.code is code and caught.value.transient is True
            assert len(calls) == 1
            assert_sanitized(caught.value, caplog)


@pytest.mark.parametrize("slow_body", [False, True])
async def test_external_timeout_cancels_entire_request_including_response_body(
    slow_body, caplog
) -> None:
    cancelled = asyncio.Event()

    async def hang():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    class SlowBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"text":"'
            await hang()

    async def handler(_):
        if slow_body:
            return httpx.Response(
                200, stream=SlowBody(), headers={"content-type": "application/json"}
            )
        await hang()

    async with transcriber(handler, timeout_seconds=0.03) as (instance, calls, _):
        with pytest.raises(TranscriptionError) as caught:
            await instance.transcribe(MEDIA)
        assert caught.value.code is TranscriptionErrorCode.TIMEOUT
        assert caught.value.transient is True
        assert cancelled.is_set() and len(calls) == 1
        assert_sanitized(caught.value, caplog)


async def test_caller_cancellation_is_not_converted_or_retried() -> None:
    entered = asyncio.Event()

    async def handler(_):
        entered.set()
        await asyncio.Event().wait()

    async with transcriber(handler) as (instance, calls, _):
        task = asyncio.create_task(instance.transcribe(MEDIA))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(calls) == 1


@pytest.mark.parametrize(
    "payload",
    [{}, {"text": ""}, {"text": " \n "}, {"text": None}, {"text": 42}, {"text": []}, [], None],
)
async def test_invalid_json_shape_is_terminal(payload, caplog) -> None:
    async with transcriber(lambda _: httpx.Response(200, json=payload)) as (instance, calls, _):
        with pytest.raises(TranscriptionError) as caught:
            await instance.transcribe(MEDIA)
        assert caught.value.code is TranscriptionErrorCode.INVALID_RESPONSE
        assert caught.value.transient is False and len(calls) == 1
        assert_sanitized(caught.value, caplog)


@pytest.mark.parametrize("content_type", ["application/json", "text/plain", "text/html"])
async def test_malformed_or_non_json_body_is_terminal(content_type, caplog) -> None:
    with caplog.at_level(logging.DEBUG):
        async with transcriber(
            lambda _: httpx.Response(
                200, content=PRIVATE_DETAIL, headers={"content-type": content_type}
            )
        ) as (instance, calls, _):
            with pytest.raises(TranscriptionError) as caught:
                await instance.transcribe(MEDIA)
            assert caught.value.code is TranscriptionErrorCode.INVALID_RESPONSE
            assert caught.value.transient is False and len(calls) == 1
            assert_sanitized(caught.value, caplog)


@pytest.mark.parametrize("length", [MAX_TRANSCRIPT_CHARACTERS, MAX_TRANSCRIPT_CHARACTERS + 1])
async def test_transcript_length_boundary(length) -> None:
    async with transcriber(lambda _: httpx.Response(200, json={"text": "a" * length})) as (
        instance,
        calls,
        _,
    ):
        if length == MAX_TRANSCRIPT_CHARACTERS:
            assert len((await instance.transcribe(MEDIA)).transcript) == length
        else:
            with pytest.raises(TranscriptionError) as caught:
                await instance.transcribe(MEDIA)
            assert caught.value.code is TranscriptionErrorCode.TOO_LONG
            assert caught.value.transient is False
        assert len(calls) == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"api_key": None},
        {"api_key": ""},
        {"api_key": "   "},
        {"api_key": "synthetic\nkey"},
        {"model": ""},
        {"model": PRIVATE_DETAIL},
        {"timeout_seconds": 0},
        {"timeout_seconds": -1},
        {"timeout_seconds": float("nan")},
        {"timeout_seconds": float("inf")},
        {"timeout_seconds": True},
        {"timeout_seconds": None},
        {"language": "pt-BR"},
        {"language": ""},
        {"max_audio_bytes": 0},
        {"max_audio_bytes": True},
        {"max_duration_seconds": 0},
        {"max_duration_seconds": 1.5},
    ],
)
def test_invalid_configuration_never_constructs_client(overrides, monkeypatch, caplog) -> None:
    constructor = AsyncMock(side_effect=AssertionError("client must not be created"))
    monkeypatch.setattr("oink_finai.services.openai_audio_transcriber.AsyncOpenAI", constructor)
    with pytest.raises(TranscriptionError) as caught:
        OpenAIAudioTranscriber(**{"api_key": PRIVATE_KEY, **overrides})
    assert caught.value.code is TranscriptionErrorCode.CONFIGURATION
    assert caught.value.transient is False
    constructor.assert_not_called()
    assert_sanitized(caught.value, caplog)


@pytest.mark.parametrize(
    ("media", "code"),
    [
        (ValidatedAudio(b"", "audio/ogg"), TranscriptionErrorCode.INVALID_RESPONSE),
        (ValidatedAudio(b"OggS", "video/mp4"), TranscriptionErrorCode.INVALID_RESPONSE),
        (ValidatedAudio(b"invalid", "audio/ogg"), TranscriptionErrorCode.INVALID_RESPONSE),
        (ValidatedAudio(b"OpusHead", "audio/opus"), TranscriptionErrorCode.INVALID_RESPONSE),
        (ValidatedAudio(b"\xff\xf1", "audio/aac"), TranscriptionErrorCode.INVALID_RESPONSE),
        (ValidatedAudio(b"OggS", "audio/ogg", 301), TranscriptionErrorCode.TOO_LONG),
        (ValidatedAudio(b"OggS", "audio/ogg", -1), TranscriptionErrorCode.TOO_LONG),
        (ValidatedAudio(b"OggS", "audio/ogg", True), TranscriptionErrorCode.TOO_LONG),
        (ValidatedAudio(b"OggS" + b"x" * 30, "audio/ogg"), TranscriptionErrorCode.TOO_LONG),
    ],
)
async def test_invalid_audio_never_calls_provider(media, code) -> None:
    async with transcriber(lambda _: httpx.Response(200), max_audio_bytes=30) as (
        instance,
        calls,
        _,
    ):
        with pytest.raises(TranscriptionError) as caught:
            await instance.transcribe(media)
        assert caught.value.code is code and caught.value.transient is False
        assert calls == []


async def test_ogg_opus_at_size_and_duration_limits() -> None:
    async with transcriber(
        lambda _: httpx.Response(200, json={"text": PRIVATE_TEXT}), max_audio_bytes=4
    ) as (instance, calls, _):
        await instance.transcribe(ValidatedAudio(b"OggS", "audio/opus", 300))
        assert len(calls) == 1
        assert b'filename="audio.ogg"' in calls[0].content
        assert b"Content-Type: audio/ogg" in calls[0].content


async def test_owned_client_has_no_retries_or_redirects_and_is_closed(monkeypatch) -> None:
    captured = {}

    def constructor(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(close=AsyncMock())

    monkeypatch.setattr("oink_finai.services.openai_audio_transcriber.AsyncOpenAI", constructor)
    instance = OpenAIAudioTranscriber(api_key=PRIVATE_KEY, timeout_seconds=12.5)
    try:
        assert captured["max_retries"] == 0
        assert captured["timeout"] == 12.5
        assert captured["http_client"].follow_redirects is False
        await instance.aclose()
        await instance.aclose()
        instance._client.close.assert_awaited_once()
    finally:
        await captured["http_client"].aclose()


async def test_borrowed_client_remains_open_after_adapter_closes() -> None:
    async with transcriber(lambda _: httpx.Response(200)) as (instance, calls, client):
        await instance.aclose()
        assert not client.is_closed()
        with pytest.raises(TranscriptionError) as caught:
            await instance.transcribe(MEDIA)
        assert caught.value.code is TranscriptionErrorCode.CONFIGURATION
        assert calls == []


async def test_injected_redirecting_client_is_rejected_without_mutation_or_requests() -> None:
    handler = Mock(side_effect=AssertionError("request must not run"))
    async with AsyncOpenAI(
        api_key=PRIVATE_KEY,
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler), follow_redirects=True
        ),
    ) as client:
        with pytest.raises(TranscriptionError) as caught:
            OpenAIAudioTranscriber(api_key=PRIVATE_KEY, client=client)
        assert caught.value.code is TranscriptionErrorCode.CONFIGURATION
        assert caught.value.transient is False
        assert client._client.follow_redirects is True
        assert not client.is_closed()
        handler.assert_not_called()


def test_client_initialization_failure_drops_raw_exception(monkeypatch, caplog) -> None:
    client = Mock()
    client._client.follow_redirects = False
    client.with_options.side_effect = ValueError(PRIVATE_DETAIL)
    with caplog.at_level(logging.DEBUG), pytest.raises(TranscriptionError) as caught:
        OpenAIAudioTranscriber(api_key=PRIVATE_KEY, client=client)
    assert caught.value.code is TranscriptionErrorCode.CONFIGURATION
    assert caught.value.transient is False
    assert_sanitized(caught.value, caplog)


async def test_close_failure_is_sanitized(monkeypatch, caplog) -> None:
    async with transcriber(lambda _: httpx.Response(200)) as (instance, _, client):
        instance._owns_client = True
        with monkeypatch.context() as patch:
            patch.setattr(
                instance._client, "close", AsyncMock(side_effect=RuntimeError(PRIVATE_DETAIL))
            )
            with pytest.raises(TranscriptionError) as caught:
                await instance.aclose()
            assert_sanitized(caught.value, caplog)
        await instance.aclose()
        assert client.is_closed()


async def test_logging_suppression_is_context_local_and_restored(caplog) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def handler(_):
        logging.getLogger("httpcore.connection").debug(PRIVATE_DETAIL)
        entered.set()
        await release.wait()
        return httpx.Response(200, json={"text": PRIVATE_TEXT})

    with caplog.at_level(logging.DEBUG):
        async with transcriber(handler) as (instance, _, _):
            task = asyncio.create_task(instance.transcribe(MEDIA))
            await entered.wait()
            logging.getLogger("httpx").info("unrelated concurrent operation")
            release.set()
            await task
            logging.getLogger("httpcore.connection").info("logging restored")
    assert "unrelated concurrent operation" in caplog.text
    assert "logging restored" in caplog.text
    assert PRIVATE_DETAIL not in caplog.text and PRIVATE_TEXT not in caplog.text


async def test_transcriber_creates_no_financial_or_outbox_records(session: AsyncSession) -> None:
    async with transcriber(lambda _: httpx.Response(200, json={"text": PRIVATE_TEXT})) as (
        instance,
        _,
        _,
    ):
        await instance.transcribe(MEDIA)
    for model in (ProcessedMessage, Expense, OutboundMessage):
        assert await session.scalar(select(func.count()).select_from(model)) == 0
