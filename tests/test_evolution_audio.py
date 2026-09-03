import base64
import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from oink_finai.api.dependencies import get_evolution_provider
from oink_finai.config.settings import get_settings
from oink_finai.database.models import Expense, OutboundMessage, ProcessedMessage, User
from oink_finai.database.session import get_session
from oink_finai.main import app
from oink_finai.providers.whatsapp.evolution import (
    EvolutionMediaReference,
    EvolutionWhatsAppProvider,
)
from oink_finai.providers.whatsapp.media_errors import MediaError, MediaErrorCode

FIXTURES = Path(__file__).parent / "fixtures" / "evolution"
OGG = b"OggS" + b"safe-audio"
MP3 = b"ID3" + b"safe-audio"
MP4 = b"\x00\x00\x00\x18ftypM4A " + b"safe-audio"
WAV = b"RIFF\x10\x00\x00\x00WAVE" + b"safe-audio"


def load_audio(name: str = "messages_upsert_audio_ptt.json") -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def make_provider(
    transport: httpx.AsyncBaseTransport | None = None, **overrides: object
) -> tuple[EvolutionWhatsAppProvider, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=transport)
    provider = EvolutionWhatsAppProvider(
        "https://evolution.invalid/root/",
        "private-api-key",
        "finance-instance",
        client=client,
        **overrides,
    )
    return provider, client


@pytest.fixture
def audio_webhook_client(
    session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[TestClient]:
    monkeypatch.setenv("EVOLUTION_WEBHOOK_SECRET", "sanitized-webhook-secret")
    monkeypatch.setenv("WHATSAPP_ACCESS_MODE", "allowlist")
    monkeypatch.setenv("WHATSAPP_ALLOWED_NUMBERS", "5511999999999")
    monkeypatch.setenv("WHATSAPP_SELF_TEST_ENABLED", "false")
    get_settings.cache_clear()
    provider = EvolutionWhatsAppProvider(
        "https://evolution.invalid", "sanitized-key", "finance-instance"
    )

    async def override_session() -> AsyncIterator[AsyncSession]:
        yield session

    app.dependency_overrides[get_session] = override_session
    app.dependency_overrides[get_evolution_provider] = lambda: provider
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()
    get_settings.cache_clear()


@pytest.mark.parametrize(
    ("fixture", "mime", "duration", "ptt"),
    [
        ("messages_upsert_audio_ptt.json", "audio/ogg; codecs=opus", 12, True),
        ("messages_upsert_audio_file.json", "audio/mpeg", 42, False),
        ("messages_upsert_audio_wrapped.json", "audio/ogg; codecs=opus", 8, True),
    ],
)
async def test_normalizes_sanitized_v237_audio_fixtures(
    fixture: str, mime: str, duration: int, ptt: bool
) -> None:
    provider, client = make_provider()
    message = await provider.parse_webhook(load_audio(fixture))
    await client.aclose()

    assert message is not None and message.media is not None
    assert message.message_type in {"audioMessage", "ephemeralMessage"}
    assert message.text_content is None
    assert message.media.media_type == "audio"
    assert message.media.declared_mime_type == mime
    assert message.media.declared_duration_seconds == duration
    assert message.media.is_voice_note is ptt
    assert "mediaKey" not in repr(message.media.reference)
    assert "untrusted" not in repr(message.media.reference)


@pytest.mark.parametrize(
    "wrapper",
    ["ephemeralMessage", "documentWithCaptionMessage", "viewOnceMessage", "viewOnceMessageV2"],
)
async def test_supports_only_confirmed_v237_wrappers(wrapper: str) -> None:
    payload = load_audio()
    audio = payload["data"]["message"]
    payload["data"]["message"] = {wrapper: {"message": audio}}
    provider, client = make_provider()
    message = await provider.parse_webhook(payload)
    await client.aclose()
    assert message is not None and message.media is not None


@pytest.mark.parametrize(
    ("mime", "content"),
    [
        ("audio/ogg; codecs=opus", OGG),
        ("audio/opus", b"OpusHead" + b"safe-audio"),
        ("audio/mpeg", MP3),
        ("audio/mp4", MP4),
        ("audio/aac", b"\xff\xf1safe-audio"),
        ("audio/wav", WAV),
    ],
)
async def test_downloads_allowed_audio_once_with_minimal_key(mime: str, content: bytes) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            201, json={"mimetype": mime, "base64": base64.b64encode(content).decode()}
        )

    provider, client = make_provider(httpx.MockTransport(handler))
    payload = load_audio()
    payload["data"]["message"]["audioMessage"]["mimetype"] = mime
    message = await provider.parse_webhook(payload)
    assert message is not None and message.media is not None
    result = await provider.download_media(message.media)
    await client.aclose()

    assert result == content and len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert request.url == httpx.URL(
        "https://evolution.invalid/root/chat/getBase64FromMediaMessage/finance-instance"
    )
    assert request.headers["apikey"] == "private-api-key"
    assert request.extensions["timeout"]["read"] == 15.0
    body = json.loads(request.content)
    assert body == {
        "message": {
            "key": {
                "id": "AUDIO-PTT-001",
                "remoteJid": "5511999999999@s.whatsapp.net",
                "fromMe": False,
            }
        },
        "convertToMp4": False,
    }
    request_content = request.content.decode()
    assert "mediaKey" not in request_content and "untrusted" not in request_content


@pytest.mark.parametrize(
    ("status", "code", "transient"),
    [
        (401, MediaErrorCode.AUTHENTICATION, False),
        (403, MediaErrorCode.AUTHENTICATION, False),
        (404, MediaErrorCode.NOT_FOUND, False),
        (429, MediaErrorCode.UNAVAILABLE, True),
        (500, MediaErrorCode.UNAVAILABLE, True),
        (400, MediaErrorCode.UNAVAILABLE, False),
    ],
)
async def test_download_maps_sanitized_http_errors(
    status: int, code: MediaErrorCode, transient: bool
) -> None:
    provider, client = make_provider(
        httpx.MockTransport(lambda _: httpx.Response(status, text="private response"))
    )
    message = await provider.parse_webhook(load_audio())
    with pytest.raises(MediaError) as caught:
        await provider.download_media(message.media)
    await client.aclose()
    assert caught.value.code is code and caught.value.transient is transient
    assert "private response" not in str(caught.value)


@pytest.mark.parametrize(
    ("response_body", "code"),
    [
        ({"mimetype": "audio/ogg", "base64": "%%%"}, MediaErrorCode.INVALID_BASE64),
        ({"mimetype": "audio/ogg"}, MediaErrorCode.INVALID_BASE64),
        (
            {"mimetype": "audio/mpeg", "base64": base64.b64encode(OGG).decode()},
            MediaErrorCode.CONTENT_MISMATCH,
        ),
        (
            {"mimetype": "audio/ogg", "base64": base64.b64encode(MP3).decode()},
            MediaErrorCode.CONTENT_MISMATCH,
        ),
    ],
)
async def test_download_rejects_invalid_or_mismatched_content(
    response_body: dict[str, str], code: MediaErrorCode
) -> None:
    provider, client = make_provider(
        httpx.MockTransport(lambda _: httpx.Response(201, json=response_body))
    )
    message = await provider.parse_webhook(load_audio())
    with pytest.raises(MediaError) as caught:
        await provider.download_media(message.media)
    await client.aclose()
    assert caught.value.code is code


@pytest.mark.parametrize(
    ("encoded", "limit"),
    [(base64.b64encode(OGG + b"x" * 20).decode(), 4), ("T2dnU3g=", 4)],
)
async def test_download_checks_size_before_and_after_decode(encoded: str, limit: int) -> None:
    provider, client = make_provider(
        httpx.MockTransport(
            lambda _: httpx.Response(201, json={"mimetype": "audio/ogg", "base64": encoded})
        ),
        media_max_bytes=limit,
    )
    message = await provider.parse_webhook(load_audio())
    with pytest.raises(MediaError) as caught:
        await provider.download_media(message.media)
    await client.aclose()
    assert caught.value.code is MediaErrorCode.TOO_LARGE


@pytest.mark.parametrize(
    ("mime", "seconds", "code"),
    [
        ("video/mp4", 1, MediaErrorCode.UNSUPPORTED_TYPE),
        ("audio/ogg", 301, MediaErrorCode.TOO_LONG),
    ],
)
async def test_download_rejects_metadata_before_request(
    mime: str, seconds: int, code: MediaErrorCode
) -> None:
    calls = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(201)

    provider, client = make_provider(httpx.MockTransport(handler))
    payload = load_audio()
    audio = payload["data"]["message"]["audioMessage"]
    audio.update({"mimetype": mime, "seconds": seconds})
    message = await provider.parse_webhook(payload)
    with pytest.raises(MediaError) as caught:
        await provider.download_media(message.media)
    await client.aclose()
    assert caught.value.code is code and calls == 0


async def test_download_timeout_is_single_transient_attempt() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("private content", request=request)

    provider, client = make_provider(httpx.MockTransport(handler), max_retries=3)
    message = await provider.parse_webhook(load_audio())
    with pytest.raises(MediaError) as caught:
        await provider.download_media(message.media)
    await client.aclose()
    assert caught.value.code is MediaErrorCode.TIMEOUT
    assert caught.value.transient is True and calls == 1
    assert "private content" not in str(caught.value)


async def test_audio_webhook_persists_only_durable_pending_reference(
    audio_webhook_client: TestClient, session: AsyncSession
) -> None:
    response = audio_webhook_client.post(
        "/api/v1/webhooks/evolution",
        json=load_audio(),
        headers={"X-Evolution-Webhook-Secret": "sanitized-webhook-secret"},
    )
    assert response.status_code == 200 and response.json() == {"status": "accepted"}
    assert await session.scalar(select(func.count()).select_from(User)) == 1
    assert await session.scalar(select(func.count()).select_from(ProcessedMessage)) == 1
    assert await session.scalar(select(func.count()).select_from(Expense)) == 0
    assert await session.scalar(select(func.count()).select_from(OutboundMessage)) == 0
    message = await session.scalar(select(ProcessedMessage))
    assert message is not None
    assert message.source_type == "AUDIO"
    assert message.accepted_text == ""
    assert message.media_remote_jid == "5511999999999@s.whatsapp.net"
    assert message.media_mime_type == "audio/ogg"
    assert message.transcribed_at is None


@pytest.mark.parametrize(
    ("jid", "from_me"),
    [
        ("120@g.us", False),
        ("status@broadcast", False),
        ("123@newsletter", False),
        ("5511888888888@s.whatsapp.net", False),
        ("5511999999999@s.whatsapp.net", True),
    ],
)
async def test_rejected_audio_never_downloads(
    audio_webhook_client: TestClient,
    session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    jid: str,
    from_me: bool,
) -> None:
    calls = 0

    async def forbidden_download(*_args: object) -> bytes:
        nonlocal calls
        calls += 1
        raise AssertionError("download called")

    monkeypatch.setattr(EvolutionWhatsAppProvider, "download_media", forbidden_download)
    payload = load_audio()
    payload["data"]["key"].update({"remoteJid": jid, "fromMe": from_me})
    response = audio_webhook_client.post(
        "/api/v1/webhooks/evolution",
        json=payload,
        headers={"X-Evolution-Webhook-Secret": "sanitized-webhook-secret"},
    )
    assert response.json() == {"status": "ignored"} and calls == 0
    assert await session.scalar(select(func.count()).select_from(ProcessedMessage)) == 0


def test_media_reference_and_errors_do_not_reveal_content(caplog) -> None:
    reference = EvolutionMediaReference("private-id", "private-jid", False)
    error = MediaError(MediaErrorCode.UNAVAILABLE, transient=True)
    with caplog.at_level(logging.DEBUG):
        logging.getLogger("audio-test").debug("%r %r", reference, error)
    rendered = repr(reference) + repr(error) + str(error) + caplog.text
    assert "private-id" not in rendered and "private-jid" not in rendered
