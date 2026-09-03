import base64
import logging

import httpx
import pytest
from fastapi.testclient import TestClient

from oink_finai.config.settings import Settings
from oink_finai.domain.enums import MessageType
from oink_finai.providers.whatsapp.errors import MediaError, MediaErrorCode
from oink_finai.providers.whatsapp.evolution import (
    EvolutionMediaReference,
    EvolutionWhatsAppProvider,
)
from oink_finai.services.whatsapp_webhook import (
    WebhookAccessError,
    WebhookDisposition,
    WhatsAppWebhookService,
)
from tests.fixtures.evolution_audio import audio_webhook, text_webhook

OGG = b"OggS" + b"safe-audio"
MP3 = b"ID3" + b"safe-audio"
MP4 = b"\x00\x00\x00\x18ftypM4A " + b"safe-audio"
WAV = b"RIFF\x10\x00\x00\x00WAVE" + b"safe-audio"


def provider(transport: httpx.AsyncBaseTransport | None = None, **kwargs: object):
    client = httpx.AsyncClient(transport=transport) if transport else None
    return EvolutionWhatsAppProvider(
        base_url="https://evolution.invalid/root/",
        instance="test-instance",
        api_key="super-secret-key",
        client=client,
        **kwargs,
    ), client


@pytest.mark.parametrize("wrapper", [None, "ephemeralMessage", "viewOnceMessageV2"])
async def test_normalizes_voice_note_and_confirmed_wrappers(wrapper: str | None) -> None:
    instance, _ = provider()
    message = await instance.parse_webhook(audio_webhook(wrapper=wrapper))
    assert message.kind is MessageType.AUDIO
    assert message.media is not None
    assert message.media.declared_mime_type == "audio/ogg; codecs=opus"
    assert message.media.declared_duration_seconds == 12
    assert message.media.is_voice_note is True
    assert "mediaKey" not in repr(message.media.reference)
    assert "untrusted" not in repr(message.media.reference)
    assert message.media.reference.message == {
        "key": {
            "id": "MSG-001",
            "remoteJid": "5511999999999@s.whatsapp.net",
            "fromMe": False,
        }
    }


async def test_normalizes_regular_audio_and_keeps_text_working() -> None:
    instance, _ = provider()
    audio = await instance.parse_webhook(audio_webhook(mime="audio/mpeg", ptt=False))
    text = await instance.parse_webhook(text_webhook())
    assert audio.kind is MessageType.AUDIO and audio.media and not audio.media.is_voice_note
    assert text.kind is MessageType.TEXT and text.text == "cafe 10" and text.media is None


@pytest.mark.parametrize(
    ("mime", "content"),
    [
        ("audio/ogg; codecs=opus", OGG),
        ("audio/mpeg", MP3),
        ("audio/mp4", MP4),
        ("audio/wav", WAV),
        ("audio/aac", b"\xff\xf1aac"),
    ],
)
async def test_downloads_allowed_audio_with_injected_client(mime: str, content: bytes) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert (
            request.url
            == "https://evolution.invalid/root/chat/getBase64FromMediaMessage/test-instance"
        )
        assert request.headers["apikey"] == "super-secret-key"
        assert request.extensions["timeout"]["read"] == 15.0
        assert b'"convertToMp4":false' in request.content
        assert b'"mediaKey"' not in request.content and b'"url"' not in request.content
        return httpx.Response(
            201, json={"mimetype": mime, "base64": base64.b64encode(content).decode()}
        )

    instance, client = provider(httpx.MockTransport(handler))
    parsed = await instance.parse_webhook(audio_webhook(mime=mime))
    assert parsed.media and await instance.download_media(parsed.media) == content
    await client.aclose()


@pytest.mark.parametrize(
    ("status", "code", "transient"),
    [
        (401, MediaErrorCode.AUTHENTICATION, False),
        (403, MediaErrorCode.AUTHENTICATION, False),
        (404, MediaErrorCode.NOT_FOUND, False),
        (429, MediaErrorCode.UNAVAILABLE, True),
        (500, MediaErrorCode.UNAVAILABLE, True),
    ],
)
async def test_maps_http_errors(status: int, code: MediaErrorCode, transient: bool) -> None:
    instance, client = provider(
        httpx.MockTransport(lambda _: httpx.Response(status, text="secret body"))
    )
    parsed = await instance.parse_webhook(audio_webhook())
    with pytest.raises(MediaError) as caught:
        await instance.download_media(parsed.media)
    assert caught.value.code is code and caught.value.transient is transient
    assert "secret body" not in str(caught.value) and "secret body" not in repr(caught.value)
    await client.aclose()


@pytest.mark.parametrize(
    ("response", "code"),
    [
        ({"mimetype": "audio/ogg", "base64": "%%%"}, MediaErrorCode.INVALID_BASE64),
        (
            {"mimetype": "audio/ogg", "base64": base64.b64encode(MP3).decode()},
            MediaErrorCode.CONTENT_MISMATCH,
        ),
        (
            {"mimetype": "audio/mpeg", "base64": base64.b64encode(OGG).decode()},
            MediaErrorCode.CONTENT_MISMATCH,
        ),
        ({"other": "missing"}, MediaErrorCode.INVALID_BASE64),
    ],
)
async def test_rejects_invalid_response_content(
    response: dict[str, str], code: MediaErrorCode
) -> None:
    instance, client = provider(httpx.MockTransport(lambda _: httpx.Response(201, json=response)))
    parsed = await instance.parse_webhook(audio_webhook())
    with pytest.raises(MediaError) as caught:
        await instance.download_media(parsed.media)
    assert caught.value.code is code
    await client.aclose()


async def test_rejects_size_before_and_after_decode() -> None:
    for encoded in (base64.b64encode(OGG + b"x" * 20).decode(), "T2dnU3g="):
        instance, client = provider(
            httpx.MockTransport(
                lambda _, value=encoded: httpx.Response(
                    201, json={"base64": value, "mimetype": "audio/ogg"}
                )
            ),
            max_bytes=4,
        )
        parsed = await instance.parse_webhook(audio_webhook())
        with pytest.raises(MediaError) as caught:
            await instance.download_media(parsed.media)
        assert caught.value.code is MediaErrorCode.TOO_LARGE
        await client.aclose()


async def test_rejects_duration_and_type_without_http_call() -> None:
    calls = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(201)

    instance, client = provider(httpx.MockTransport(handler))
    for payload, code in [
        (audio_webhook(seconds=301), MediaErrorCode.TOO_LONG),
        (audio_webhook(mime="video/mp4"), MediaErrorCode.UNSUPPORTED_TYPE),
    ]:
        parsed = await instance.parse_webhook(payload)
        with pytest.raises(MediaError) as caught:
            await instance.download_media(parsed.media)
        assert caught.value.code is code
    assert calls == 0
    await client.aclose()


async def test_timeout_is_transient_and_is_not_retried() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("contains-sensitive-content", request=request)

    instance, client = provider(httpx.MockTransport(handler))
    parsed = await instance.parse_webhook(audio_webhook())
    with pytest.raises(MediaError) as caught:
        await instance.download_media(parsed.media)
    assert caught.value.code is MediaErrorCode.TIMEOUT and caught.value.transient and calls == 1
    assert "sensitive" not in str(caught.value)
    await client.aclose()


@pytest.mark.parametrize(
    "payload",
    [
        audio_webhook(jid="120@g.us"),
        audio_webhook(jid="status@broadcast"),
        audio_webhook(jid="123@newsletter"),
        audio_webhook(from_me=True),
        audio_webhook(jid="5511888888888@s.whatsapp.net"),
    ],
)
async def test_access_filter_rejects_without_download(payload: dict[str, object]) -> None:
    class CountingProvider(EvolutionWhatsAppProvider):
        downloads = 0

        async def download_media(self, media):
            self.downloads += 1
            return b""

    instance = CountingProvider(
        base_url="https://evolution.invalid", instance="test-instance", api_key="key"
    )
    service = WhatsAppWebhookService(instance, webhook_settings())
    result = await service.handle(payload, "webhook-secret")
    assert result.disposition is WebhookDisposition.IGNORED and instance.downloads == 0


async def test_secret_and_instance_are_checked_before_normalization() -> None:
    instance, _ = provider()
    service = WhatsAppWebhookService(instance, webhook_settings())
    with pytest.raises(WebhookAccessError):
        await service.handle(audio_webhook(), "wrong")
    wrong_instance = audio_webhook()
    wrong_instance["instance"] = "other"
    with pytest.raises(WebhookAccessError):
        await service.handle(wrong_instance, "webhook-secret")


async def test_accepted_audio_is_only_recognized_not_downloaded() -> None:
    class CountingProvider(EvolutionWhatsAppProvider):
        downloads = 0

        async def download_media(self, media):
            self.downloads += 1
            return b""

    instance = CountingProvider(
        base_url="https://evolution.invalid", instance="test-instance", api_key="key"
    )
    result = await WhatsAppWebhookService(instance, webhook_settings()).handle(
        audio_webhook(), "webhook-secret"
    )
    assert result.disposition is WebhookDisposition.AUDIO_RECOGNIZED
    assert result.message and result.message.kind is MessageType.AUDIO
    assert instance.downloads == 0


async def test_malformed_payload_is_ignored() -> None:
    instance, _ = provider()
    result = await WhatsAppWebhookService(instance, webhook_settings()).handle(
        {"event": "messages.upsert", "instance": "test-instance", "data": {}}, "webhook-secret"
    )
    assert result.disposition is WebhookDisposition.IGNORED


def test_secrets_and_content_are_not_represented_or_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = webhook_settings(
        evolution_api_key="api-private", evolution_webhook_secret="hook-private"
    )
    reference = EvolutionMediaReference({"key": {"id": "private-content"}})
    with caplog.at_level(logging.DEBUG):
        logging.getLogger("test").debug("%r %r", settings, reference)
    rendered = repr(settings) + repr(reference) + caplog.text
    assert "api-private" not in rendered
    assert "hook-private" not in rendered
    assert "private-content" not in rendered


def webhook_settings(**overrides: object) -> Settings:
    values = {
        "_env_file": None,
        "evolution_instance_id": "test-instance",
        "evolution_webhook_secret": "webhook-secret",
        "whatsapp_allowed_numbers": "5511999999999",
    }
    values.update(overrides)
    return Settings(**values)


def test_webhook_route_acknowledges_audio_without_external_calls(monkeypatch) -> None:
    from oink_finai.api.routes import whatsapp
    from oink_finai.main import app

    monkeypatch.setattr(whatsapp, "get_settings", webhook_settings)
    client = TestClient(app)
    response = client.post(
        "/webhooks/evolution",
        headers={"x-webhook-secret": "webhook-secret"},
        json=audio_webhook(),
    )
    assert response.status_code == 200
    assert response.json() == {"status": "audio_recognized"}


def test_webhook_route_rejects_bad_secret_without_details(monkeypatch) -> None:
    from oink_finai.api.routes import whatsapp
    from oink_finai.main import app

    monkeypatch.setattr(whatsapp, "get_settings", webhook_settings)
    response = TestClient(app).post(
        "/webhooks/evolution",
        headers={"x-webhook-secret": "sensitive-wrong-secret"},
        json=audio_webhook(),
    )
    assert response.status_code == 401
    assert response.json() == {"detail": "invalid webhook"}
    assert "sensitive" not in response.text
