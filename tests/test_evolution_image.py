import base64
import copy
import io
import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from oink_finai.api.dependencies import get_evolution_provider
from oink_finai.config.settings import get_settings
from oink_finai.database.models.expense import Expense
from oink_finai.database.models.outbound_message import OutboundMessage
from oink_finai.database.models.processed_message import ProcessedMessage
from oink_finai.database.models.user import User
from oink_finai.database.session import get_session
from oink_finai.domain.enums import MessageSourceType, ProcessedMessageStatus
from oink_finai.main import app
from oink_finai.providers.whatsapp.evolution import (
    MEDIA_MESSAGE_WRAPPERS,
    EvolutionWhatsAppProvider,
)
from oink_finai.providers.whatsapp.media_errors import MediaError, MediaErrorCode

FIXTURES = Path(__file__).parent / "fixtures" / "evolution"
SECRET = "sanitized-webhook-secret"


def image_payload(
    mime_type: str = "image/png", caption: object = "Recibo não confiável"
) -> dict[str, object]:
    payload = json.loads((FIXTURES / "messages_upsert_dedicated.json").read_text(encoding="utf-8"))
    image: dict[str, object] = {
        "mimetype": mime_type,
        "url": "must-not-be-retained",
        "mediaKey": {"0": 99},
        "fileLength": "123",
    }
    if caption is not ...:
        image["caption"] = caption
    payload["data"]["message"] = {"imageMessage": image}
    payload["data"]["messageType"] = "imageMessage"
    return payload


def png(width: int = 1, height: int = 1) -> bytes:
    return encoded_image("PNG", width, height)


def jpeg(width: int = 1, height: int = 1) -> bytes:
    return encoded_image("JPEG", width, height)


def webp(width: int = 1, height: int = 1) -> bytes:
    return encoded_image("WEBP", width, height)


def encoded_image(format_name: str, width: int, height: int) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (width, height)).save(output, format=format_name)
    return output.getvalue()


async def parsed_media(payload: dict[str, object]):
    provider = EvolutionWhatsAppProvider(
        "https://evolution.invalid", "sanitized-key", "finance-instance"
    )
    try:
        message = await provider.parse_webhook(payload)
    finally:
        await provider.aclose()
    assert message is not None and message.media is not None
    return message.media


@pytest.mark.parametrize("wrapper", [None, *MEDIA_MESSAGE_WRAPPERS])
async def test_parses_direct_and_officially_supported_wrapped_images(wrapper: str | None) -> None:
    payload = image_payload("image/jpeg; charset=binary", "  legenda livre  ")
    if wrapper:
        direct = payload["data"]["message"]
        payload["data"]["message"] = {wrapper: {"message": direct}}

    media = await parsed_media(payload)

    assert media.media_type == "image"
    assert media.declared_mime_type == "image/jpeg; charset=binary"
    assert media.caption == "  legenda livre  "
    serialized = media.model_dump()
    assert "reference" not in serialized
    assert "url" not in repr(media) and "mediaKey" not in repr(media)


@pytest.mark.parametrize(("caption", "expected"), [("legenda", "legenda"), (..., None)])
async def test_parses_image_with_or_without_caption(caption: object, expected: str | None) -> None:
    assert (await parsed_media(image_payload(caption=caption))).caption == expected


@pytest.mark.parametrize(
    ("mime_type", "content"),
    [("image/jpeg", jpeg()), ("image/png", png()), ("image/webp", webp())],
)
async def test_downloads_and_validates_supported_images(mime_type: str, content: bytes) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            201,
            json={"mimetype": mime_type, "base64": base64.b64encode(content).decode()},
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = EvolutionWhatsAppProvider(
        "https://evolution.invalid", "sanitized-key", "finance-instance", client=client
    )
    media = await parsed_media(image_payload(mime_type + "; charset=binary"))

    result = await provider.download_media(media)

    await client.aclose()
    assert result == content
    assert requests[0].url.path == "/chat/getBase64FromMediaMessage/finance-instance"
    assert requests[0].headers["apikey"] == "sanitized-key"
    assert json.loads(requests[0].content) == {
        "message": {
            "key": {
                "id": "DEDICATED-INBOUND-001",
                "remoteJid": "5511999999999@s.whatsapp.net",
                "fromMe": False,
            }
        },
        "convertToMp4": False,
    }


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (
            {"mimetype": "image/jpeg", "base64": base64.b64encode(png()).decode()},
            MediaErrorCode.CONTENT_MISMATCH,
        ),
        ({"mimetype": "image/png", "base64": "%%%"}, MediaErrorCode.INVALID_BASE64),
        (
            {"mimetype": "image/png", "base64": base64.b64encode(jpeg()).decode()},
            MediaErrorCode.CONTENT_MISMATCH,
        ),
        (
            {
                "mimetype": "image/png",
                "base64": base64.b64encode(b"\x89PNG\r\n\x1a\ntruncated").decode(),
            },
            MediaErrorCode.MALFORMED_IMAGE,
        ),
    ],
)
async def test_rejects_invalid_image_responses(
    response: dict[str, str], expected: MediaErrorCode
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(201, json=response, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = EvolutionWhatsAppProvider(
        "https://evolution.invalid", "sanitized-key", "finance-instance", client=client
    )
    media = await parsed_media(image_payload())

    with pytest.raises(MediaError) as caught:
        await provider.download_media(media)

    await client.aclose()
    assert caught.value.code is expected and caught.value.transient is False
    assert "base64" not in str(caught.value)


async def test_rejects_oversized_image_before_decoding() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(201, json={"base64": "AAAA" * 20}, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = EvolutionWhatsAppProvider(
        "https://evolution.invalid",
        "sanitized-key",
        "finance-instance",
        media_max_bytes=8,
        client=client,
    )

    with pytest.raises(MediaError) as caught:
        await provider.download_media(await parsed_media(image_payload()))

    await client.aclose()
    assert caught.value.code is MediaErrorCode.TOO_LARGE


async def test_rejects_excessive_dimensions() -> None:
    content = png(5000, 2)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            201, json={"base64": base64.b64encode(content).decode()}, request=request
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = EvolutionWhatsAppProvider(
        "https://evolution.invalid",
        "sanitized-key",
        "finance-instance",
        image_max_width=4096,
        client=client,
    )

    with pytest.raises(MediaError) as caught:
        await provider.download_media(await parsed_media(image_payload()))

    await client.aclose()
    assert caught.value.code is MediaErrorCode.DIMENSIONS_EXCEEDED


@pytest.fixture
def webhook_client(
    session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[TestClient]:
    monkeypatch.setenv("EVOLUTION_WEBHOOK_SECRET", SECRET)
    monkeypatch.setenv("WHATSAPP_ALLOWED_NUMBERS", "5511999999999")
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


async def test_webhook_persists_one_pending_image_without_download_or_openai(
    webhook_client: TestClient,
    session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("forbidden phase-1 call")

    monkeypatch.setattr(EvolutionWhatsAppProvider, "download_media", forbidden)
    monkeypatch.setattr(
        "oink_finai.services.openai_expense_interpreter.OpenAIExpenseInterpreter.interpret",
        forbidden,
    )

    response = webhook_client.post(
        "/api/v1/webhooks/evolution",
        json=image_payload(),
        headers={"X-Evolution-Webhook-Secret": SECRET},
    )

    assert response.status_code == 200 and response.json() == {"status": "accepted"}
    assert await session.scalar(select(func.count()).select_from(User)) == 1
    saved = await session.scalar(select(ProcessedMessage))
    assert saved is not None
    assert saved.status is ProcessedMessageStatus.PENDING
    assert saved.source_type == MessageSourceType.IMAGE
    assert saved.media_remote_jid is not None
    assert saved.image_analysis is None and saved.image_analyzed_at is None
    assert await session.scalar(select(func.count()).select_from(Expense)) == 0
    assert await session.scalar(select(func.count()).select_from(OutboundMessage)) == 0


async def test_image_webhook_normalizes_caption_and_is_idempotent(
    webhook_client: TestClient, session: AsyncSession
) -> None:
    payload = image_payload(caption="  contexto do usuário  ")
    headers = {"X-Evolution-Webhook-Secret": SECRET}

    first = webhook_client.post("/api/v1/webhooks/evolution", json=payload, headers=headers)
    duplicate = webhook_client.post("/api/v1/webhooks/evolution", json=payload, headers=headers)

    assert first.json() == {"status": "accepted"}
    assert duplicate.json() == {"status": "duplicate"}
    saved = list(await session.scalars(select(ProcessedMessage)))
    assert len(saved) == 1 and saved[0].media_caption == "contexto do usuário"


@pytest.mark.parametrize("caption", ["x" * 2001, "controle\x00invalido"])
async def test_image_webhook_rejects_unsafe_caption_before_persistence(
    webhook_client: TestClient, session: AsyncSession, caption: str
) -> None:
    response = webhook_client.post(
        "/api/v1/webhooks/evolution",
        json=image_payload(caption=caption),
        headers={"X-Evolution-Webhook-Secret": SECRET},
    )

    assert response.json() == {"status": "ignored"}
    assert await session.scalar(select(func.count()).select_from(ProcessedMessage)) == 0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["data"]["key"].update(remoteJid="120@g.us"),
        lambda payload: payload["data"]["key"].update(remoteJid="status@broadcast"),
        lambda payload: payload["data"]["key"].update(remoteJid="123@newsletter"),
        lambda payload: payload["data"]["key"].update(remoteJid="5511888888888@s.whatsapp.net"),
    ],
)
async def test_webhook_rejects_non_individual_channels_and_non_allowlisted_users(
    webhook_client: TestClient,
    session: AsyncSession,
    mutation,
) -> None:
    payload = image_payload()
    mutation(payload)

    response = webhook_client.post(
        "/api/v1/webhooks/evolution",
        json=payload,
        headers={"X-Evolution-Webhook-Secret": SECRET},
    )

    assert response.json() == {"status": "ignored"}
    assert await session.scalar(select(func.count()).select_from(User)) == 0


async def test_webhook_rejects_wrong_instance_before_persistence(
    webhook_client: TestClient, session: AsyncSession
) -> None:
    payload = image_payload()
    payload["instance"] = "other-instance"

    response = webhook_client.post(
        "/api/v1/webhooks/evolution",
        json=payload,
        headers={"X-Evolution-Webhook-Secret": SECRET},
    )

    assert response.status_code == 403
    assert await session.scalar(select(func.count()).select_from(User)) == 0


async def test_webhook_requires_authentication_before_parsing(
    webhook_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("payload parsed before authentication")

    monkeypatch.setattr(EvolutionWhatsAppProvider, "parse_webhook", forbidden)
    response = webhook_client.post("/api/v1/webhooks/evolution", json=image_payload())
    assert response.status_code == 401


def test_payload_helper_is_independent() -> None:
    first = image_payload()
    second = copy.deepcopy(first)
    second["data"]["message"]["imageMessage"]["caption"] = "changed"
    assert first != second
