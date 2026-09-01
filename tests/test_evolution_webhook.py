import copy
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from oink_finai.api.dependencies import get_evolution_provider
from oink_finai.config.settings import get_settings
from oink_finai.database.models.processed_message import ProcessedMessage
from oink_finai.database.models.user import User
from oink_finai.database.session import get_session
from oink_finai.main import app
from oink_finai.providers.whatsapp.evolution import EvolutionWhatsAppProvider

FIXTURES = Path(__file__).parent / "fixtures" / "evolution"
SECRET = "sanitized-webhook-secret"


def load_payload(name: str = "messages_upsert_conversation.json") -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture
def webhook_client(
    session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[TestClient]:
    monkeypatch.setenv("EVOLUTION_WEBHOOK_SECRET", SECRET)
    monkeypatch.setenv("WHATSAPP_ACCESS_MODE", "allowlist")
    monkeypatch.setenv("WHATSAPP_ALLOWED_NUMBERS", "5511999999999")
    monkeypatch.setenv("WHATSAPP_SELF_TEST_ENABLED", "true")
    monkeypatch.setenv("WHATSAPP_SELF_TEST_NUMBER", "5511999999999")
    monkeypatch.setenv("WHATSAPP_SELF_TEST_PREFIX", "!oink")
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


def post(webhook_client: TestClient, payload: dict[str, object], secret: str = SECRET):
    return webhook_client.post(
        "/api/v1/webhooks/evolution",
        json=payload,
        headers={"X-Evolution-Webhook-Secret": secret},
    )


def authorized_payload() -> dict[str, object]:
    payload = load_payload()
    payload["data"]["key"]["fromMe"] = True
    payload["data"]["message"]["conversation"] = "!oink Almoço 42,50"
    return payload


@pytest.mark.parametrize(
    "fixture_name",
    ["messages_upsert_conversation.json", "messages_upsert_extended_text.json"],
)
async def test_accepts_supported_text_and_records_processed_message(
    webhook_client: TestClient, session: AsyncSession, fixture_name: str
) -> None:
    payload = load_payload(fixture_name)
    payload["data"]["key"]["fromMe"] = True
    payload["data"]["key"]["remoteJid"] = "5511999999999@s.whatsapp.net"
    if fixture_name == "messages_upsert_conversation.json":
        payload["data"]["message"]["conversation"] = "!oink Almoço 42,50"
    else:
        payload["data"]["message"]["extendedTextMessage"]["text"] = "!oink Mercado 120,00"
    response = post(webhook_client, payload)

    assert response.status_code == 200
    assert response.json() == {"status": "accepted"}
    count = await session.scalar(select(func.count()).select_from(ProcessedMessage))
    assert count == 1


@pytest.mark.parametrize(
    "ignored_kind",
    ["not_from_me", "group", "status", "channel", "broadcast", "irrelevant", "media"],
)
async def test_ignores_unsupported_webhooks(
    webhook_client: TestClient, session: AsyncSession, ignored_kind: str
) -> None:
    payload = copy.deepcopy(authorized_payload())
    data = payload["data"]
    if ignored_kind == "not_from_me":
        data["key"]["fromMe"] = False
    elif ignored_kind == "group":
        data["key"]["remoteJid"] = "120363000000000000@g.us"
    elif ignored_kind == "status":
        data["key"]["remoteJid"] = "status@broadcast"
    elif ignored_kind == "channel":
        data["key"]["remoteJid"] = "120363000000000000@newsletter"
    elif ignored_kind == "broadcast":
        data["key"]["remoteJid"] = "5511999999999@broadcast"
    elif ignored_kind == "irrelevant":
        payload["event"] = "presence.update"
    else:
        data["message"] = {"imageMessage": {"mimetype": "image/jpeg"}}
        data["messageType"] = "imageMessage"

    response = post(webhook_client, payload)

    assert response.status_code == 200
    assert response.json() == {"status": "ignored"}
    count = await session.scalar(select(func.count()).select_from(ProcessedMessage))
    assert count == 0


async def test_duplicate_webhook_succeeds_without_processing_twice(
    webhook_client: TestClient, session: AsyncSession
) -> None:
    payload = authorized_payload()

    first = post(webhook_client, payload)
    duplicate = post(webhook_client, payload)

    assert first.json() == {"status": "accepted"}
    assert duplicate.status_code == 200
    assert duplicate.json() == {"status": "duplicate"}
    count = await session.scalar(select(func.count()).select_from(ProcessedMessage))
    assert count == 1


def test_rejects_invalid_secret(webhook_client: TestClient) -> None:
    response = post(webhook_client, authorized_payload(), secret="wrong-secret")

    assert response.status_code == 401


def test_invalid_timestamp_returns_controlled_response(webhook_client: TestClient) -> None:
    payload = authorized_payload()
    payload["data"]["messageTimestamp"] = 10**100
    payload["date_time"] = "invalid-date-time"

    response = post(webhook_client, payload)

    assert response.status_code == 200
    assert response.json() == {"status": "accepted"}


@pytest.mark.parametrize("field", ["instance", "instanceId", "missing"])
async def test_rejects_invalid_webhook_instance_without_persistence(
    webhook_client: TestClient, session: AsyncSession, field: str
) -> None:
    payload = authorized_payload()
    payload.pop("instance", None)
    payload["data"].pop("instanceId", None)
    if field == "instance":
        payload["instance"] = "other-instance"
    elif field == "instanceId":
        payload["data"]["instanceId"] = "other-instance"

    response = post(webhook_client, payload)

    assert response.status_code == 403
    assert response.json() == {"detail": "Webhook instance is not allowed"}
    processed_count = await session.scalar(select(func.count()).select_from(ProcessedMessage))
    user_count = await session.scalar(select(func.count()).select_from(User))
    assert processed_count == 0
    assert user_count == 0


@pytest.mark.parametrize(
    ("mutation", "environment"),
    [
        ("missing_prefix", {}),
        ("wrong_case_prefix", {}),
        ("outside_allowlist", {"WHATSAPP_ALLOWED_NUMBERS": "5511888888888"}),
        ("wrong_self_chat", {}),
        ("self_test_disabled", {"WHATSAPP_SELF_TEST_ENABLED": "false"}),
    ],
)
async def test_ignored_personal_messages_never_persist(
    webhook_client: TestClient,
    session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    environment: dict[str, str],
) -> None:
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    payload = authorized_payload()
    if mutation == "missing_prefix":
        payload["data"]["message"]["conversation"] = "Almoço 42,50"
    elif mutation == "wrong_case_prefix":
        payload["data"]["message"]["conversation"] = "!OINK Almoço 42,50"
    elif mutation == "wrong_self_chat":
        payload["data"]["key"]["remoteJid"] = "5511888888888@s.whatsapp.net"

    response = post(webhook_client, payload)

    assert response.json() == {"status": "ignored"}
    assert await session.scalar(select(func.count()).select_from(ProcessedMessage)) == 0
    assert await session.scalar(select(func.count()).select_from(User)) == 0


async def test_bot_generated_message_without_prefix_cannot_loop(
    webhook_client: TestClient, session: AsyncSession
) -> None:
    payload = authorized_payload()
    payload["data"]["key"]["id"] = "BOT-OUTPUT-001"
    payload["data"]["message"]["conversation"] = "Resposta gerada pelo Oink"

    response = post(webhook_client, payload)

    assert response.json() == {"status": "ignored"}
    assert await session.scalar(select(func.count()).select_from(ProcessedMessage)) == 0


async def test_accepts_self_chat_using_remote_jid_alt(
    webhook_client: TestClient, session: AsyncSession
) -> None:
    payload = authorized_payload()
    payload["data"]["key"]["remoteJid"] = "123456789012345@lid"
    payload["data"]["key"]["remoteJidAlt"] = "5511999999999@s.whatsapp.net"

    response = post(webhook_client, payload)

    assert response.json() == {"status": "accepted"}
    assert await session.scalar(select(func.count()).select_from(ProcessedMessage)) == 1
