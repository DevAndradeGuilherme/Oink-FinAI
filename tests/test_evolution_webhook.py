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


@pytest.mark.parametrize(
    "fixture_name",
    ["messages_upsert_conversation.json", "messages_upsert_extended_text.json"],
)
async def test_accepts_supported_text_and_records_processed_message(
    webhook_client: TestClient, session: AsyncSession, fixture_name: str
) -> None:
    response = post(webhook_client, load_payload(fixture_name))

    assert response.status_code == 200
    assert response.json() == {"status": "accepted"}
    count = await session.scalar(select(func.count()).select_from(ProcessedMessage))
    assert count == 1


@pytest.mark.parametrize("ignored_kind", ["from_me", "group", "irrelevant", "media"])
async def test_ignores_unsupported_webhooks(
    webhook_client: TestClient, session: AsyncSession, ignored_kind: str
) -> None:
    payload = copy.deepcopy(load_payload())
    data = payload["data"]
    if ignored_kind == "from_me":
        data["key"]["fromMe"] = True
    elif ignored_kind == "group":
        data["key"]["remoteJid"] = "120363000000000000@g.us"
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
    payload = load_payload()

    first = post(webhook_client, payload)
    duplicate = post(webhook_client, payload)

    assert first.json() == {"status": "accepted"}
    assert duplicate.status_code == 200
    assert duplicate.json() == {"status": "duplicate"}
    count = await session.scalar(select(func.count()).select_from(ProcessedMessage))
    assert count == 1


def test_rejects_invalid_secret(webhook_client: TestClient) -> None:
    response = post(webhook_client, load_payload(), secret="wrong-secret")

    assert response.status_code == 401


def test_invalid_timestamp_returns_controlled_response(webhook_client: TestClient) -> None:
    payload = load_payload()
    payload["data"]["messageTimestamp"] = 10**100
    payload["date_time"] = "invalid-date-time"

    response = post(webhook_client, payload)

    assert response.status_code == 200
    assert response.json() == {"status": "accepted"}


@pytest.mark.parametrize("field", ["instance", "instanceId", "missing"])
async def test_rejects_invalid_webhook_instance_without_persistence(
    webhook_client: TestClient, session: AsyncSession, field: str
) -> None:
    payload = load_payload()
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
