import asyncio
import copy
import json
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from oink_finai.api.dependencies import get_evolution_provider
from oink_finai.config.settings import get_settings
from oink_finai.database.models.conversation_state import ConversationState
from oink_finai.database.models.expense import Expense
from oink_finai.database.models.outbound_message import OutboundMessage
from oink_finai.database.models.processed_message import ProcessedMessage
from oink_finai.database.models.user import User
from oink_finai.database.session import get_session
from oink_finai.domain.enums import ProcessedMessageStatus
from oink_finai.main import app
from oink_finai.providers.whatsapp.evolution import EvolutionWhatsAppProvider
from oink_finai.services.expense_commands import (
    ExpenseCommand,
    ExpenseCommandType,
    encode_expense_action,
)

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


def dedicated_payload() -> dict[str, object]:
    return load_payload("messages_upsert_dedicated.json")


async def assert_no_business_records(session: AsyncSession) -> None:
    assert await session.scalar(select(func.count()).select_from(User)) == 0
    assert await session.scalar(select(func.count()).select_from(ProcessedMessage)) == 0
    assert await session.scalar(select(func.count()).select_from(Expense)) == 0
    assert await session.scalar(select(func.count()).select_from(OutboundMessage)) == 0


@pytest.mark.parametrize("self_test_enabled", ["false", "true"])
async def test_accepts_allowlisted_dedicated_inbound_without_prefix(
    webhook_client: TestClient,
    session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    self_test_enabled: str,
) -> None:
    monkeypatch.setenv("WHATSAPP_SELF_TEST_ENABLED", self_test_enabled)
    get_settings.cache_clear()
    payload = dedicated_payload()
    original_text = payload["data"]["message"]["conversation"]

    response = post(webhook_client, payload)

    saved = await session.scalar(select(ProcessedMessage))
    assert response.json() == {"status": "accepted"}
    assert saved is not None
    assert saved.accepted_text == original_text
    assert saved.status == ProcessedMessageStatus.PENDING
    assert await session.scalar(select(func.count()).select_from(User)) == 1
    assert await session.scalar(select(func.count()).select_from(ConversationState)) == 1


async def test_accepts_allowlisted_dedicated_lid_using_remote_jid_alt(
    webhook_client: TestClient, session: AsyncSession
) -> None:
    payload = dedicated_payload()
    payload["data"]["key"].update(
        {
            "remoteJid": "123456789012345@lid",
            "remoteJidAlt": "5511999999999@s.whatsapp.net",
        }
    )

    response = post(webhook_client, payload)

    assert response.json() == {"status": "accepted"}
    assert await session.scalar(select(func.count()).select_from(ProcessedMessage)) == 1


async def test_dedicated_inbound_preserves_self_test_prefix_as_normal_text(
    webhook_client: TestClient, session: AsyncSession
) -> None:
    payload = dedicated_payload()
    payload["data"]["message"]["conversation"] = "!oink texto normal"

    response = post(webhook_client, payload)

    saved = await session.scalar(select(ProcessedMessage))
    assert response.json() == {"status": "accepted"}
    assert saved is not None
    assert saved.accepted_text == "!oink texto normal"


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


async def test_webhook_only_persists_accepted_text_and_never_waits_for_gemini(
    webhook_client: TestClient,
    session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def forbidden_interpretation(*_args: object, **_kwargs: object) -> None:
        await asyncio.sleep(10)
        raise AssertionError("webhook called Gemini")

    monkeypatch.setattr(
        "oink_finai.services.gemini_expense_interpreter.GeminiExpenseInterpreter.interpret",
        forbidden_interpretation,
    )
    payload = authorized_payload()
    payload["private_raw_field"] = "must-not-be-persisted"

    response = post(webhook_client, payload)

    saved = await session.scalar(select(ProcessedMessage))
    assert response.status_code == 200
    assert response.json() == {"status": "accepted"}
    assert saved is not None
    assert saved.status == ProcessedMessageStatus.PENDING
    original_text = payload["data"]["message"]["conversation"]
    assert saved.accepted_text == original_text[len("!oink ") :]
    assert "must-not-be-persisted" not in repr(saved.__dict__)


async def test_interactive_click_is_normalized_before_persistence(
    webhook_client: TestClient, session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WHATSAPP_SELF_TEST_ENABLED", "false")
    get_settings.cache_clear()
    expense_id = uuid4()
    action_id = encode_expense_action(ExpenseCommand(ExpenseCommandType.REMOVE, expense_id))
    payload = dedicated_payload()
    payload["data"]["message"] = {
        "interactiveResponseMessage": {
            "nativeFlowResponseMessage": {
                "name": "quick_reply",
                "paramsJson": json.dumps({"id": action_id, "display_text": "ignored"}),
            }
        }
    }
    payload["data"]["messageType"] = "interactiveResponseMessage"

    response = post(webhook_client, payload)

    saved = await session.scalar(select(ProcessedMessage))
    assert response.json() == {"status": "accepted"}
    assert saved is not None
    assert saved.accepted_text == f"remover {expense_id}"
    assert action_id not in repr(saved.__dict__)


@pytest.mark.parametrize(
    "params_json",
    ["{", "{}", json.dumps({"id": "unknown-action"})],
)
async def test_malformed_or_unknown_interactive_click_is_ignored(
    webhook_client: TestClient,
    session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    params_json: str,
) -> None:
    monkeypatch.setenv("WHATSAPP_SELF_TEST_ENABLED", "false")
    get_settings.cache_clear()
    payload = dedicated_payload()
    payload["data"]["message"] = {
        "interactiveResponseMessage": {
            "nativeFlowResponseMessage": {
                "name": "quick_reply",
                "paramsJson": params_json,
            }
        }
    }
    payload["data"]["messageType"] = "interactiveResponseMessage"

    response = post(webhook_client, payload)

    assert response.json() == {"status": "ignored"}
    assert await session.scalar(select(func.count()).select_from(ProcessedMessage)) == 0


@pytest.mark.parametrize(
    "ignored_kind",
    [
        "group",
        "status",
        "channel",
        "broadcast",
        "unknown_jid",
        "missing_from_me",
        "invalid_from_me",
        "irrelevant",
        "media",
    ],
)
async def test_ignores_unsupported_webhooks(
    webhook_client: TestClient, session: AsyncSession, ignored_kind: str
) -> None:
    payload = copy.deepcopy(authorized_payload())
    data = payload["data"]
    if ignored_kind == "group":
        data["key"]["remoteJid"] = "120363000000000000@g.us"
    elif ignored_kind == "status":
        data["key"]["remoteJid"] = "status@broadcast"
    elif ignored_kind == "channel":
        data["key"]["remoteJid"] = "120363000000000000@newsletter"
    elif ignored_kind == "broadcast":
        data["key"]["remoteJid"] = "5511999999999@broadcast"
    elif ignored_kind == "unknown_jid":
        data["key"]["remoteJid"] = "opaque@unknown"
    elif ignored_kind == "missing_from_me":
        data["key"].pop("fromMe")
    elif ignored_kind == "invalid_from_me":
        data["key"]["fromMe"] = "false"
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
    payload = dedicated_payload()

    first = post(webhook_client, payload)
    duplicate = post(webhook_client, payload)

    assert first.json() == {"status": "accepted"}
    assert duplicate.status_code == 200
    assert duplicate.json() == {"status": "duplicate"}
    count = await session.scalar(select(func.count()).select_from(ProcessedMessage))
    assert count == 1


@pytest.mark.parametrize("identity_case", ["outside_allowlist", "participant_alt_bypass", "lid"])
async def test_rejects_untrusted_dedicated_identity_before_database_and_gemini(
    webhook_client: TestClient,
    session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    identity_case: str,
) -> None:
    calls = 0

    async def forbidden_interpretation(*_args: object, **_kwargs: object) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(
        "oink_finai.services.gemini_expense_interpreter.GeminiExpenseInterpreter.interpret",
        forbidden_interpretation,
    )
    payload = dedicated_payload()
    if identity_case == "outside_allowlist":
        payload["data"]["key"]["remoteJid"] = "5511888888888@s.whatsapp.net"
    elif identity_case == "participant_alt_bypass":
        payload["data"]["key"].update(
            {
                "remoteJid": "5511888888888@s.whatsapp.net",
                "participantAlt": "5511999999999@s.whatsapp.net",
            }
        )
    else:
        payload["data"]["key"].update(
            {
                "remoteJid": "123456789012345@lid",
                "remoteJidAlt": "untrusted@lid",
                "participantAlt": "5511999999999@s.whatsapp.net",
            }
        )

    response = post(webhook_client, payload)

    assert response.json() == {"status": "ignored"}
    await assert_no_business_records(session)
    assert calls == 0


async def test_outbound_confirmation_cannot_return_as_new_expense(
    webhook_client: TestClient,
    session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WHATSAPP_SELF_TEST_ENABLED", "false")
    get_settings.cache_clear()
    payload = dedicated_payload()
    payload["data"]["key"]["fromMe"] = True
    payload["data"]["message"]["conversation"] = "Despesa registrada"

    response = post(webhook_client, payload)

    assert response.json() == {"status": "ignored"}
    await assert_no_business_records(session)


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
