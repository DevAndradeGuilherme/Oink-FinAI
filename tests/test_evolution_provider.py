import json
import math
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from oink_finai.providers.whatsapp.evolution import (
    EvolutionProviderError,
    EvolutionWebhookInstanceError,
    EvolutionWhatsAppProvider,
)

FIXTURES = Path(__file__).parent / "fixtures" / "evolution"


def load_payload(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"base_url": "/relative"}, "base_url"),
        ({"api_key": " "}, "api_key"),
        ({"instance": ""}, "instance"),
        ({"timeout_seconds": 0}, "timeout_seconds"),
    ],
)
def test_configuration_is_validated(overrides: dict[str, object], message: str) -> None:
    arguments: dict[str, object] = {
        "base_url": "https://evolution.invalid",
        "api_key": "sanitized-key",
        "instance": "finance-instance",
        **overrides,
    }

    with pytest.raises(ValueError, match=message):
        EvolutionWhatsAppProvider(**arguments)


@pytest.mark.parametrize("max_retries", [0, 1, 3])
async def test_accepts_non_negative_integer_retry_counts(max_retries: int) -> None:
    provider = EvolutionWhatsAppProvider(
        "https://evolution.invalid",
        "sanitized-key",
        "finance-instance",
        max_retries=max_retries,
    )

    assert provider._max_retries == max_retries
    await provider.aclose()


@pytest.mark.parametrize("max_retries", [-1, True, False])
def test_rejects_invalid_retry_counts(max_retries: int) -> None:
    with pytest.raises(ValueError, match="max_retries must be a non-negative integer"):
        EvolutionWhatsAppProvider(
            "https://evolution.invalid",
            "sanitized-key",
            "finance-instance",
            max_retries=max_retries,
        )


@pytest.mark.parametrize(
    ("fixture_name", "expected_type", "expected_text"),
    [
        ("messages_upsert_conversation.json", "conversation", "Almoço 42,50"),
        ("messages_upsert_extended_text.json", "extendedTextMessage", "Mercado 120,00"),
    ],
)
async def test_parse_supported_text_messages(
    fixture_name: str, expected_type: str, expected_text: str
) -> None:
    provider = EvolutionWhatsAppProvider(
        "https://evolution.invalid", "sanitized-key", "finance-instance"
    )
    try:
        message = await provider.parse_webhook(load_payload(fixture_name))
    finally:
        await provider.aclose()

    assert message is not None
    assert message.external_message_id
    assert message.instance_id == "finance-instance"
    assert message.phone_number == message.remote_jid.split("@")[0]
    assert message.message_type == expected_type
    assert message.text_content == expected_text


async def test_parse_lid_message_uses_remote_jid_alt_as_phone_number() -> None:
    payload = load_payload("messages_upsert_conversation.json")
    payload["data"]["key"].update(
        {
            "remoteJid": "123456789012345@lid",
            "remoteJidAlt": "5511999999999@s.whatsapp.net",
            "participant": "987654321012345@lid",
            "participantAlt": "5511888888888@s.whatsapp.net",
        }
    )
    provider = EvolutionWhatsAppProvider(
        "https://evolution.invalid", "sanitized-key", "finance-instance"
    )

    try:
        message = await provider.parse_webhook(payload)
    finally:
        await provider.aclose()

    assert message is not None
    assert message.remote_jid_alt == "5511999999999@s.whatsapp.net"
    assert message.participant_jid == "987654321012345@lid"
    assert message.participant_jid_alt == "5511888888888@s.whatsapp.net"
    assert message.phone_number == "5511999999999"


async def test_send_text_uses_v237_contract() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201, json={"key": {"id": "sent-id"}})

    client = httpx.AsyncClient(
        headers={"x-shared-header": "preserved"}, transport=httpx.MockTransport(handler)
    )
    provider = EvolutionWhatsAppProvider(
        "https://evolution.invalid/api/",
        "sanitized-key",
        "finance-instance",
        timeout_seconds=7.5,
        client=client,
    )
    await provider.send_text("5511999999999", "Olá")
    await provider.aclose()

    assert len(requests) == 1
    assert requests[0].url == httpx.URL(
        "https://evolution.invalid/api/message/sendText/finance-instance"
    )
    assert json.loads(requests[0].content) == {"number": "5511999999999", "text": "Olá"}
    assert requests[0].headers["apikey"] == "sanitized-key"
    assert requests[0].headers["x-shared-header"] == "preserved"
    assert requests[0].extensions["timeout"] == {
        "connect": 7.5,
        "read": 7.5,
        "write": 7.5,
        "pool": 7.5,
    }
    assert "apikey" not in client.headers
    assert not client.is_closed
    await client.aclose()


async def test_provider_closes_client_it_creates() -> None:
    provider = EvolutionWhatsAppProvider(
        "https://evolution.invalid", "sanitized-key", "finance-instance"
    )
    owned_client = provider._client

    await provider.aclose()

    assert owned_client.is_closed


async def test_send_text_without_injection_still_uses_provider_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201, request=request)

    real_async_client = httpx.AsyncClient

    def create_client(*args: object, **kwargs: object) -> httpx.AsyncClient:
        return real_async_client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(httpx, "AsyncClient", create_client)
    provider = EvolutionWhatsAppProvider(
        "https://evolution.invalid", "sanitized-key", "finance-instance"
    )

    async with provider:
        await provider.send_text("5511999999999", "hello")

    assert requests[0].url == httpx.URL(
        "https://evolution.invalid/message/sendText/finance-instance"
    )
    assert requests[0].headers["apikey"] == "sanitized-key"
    assert provider._client.is_closed


@pytest.mark.parametrize(
    "error_type",
    [httpx.ConnectTimeout, httpx.ConnectError, httpx.PoolTimeout],
)
async def test_send_text_retries_only_safe_connection_failures(
    error_type: type[httpx.RequestError],
) -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise error_type("connection unavailable", request=request)
        return httpx.Response(201, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = EvolutionWhatsAppProvider(
        "https://evolution.invalid",
        "sanitized-key",
        "finance-instance",
        max_retries=1,
        client=client,
    )
    await provider.send_text("5511999999999", "hello")
    await client.aclose()

    assert attempts == 2


@pytest.mark.parametrize(
    "error_type",
    [httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError],
)
async def test_send_text_does_not_retry_ambiguous_transport_failures(
    error_type: type[httpx.RequestError],
) -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise error_type("ambiguous transport failure", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = EvolutionWhatsAppProvider(
        "https://evolution.invalid",
        "sanitized-key",
        "finance-instance",
        max_retries=3,
        client=client,
    )

    with pytest.raises(EvolutionProviderError) as error:
        await provider.send_text("5511999999999", "sensitive-message")
    await client.aclose()

    assert attempts == 1
    assert error.value.outcome_unknown is True
    assert "sanitized-key" not in str(error.value)
    assert "sensitive-message" not in str(error.value)


@pytest.mark.parametrize("status_code", [500, 429])
async def test_send_text_does_not_retry_http_responses(status_code: int) -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(status_code, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = EvolutionWhatsAppProvider(
        "https://evolution.invalid",
        "sanitized-key",
        "finance-instance",
        max_retries=3,
        client=client,
    )

    with pytest.raises(EvolutionProviderError) as error:
        await provider.send_text("5511999999999", "sensitive-message")
    await client.aclose()

    assert attempts == 1
    assert error.value.outcome_unknown is True


async def test_send_text_with_zero_retries_still_attempts_once() -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(201, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = EvolutionWhatsAppProvider(
        "https://evolution.invalid",
        "sanitized-key",
        "finance-instance",
        max_retries=0,
        client=client,
    )

    await provider.send_text("5511999999999", "hello")
    await client.aclose()

    assert attempts == 1


async def test_send_text_with_zero_retries_does_not_repeat_connection_failure() -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectTimeout("connection unavailable", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = EvolutionWhatsAppProvider(
        "https://evolution.invalid",
        "sanitized-key",
        "finance-instance",
        max_retries=0,
        client=client,
    )

    with pytest.raises(EvolutionProviderError) as error:
        await provider.send_text("5511999999999", "hello")
    await client.aclose()

    assert attempts == 1
    assert error.value.outcome_unknown is False


@pytest.mark.parametrize("field", ["instance", "instanceId"])
async def test_parse_webhook_accepts_configured_instance_from_supported_fields(field: str) -> None:
    payload = load_payload("messages_upsert_conversation.json")
    payload.pop("instance", None)
    payload["data"].pop("instanceId", None)
    if field == "instance":
        payload["instance"] = "finance-instance"
    else:
        payload["data"]["instanceId"] = "finance-instance"
    provider = EvolutionWhatsAppProvider(
        "https://evolution.invalid", "sanitized-key", "finance-instance"
    )

    message = await provider.parse_webhook(payload)
    await provider.aclose()

    assert message is not None
    assert message.instance_id == "finance-instance"


@pytest.mark.parametrize("field", ["instance", "instanceId"])
async def test_parse_webhook_rejects_another_instance(field: str) -> None:
    payload = load_payload("messages_upsert_conversation.json")
    payload.pop("instance", None)
    payload["data"].pop("instanceId", None)
    if field == "instance":
        payload["instance"] = "other-instance"
    else:
        payload["data"]["instanceId"] = "other-instance"
    provider = EvolutionWhatsAppProvider(
        "https://evolution.invalid", "sanitized-key", "finance-instance"
    )

    with pytest.raises(EvolutionWebhookInstanceError):
        await provider.parse_webhook(payload)
    await provider.aclose()


async def test_parse_webhook_rejects_missing_instance() -> None:
    payload = load_payload("messages_upsert_conversation.json")
    payload.pop("instance", None)
    payload["data"].pop("instanceId", None)
    provider = EvolutionWhatsAppProvider(
        "https://evolution.invalid", "sanitized-key", "finance-instance"
    )

    with pytest.raises(EvolutionWebhookInstanceError):
        await provider.parse_webhook(payload)
    await provider.aclose()


@pytest.mark.parametrize(
    "raw_timestamp",
    [10**100, -(10**100), str(10**100), math.nan, math.inf, -math.inf],
)
def test_invalid_unix_timestamp_falls_back_to_valid_date_time(raw_timestamp: object) -> None:
    result = EvolutionWhatsAppProvider._parse_timestamp(raw_timestamp, "2026-08-31T12:30:00-03:00")

    assert result == datetime.fromisoformat("2026-08-31T12:30:00-03:00")
    assert result.tzinfo is not None


@pytest.mark.parametrize("fallback", [None, "invalid-date-time"])
def test_invalid_timestamp_uses_current_utc_time(fallback: object) -> None:
    before = datetime.now(UTC)

    result = EvolutionWhatsAppProvider._parse_timestamp(math.nan, fallback)

    after = datetime.now(UTC)
    assert before <= result <= after
    assert result.tzinfo is UTC
