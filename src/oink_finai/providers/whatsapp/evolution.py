import asyncio
import math
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

import httpx

from oink_finai.providers.whatsapp.base import WhatsAppProvider
from oink_finai.schemas.whatsapp import InboundWhatsAppMessage


class EvolutionProviderError(RuntimeError):
    """Raised when Evolution cannot send a message."""

    def __init__(self, message: str, *, outcome_unknown: bool = False) -> None:
        super().__init__(message)
        self.outcome_unknown = outcome_unknown


class EvolutionWebhookInstanceError(ValueError):
    """Raised when a webhook does not identify the configured Evolution instance."""


class EvolutionWhatsAppProvider(WhatsAppProvider):
    """Evolution client with retries limited to failures known to precede transmission.

    ``max_retries`` controls only retries while acquiring a pooled connection or establishing
    a connection. A sent or potentially sent ``sendText`` request is never retried.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        instance: str,
        *,
        timeout_seconds: float = 10.0,
        max_retries: int = 2,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        normalized_base_url = base_url.strip().rstrip("/")
        parsed_base_url = httpx.URL(normalized_base_url)
        if parsed_base_url.scheme not in {"http", "https"} or not parsed_base_url.host:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if not api_key.strip():
            raise ValueError("api_key must not be empty")
        if not instance.strip():
            raise ValueError("instance must not be empty")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive finite number")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")

        self._base_url = normalized_base_url
        self._api_key = api_key
        self._instance = instance
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient()

    async def __aenter__(self) -> "EvolutionWhatsAppProvider":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def send_text(self, phone_number: str, text: str) -> None:
        url = f"{self._base_url}/message/sendText/{quote(self._instance, safe='')}"
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post(
                    url,
                    json={"number": phone_number, "text": text},
                    headers={"apikey": self._api_key},
                    timeout=self._timeout_seconds,
                )
                response.raise_for_status()
                return
            except httpx.HTTPStatusError:
                raise EvolutionProviderError(
                    "Evolution send result is unknown after an HTTP response",
                    outcome_unknown=True,
                ) from None
            except (httpx.ConnectTimeout, httpx.ConnectError, httpx.PoolTimeout):
                pass
            except httpx.HTTPError:
                raise EvolutionProviderError(
                    "Evolution send result is unknown after a transport failure",
                    outcome_unknown=True,
                ) from None

            if attempt == self._max_retries:
                raise EvolutionProviderError(
                    "Evolution connection failed after limited retries"
                ) from None
            await asyncio.sleep(0.1 * (2**attempt))

    async def parse_webhook(self, payload: dict[str, object]) -> InboundWhatsAppMessage | None:
        if payload.get("event") != "messages.upsert":
            return None

        data = payload.get("data")
        if not isinstance(data, dict):
            return None
        instance_id = self._extract_instance(payload, data)
        if instance_id is None:
            raise EvolutionWebhookInstanceError("Evolution webhook instance is missing")
        if instance_id != self._instance:
            raise EvolutionWebhookInstanceError("Evolution webhook instance is not allowed")

        key = data.get("key")
        message = data.get("message")
        if not isinstance(key, dict) or not isinstance(message, dict):
            return None

        external_id = key.get("id")
        remote_jid = key.get("remoteJid")
        if not all(isinstance(value, str) and value for value in (external_id, remote_jid)):
            return None

        from_me = bool(key.get("fromMe", False))
        remote_jid_alt = self._first_string(
            key, data, names=("remoteJidAlt", "remoteJidAlternative")
        )
        participant_jid = self._first_string(key, data, names=("participant", "participantJid"))
        participant_jid_alt = self._first_string(
            key, data, names=("participantAlt", "participantJidAlt")
        )
        message_type, text = self._extract_content(message, data.get("messageType"))
        timestamp = self._parse_timestamp(data.get("messageTimestamp"), payload.get("date_time"))
        phone_number = self._phone_from_jids(remote_jid_alt, remote_jid)

        return InboundWhatsAppMessage(
            external_message_id=external_id,
            instance_id=instance_id,
            remote_jid=remote_jid,
            remote_jid_alt=remote_jid_alt,
            participant_jid=participant_jid,
            participant_jid_alt=participant_jid_alt,
            phone_number=phone_number,
            from_me=from_me,
            message_type=message_type,
            text_content=text,
            timestamp=timestamp,
        )

    @staticmethod
    def _first_string(*containers: dict[str, Any], names: tuple[str, ...]) -> str | None:
        for container in containers:
            for name in names:
                value = container.get(name)
                if isinstance(value, str) and value:
                    return value
        return None

    @staticmethod
    def _phone_from_jids(*jids: str | None) -> str:
        for jid in jids:
            if jid and jid.endswith("@s.whatsapp.net"):
                return jid.partition("@")[0]
        return next(jid.partition("@")[0] for jid in jids if jid)

    @staticmethod
    def _extract_instance(payload: dict[str, object], data: dict[str, Any]) -> str | None:
        envelope_instance = payload.get("instance")
        if isinstance(envelope_instance, str) and envelope_instance:
            return envelope_instance
        data_instance = data.get("instanceId")
        if isinstance(data_instance, str) and data_instance:
            return data_instance
        return None

    @staticmethod
    def _extract_content(message: dict[str, Any], reported_type: object) -> tuple[str, str | None]:
        conversation = message.get("conversation")
        if isinstance(conversation, str):
            return "conversation", conversation
        extended = message.get("extendedTextMessage")
        if isinstance(extended, dict) and isinstance(extended.get("text"), str):
            return "extendedTextMessage", extended["text"]
        if isinstance(reported_type, str) and reported_type:
            return reported_type, None
        return next(iter(message), "unknown"), None

    @staticmethod
    def _parse_timestamp(raw: object, fallback: object) -> datetime:
        timestamp = EvolutionWhatsAppProvider._safe_unix_timestamp(raw)
        if timestamp is not None:
            return timestamp
        if isinstance(fallback, str):
            try:
                parsed_fallback = datetime.fromisoformat(fallback.replace("Z", "+00:00"))
                if parsed_fallback.tzinfo is None:
                    return parsed_fallback.replace(tzinfo=UTC)
                return parsed_fallback
            except ValueError:
                pass
        return datetime.now(UTC)

    @staticmethod
    def _safe_unix_timestamp(raw: object) -> datetime | None:
        if isinstance(raw, bool):
            return None
        value: int | float
        if isinstance(raw, (int, float)):
            value = raw
        elif isinstance(raw, str) and raw.lstrip("+-").isdigit():
            try:
                value = int(raw)
            except ValueError:
                return None
        else:
            return None

        try:
            return datetime.fromtimestamp(value, tz=UTC)
        except (OverflowError, OSError, ValueError, TypeError):
            return None
