import secrets
from dataclasses import dataclass
from enum import StrEnum

from oink_finai.config.settings import Settings
from oink_finai.domain.enums import MessageType
from oink_finai.providers.whatsapp.base import IncomingMessage, WhatsAppProvider


class WebhookDisposition(StrEnum):
    TEXT = "text"
    AUDIO_RECOGNIZED = "audio_recognized"
    IGNORED = "ignored"


@dataclass(frozen=True, slots=True)
class WebhookResult:
    disposition: WebhookDisposition
    message: IncomingMessage | None = None


class WebhookAccessError(Exception):
    pass


class WhatsAppWebhookService:
    """Authenticates and normalizes messages; phase 1 deliberately does not download audio."""

    def __init__(self, provider: WhatsAppProvider, settings: Settings) -> None:
        self._provider = provider
        self._settings = settings

    async def handle(self, payload: dict[str, object], secret: str | None) -> WebhookResult:
        configured_secret = self._settings.evolution_webhook_secret
        if (
            not configured_secret
            or not secret
            or not secrets.compare_digest(secret, configured_secret)
        ):
            raise WebhookAccessError
        if payload.get("instance") != self._settings.evolution_instance_id:
            raise WebhookAccessError
        if payload.get("event") != "messages.upsert":
            return WebhookResult(WebhookDisposition.IGNORED)

        data = payload.get("data")
        key = data.get("key") if isinstance(data, dict) else None
        if not isinstance(key, dict):
            return WebhookResult(WebhookDisposition.IGNORED)
        jid = key.get("remoteJid")
        if not isinstance(jid, str) or key.get("fromMe") is not False:
            return WebhookResult(WebhookDisposition.IGNORED)
        if not jid.endswith("@s.whatsapp.net"):
            return WebhookResult(WebhookDisposition.IGNORED)
        phone = jid.removesuffix("@s.whatsapp.net")
        if phone not in self._settings.allowed_numbers:
            return WebhookResult(WebhookDisposition.IGNORED)

        try:
            message = await self._provider.parse_webhook(payload)
        except (TypeError, ValueError, OverflowError):
            return WebhookResult(WebhookDisposition.IGNORED)
        if message.kind is MessageType.AUDIO:
            return WebhookResult(WebhookDisposition.AUDIO_RECOGNIZED, message)
        return WebhookResult(WebhookDisposition.TEXT, message)
