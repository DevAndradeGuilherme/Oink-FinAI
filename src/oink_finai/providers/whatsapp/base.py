from abc import ABC, abstractmethod

from oink_finai.schemas.whatsapp import InboundWhatsAppMessage


class WhatsAppProvider(ABC):
    @abstractmethod
    async def parse_webhook(self, payload: dict[str, object]) -> InboundWhatsAppMessage | None:
        """Convert provider payload into the internal message contract."""

    @abstractmethod
    async def send_text(self, phone_number: str, text: str) -> str | None:
        """Send a text message through the configured provider."""
