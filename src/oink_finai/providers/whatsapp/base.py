from abc import ABC, abstractmethod
from dataclasses import dataclass

from oink_finai.schemas.whatsapp import InboundWhatsAppMessage


@dataclass(frozen=True)
class InteractiveAction:
    id: str
    label: str


@dataclass(frozen=True)
class InteractiveMessage:
    title: str
    body: str
    actions: tuple[InteractiveAction, ...]


class InteractiveMessageUnsupportedError(RuntimeError):
    """Provider deterministically rejected or does not support interactive messages."""


class WhatsAppProvider(ABC):
    @abstractmethod
    async def parse_webhook(self, payload: dict[str, object]) -> InboundWhatsAppMessage | None:
        """Convert provider payload into the internal message contract."""

    @abstractmethod
    async def send_text(self, phone_number: str, text: str) -> str | None:
        """Send a text message through the configured provider."""

    async def send_interactive(self, phone_number: str, message: InteractiveMessage) -> str | None:
        """Send provider-neutral reply actions when supported."""
        raise InteractiveMessageUnsupportedError
