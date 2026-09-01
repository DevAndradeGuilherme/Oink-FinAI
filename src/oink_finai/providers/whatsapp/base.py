from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass(frozen=True, slots=True)
class IncomingMessage:
    provider: str
    instance_id: str
    external_message_id: str
    sender_phone: str
    sent_at: datetime
    kind: Literal["text", "audio", "image"]
    content: str | bytes
    from_bot: bool


class WhatsAppProvider(ABC):
    @abstractmethod
    async def parse_webhook(self, payload: dict[str, object]) -> IncomingMessage:
        """Convert provider payload into the internal message contract."""

    @abstractmethod
    async def send_text(self, phone_number: str, text: str) -> None:
        """Send a text message through the configured provider."""
