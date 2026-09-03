from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from oink_finai.domain.enums import MediaType, MessageType


class OpaqueMediaReference(Protocol):
    """Provider-owned token; consumers must only pass it back to the provider."""


@dataclass(frozen=True, slots=True)
class MediaMetadata:
    media_type: MediaType
    declared_mime_type: str
    declared_duration_seconds: int | None
    is_voice_note: bool
    reference: OpaqueMediaReference


@dataclass(frozen=True, slots=True)
class IncomingMessage:
    provider: str
    instance_id: str
    external_message_id: str
    sender_phone: str
    sent_at: datetime
    kind: MessageType
    from_bot: bool
    content: str | None = None
    media: MediaMetadata | None = None

    @property
    def text(self) -> str | None:
        """Backward-compatible semantic alias for textual content."""
        return self.content


class WhatsAppProvider(ABC):
    @abstractmethod
    async def parse_webhook(self, payload: dict[str, object]) -> IncomingMessage:
        """Convert provider payload into the internal message contract."""

    @abstractmethod
    async def send_text(self, phone_number: str, text: str) -> None:
        """Send a text message through the configured provider."""

    @abstractmethod
    async def download_media(self, media: MediaMetadata) -> bytes:
        """Download and validate media represented by an opaque provider reference."""
