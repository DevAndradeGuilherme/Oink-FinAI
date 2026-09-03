from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class InboundMedia(BaseModel):
    """Provider-neutral metadata with a non-serializable opaque retrieval reference."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    media_type: Literal["audio"]
    declared_mime_type: str
    declared_duration_seconds: int | None
    is_voice_note: bool
    reference: object = Field(exclude=True, repr=False)


class InboundWhatsAppMessage(BaseModel):
    provider: Literal["evolution"] = "evolution"
    external_message_id: str
    instance_id: str
    remote_jid: str
    remote_jid_alt: str | None = None
    participant_jid: str | None = None
    participant_jid_alt: str | None = None
    phone_number: str
    from_me: bool
    message_type: str
    text_content: str | None
    interaction_id: str | None = None
    media: InboundMedia | None = None
    timestamp: datetime
