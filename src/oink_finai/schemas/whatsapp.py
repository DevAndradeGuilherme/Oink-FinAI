from datetime import datetime
from typing import Literal

from pydantic import BaseModel


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
    timestamp: datetime
