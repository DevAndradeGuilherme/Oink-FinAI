from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class InboundWhatsAppMessage(BaseModel):
    provider: Literal["evolution"] = "evolution"
    external_message_id: str
    instance_id: str
    remote_jid: str
    phone_number: str
    from_me: bool
    message_type: str
    text_content: str | None
    timestamp: datetime
