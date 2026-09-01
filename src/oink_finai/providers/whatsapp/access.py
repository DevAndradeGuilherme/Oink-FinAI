from dataclasses import dataclass

from oink_finai.config.settings import Settings
from oink_finai.schemas.whatsapp import InboundWhatsAppMessage

BLOCKED_JID_SUFFIXES = ("@g.us", "@broadcast", "@newsletter")
INDIVIDUAL_JID_SUFFIXES = ("@s.whatsapp.net", "@lid")


@dataclass(frozen=True)
class WhatsAppAccessDecision:
    accepted: bool
    message: InboundWhatsAppMessage | None = None


def filter_inbound_message(
    message: InboundWhatsAppMessage, settings: Settings
) -> WhatsAppAccessDecision:
    """Authorize self-test traffic without logging or retaining message data."""
    jids = tuple(
        jid
        for jid in (
            message.remote_jid,
            message.remote_jid_alt,
            message.participant_jid,
            message.participant_jid_alt,
        )
        if jid
    )
    if not _is_individual_chat(message.remote_jid) or any(
        jid.endswith(BLOCKED_JID_SUFFIXES) for jid in jids
    ):
        return WhatsAppAccessDecision(accepted=False)

    if not message.from_me or not settings.whatsapp_self_test_enabled:
        return WhatsAppAccessDecision(accepted=False)

    self_number = _normalize_number(settings.whatsapp_self_test_number)
    conversation_jids = tuple(jid for jid in (message.remote_jid, message.remote_jid_alt) if jid)
    jid_numbers = {_normalize_jid_number(jid) for jid in conversation_jids}
    if not self_number or self_number not in settings.whatsapp_allowed_number_set:
        return WhatsAppAccessDecision(accepted=False)
    if self_number not in jid_numbers:
        return WhatsAppAccessDecision(accepted=False)

    text = message.text_content
    prefix = settings.whatsapp_self_test_prefix
    if text is None or not prefix or not text.startswith(prefix):
        return WhatsAppAccessDecision(accepted=False)

    return WhatsAppAccessDecision(
        accepted=True,
        message=message.model_copy(update={"text_content": text[len(prefix) :].lstrip()}),
    )


def _is_individual_chat(jid: str) -> bool:
    return jid.endswith(INDIVIDUAL_JID_SUFFIXES)


def _normalize_jid_number(jid: str) -> str:
    return _normalize_number(jid.partition("@")[0])


def _normalize_number(number: str | None) -> str:
    if not number:
        return ""
    return "".join(character for character in number if character.isdigit())
