from dataclasses import dataclass

from oink_finai.config.settings import Settings
from oink_finai.schemas.whatsapp import InboundWhatsAppMessage

BLOCKED_JID_SUFFIXES = ("@g.us", "@broadcast", "@newsletter")
INDIVIDUAL_JID_SUFFIXES = ("@s.whatsapp.net", "@lid")
KNOWN_JID_SUFFIXES = INDIVIDUAL_JID_SUFFIXES + BLOCKED_JID_SUFFIXES


@dataclass(frozen=True)
class WhatsAppAccessDecision:
    accepted: bool
    message: InboundWhatsAppMessage | None = None


def filter_inbound_message(
    message: InboundWhatsAppMessage, settings: Settings
) -> WhatsAppAccessDecision:
    """Authorize dedicated inbound traffic and the explicit self-test exception."""
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
    if (
        not _is_individual_chat(message.remote_jid)
        or any(jid.endswith(BLOCKED_JID_SUFFIXES) for jid in jids)
        or any(not jid.endswith(KNOWN_JID_SUFFIXES) for jid in jids)
    ):
        return WhatsAppAccessDecision(accepted=False)

    if not message.from_me:
        phone_number = _trusted_chat_number(message.remote_jid, message.remote_jid_alt)
        if not phone_number or phone_number not in settings.whatsapp_allowed_number_set:
            return WhatsAppAccessDecision(accepted=False)
        return WhatsAppAccessDecision(
            accepted=True,
            message=message.model_copy(update={"phone_number": phone_number}),
        )

    if not settings.whatsapp_self_test_enabled:
        return WhatsAppAccessDecision(accepted=False)

    self_number = _normalize_number(settings.whatsapp_self_test_number)
    if not self_number or self_number not in settings.whatsapp_allowed_number_set:
        return WhatsAppAccessDecision(accepted=False)
    if _trusted_chat_number(message.remote_jid, message.remote_jid_alt) != self_number:
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
    number = jid.partition("@")[0]
    return number if number.isdigit() else ""


def _trusted_chat_number(remote_jid: str, remote_jid_alt: str | None) -> str:
    if remote_jid.endswith("@s.whatsapp.net"):
        return _normalize_jid_number(remote_jid)
    if (
        remote_jid.endswith("@lid")
        and remote_jid_alt
        and remote_jid_alt.endswith("@s.whatsapp.net")
    ):
        return _normalize_jid_number(remote_jid_alt)
    return ""


def _normalize_number(number: str | None) -> str:
    if not number:
        return ""
    return "".join(character for character in number if character.isdigit())
