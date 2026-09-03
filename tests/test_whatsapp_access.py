from datetime import UTC, datetime

from oink_finai.config.settings import Settings
from oink_finai.providers.whatsapp.access import filter_inbound_message
from oink_finai.schemas.whatsapp import InboundWhatsAppMessage


def message(**updates: object) -> InboundWhatsAppMessage:
    original = InboundWhatsAppMessage(
        external_message_id="safe-id",
        instance_id="finance-instance",
        remote_jid="5511999999999@s.whatsapp.net",
        phone_number="5511999999999",
        from_me=True,
        message_type="conversation",
        text_content="!oink   almoço 42,50",
        timestamp=datetime.now(UTC),
    )
    return original.model_copy(update=updates)


def enabled_settings() -> Settings:
    return Settings(
        _env_file=None,
        whatsapp_allowed_numbers="+55 (11) 99999-9999",
        whatsapp_self_test_enabled=True,
        whatsapp_self_test_number="5511999999999",
    )


def test_authorized_message_has_exact_prefix_removed() -> None:
    original = message()

    decision = filter_inbound_message(original, enabled_settings())

    assert decision.accepted is True
    assert decision.message is not None
    assert decision.message.text_content == "almoço 42,50"
    assert original.text_content == "!oink   almoço 42,50"


def test_from_me_is_ignored_with_default_self_test_configuration() -> None:
    settings = Settings(
        _env_file=None,
        whatsapp_allowed_numbers="5511999999999",
        whatsapp_self_test_number="5511999999999",
    )

    assert filter_inbound_message(message(), settings).accepted is False


def test_dedicated_inbound_is_allowed_without_self_test_or_prefix() -> None:
    settings = Settings(
        _env_file=None,
        whatsapp_allowed_numbers="+55 (11) 99999-9999",
        whatsapp_self_test_enabled=False,
    )
    original = message(from_me=False, text_content="  !oink almoço 42,50  ")

    decision = filter_inbound_message(original, settings)

    assert decision.accepted is True
    assert decision.message is not None
    assert decision.message.text_content == "  !oink almoço 42,50  "


def test_dedicated_lid_uses_trusted_remote_jid_alt() -> None:
    original = message(
        from_me=False,
        remote_jid="123456789012345@lid",
        remote_jid_alt="5511999999999@s.whatsapp.net",
        phone_number="123456789012345",
    )

    decision = filter_inbound_message(original, enabled_settings())

    assert decision.accepted is True
    assert decision.message is not None
    assert decision.message.phone_number == "5511999999999"


def test_participant_alt_cannot_authorize_dedicated_chat() -> None:
    other_chat = message(
        from_me=False,
        remote_jid="5511888888888@s.whatsapp.net",
        participant_jid_alt="5511999999999@s.whatsapp.net",
    )

    assert filter_inbound_message(other_chat, enabled_settings()).accepted is False


def test_lid_without_trusted_remote_jid_alt_is_rejected() -> None:
    lid_chat = message(
        from_me=False,
        remote_jid="123456789012345@lid",
        remote_jid_alt=None,
        participant_jid_alt="5511999999999@s.whatsapp.net",
    )

    assert filter_inbound_message(lid_chat, enabled_settings()).accepted is False


def test_participant_cannot_disguise_conversation_with_another_contact() -> None:
    other_chat = message(
        remote_jid="5511888888888@s.whatsapp.net",
        participant_jid="5511999999999@s.whatsapp.net",
    )

    assert filter_inbound_message(other_chat, enabled_settings()).accepted is False
