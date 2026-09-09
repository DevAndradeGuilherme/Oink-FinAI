import secrets
from datetime import UTC, datetime
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from oink_finai.api.dependencies import get_evolution_provider
from oink_finai.config.settings import Settings, get_settings
from oink_finai.database.models.conversation_state import ConversationState
from oink_finai.database.models.outbound_message import OutboundMessage
from oink_finai.database.models.processed_message import ProcessedMessage
from oink_finai.database.models.user import User
from oink_finai.database.session import get_session
from oink_finai.domain.enums import (
    ConversationStatus,
    MessageSourceType,
    OutboundMessageKind,
    OutboundMessageStatus,
    ProcessedMessageStatus,
)
from oink_finai.providers.whatsapp.access import filter_inbound_message
from oink_finai.providers.whatsapp.evolution import (
    EvolutionMediaReference,
    EvolutionWebhookInstanceError,
    EvolutionWhatsAppProvider,
)
from oink_finai.services.expense_commands import (
    expense_command_text,
    parse_expense_action,
)
from oink_finai.services.image_analyzer import normalize_image_caption
from oink_finai.services.pipeline_timing import PipelineTiming
from oink_finai.services.usage_control import UsageControl

router = APIRouter(prefix="/api/v1/webhooks", tags=["webhooks"])
RATE_LIMIT_GUIDANCE_TEXT = "Recebi muitas solicitações em pouco tempo. Tente novamente mais tarde."


class WebhookResponse(BaseModel):
    status: str


def _retire_historical_clarification(state: ConversationState) -> None:
    if state.status is ConversationStatus.WAITING_EXPENSE_CLARIFICATION:
        state.status = ConversationStatus.IDLE
        state.active_expense_id = None
        state.context = None
        state.expires_at = None


def verify_webhook_secret(
    webhook_secret: Annotated[str | None, Header(alias="X-Evolution-Webhook-Secret")] = None,
) -> None:
    configured_secret = get_settings().evolution_webhook_secret_value
    if not configured_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Evolution webhook secret is not configured",
        )
    if webhook_secret is None or not secrets.compare_digest(webhook_secret, configured_secret):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid webhook secret"
        )


@router.post("/evolution", response_model=WebhookResponse)
async def evolution_webhook(
    payload: dict[str, Any],
    _: Annotated[None, Depends(verify_webhook_secret)],
    session: Annotated[AsyncSession, Depends(get_session)],
    provider: Annotated[EvolutionWhatsAppProvider, Depends(get_evolution_provider)],
) -> WebhookResponse:
    settings = get_settings()
    correlation_id = uuid4()
    timing = PipelineTiming(settings.pipeline_timing_enabled)
    timing_fields = {
        "processed_message_id": correlation_id,
        "operation": "INBOUND_MESSAGE",
    }
    timing.event("webhook_received", correlation_id, **timing_fields)
    async with timing.span(None, "webhook_completed", correlation_id, **timing_fields):
        return await _handle_evolution_webhook(
            payload, session, provider, correlation_id, timing, settings
        )


async def _handle_evolution_webhook(
    payload: dict[str, Any],
    session: AsyncSession,
    provider: EvolutionWhatsAppProvider,
    correlation_id: UUID,
    timing: PipelineTiming,
    settings: Settings,
) -> WebhookResponse:
    try:
        message = await provider.parse_webhook(payload)
    except EvolutionWebhookInstanceError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Webhook instance is not allowed",
        ) from None
    if message is None:
        return WebhookResponse(status="ignored")
    async with timing.span(None, "access_filter_completed", correlation_id):
        decision = filter_inbound_message(message, settings)
    if not decision.accepted or decision.message is None:
        return WebhookResponse(status="ignored")
    message = decision.message

    media = message.media
    media_reference: EvolutionMediaReference | None = None
    if media is not None:
        if not isinstance(media.reference, EvolutionMediaReference):
            return WebhookResponse(status="ignored")
        media_reference = media.reference
        accepted_text = ""
        if media.media_type == "image":
            try:
                media_caption = normalize_image_caption(media.caption)
            except ValueError:
                return WebhookResponse(status="ignored")
            source_type = MessageSourceType.IMAGE
        else:
            media_caption = None
            source_type = MessageSourceType.AUDIO
    elif message.interaction_id is not None:
        command = parse_expense_action(message.interaction_id)
        if command is None:
            return WebhookResponse(status="ignored")
        accepted_text = expense_command_text(command)
        media_caption = None
        source_type = MessageSourceType.TEXT
    else:
        accepted_text = (message.text_content or "").strip()
        media_caption = None
        source_type = MessageSourceType.TEXT
    if source_type is MessageSourceType.TEXT and not accepted_text:
        return WebhookResponse(status="ignored")
    accepted_text = accepted_text[: settings.inbound_message_max_length]

    try:
        user = await session.scalar(select(User).where(User.phone_number == message.phone_number))
        if user is None:
            try:
                async with session.begin_nested():
                    user = User(
                        phone_number=message.phone_number, timezone=settings.default_timezone
                    )
                    session.add(user)
                    await session.flush()
            except IntegrityError:
                user = await session.scalar(
                    select(User).where(User.phone_number == message.phone_number)
                )
                if user is None:
                    raise
        now = datetime.now(UTC)
        processed_message = ProcessedMessage(
            id=correlation_id,
            provider=message.provider,
            instance_id=message.instance_id,
            external_message_id=message.external_message_id,
            user_id=user.id,
            clarification_origin_message_id=None,
            accepted_text=accepted_text,
            source_type=source_type,
            media_remote_jid=(media_reference.remote_jid if media_reference else None),
            media_mime_type=(
                media.declared_mime_type.partition(";")[0].strip().lower() if media else None
            ),
            media_duration_seconds=(media.declared_duration_seconds if media else None),
            media_is_voice_note=(media.is_voice_note if media else None),
            media_caption=media_caption,
            message_timestamp=message.timestamp,
            status=ProcessedMessageStatus.PENDING,
            available_at=now,
        )
        session.add(processed_message)
        await session.flush()
        admission = await UsageControl(settings).admit_inbound(
            session, message_id=processed_message.id, user_id=user.id
        )
        if not admission.allowed:
            processed_message.status = ProcessedMessageStatus.FAILED
            processed_message.error_code = "INBOUND_RATE_LIMITED"
            processed_message.last_error_code = "INBOUND_RATE_LIMITED"
            processed_message.next_attempt_at = None
            window_marker = (admission.retry_at or now).astimezone(UTC).isoformat()
            guidance_key = f"usage-limit:{user.id}:{window_marker}"
            existing_guidance = await session.scalar(
                select(OutboundMessage.id).where(OutboundMessage.dedup_key == guidance_key)
            )
            if existing_guidance is None:
                session.add(
                    OutboundMessage(
                        user_id=user.id,
                        processed_message_id=processed_message.id,
                        destination=user.phone_number,
                        content=RATE_LIMIT_GUIDANCE_TEXT,
                        content_type="TEXT",
                        kind=OutboundMessageKind.RATE_LIMIT_GUIDANCE,
                        dedup_key=guidance_key,
                        status=OutboundMessageStatus.PENDING,
                        available_at=now,
                    )
                )
        else:
            conversation_state = await session.scalar(
                select(ConversationState)
                .where(ConversationState.user_id == user.id)
                .with_for_update(of=ConversationState)
            )
            if conversation_state is not None:
                _retire_historical_clarification(conversation_state)
        async with timing.span(
            None,
            "inbound_persisted",
            correlation_id,
            source_type=source_type.value,
            mime_type=(
                media.declared_mime_type.partition(";")[0].strip().lower() if media else None
            ),
            audio_duration_seconds=(media.declared_duration_seconds if media else None),
        ):
            await session.commit()
    except IntegrityError:
        await session.rollback()
        return WebhookResponse(status="duplicate")
    return WebhookResponse(status="accepted" if admission.allowed else "rate_limited")
