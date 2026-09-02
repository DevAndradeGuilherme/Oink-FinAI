import secrets
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from oink_finai.api.dependencies import get_evolution_provider
from oink_finai.config.settings import get_settings
from oink_finai.database.models.conversation_state import ConversationState
from oink_finai.database.models.processed_message import ProcessedMessage
from oink_finai.database.models.user import User
from oink_finai.database.session import get_session
from oink_finai.domain.enums import ProcessedMessageStatus
from oink_finai.providers.whatsapp.access import filter_inbound_message
from oink_finai.providers.whatsapp.evolution import (
    EvolutionWebhookInstanceError,
    EvolutionWhatsAppProvider,
)

router = APIRouter(prefix="/api/v1/webhooks", tags=["webhooks"])


class WebhookResponse(BaseModel):
    status: str


def verify_webhook_secret(
    webhook_secret: Annotated[str | None, Header(alias="X-Evolution-Webhook-Secret")] = None,
) -> None:
    configured_secret = get_settings().evolution_webhook_secret
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
    try:
        message = await provider.parse_webhook(payload)
    except EvolutionWebhookInstanceError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Webhook instance is not allowed",
        ) from None
    if message is None:
        return WebhookResponse(status="ignored")
    decision = filter_inbound_message(message, get_settings())
    if not decision.accepted or decision.message is None:
        return WebhookResponse(status="ignored")
    message = decision.message

    settings = get_settings()
    accepted_text = (message.text_content or "").strip()
    if not accepted_text:
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
                    session.add(ConversationState(user_id=user.id))
            except IntegrityError:
                user = await session.scalar(
                    select(User).where(User.phone_number == message.phone_number)
                )
                if user is None:
                    raise
        session.add(
            ProcessedMessage(
                provider=message.provider,
                instance_id=message.instance_id,
                external_message_id=message.external_message_id,
                user_id=user.id,
                accepted_text=accepted_text,
                message_timestamp=message.timestamp,
                status=ProcessedMessageStatus.PENDING,
                available_at=datetime.now(UTC),
            )
        )
        await session.commit()
    except IntegrityError:
        await session.rollback()
        return WebhookResponse(status="duplicate")
    return WebhookResponse(status="accepted")
