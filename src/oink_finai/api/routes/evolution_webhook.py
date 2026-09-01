import secrets
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from oink_finai.api.dependencies import get_evolution_provider
from oink_finai.config.settings import get_settings
from oink_finai.database.models.processed_message import ProcessedMessage
from oink_finai.database.session import get_session
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
    if message.from_me or message.remote_jid.endswith("@g.us"):
        return WebhookResponse(status="ignored")
    if message.text_content is None:
        return WebhookResponse(status="ignored")

    session.add(
        ProcessedMessage(
            provider=message.provider,
            instance_id=message.instance_id,
            external_message_id=message.external_message_id,
        )
    )
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        return WebhookResponse(status="duplicate")
    return WebhookResponse(status="accepted")
