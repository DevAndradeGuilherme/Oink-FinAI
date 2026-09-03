from fastapi import APIRouter, Header, HTTPException, Request, status

from oink_finai.config.settings import get_settings
from oink_finai.providers.whatsapp.evolution import EvolutionWhatsAppProvider
from oink_finai.services.whatsapp_webhook import WebhookAccessError, WhatsAppWebhookService

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


@router.post("/evolution")
async def evolution_webhook(
    request: Request,
    x_webhook_secret: str | None = Header(default=None),
) -> dict[str, str]:
    settings = get_settings()
    provider = EvolutionWhatsAppProvider(
        base_url=settings.evolution_api_url,
        instance=settings.evolution_instance_id,
        api_key=settings.evolution_api_key,
        max_bytes=settings.media_max_bytes,
        max_duration_seconds=settings.media_max_duration_seconds,
        timeout_seconds=settings.evolution_media_timeout_seconds,
    )
    service = WhatsAppWebhookService(provider, settings)
    try:
        payload = await request.json()
    except ValueError:
        return {"status": "ignored"}
    if not isinstance(payload, dict):
        return {"status": "ignored"}
    try:
        result = await service.handle(payload, x_webhook_secret)
    except WebhookAccessError as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid webhook"
        ) from error
    return {"status": result.disposition.value}
