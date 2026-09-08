from collections.abc import AsyncIterator

from fastapi import HTTPException, status

from oink_finai.config.settings import get_settings
from oink_finai.providers.whatsapp.evolution import EvolutionWhatsAppProvider


async def get_evolution_provider() -> AsyncIterator[EvolutionWhatsAppProvider]:
    settings = get_settings()
    evolution_api_key = settings.evolution_api_key_value
    if not all((settings.evolution_base_url, evolution_api_key, settings.evolution_instance)):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Evolution provider is not configured",
        )
    provider = EvolutionWhatsAppProvider(
        base_url=settings.evolution_base_url,
        api_key=evolution_api_key,
        instance=settings.evolution_instance,
        timeout_seconds=settings.evolution_timeout_seconds,
        media_timeout_seconds=settings.evolution_media_timeout_seconds,
        media_max_bytes=settings.media_max_bytes,
        media_max_duration_seconds=settings.media_max_duration_seconds,
        image_max_width=settings.image_max_width,
        image_max_height=settings.image_max_height,
        image_max_pixels=settings.image_max_pixels,
        max_retries=0,
    )
    try:
        yield provider
    finally:
        await provider.aclose()
