from collections.abc import AsyncIterator

from fastapi import HTTPException, status

from oink_finai.config.settings import get_settings
from oink_finai.providers.whatsapp.evolution import EvolutionWhatsAppProvider


async def get_evolution_provider() -> AsyncIterator[EvolutionWhatsAppProvider]:
    settings = get_settings()
    if not all(
        (settings.evolution_base_url, settings.evolution_api_key, settings.evolution_instance)
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Evolution provider is not configured",
        )
    provider = EvolutionWhatsAppProvider(
        base_url=settings.evolution_base_url,
        api_key=settings.evolution_api_key,
        instance=settings.evolution_instance,
    )
    try:
        yield provider
    finally:
        await provider.aclose()
