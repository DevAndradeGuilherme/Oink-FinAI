from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from oink_finai.config.settings import get_settings
from oink_finai.database.session import SessionFactory
from oink_finai.schemas.health import HealthResponse
from oink_finai.services.readiness import DatabaseReadiness

router = APIRouter(tags=["health"])


def get_database_readiness() -> DatabaseReadiness:
    settings = get_settings()
    return DatabaseReadiness(
        SessionFactory, timeout_seconds=settings.readiness_database_timeout_seconds
    )


@router.get("/live", response_model=HealthResponse)
async def live() -> HealthResponse:
    return HealthResponse(status="ok")


async def _readiness_response(checker: DatabaseReadiness) -> HealthResponse | JSONResponse:
    if await checker.is_ready():
        return HealthResponse(status="ok")
    return JSONResponse(status_code=503, content={"status": "unavailable"})


@router.get("/ready", response_model=HealthResponse)
async def ready(
    checker: Annotated[DatabaseReadiness, Depends(get_database_readiness)],
) -> HealthResponse | JSONResponse:
    return await _readiness_response(checker)


@router.get("/health", response_model=HealthResponse)
async def health(
    checker: Annotated[DatabaseReadiness, Depends(get_database_readiness)],
) -> HealthResponse | JSONResponse:
    """Compatibility alias for readiness; success body remains {"status": "ok"}."""
    return await _readiness_response(checker)
