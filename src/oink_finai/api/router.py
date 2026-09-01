from fastapi import APIRouter

from oink_finai.api.routes.evolution_webhook import router as evolution_webhook_router
from oink_finai.api.routes.health import router as health_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(evolution_webhook_router)
