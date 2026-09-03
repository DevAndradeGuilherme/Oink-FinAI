from fastapi import APIRouter

from oink_finai.api.routes.health import router as health_router
from oink_finai.api.routes.whatsapp import router as whatsapp_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(whatsapp_router)
