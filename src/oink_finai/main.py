from fastapi import FastAPI

from oink_finai.api.router import api_router
from oink_finai.config.settings import get_settings

settings = get_settings()

app = FastAPI(title=settings.app_name, debug=settings.app_debug)
app.include_router(api_router)
