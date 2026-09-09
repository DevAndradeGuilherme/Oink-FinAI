from fastapi import FastAPI

from oink_finai.api.middleware import EvolutionWebhookGuardMiddleware
from oink_finai.api.router import api_router
from oink_finai.config.settings import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    runtime_settings = settings or get_settings()
    production = runtime_settings.app_env == "production"
    application = FastAPI(
        title=runtime_settings.app_name,
        debug=runtime_settings.app_debug,
        docs_url=None if production else "/docs",
        redoc_url=None if production else "/redoc",
        openapi_url=None if production else "/openapi.json",
    )
    application.add_middleware(
        EvolutionWebhookGuardMiddleware,
        max_body_bytes=runtime_settings.evolution_webhook_max_body_bytes,
        timeout_seconds=runtime_settings.evolution_webhook_http_timeout_seconds,
        max_concurrency=runtime_settings.evolution_webhook_max_concurrency,
    )
    application.include_router(api_router)
    return application


app = create_app()
