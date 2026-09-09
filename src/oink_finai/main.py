from fastapi import FastAPI

from oink_finai.api.middleware import EvolutionWebhookGuardMiddleware
from oink_finai.api.router import api_router
from oink_finai.config.settings import Settings, get_settings
from oink_finai.observability import configure_application_logging


def create_app(
    settings: Settings | None = None, *, configure_runtime_logging: bool = False
) -> FastAPI:
    runtime_settings = settings or get_settings()
    production = runtime_settings.app_env == "production"
    if production and configure_runtime_logging:
        configure_application_logging(
            log_format=runtime_settings.log_format,
            level=runtime_settings.log_level,
            service="api",
            include_traceback=runtime_settings.log_include_traceback,
        )
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
        log_requests=production,
    )
    application.include_router(api_router)
    return application


app = create_app(configure_runtime_logging=True)
