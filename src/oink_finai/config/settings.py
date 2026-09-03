from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Oink FinAI"
    app_env: str = "development"
    app_debug: bool = False
    database_url: str = "postgresql+asyncpg://oink:oink@localhost:5432/oink"
    redis_url: str = "redis://localhost:6379/0"
    default_timezone: str = "America/Sao_Paulo"
    evolution_api_url: str | None = None
    evolution_api_key: str | None = Field(default=None, repr=False)
    evolution_instance_id: str | None = None
    evolution_webhook_secret: str | None = Field(default=None, repr=False)
    whatsapp_allowed_numbers: str = ""
    media_max_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    media_max_duration_seconds: int = Field(default=5 * 60, gt=0)
    evolution_media_timeout_seconds: float = Field(default=15.0, gt=0)

    @property
    def allowed_numbers(self) -> frozenset[str]:
        return frozenset(
            value.strip() for value in self.whatsapp_allowed_numbers.split(",") if value.strip()
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
