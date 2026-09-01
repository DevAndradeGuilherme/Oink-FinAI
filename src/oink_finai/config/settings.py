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


@lru_cache
def get_settings() -> Settings:
    return Settings()
