from functools import lru_cache
from typing import Literal

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
    gemini_api_key: str | None = Field(default=None, repr=False)
    gemini_model: str | None = None
    gemini_timeout_seconds: float = Field(default=30.0, gt=0)
    evolution_base_url: str | None = None
    evolution_api_key: str | None = Field(default=None, repr=False)
    evolution_instance: str | None = None
    evolution_webhook_secret: str | None = Field(default=None, repr=False)
    whatsapp_access_mode: Literal["allowlist"] = "allowlist"
    whatsapp_allowed_numbers: str = ""
    whatsapp_self_test_enabled: bool = False
    whatsapp_self_test_number: str | None = Field(default=None, repr=False)
    whatsapp_self_test_prefix: str = "!oink"
    inbound_message_max_length: int = Field(default=2000, ge=1, le=10000)
    worker_poll_interval_seconds: float = Field(default=1.0, gt=0)
    worker_batch_size: int = Field(default=10, ge=1, le=100)
    worker_processing_lock_timeout_seconds: float = Field(default=300.0, gt=0)
    gemini_max_attempts: int = Field(default=3, ge=1, le=5)
    gemini_retry_base_seconds: float = Field(default=0.5, gt=0)
    gemini_retry_max_seconds: float = Field(default=5.0, gt=0)
    outbox_max_attempts: int = Field(default=3, ge=1, le=10)
    outbox_retry_base_seconds: float = Field(default=1.0, gt=0)
    outbox_state_timeout_seconds: float = Field(default=300.0, gt=0)
    evolution_timeout_seconds: float = Field(default=10.0, gt=0)

    @property
    def whatsapp_allowed_number_set(self) -> frozenset[str]:
        return frozenset(
            normalized
            for value in self.whatsapp_allowed_numbers.split(",")
            if (normalized := "".join(character for character in value if character.isdigit()))
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
