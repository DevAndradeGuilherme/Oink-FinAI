from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    app_name: str = "Oink FinAI"
    app_env: str = "development"
    app_debug: bool = False
    pipeline_timing_enabled: bool = False
    database_url: str = "postgresql+asyncpg://oink:oink@localhost:5432/oink"
    redis_url: str = "redis://localhost:6379/0"
    default_timezone: str = "America/Sao_Paulo"
    openai_api_key: str | None = Field(default=None, repr=False)
    openai_expense_model: str = "gpt-4.1-mini"
    openai_expense_timeout_seconds: float = Field(default=90.0, gt=0, allow_inf_nan=False)
    openai_image_model: str = "gpt-4.1-mini"
    openai_image_timeout_seconds: float = Field(default=90.0, gt=0, allow_inf_nan=False)
    openai_audio_transcription_model: str = "gpt-transcribe"
    openai_audio_transcription_timeout_seconds: float = Field(
        default=90.0, gt=0, allow_inf_nan=False
    )
    openai_audio_transcription_language: str = Field(default="pt", pattern=r"^[a-z]{2}$")
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
    expense_processing_max_attempts: int = Field(default=4, ge=1, le=10)
    expense_retry_base_seconds: float = Field(default=30.0, gt=0)
    expense_retry_max_seconds: float = Field(default=300.0, gt=0)
    outbox_max_attempts: int = Field(default=3, ge=1, le=10)
    outbox_retry_base_seconds: float = Field(default=1.0, gt=0)
    outbox_state_timeout_seconds: float = Field(default=300.0, gt=0)
    evolution_timeout_seconds: float = Field(default=10.0, gt=0)
    evolution_media_timeout_seconds: float = Field(default=15.0, gt=0)
    media_max_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    media_max_duration_seconds: int = Field(default=300, gt=0)
    image_max_width: int = Field(default=4096, gt=0)
    image_max_height: int = Field(default=4096, gt=0)
    image_max_pixels: int = Field(default=16_000_000, gt=0)
    expense_delete_confirmation_ttl_seconds: float = Field(default=600.0, gt=0)
    expense_clarification_ttl_seconds: float = Field(default=900.0, gt=0)
    expense_clarification_min_confidence: float = Field(default=0.75, ge=0, le=1)

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
