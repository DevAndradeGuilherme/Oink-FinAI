import re
from functools import lru_cache
from typing import Any, Literal
from urllib.parse import unquote, urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, SecretStr, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
        hide_input_in_errors=True,
    )

    def __init__(self, **values: Any) -> None:
        try:
            super().__init__(**values)
        except ValidationError as exc:
            sanitized_errors = exc.errors(include_url=False, include_input=False)
            raise ValidationError.from_exception_data(
                self.__class__.__name__, sanitized_errors
            ) from None

    app_name: str = "Oink FinAI"
    app_env: Literal["development", "test", "production"] = "development"
    app_debug: bool = False
    app_reload: bool = False
    pipeline_timing_enabled: bool = False
    app_release: str | None = Field(default=None, max_length=128)
    database_url: SecretStr = SecretStr("postgresql+asyncpg://oink:oink@localhost:5432/oink")
    redis_url: SecretStr = SecretStr("redis://localhost:6379/0")
    default_timezone: str = "America/Sao_Paulo"
    openai_api_key: SecretStr | None = None
    openai_expense_model: str = "gpt-4.1-mini"
    openai_expense_timeout_seconds: float = Field(default=90.0, gt=0, le=120, allow_inf_nan=False)
    openai_expense_max_output_tokens: int = Field(default=600, ge=64, le=4096)
    openai_query_model: str = "gpt-4.1-mini"
    openai_query_timeout_seconds: float = Field(default=90.0, gt=0, le=120, allow_inf_nan=False)
    openai_query_max_output_tokens: int = Field(default=600, ge=64, le=4096)
    openai_image_model: str = "gpt-4.1-mini"
    openai_image_timeout_seconds: float = Field(default=90.0, gt=0, le=120, allow_inf_nan=False)
    openai_image_max_output_tokens: int = Field(default=500, ge=64, le=4096)
    openai_audio_transcription_model: str = "gpt-transcribe"
    openai_audio_transcription_timeout_seconds: float = Field(
        default=90.0, gt=0, le=120, allow_inf_nan=False
    )
    openai_audio_transcription_language: str = Field(default="pt", pattern=r"^[a-z]{2}$")
    evolution_base_url: str | None = None
    evolution_api_key: SecretStr | None = None
    evolution_instance: str | None = None
    evolution_webhook_secret: SecretStr | None = None
    whatsapp_access_mode: Literal["allowlist"] = "allowlist"
    whatsapp_allowed_numbers: str = ""
    whatsapp_self_test_enabled: bool = False
    whatsapp_self_test_number: str | None = Field(default=None, repr=False)
    whatsapp_self_test_prefix: str = "!oink"
    inbound_message_max_length: int = Field(default=2000, ge=1, le=10000)
    evolution_webhook_max_body_bytes: int = Field(default=262_144, ge=1_024, le=1_048_576)
    evolution_webhook_http_timeout_seconds: float = Field(
        default=10.0, gt=0, le=30, allow_inf_nan=False
    )
    evolution_webhook_max_concurrency: int = Field(default=8, ge=1, le=64)
    readiness_database_timeout_seconds: float = Field(default=2.0, gt=0, le=5, allow_inf_nan=False)
    usage_window_timezone: Literal["UTC"] = "UTC"
    inbound_user_per_minute_limit: int = Field(default=10, ge=1, le=120)
    inbound_user_per_day_limit: int = Field(default=150, ge=1, le=2_000)
    inbound_global_per_day_limit: int = Field(default=600, ge=1, le=10_000)
    openai_user_per_day_limit: int = Field(default=60, ge=1, le=1_000)
    openai_global_per_day_limit: int = Field(default=240, ge=1, le=5_000)
    openai_global_concurrency_limit: int = Field(default=2, ge=1, le=32)
    openai_text_user_per_day_limit: int = Field(default=40, ge=1, le=1_000)
    openai_text_global_per_day_limit: int = Field(default=160, ge=1, le=5_000)
    openai_query_user_per_day_limit: int = Field(default=20, ge=1, le=1_000)
    openai_query_global_per_day_limit: int = Field(default=80, ge=1, le=5_000)
    openai_image_user_per_day_limit: int = Field(default=12, ge=1, le=1_000)
    openai_image_global_per_day_limit: int = Field(default=48, ge=1, le=5_000)
    openai_audio_user_per_day_limit: int = Field(default=10, ge=1, le=1_000)
    openai_audio_global_per_day_limit: int = Field(default=40, ge=1, le=5_000)
    openai_reservation_stale_seconds: float = Field(default=330.0, gt=0, le=3600)
    openai_concurrency_retry_seconds: float = Field(default=30.0, gt=0, le=600)
    worker_poll_interval_seconds: float = Field(default=1.0, gt=0, le=60)
    worker_batch_size: int = Field(default=10, ge=1, le=100)
    worker_processing_lock_timeout_seconds: float = Field(default=300.0, gt=0, le=3600)
    worker_heartbeat_interval_seconds: float = Field(default=15.0, ge=5, le=60)
    worker_heartbeat_stale_seconds: float = Field(default=60.0, ge=15, le=600)
    worker_heartbeat_database_timeout_seconds: float = Field(default=3.0, gt=0, le=10)
    worker_heartbeat_retention_days: int = Field(default=7, ge=1, le=90)
    worker_heartbeat_id_path: str = "/tmp/oink-finai-worker-id"
    expense_processing_max_attempts: int = Field(default=4, ge=1, le=10)
    expense_retry_base_seconds: float = Field(default=30.0, gt=0, le=3600)
    expense_retry_max_seconds: float = Field(default=300.0, gt=0, le=86400)
    outbox_max_attempts: int = Field(default=3, ge=1, le=10)
    outbox_retry_base_seconds: float = Field(default=1.0, gt=0, le=3600)
    outbox_state_timeout_seconds: float = Field(default=300.0, gt=0, le=3600)
    evolution_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    evolution_media_timeout_seconds: float = Field(default=15.0, gt=0, le=60)
    media_max_bytes: int = Field(default=10 * 1024 * 1024, gt=0, le=25 * 1024 * 1024)
    media_max_duration_seconds: int = Field(default=300, gt=0, le=600)
    image_max_width: int = Field(default=4096, gt=0, le=8192)
    image_max_height: int = Field(default=4096, gt=0, le=8192)
    image_max_pixels: int = Field(default=16_000_000, gt=0, le=32_000_000)
    expense_delete_confirmation_ttl_seconds: float = Field(default=600.0, gt=0, le=86400)
    whatsapp_query_message_max_chars: int = Field(default=3500, ge=160, le=4096)
    whatsapp_query_max_pages: int = Field(default=10, ge=1, le=20)

    @property
    def whatsapp_allowed_number_set(self) -> frozenset[str]:
        return frozenset(
            normalized
            for value in self.whatsapp_allowed_numbers.split(",")
            if (normalized := "".join(character for character in value if character.isdigit()))
        )

    @property
    def database_url_value(self) -> str:
        return self.database_url.get_secret_value()

    @property
    def redis_url_value(self) -> str:
        return self.redis_url.get_secret_value()

    @property
    def openai_api_key_value(self) -> str | None:
        return self.openai_api_key.get_secret_value() if self.openai_api_key else None

    @property
    def evolution_api_key_value(self) -> str | None:
        return self.evolution_api_key.get_secret_value() if self.evolution_api_key else None

    @property
    def evolution_webhook_secret_value(self) -> str | None:
        if self.evolution_webhook_secret is None:
            return None
        return self.evolution_webhook_secret.get_secret_value()

    @model_validator(mode="after")
    def validate_runtime_configuration(self) -> "Settings":
        if self.expense_retry_base_seconds > self.expense_retry_max_seconds:
            raise ValueError("expense retry base must not exceed its maximum")

        longest_pipeline = max(
            self.evolution_media_timeout_seconds
            + self.openai_audio_transcription_timeout_seconds
            + self.openai_expense_timeout_seconds,
            self.evolution_media_timeout_seconds
            + self.openai_image_timeout_seconds
            + self.openai_expense_timeout_seconds,
            self.openai_expense_timeout_seconds + self.openai_query_timeout_seconds,
        )
        if self.worker_processing_lock_timeout_seconds <= longest_pipeline + 30:
            raise ValueError("processing lock timeout must exceed the longest pipeline timeout")
        if self.outbox_state_timeout_seconds <= self.evolution_timeout_seconds + 5:
            raise ValueError("outbox state timeout must exceed the provider timeout")
        if self.image_max_pixels > self.image_max_width * self.image_max_height:
            raise ValueError("image pixel limit must not exceed the dimension limit")
        if self.worker_heartbeat_stale_seconds < 3 * self.worker_heartbeat_interval_seconds:
            raise ValueError("worker heartbeat stale threshold must be at least three intervals")
        if self.worker_heartbeat_database_timeout_seconds >= self.worker_heartbeat_interval_seconds:
            raise ValueError("worker heartbeat database timeout must be shorter than its interval")
        if not re.fullmatch(r"/tmp/[A-Za-z0-9._-]{1,100}", self.worker_heartbeat_id_path):
            raise ValueError("worker heartbeat ID path must be a safe file in /tmp")
        if self.app_release is not None and not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", self.app_release
        ):
            raise ValueError("APP_RELEASE must be an opaque identifier")
        self._validate_usage_limits()

        try:
            ZoneInfo(self.default_timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("default timezone is invalid") from exc

        if self.app_env == "production":
            self._validate_production()
        return self

    def _validate_production(self) -> None:
        if self.app_debug or self.app_reload:
            raise ValueError("debug and reload must be disabled in production")
        if self.whatsapp_self_test_enabled:
            raise ValueError("WhatsApp self-test must be disabled in production")

        self._require_production_secret("OPENAI_API_KEY", self.openai_api_key, minimum=20)
        self._require_production_secret("EVOLUTION_API_KEY", self.evolution_api_key, minimum=16)
        self._require_production_secret(
            "EVOLUTION_WEBHOOK_SECRET", self.evolution_webhook_secret, minimum=32
        )
        webhook_secret = self.evolution_webhook_secret_value
        if webhook_secret is None or len(set(webhook_secret)) < 12:
            raise ValueError("EVOLUTION_WEBHOOK_SECRET is weak in production")

        database_url = self.database_url_value
        try:
            parsed_database = urlsplit(database_url)
            _ = parsed_database.port
        except ValueError:
            raise ValueError("DATABASE_URL is invalid") from None
        if (
            parsed_database.scheme != "postgresql+asyncpg"
            or not parsed_database.hostname
            or not parsed_database.username
            or parsed_database.password is None
            or not parsed_database.password
            or parsed_database.path in {"", "/"}
            or parsed_database.fragment
            or any(character.isspace() for character in database_url)
        ):
            raise ValueError("DATABASE_URL is invalid")
        default_credentials = {"oink", "postgres", "password", "changeme", "change-me"}
        database_username = unquote(parsed_database.username).casefold()
        database_password = unquote(parsed_database.password).casefold()
        if database_username in default_credentials or database_password in default_credentials:
            raise ValueError("DATABASE_URL uses default credentials")

        if self.evolution_base_url is None:
            raise ValueError("EVOLUTION_BASE_URL is required in production")
        try:
            parsed_evolution = urlsplit(self.evolution_base_url)
            _ = parsed_evolution.port
        except ValueError as exc:
            raise ValueError("EVOLUTION_BASE_URL is invalid") from exc
        if (
            parsed_evolution.scheme != "https"
            or not parsed_evolution.hostname
            or parsed_evolution.username is not None
            or parsed_evolution.password is not None
            or parsed_evolution.fragment
            or any(character.isspace() for character in self.evolution_base_url)
        ):
            raise ValueError("EVOLUTION_BASE_URL must be an HTTPS URL in production")
        if self._is_placeholder(self.evolution_instance):
            raise ValueError("EVOLUTION_INSTANCE is invalid in production")
        if not self.whatsapp_allowed_number_set:
            raise ValueError("WHATSAPP_ALLOWED_NUMBERS is required in production")
        if self._is_placeholder(self.openai_expense_model) or self._is_placeholder(
            self.openai_image_model
        ):
            raise ValueError("OpenAI model names are required in production")
        if self._is_placeholder(self.openai_query_model):
            raise ValueError("OpenAI model names are required in production")
        if self._is_placeholder(self.openai_audio_transcription_model):
            raise ValueError("OpenAI model names are required in production")
        for model in (
            self.openai_expense_model,
            self.openai_query_model,
            self.openai_image_model,
            self.openai_audio_transcription_model,
        ):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", model):
                raise ValueError("OpenAI model name is invalid in production")

    def _validate_usage_limits(self) -> None:
        if self.inbound_user_per_day_limit > self.inbound_global_per_day_limit:
            raise ValueError("per-user inbound limit must not exceed the global limit")
        if self.openai_user_per_day_limit > self.openai_global_per_day_limit:
            raise ValueError("per-user OpenAI limit must not exceed the global limit")
        if self.openai_global_concurrency_limit > self.openai_global_per_day_limit:
            raise ValueError("OpenAI concurrency must not exceed the global daily limit")
        typed_limits = (
            (self.openai_text_user_per_day_limit, self.openai_text_global_per_day_limit),
            (self.openai_query_user_per_day_limit, self.openai_query_global_per_day_limit),
            (self.openai_image_user_per_day_limit, self.openai_image_global_per_day_limit),
            (self.openai_audio_user_per_day_limit, self.openai_audio_global_per_day_limit),
        )
        for user_limit, global_limit in typed_limits:
            if user_limit > global_limit:
                raise ValueError("per-user operation limit must not exceed its global limit")
            if user_limit > self.openai_user_per_day_limit:
                raise ValueError("operation limit must not exceed the per-user OpenAI limit")
            if global_limit > self.openai_global_per_day_limit:
                raise ValueError("operation limit must not exceed the global OpenAI limit")

    @classmethod
    def _require_production_secret(
        cls,
        name: str,
        secret: SecretStr | None,
        *,
        minimum: int,
    ) -> None:
        if secret is None:
            raise ValueError(f"{name} is required in production")
        value = secret.get_secret_value()
        if len(value) < minimum or cls._is_placeholder(value):
            raise ValueError(f"{name} is invalid in production")
        if not all(33 <= ord(character) <= 126 for character in value):
            raise ValueError(f"{name} must contain printable non-space ASCII")

    @staticmethod
    def _is_placeholder(value: str | None) -> bool:
        if value is None:
            return True
        normalized = value.strip().casefold()
        return (
            not normalized
            or normalized in {"test", "testing", "dummy", "placeholder"}
            or normalized.startswith("your-")
            or any(marker in normalized for marker in ("change-me", "changeme", "replace-me"))
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
