from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from oink_finai.config.settings import Settings

ENV_EXAMPLE = Path(__file__).parents[1] / ".env.example"
RETRY_ENV_NAMES = (
    "EXPENSE_PROCESSING_MAX_ATTEMPTS",
    "EXPENSE_RETRY_BASE_SECONDS",
    "EXPENSE_RETRY_MAX_SECONDS",
)


def clear_retry_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in RETRY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_settings_loads_env_example(monkeypatch: pytest.MonkeyPatch) -> None:
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            monkeypatch.delenv(line.partition("=")[0], raising=False)

    settings = Settings(_env_file=ENV_EXAMPLE)

    assert settings.openai_expense_model == "gpt-4.1-mini"
    assert settings.openai_image_model == "gpt-4.1-mini"
    assert settings.openai_audio_transcription_model == "gpt-transcribe"
    assert settings.outbox_state_timeout_seconds == 300.0
    assert settings.expense_processing_max_attempts == 3
    assert settings.expense_retry_base_seconds == 0.5
    assert settings.expense_retry_max_seconds == 5.0


def test_default_openai_models_and_private_key_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_EXPENSE_MODEL",
        "OPENAI_IMAGE_MODEL",
        "OPENAI_AUDIO_TRANSCRIPTION_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings(_env_file=None)

    assert settings.openai_api_key is None
    assert settings.openai_expense_model == "gpt-4.1-mini"
    assert settings.openai_image_model == "gpt-4.1-mini"
    assert settings.openai_audio_transcription_model == "gpt-transcribe"
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-private-key")
    assert "synthetic-private-key" not in repr(Settings(_env_file=None))
    assert isinstance(Settings(_env_file=None).database_url, SecretStr)


def test_canonical_retry_environment_names_configure_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_retry_environment(monkeypatch)
    monkeypatch.setenv("EXPENSE_PROCESSING_MAX_ATTEMPTS", "6")
    monkeypatch.setenv("EXPENSE_RETRY_BASE_SECONDS", "2.5")
    monkeypatch.setenv("EXPENSE_RETRY_MAX_SECONDS", "45")

    settings = Settings(_env_file=None)

    assert settings.expense_processing_max_attempts == 6
    assert settings.expense_retry_base_seconds == 2.5
    assert settings.expense_retry_max_seconds == 45.0


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("OUTBOX_STATE_TIMEOUT_SECONDS", "300e"),
        ("EXPENSE_PROCESSING_MAX_ATTEMPTS", "invalid"),
        ("EXPENSE_RETRY_BASE_SECONDS", "invalid"),
        ("EXPENSE_RETRY_MAX_SECONDS", "invalid"),
        ("OPENAI_EXPENSE_TIMEOUT_SECONDS", "nan"),
        ("OPENAI_IMAGE_TIMEOUT_SECONDS", "inf"),
        ("OPENAI_AUDIO_TRANSCRIPTION_TIMEOUT_SECONDS", "0"),
        ("OPENAI_AUDIO_TRANSCRIPTION_LANGUAGE", "pt-BR"),
    ],
)
def test_invalid_numeric_environment_values_are_rejected(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    clear_retry_environment(monkeypatch)
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_all_documented_expense_variables_are_consumed() -> None:
    documented_names = {
        line.partition("=")[0]
        for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
        if line.startswith("EXPENSE_")
    }
    assert documented_names == {
        "EXPENSE_PROCESSING_MAX_ATTEMPTS",
        "EXPENSE_RETRY_BASE_SECONDS",
        "EXPENSE_RETRY_MAX_SECONDS",
    }
    assert all(name.casefold() in Settings.model_fields for name in documented_names)


def test_openai_environment_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "OPENAI_API_KEY": "synthetic-private-key",
        "OPENAI_EXPENSE_MODEL": "expense-model",
        "OPENAI_EXPENSE_TIMEOUT_SECONDS": "11.5",
        "OPENAI_IMAGE_MODEL": "image-model",
        "OPENAI_IMAGE_TIMEOUT_SECONDS": "12.5",
        "OPENAI_AUDIO_TRANSCRIPTION_MODEL": "gpt-transcribe",
        "OPENAI_AUDIO_TRANSCRIPTION_TIMEOUT_SECONDS": "15.5",
        "OPENAI_AUDIO_TRANSCRIPTION_LANGUAGE": "en",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    settings = Settings(_env_file=None)

    assert settings.openai_api_key_value == "synthetic-private-key"
    assert settings.openai_expense_model == "expense-model"
    assert settings.openai_expense_timeout_seconds == 11.5
    assert settings.openai_image_model == "image-model"
    assert settings.openai_image_timeout_seconds == 12.5
    assert settings.openai_audio_transcription_model == "gpt-transcribe"
    assert settings.openai_audio_transcription_timeout_seconds == 15.5
    assert settings.openai_audio_transcription_language == "en"


def production_settings(**overrides) -> Settings:
    values = {
        "app_env": "production",
        "database_url": (
            "postgresql+asyncpg://oink_runtime:strong-db-credential@postgres:5432/oink"
        ),
        "openai_api_key": "sk-runtime-validation-0123456789",
        "evolution_base_url": "https://evolution.test",
        "evolution_api_key": "evolution-runtime-key-0123456789",
        "evolution_instance": "primary-instance",
        "evolution_webhook_secret": "webhook-runtime-secret-0123456789abcdef",
        "whatsapp_allowed_numbers": "+5511999999999",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_production_settings_accept_a_complete_safe_configuration() -> None:
    settings = production_settings()

    assert settings.app_env == "production"
    assert settings.database_url_value.startswith("postgresql+asyncpg://")
    assert settings.usage_window_timezone == "UTC"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("INBOUND_USER_PER_MINUTE_LIMIT", "0"),
        ("INBOUND_GLOBAL_PER_DAY_LIMIT", "10001"),
        ("OPENAI_GLOBAL_CONCURRENCY_LIMIT", "0"),
        ("OPENAI_TEXT_USER_PER_DAY_LIMIT", "1001"),
        ("OPENAI_EXPENSE_MAX_OUTPUT_TOKENS", "0"),
        ("USAGE_WINDOW_TIMEZONE", "America/Sao_Paulo"),
        ("EVOLUTION_WEBHOOK_MAX_BODY_BYTES", "0"),
        ("EVOLUTION_WEBHOOK_MAX_BODY_BYTES", "1048577"),
        ("EVOLUTION_WEBHOOK_HTTP_TIMEOUT_SECONDS", "0"),
        ("EVOLUTION_WEBHOOK_MAX_CONCURRENCY", "0"),
        ("READINESS_DATABASE_TIMEOUT_SECONDS", "0"),
        ("WORKER_HEARTBEAT_INTERVAL_SECONDS", "0"),
        ("WORKER_HEARTBEAT_RETENTION_DAYS", "0"),
        ("WORKER_HEARTBEAT_ID_PATH", "../unsafe"),
    ],
)
def test_unsafe_usage_limits_are_rejected(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_incoherent_usage_limits_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            openai_user_per_day_limit=10,
            openai_text_user_per_day_limit=11,
        )


def test_incoherent_heartbeat_thresholds_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            worker_heartbeat_interval_seconds=15,
            worker_heartbeat_stale_seconds=44,
        )
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            worker_heartbeat_interval_seconds=5,
            worker_heartbeat_database_timeout_seconds=5,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"app_debug": True},
        {"app_reload": True},
        {"whatsapp_self_test_enabled": True},
        {"openai_api_key": None},
        {"openai_api_key": "change-me-openai-key-012345"},
        {"evolution_api_key": None},
        {"evolution_webhook_secret": "weak"},
        {"evolution_webhook_secret": "a" * 32},
        {"database_url": "postgresql+asyncpg://oink:oink@postgres:5432/oink"},
        {"database_url": "postgresql+asyncpg://%6fink:strong-password@postgres:5432/oink"},
        {"database_url": "not-a-database-url"},
        {"evolution_base_url": "http://evolution.test"},
        {"evolution_base_url": "https://user:password@evolution.test"},
        {"evolution_instance": "your-instance"},
        {"whatsapp_allowed_numbers": ""},
    ],
)
def test_production_rejects_unsafe_configuration(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        production_settings(**overrides)


def test_validation_error_does_not_render_secret_input() -> None:
    secret = "change-me-private-value-0123456789"

    with pytest.raises(ValidationError) as caught:
        production_settings(evolution_webhook_secret=secret)

    assert secret not in str(caught.value)

    database_secret = "database-secret-in-invalid-port"
    with pytest.raises(ValidationError) as database_error:
        production_settings(
            database_url=(
                f"postgresql+asyncpg://runtime:strong-password@postgres:{database_secret}/oink"
            )
        )

    rendered_error = str(database_error.value) + repr(database_error.value.errors())
    assert database_secret not in rendered_error


def test_secret_fields_are_masked_in_repr_and_json() -> None:
    values = {
        "database_url": "postgresql+asyncpg://private:database-secret@localhost/oink",
        "redis_url": "redis://:redis-secret@localhost/0",
        "openai_api_key": "private-openai-key",
        "evolution_api_key": "private-evolution-key",
        "evolution_webhook_secret": "private-webhook-secret",
    }
    settings = Settings(_env_file=None, **values)
    rendered = repr(settings) + settings.model_dump_json()

    assert all(value not in rendered for value in values.values())


def test_runtime_relationships_are_validated() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, expense_retry_base_seconds=10, expense_retry_max_seconds=5)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, worker_processing_lock_timeout_seconds=200)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, outbox_state_timeout_seconds=15)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, image_max_width=100, image_max_height=100)


def test_unknown_app_environment_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, app_env="staging")
