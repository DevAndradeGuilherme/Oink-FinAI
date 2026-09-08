from pathlib import Path

import pytest
from pydantic import ValidationError

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
    assert Settings.model_fields["openai_api_key"].repr is False


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

    assert settings.openai_api_key == "synthetic-private-key"
    assert settings.openai_expense_model == "expense-model"
    assert settings.openai_expense_timeout_seconds == 11.5
    assert settings.openai_image_model == "image-model"
    assert settings.openai_image_timeout_seconds == 12.5
    assert settings.openai_audio_transcription_model == "gpt-transcribe"
    assert settings.openai_audio_transcription_timeout_seconds == 15.5
    assert settings.openai_audio_transcription_language == "en"
