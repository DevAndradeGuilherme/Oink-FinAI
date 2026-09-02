from pathlib import Path

import pytest
from pydantic import ValidationError

from oink_finai.config.settings import Settings

ENV_EXAMPLE = Path(__file__).parents[1] / ".env.example"
RETRY_ENV_NAMES = (
    "EXPENSE_PROCESSING_MAX_ATTEMPTS",
    "EXPENSE_RETRY_BASE_SECONDS",
    "EXPENSE_RETRY_MAX_SECONDS",
    "GEMINI_MAX_ATTEMPTS",
    "GEMINI_RETRY_BASE_SECONDS",
    "GEMINI_RETRY_MAX_SECONDS",
)


def clear_retry_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in RETRY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_settings_loads_env_example(monkeypatch: pytest.MonkeyPatch) -> None:
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            monkeypatch.delenv(line.partition("=")[0], raising=False)

    settings = Settings(_env_file=ENV_EXAMPLE)

    assert settings.outbox_state_timeout_seconds == 300.0
    assert settings.expense_processing_max_attempts == 3
    assert settings.expense_retry_base_seconds == 0.5
    assert settings.expense_retry_max_seconds == 5.0


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


def test_legacy_retry_environment_names_remain_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_retry_environment(monkeypatch)
    monkeypatch.setenv("GEMINI_MAX_ATTEMPTS", "5")
    monkeypatch.setenv("GEMINI_RETRY_BASE_SECONDS", "1.5")
    monkeypatch.setenv("GEMINI_RETRY_MAX_SECONDS", "20")

    settings = Settings(_env_file=None)

    assert settings.expense_processing_max_attempts == 5
    assert settings.expense_retry_base_seconds == 1.5
    assert settings.expense_retry_max_seconds == 20.0


def test_canonical_retry_names_take_priority_over_legacy_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_retry_environment(monkeypatch)
    values = {
        "EXPENSE_PROCESSING_MAX_ATTEMPTS": "7",
        "EXPENSE_RETRY_BASE_SECONDS": "3",
        "EXPENSE_RETRY_MAX_SECONDS": "60",
        "GEMINI_MAX_ATTEMPTS": "2",
        "GEMINI_RETRY_BASE_SECONDS": "0.25",
        "GEMINI_RETRY_MAX_SECONDS": "4",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    settings = Settings(_env_file=None)

    assert settings.expense_processing_max_attempts == 7
    assert settings.expense_retry_base_seconds == 3.0
    assert settings.expense_retry_max_seconds == 60.0


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("OUTBOX_STATE_TIMEOUT_SECONDS", "300e"),
        ("EXPENSE_PROCESSING_MAX_ATTEMPTS", "invalid"),
        ("EXPENSE_RETRY_BASE_SECONDS", "invalid"),
        ("EXPENSE_RETRY_MAX_SECONDS", "invalid"),
    ],
)
def test_invalid_numeric_environment_values_are_rejected(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    clear_retry_environment(monkeypatch)
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_all_documented_retry_variables_are_consumed() -> None:
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
    for name in documented_names:
        assert any(
            name in getattr(field.validation_alias, "choices", ())
            for field in Settings.model_fields.values()
        )
