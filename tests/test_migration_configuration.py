import pytest

from oink_finai.config.migration import migration_database_url

MIGRATION_URL = (
    "postgresql+asyncpg://migrator_role:synthetic-strong-password@postgres:5432/"
    "application_database"
)
RUNTIME_URL = (
    "postgresql+asyncpg://runtime_role:synthetic-runtime-password@postgres:5432/"
    "application_database"
)


def test_production_alembic_requires_dedicated_migration_url() -> None:
    with pytest.raises(RuntimeError, match="MIGRATION_DATABASE_URL is required"):
        migration_database_url({"APP_ENV": "production", "DATABASE_URL": RUNTIME_URL})


def test_production_alembic_uses_migration_url() -> None:
    assert (
        migration_database_url(
            {
                "APP_ENV": "production",
                "DATABASE_URL": RUNTIME_URL,
                "MIGRATION_DATABASE_URL": MIGRATION_URL,
            }
        )
        == MIGRATION_URL
    )


def test_development_alembic_retains_database_url_compatibility() -> None:
    assert migration_database_url({"APP_ENV": "development", "DATABASE_URL": RUNTIME_URL}) == (
        RUNTIME_URL
    )


@pytest.mark.parametrize(
    "url",
    [
        "postgresql+asyncpg://migrator_role:change-me@postgres:5432/application_database",
        "postgresql+asyncpg://postgres:strong-password@postgres:5432/application_database",
        "not-a-url",
    ],
)
def test_alembic_rejects_placeholders_without_echoing_url(url: str) -> None:
    with pytest.raises(RuntimeError) as captured:
        migration_database_url({"APP_ENV": "production", "MIGRATION_DATABASE_URL": url})

    assert url not in str(captured.value)
