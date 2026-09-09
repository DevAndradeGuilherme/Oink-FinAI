import os
from collections.abc import Mapping
from urllib.parse import unquote, urlsplit


def migration_database_url(environ: Mapping[str, str] | None = None) -> str:
    """Return the Alembic URL without loading application-only settings."""
    values = os.environ if environ is None else environ
    production = values.get("APP_ENV", "development").strip().casefold() == "production"
    variable = "MIGRATION_DATABASE_URL" if production else None
    url = values.get("MIGRATION_DATABASE_URL")
    if not url and not production:
        variable = "DATABASE_URL"
        url = values.get("DATABASE_URL")

    if not url:
        required = variable or "MIGRATION_DATABASE_URL or DATABASE_URL"
        raise RuntimeError(f"{required} is required for Alembic")

    try:
        parsed = urlsplit(url)
        _ = parsed.port
    except ValueError:
        raise RuntimeError("the Alembic database URL is invalid") from None

    username = unquote(parsed.username or "").strip().casefold()
    password = unquote(parsed.password or "").strip().casefold()
    placeholders = {"", "postgres", "oink", "password", "change-me", "changeme"}
    if (
        parsed.scheme != "postgresql+asyncpg"
        or not parsed.hostname
        or username in placeholders
        or password in placeholders
        or parsed.path in {"", "/"}
        or parsed.fragment
        or any(character.isspace() for character in url)
    ):
        raise RuntimeError("the Alembic database URL is invalid")
    return url
