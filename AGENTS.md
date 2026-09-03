# Repository Guidelines

## Project Structure & Module Organization

Production code lives under `src/oink_finai/`, split into API, configuration, database, domain, schemas, services, repositories, and providers. Tests live in `tests/`; Alembic revisions live in `migrations/`. Keep provider-specific WhatsApp code behind `providers/whatsapp/WhatsAppProvider`.

## Build, Test, and Development Commands

Use `pip install -e ".[dev]"` for local dependencies, `pytest` for tests, `ruff check .` for linting, and `ruff format --check .` for formatting verification. Run the stack with `docker compose up --build`. Apply schema changes with `alembic upgrade head`.

## Coding Style & Naming Conventions

Target Python 3.12 with four-space indentation, type hints, 100-character lines, and Ruff rules configured in `pyproject.toml`. Use `snake_case` for modules/functions, `PascalCase` for classes, and `UPPER_CASE` for constants. Test files use `test_*.py`. Use `Decimal`, never `float`, for money.

## Testing Guidelines

Add Pytest coverage with every behavioral change or bug fix. Cover normal behavior, boundaries, constraints, and failure paths. Tests must remain deterministic and independent of credentials or external services. Model tests use async SQLite; validate PostgreSQL-specific migrations separately.

## Commit & Pull Request Guidelines

History currently contains one descriptive commit (`First commit > add readme`), so no established convention exists. Use short, imperative subjects such as `Add transaction import validation`. Keep commits focused. Pull requests should explain purpose, key changes, verification performed, and remaining risks; link relevant issues and include screenshots for visible UI changes. Never commit secrets, local environment files, generated artifacts, or dependency caches.
