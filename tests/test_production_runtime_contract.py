import re
from pathlib import Path

from oink_finai.config.settings import Settings
from oink_finai.main import create_app

ROOT = Path(__file__).parents[1]


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


def service_block(compose: str, name: str) -> str:
    match = re.search(
        rf"^  {re.escape(name)}:\n(?P<body>(?:^(?:    |\s*$).*\n?)*)",
        compose,
        flags=re.MULTILINE,
    )
    assert match is not None, f"service {name} is missing"
    return match.group("body")


def test_production_compose_isolated_runtime_contract() -> None:
    compose = (ROOT / "docker-compose.prod.yml").read_text(encoding="utf-8")
    api = service_block(compose, "api")
    worker = service_block(compose, "worker")
    migrate = service_block(compose, "migrate")
    postgres = service_block(compose, "postgres")

    assert "build:" not in compose
    assert "container_name:" not in compose
    assert "redis:" not in compose
    assert "@${OINK_IMAGE_DIGEST:" in compose
    assert "./src" not in compose and "./migrations" not in compose
    assert "--reload" not in api and "alembic" not in api
    assert '"127.0.0.1:8000:8000"' in api
    assert "ports:" not in worker and "ports:" not in migrate and "ports:" not in postgres
    assert 'command: ["alembic", "upgrade", "head"]' in migrate
    assert 'restart: "no"' in migrate
    assert "OINK_MIGRATION_ENV_FILE" in migrate
    assert "OINK_RUNTIME_ENV_FILE" not in migrate
    assert "DATABASE_URL:" not in migrate
    assert "OPENAI_" not in migrate and "EVOLUTION_" not in migrate
    assert "POSTGRES_PASSWORD_FILE" in postgres
    assert "POSTGRES_PASSWORD:" not in postgres
    assert "--auth-local=scram-sha-256" in postgres
    for process in (api, worker):
        assert "scale: 1" in process
        assert "init: true" in process
        assert "stop_grace_period: 330s" in process
        assert "condition: service_completed_successfully" in process
        assert "replicas: 1" in process
        assert "OINK_RUNTIME_ENV_FILE" in process
        assert "MIGRATION_DATABASE_URL" not in process
    assert "urllib.request" in api and "127.0.0.1:8000/ready" in api
    assert "oink_finai.worker_healthcheck" in worker
    assert "curl" not in api + worker and "wget" not in api + worker
    assert "start_period: 45s" in worker


def test_manual_postgres_admin_service_is_isolated() -> None:
    compose = (ROOT / "docker-compose.postgres-admin.yml").read_text(encoding="utf-8")
    admin = service_block(compose, "postgres-admin")

    assert 'profiles: ["postgres-admin"]' in admin
    assert 'restart: "no"' in admin
    assert "ports:" not in admin
    assert "OINK_RUNTIME_ENV_FILE" not in admin
    assert "OINK_MIGRATION_ENV_FILE" not in admin
    assert "OPENAI_" not in admin and "EVOLUTION_" not in admin
    assert "/run/secrets/postgres_bootstrap_password" in admin
    assert "./scripts/postgres:/opt/oink/postgres:ro" in admin


def test_postgres_provisioning_contract_has_no_embedded_passwords_or_reassign_owned() -> None:
    script_directory = ROOT / "scripts" / "postgres"
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(script_directory.glob("*"))
        if path.is_file()
    )

    assert "REASSIGN OWNED" not in sources.upper()
    assert "PASSWORD NULL" in sources
    assert "PASSWORD :'" not in sources
    assert "PGPASSWORD=" not in sources
    assert "ALLOW_EXISTING_DATABASE_ADAPTATION" in sources
    assert "REVOKE ALL ON SCHEMA" in sources
    assert "ALTER DEFAULT PRIVILEGES" in sources
    assert "GRANT SELECT ON TABLE %I.alembic_version" in sources


def test_image_declares_a_non_root_runtime_user() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "USER oink:oink" in dockerfile
    assert dockerfile.index("--no-compile .") < dockerfile.index("USER oink:oink")
    assert "chmod 777" not in dockerfile
    assert 'python -c "import openai; import oink_finai"' in dockerfile


def test_docker_context_excludes_private_and_local_files() -> None:
    patterns = set((ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines())

    assert {".git", ".env", ".env.*", "*.sql", "*.dump", "*.backup"} <= patterns
    assert {"__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "venv"} <= patterns
    assert not any(pattern.startswith("!.env") for pattern in patterns)


def test_production_disables_schema_documentation_only() -> None:
    production_app = create_app(production_settings())

    assert production_app.docs_url is None
    assert production_app.redoc_url is None
    assert production_app.openapi_url is None

    development_app = create_app(production_settings(app_env="development"))
    assert development_app.docs_url == "/docs"
    assert development_app.openapi_url == "/openapi.json"
