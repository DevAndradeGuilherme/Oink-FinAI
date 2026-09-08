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
    for process in (api, worker):
        assert "scale: 1" in process
        assert "init: true" in process
        assert "stop_grace_period: 330s" in process
        assert "condition: service_completed_successfully" in process
        assert "replicas: 1" in process


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
