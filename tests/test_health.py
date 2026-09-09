import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from oink_finai.api.routes.health import get_database_readiness
from oink_finai.main import app
from oink_finai.services.readiness import EXPECTED_ALEMBIC_HEAD, DatabaseReadiness


class FixedReadiness:
    def __init__(self, ready: bool) -> None:
        self.ready = ready
        self.calls = 0

    async def is_ready(self) -> bool:
        self.calls += 1
        return self.ready


def test_live_does_not_resolve_or_access_readiness() -> None:
    def forbidden_dependency():
        raise AssertionError("live must not access the database")

    app.dependency_overrides[get_database_readiness] = forbidden_dependency
    try:
        response = TestClient(app).get("/live")
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_and_compatible_health_are_database_readiness() -> None:
    checker = FixedReadiness(True)
    app.dependency_overrides[get_database_readiness] = lambda: checker
    try:
        with TestClient(app) as client:
            ready = client.get("/ready")
            health = client.get("/health")
    finally:
        app.dependency_overrides.clear()
    assert ready.status_code == 200 and ready.json() == {"status": "ok"}
    assert health.status_code == 200 and health.json() == {"status": "ok"}
    assert checker.calls == 2


def test_ready_and_health_fail_sanitized_when_database_is_not_ready() -> None:
    checker = FixedReadiness(False)
    app.dependency_overrides[get_database_readiness] = lambda: checker
    try:
        with TestClient(app) as client:
            ready = client.get("/ready")
            health = client.get("/health")
    finally:
        app.dependency_overrides.clear()
    assert ready.status_code == 503 and ready.json() == {"status": "unavailable"}
    assert health.status_code == 503 and health.json() == {"status": "unavailable"}
    assert "database" not in ready.text.casefold()


@pytest_asyncio.fixture
async def readiness_factory(
    tmp_path: Path,
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'ready.db').as_posix()}")
    async with engine.begin() as connection:
        await connection.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32))"))
        await connection.execute(
            text("INSERT INTO alembic_version VALUES (:version)"),
            {"version": EXPECTED_ALEMBIC_HEAD},
        )
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


async def test_database_readiness_accepts_only_the_expected_schema(
    readiness_factory: async_sessionmaker[AsyncSession],
) -> None:
    checker = DatabaseReadiness(readiness_factory, timeout_seconds=0.5)
    assert await checker.is_ready()
    async with readiness_factory() as session, session.begin():
        await session.execute(text("UPDATE alembic_version SET version_num = '20260909_0013'"))
    assert not await checker.is_ready()


async def test_database_readiness_handles_unavailable_and_timeout() -> None:
    class UnavailableFactory:
        def __call__(self):
            raise OSError("private failure")

    class SlowContext:
        async def __aenter__(self):
            await asyncio.sleep(10)

        async def __aexit__(self, *_args):
            return None

    class SlowFactory:
        def __call__(self):
            return SlowContext()

    unavailable = DatabaseReadiness(UnavailableFactory(), timeout_seconds=0.01)  # type: ignore[arg-type]
    slow = DatabaseReadiness(SlowFactory(), timeout_seconds=0.01)  # type: ignore[arg-type]
    assert not await unavailable.is_ready()
    assert not await slow.is_ready()
