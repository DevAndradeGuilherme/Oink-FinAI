from collections.abc import AsyncIterator

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from oink_finai.config.settings import Settings
from oink_finai.database import models  # noqa: F401
from oink_finai.database.base import Base


def pytest_configure() -> None:
    # Prevent application imports during collection from reading local credentials.
    # Tests of documented settings may still pass an explicit, synthetic env file.
    Settings.model_config["env_file"] = None


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as database_session:
        yield database_session

    await engine.dispose()
