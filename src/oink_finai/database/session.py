import asyncio
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from oink_finai.config.settings import get_settings

settings = get_settings()
engine = create_async_engine(settings.database_url_value, pool_pre_ping=True)
SessionFactory = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionFactory() as session:
        try:
            yield session
        except asyncio.CancelledError:
            await asyncio.shield(session.rollback())
            raise
        except BaseException:
            await session.rollback()
            raise
