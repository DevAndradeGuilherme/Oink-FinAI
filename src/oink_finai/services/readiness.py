import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

EXPECTED_ALEMBIC_HEAD = "20260910_0014"


class DatabaseReadiness:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        timeout_seconds: float,
        expected_head: str = EXPECTED_ALEMBIC_HEAD,
    ) -> None:
        self._session_factory = session_factory
        self._timeout_seconds = timeout_seconds
        self._expected_head = expected_head

    async def is_ready(self) -> bool:
        try:
            async with asyncio.timeout(self._timeout_seconds):
                async with self._session_factory() as session:
                    versions = list(
                        await session.scalars(text("SELECT version_num FROM alembic_version"))
                    )
            return versions == [self._expected_head]
        except Exception:
            return False
