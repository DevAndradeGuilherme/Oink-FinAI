import asyncio
import os
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from oink_finai.config.settings import Settings
from oink_finai.database.models import ProcessedMessage, UsageLedger, User
from oink_finai.domain.enums import ProcessedMessageStatus, UsageOperation
from oink_finai.services.usage_control import AdmissionDenial, UsageControl

pytestmark = pytest.mark.skipif(
    not (
        os.environ.get("OINK_TEST_POSTGRES_URL")
        and os.environ.get("OINK_TEST_POSTGRES_DISPOSABLE") == "YES_DELETE"
    ),
    reason="explicit disposable PostgreSQL marker not configured",
)


async def test_postgres_serializes_two_worker_global_concurrency() -> None:
    engine = create_async_engine(os.environ["OINK_TEST_POSTGRES_URL"], pool_size=4)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    marker = uuid4().hex
    user_ids = []
    message_ids = []
    async with factory() as session, session.begin():
        for suffix in ("a", "b"):
            user = User(phone_number=f"disposable-{marker[:18]}-{suffix}", timezone="UTC")
            session.add(user)
            await session.flush()
            message = ProcessedMessage(
                provider="disposable-postgres-test",
                instance_id=marker,
                external_message_id=f"{marker}-{suffix}",
                user_id=user.id,
                accepted_text="synthetic",
                message_timestamp=datetime.now(UTC),
                status=ProcessedMessageStatus.PROCESSING,
                available_at=datetime.now(UTC),
                processing_attempts=1,
            )
            session.add(message)
            await session.flush()
            user_ids.append(user.id)
            message_ids.append(message.id)

    config = Settings(
        _env_file=None,
        openai_global_concurrency_limit=1,
    )
    first_worker = UsageControl(config, factory)
    second_worker = UsageControl(config, factory)
    try:
        results = await asyncio.gather(
            first_worker.reserve_openai(
                message_id=message_ids[0],
                operation=UsageOperation.TEXT_INTERPRETATION,
                model="synthetic-model",
            ),
            second_worker.reserve_openai(
                message_id=message_ids[1],
                operation=UsageOperation.TEXT_INTERPRETATION,
                model="synthetic-model",
            ),
        )
        allowed = [result for result in results if result.allowed]
        denied = [result for result in results if not result.allowed]
        assert len(allowed) == 1
        assert len(denied) == 1
        assert denied[0].denial is AdmissionDenial.GLOBAL_CONCURRENCY
        assert allowed[0].ledger_id is not None
        await first_worker.complete(allowed[0].ledger_id)

        retried = await second_worker.reserve_openai(
            message_id=(message_ids[1] if results[1] is denied[0] else message_ids[0]),
            operation=UsageOperation.TEXT_INTERPRETATION,
            model="synthetic-model",
        )
        assert retried.allowed
    finally:
        async with factory() as session, session.begin():
            await session.execute(delete(UsageLedger).where(UsageLedger.user_id.in_(user_ids)))
            await session.execute(
                delete(ProcessedMessage).where(ProcessedMessage.id.in_(message_ids))
            )
            await session.execute(delete(User).where(User.id.in_(user_ids)))
        await engine.dispose()
