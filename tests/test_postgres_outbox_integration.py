import os
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from oink_finai.database.models import OutboundMessage, User
from oink_finai.domain.enums import OutboundMessageKind, OutboundMessageStatus
from oink_finai.providers.whatsapp import WhatsAppProvider
from oink_finai.services.expense_processing import ExpenseProcessingService
from oink_finai.services.outbox_delivery import OutboxDeliveryService

pytestmark = pytest.mark.skipif(
    "OINK_TEST_POSTGRES_URL" not in os.environ,
    reason="OINK_TEST_POSTGRES_URL not configured",
)


class NoCallProvider(WhatsAppProvider):
    async def parse_webhook(self, payload: dict[str, object]):
        raise AssertionError("provider must not be called")

    async def send_text(self, phone_number: str, text: str) -> str | None:
        raise AssertionError("provider must not be called")


async def test_postgres_rejects_numeric_14_2_overflow() -> None:
    engine = create_async_engine(os.environ["OINK_TEST_POSTGRES_URL"])
    with pytest.raises(DBAPIError) as captured:
        async with engine.begin() as connection:
            await connection.execute(text("SELECT CAST(1000000000000.00 AS NUMERIC(14, 2))"))
    assert getattr(captured.value.orig, "sqlstate", None) == "22003"
    assert ExpenseProcessingService._is_data_exception(captured.value)
    await engine.dispose()


async def test_postgres_claim_and_recovery_transitions() -> None:
    engine = create_async_engine(os.environ["OINK_TEST_POSTGRES_URL"])
    factory = async_sessionmaker(engine, expire_on_commit=False)
    unique = uuid4().hex[:20]
    async with factory() as session, session.begin():
        user = User(phone_number=f"test-{unique}")
        session.add(user)
        await session.flush()
        message = OutboundMessage(
            user_id=user.id,
            destination="test-destination",
            content="synthetic",
            kind=OutboundMessageKind.CLARIFICATION,
            dedup_key=f"postgres-test-{unique}",
            status=OutboundMessageStatus.PENDING,
            available_at=datetime.now(UTC),
        )
        session.add(message)
        await session.flush()
        message_id = message.id

    delivery = OutboxDeliveryService(factory, NoCallProvider())
    claim = (await delivery.claim(1))[0]
    async with factory() as session, session.begin():
        message = await session.get(OutboundMessage, message_id)
        assert message is not None
        assert message.status == OutboundMessageStatus.CLAIMED
        message.claimed_at = datetime.now(UTC) - timedelta(hours=1)

    assert await delivery.recover_stale(datetime.now(UTC) - timedelta(minutes=5)) == (1, 0)
    claim = (await delivery.claim(1))[0]
    assert await delivery._start_sending(claim) is not None
    async with factory() as session, session.begin():
        message = await session.get(OutboundMessage, message_id)
        assert message is not None
        message.sending_at = datetime.now(UTC) - timedelta(hours=1)

    assert await delivery.recover_stale(datetime.now(UTC) - timedelta(minutes=5)) == (0, 1)
    async with factory() as session:
        message = await session.get(OutboundMessage, message_id)
        assert message is not None
        assert message.status == OutboundMessageStatus.UNKNOWN
    await engine.dispose()
