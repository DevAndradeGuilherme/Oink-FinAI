"""Destructive usage-control probe; only for the isolated PostgreSQL test runner."""

import asyncio
import os
from datetime import UTC, datetime, timedelta
from urllib.parse import urlsplit
from uuid import uuid4

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from oink_finai.config.settings import Settings
from oink_finai.database.models import ProcessedMessage, UsageLedger, User
from oink_finai.domain.enums import (
    ProcessedMessageStatus,
    UsageOperation,
    UsageReservationState,
)
from oink_finai.services.usage_control import AdmissionDenial, UsageControl

MARKER = "I_UNDERSTAND_ONLY_A_DISPOSABLE_POSTGRES_WILL_BE_DESTROYED"


async def main() -> None:
    if os.environ.get("OINK_DESTRUCTIVE_POSTGRES_TEST_MARKER") != MARKER:
        raise RuntimeError("explicit disposable-database marker is required")
    settings = Settings(_env_file=None)
    parsed = urlsplit(settings.database_url_value)
    if parsed.hostname != "postgres" or not parsed.path.removeprefix("/").startswith("oink_lp_"):
        raise RuntimeError("database target is not the isolated test database")
    engine = create_async_engine(settings.database_url_value, pool_size=4)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    marker = uuid4().hex
    user_ids = []
    message_ids = []
    try:
        async with factory() as session, session.begin():
            for suffix in ("a", "b", "c", "d"):
                user = User(phone_number=f"disposable-{marker[:18]}-{suffix}", timezone="UTC")
                session.add(user)
                await session.flush()
                message = ProcessedMessage(
                    provider="disposable-concurrency-probe",
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
            same_user_message = ProcessedMessage(
                provider="disposable-concurrency-probe",
                instance_id=marker,
                external_message_id=f"{marker}-same-user",
                user_id=user_ids[0],
                accepted_text="synthetic",
                message_timestamp=datetime.now(UTC),
                status=ProcessedMessageStatus.PROCESSING,
                available_at=datetime.now(UTC),
                processing_attempts=1,
            )
            session.add(same_user_message)
            await session.flush()
            message_ids.append(same_user_message.id)

        limited = settings.model_copy(
            update={
                "openai_global_concurrency_limit": 1,
                "openai_reservation_stale_seconds": 1.0,
                "inbound_user_per_minute_limit": 1,
            }
        )
        first_worker = UsageControl(limited, factory)
        second_worker = UsageControl(limited, factory)

        async def admit(control: UsageControl, message_id):
            async with factory() as session, session.begin():
                return await control.admit_inbound(
                    session, message_id=message_id, user_id=user_ids[0]
                )

        webhook_results = await asyncio.gather(
            admit(first_worker, message_ids[0]),
            admit(second_worker, message_ids[4]),
        )
        if sum(result.allowed for result in webhook_results) != 1:
            raise RuntimeError("same-user concurrent webhook admission exceeded its limit")

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
        if len(allowed) != 1 or len(denied) != 1:
            raise RuntimeError("concurrent admission exceeded the configured slot")
        if denied[0].denial is not AdmissionDenial.GLOBAL_CONCURRENCY:
            raise RuntimeError("concurrent denial reason is invalid")
        if allowed[0].ledger_id is None:
            raise RuntimeError("allowed reservation has no durable ledger identity")
        await first_worker.complete(allowed[0].ledger_id)

        untransmitted = await first_worker.reserve_openai(
            message_id=message_ids[2],
            operation=UsageOperation.IMAGE_ANALYSIS,
            model="synthetic-model",
        )
        if untransmitted.ledger_id is None:
            raise RuntimeError("untransmitted crash reservation was not admitted")
        async with factory() as session, session.begin():
            ledger = await session.get(UsageLedger, untransmitted.ledger_id)
            ledger.created_at = datetime.now(UTC) - timedelta(minutes=2)
        await first_worker.reconcile_stale()

        transmitted = await second_worker.reserve_openai(
            message_id=message_ids[3],
            operation=UsageOperation.AUDIO_TRANSCRIPTION,
            model="synthetic-model",
        )
        if transmitted.ledger_id is None:
            raise RuntimeError("transmitted crash reservation was not admitted")
        await second_worker.mark_transmitted(transmitted.ledger_id)
        async with factory() as session, session.begin():
            ledger = await session.get(UsageLedger, transmitted.ledger_id)
            ledger.created_at = datetime.now(UTC) - timedelta(minutes=2)
        await second_worker.reconcile_stale()

        async with factory() as session:
            released = await session.get(UsageLedger, untransmitted.ledger_id)
            ambiguous = await session.get(UsageLedger, transmitted.ledger_id)
            if released is None or released.state is not UsageReservationState.RELEASED:
                raise RuntimeError("proven pre-transmission crash was not released")
            if ambiguous is None or ambiguous.state is not UsageReservationState.AMBIGUOUS:
                raise RuntimeError("transmitted crash was not retained conservatively")
            rows = list(
                await session.scalars(
                    select(UsageLedger).where(
                        UsageLedger.user_id.in_(user_ids),
                        UsageLedger.operation != UsageOperation.INBOUND_MESSAGE,
                    )
                )
            )
            if any(item.model != "synthetic-model" for item in rows):
                raise RuntimeError("usage ledger model reconciliation failed")
        print("PostgreSQL usage concurrency and crash reconciliation passed.")
    finally:
        async with factory() as session, session.begin():
            await session.execute(delete(UsageLedger).where(UsageLedger.user_id.in_(user_ids)))
            await session.execute(
                delete(ProcessedMessage).where(ProcessedMessage.id.in_(message_ids))
            )
            await session.execute(delete(User).where(User.id.in_(user_ids)))
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
