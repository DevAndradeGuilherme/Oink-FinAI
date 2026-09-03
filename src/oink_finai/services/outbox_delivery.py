import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from oink_finai.database.models import OutboundMessage
from oink_finai.domain.enums import OutboundMessageStatus
from oink_finai.providers.whatsapp import (
    EvolutionProviderError,
    InteractiveAction,
    InteractiveMessage,
    InteractiveMessageUnsupportedError,
    WhatsAppProvider,
)


@dataclass(frozen=True)
class OutboundMessageClaim:
    message_id: UUID
    claim_token: UUID


@dataclass(frozen=True)
class OutboundDeliveryData:
    destination: str
    content: str
    content_type: str
    actions: tuple[InteractiveAction, ...]
    fallback_content: str | None
    attempt_count: int


class OutboxDeliveryService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        provider: WhatsAppProvider,
        *,
        max_attempts: int = 3,
        retry_base_seconds: float = 1.0,
    ) -> None:
        self._session_factory = session_factory
        self._provider = provider
        self._max_attempts = max_attempts
        self._retry_base_seconds = retry_base_seconds

    async def claim(self, batch_size: int) -> list[OutboundMessageClaim]:
        now = datetime.now(UTC)
        async with self._session_factory() as session, session.begin():
            messages = list(
                await session.scalars(
                    select(OutboundMessage)
                    .where(
                        OutboundMessage.status == OutboundMessageStatus.PENDING,
                        OutboundMessage.available_at <= now,
                    )
                    .order_by(OutboundMessage.created_at)
                    .limit(batch_size)
                    .with_for_update(skip_locked=True)
                )
            )
            claims = []
            for message in messages:
                token = uuid4()
                message.status = OutboundMessageStatus.CLAIMED
                message.claimed_at = now
                message.claim_token = token
                claims.append(OutboundMessageClaim(message.id, token))
            return claims

    async def send(self, claim: OutboundMessageClaim) -> None:
        message = await self._start_sending(claim)
        if message is None:
            return
        try:
            provider_message_id = await self._deliver(message)
        except asyncio.CancelledError:
            await asyncio.shield(
                self._finish(claim, OutboundMessageStatus.UNKNOWN, "OUTCOME_UNKNOWN")
            )
            raise
        except EvolutionProviderError as exc:
            if exc.outcome_unknown:
                await self._finish(claim, OutboundMessageStatus.UNKNOWN, "OUTCOME_UNKNOWN")
            elif message.attempt_count < self._max_attempts:
                await self._retry(claim, message.attempt_count)
            else:
                await self._finish(claim, OutboundMessageStatus.FAILED, "SEND_UNAVAILABLE")
            return
        except Exception:
            await self._finish(claim, OutboundMessageStatus.UNKNOWN, "OUTCOME_UNKNOWN")
            return

        async with self._session_factory() as session, session.begin():
            await session.execute(
                update(OutboundMessage)
                .where(
                    OutboundMessage.id == claim.message_id,
                    OutboundMessage.status == OutboundMessageStatus.SENDING,
                    OutboundMessage.claim_token == claim.claim_token,
                )
                .values(
                    status=OutboundMessageStatus.SENT,
                    provider_message_id=provider_message_id,
                    sent_at=datetime.now(UTC),
                    claim_token=None,
                    error_code=None,
                )
            )

    async def _deliver(self, message: OutboundDeliveryData) -> str | None:
        if message.content_type != "BUTTONS":
            return await self._provider.send_text(message.destination, message.content)
        title, separator, body = message.content.partition("\n\n")
        interactive = InteractiveMessage(
            title=title,
            body=body if separator else "",
            actions=message.actions,
        )
        try:
            return await self._provider.send_interactive(message.destination, interactive)
        except InteractiveMessageUnsupportedError:
            if message.fallback_content is None:
                raise EvolutionProviderError("Interactive message is unsupported") from None
            return await self._provider.send_text(message.destination, message.fallback_content)

    async def _start_sending(self, claim: OutboundMessageClaim) -> OutboundDeliveryData | None:
        async with self._session_factory() as session, session.begin():
            message = await session.scalar(
                select(OutboundMessage)
                .where(
                    OutboundMessage.id == claim.message_id,
                    OutboundMessage.status == OutboundMessageStatus.CLAIMED,
                    OutboundMessage.claim_token == claim.claim_token,
                )
                .with_for_update()
            )
            if message is None:
                return None
            message.status = OutboundMessageStatus.SENDING
            message.sending_at = datetime.now(UTC)
            message.attempt_count += 1
            actions = tuple(
                InteractiveAction(id=action["id"], label=action["label"])
                for action in (message.actions or [])
            )
            return OutboundDeliveryData(
                destination=message.destination,
                content=message.content,
                content_type=message.content_type,
                actions=actions,
                fallback_content=message.fallback_content,
                attempt_count=message.attempt_count,
            )

    async def _retry(self, claim: OutboundMessageClaim, attempt_count: int) -> None:
        await self._transition(
            claim,
            OutboundMessageStatus.PENDING,
            "SEND_UNAVAILABLE",
            available_at=datetime.now(UTC)
            + timedelta(seconds=self._retry_base_seconds * (2 ** (attempt_count - 1))),
        )

    async def _finish(
        self, claim: OutboundMessageClaim, status: OutboundMessageStatus, error_code: str
    ) -> None:
        await self._transition(claim, status, error_code)

    async def _transition(
        self,
        claim: OutboundMessageClaim,
        status: OutboundMessageStatus,
        error_code: str,
        *,
        available_at: datetime | None = None,
    ) -> None:
        values: dict[str, object] = {
            "status": status,
            "error_code": error_code,
            "claim_token": None,
        }
        if available_at is not None:
            values["available_at"] = available_at
        async with self._session_factory() as session, session.begin():
            await session.execute(
                update(OutboundMessage)
                .where(
                    OutboundMessage.id == claim.message_id,
                    OutboundMessage.status == OutboundMessageStatus.SENDING,
                    OutboundMessage.claim_token == claim.claim_token,
                )
                .values(**values)
            )

    async def recover_stale(self, cutoff: datetime) -> tuple[int, int]:
        async with self._session_factory() as session, session.begin():
            claimed = await session.execute(
                update(OutboundMessage)
                .where(
                    OutboundMessage.status == OutboundMessageStatus.CLAIMED,
                    OutboundMessage.claimed_at < cutoff,
                )
                .values(
                    status=OutboundMessageStatus.PENDING,
                    claimed_at=None,
                    claim_token=None,
                    error_code=None,
                )
            )
            sending = await session.execute(
                update(OutboundMessage)
                .where(
                    OutboundMessage.status == OutboundMessageStatus.SENDING,
                    OutboundMessage.sending_at < cutoff,
                )
                .values(
                    status=OutboundMessageStatus.UNKNOWN,
                    claim_token=None,
                    error_code="OUTCOME_UNKNOWN",
                )
            )
            return claimed.rowcount, sending.rowcount
