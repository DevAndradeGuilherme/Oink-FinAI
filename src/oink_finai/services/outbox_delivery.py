import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import exists, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import aliased

from oink_finai.database.models import OutboundMessage
from oink_finai.domain.enums import OutboundMessageStatus
from oink_finai.providers.whatsapp import (
    EvolutionProviderError,
    InteractiveAction,
    InteractiveMessage,
    InteractiveMessageUnsupportedError,
    WhatsAppProvider,
)
from oink_finai.services.pipeline_timing import PipelineTiming


@dataclass(frozen=True)
class OutboundMessageClaim:
    message_id: UUID
    claim_token: UUID
    correlation_id: UUID | None = None
    attempt_number: int = 0


@dataclass(frozen=True)
class OutboundDeliveryData:
    destination: str
    content: str
    content_type: str
    actions: tuple[InteractiveAction, ...]
    fallback_content: str | None
    attempt_count: int
    correlation_id: UUID


class OutboxDeliveryService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        provider: WhatsAppProvider,
        *,
        max_attempts: int = 3,
        retry_base_seconds: float = 1.0,
        timing: PipelineTiming | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._provider = provider
        self._max_attempts = max_attempts
        self._retry_base_seconds = retry_base_seconds
        self._timing = timing or PipelineTiming(False)

    async def claim(self, batch_size: int) -> list[OutboundMessageClaim]:
        now = datetime.now(UTC)
        earlier_page = aliased(OutboundMessage)
        earlier_page_pending = exists(
            select(earlier_page.id).where(
                earlier_page.processed_message_id == OutboundMessage.processed_message_id,
                earlier_page.kind == OutboundMessage.kind,
                earlier_page.sequence_no < OutboundMessage.sequence_no,
                earlier_page.status != OutboundMessageStatus.SENT,
            )
        )
        async with self._session_factory() as session, session.begin():
            messages = list(
                await session.scalars(
                    select(OutboundMessage)
                    .where(
                        OutboundMessage.status == OutboundMessageStatus.PENDING,
                        OutboundMessage.available_at <= now,
                        ~earlier_page_pending,
                    )
                    .order_by(
                        OutboundMessage.created_at,
                        OutboundMessage.sequence_no,
                        OutboundMessage.id,
                    )
                    .limit(batch_size)
                    .with_for_update(of=OutboundMessage, skip_locked=True)
                )
            )
            claims = []
            for message in messages:
                token = uuid4()
                message.status = OutboundMessageStatus.CLAIMED
                message.claimed_at = now
                message.claim_token = token
                correlation_id = self._correlation_id(message.dedup_key, message.id)
                claims.append(
                    OutboundMessageClaim(
                        message.id, token, correlation_id, message.attempt_count + 1
                    )
                )
            timing_data = [
                (
                    claim.correlation_id,
                    claim.attempt_number,
                    message.available_at,
                )
                for claim, message in zip(claims, messages, strict=True)
            ]
        for correlation_id, attempt_number, available_at in timing_data:
            assert correlation_id is not None
            self._timing.event("outbox_claimed", correlation_id, attempt_number=attempt_number)
            self._timing.event(
                "outbox_queue_wait_completed",
                correlation_id,
                duration_ms=self._timing.elapsed_ms(available_at, now),
                attempt_number=attempt_number,
            )
        return claims

    async def send(self, claim: OutboundMessageClaim) -> None:
        message = await self._start_sending(claim)
        if message is None:
            return
        send_span = self._timing.span(
            "outbound_send_started",
            "outbound_send_completed",
            message.correlation_id,
            attempt_number=message.attempt_count,
            stage="send",
        )
        async with send_span:
            try:
                provider_message_id = await self._deliver(message)
                send_span.result(outcome="success")
            except asyncio.CancelledError:
                send_span.result(outcome="transient_failure", error_code="OUTCOME_UNKNOWN")
                await asyncio.shield(
                    self._finish(claim, OutboundMessageStatus.UNKNOWN, "OUTCOME_UNKNOWN")
                )
                raise
            except EvolutionProviderError as exc:
                if exc.outcome_unknown:
                    send_span.result(outcome="terminal_failure", error_code="OUTCOME_UNKNOWN")
                    await self._finish(claim, OutboundMessageStatus.UNKNOWN, "OUTCOME_UNKNOWN")
                elif message.attempt_count < self._max_attempts:
                    next_attempt_at = await self._retry(claim, message.attempt_count)
                    send_span.result(
                        outcome="transient_failure",
                        error_code="SEND_UNAVAILABLE",
                        next_attempt_at=next_attempt_at,
                    )
                else:
                    send_span.result(outcome="terminal_failure", error_code="SEND_UNAVAILABLE")
                    await self._finish(claim, OutboundMessageStatus.FAILED, "SEND_UNAVAILABLE")
                return
            except Exception:
                send_span.result(outcome="terminal_failure", error_code="OUTCOME_UNKNOWN")
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
        self._timing.event(
            "outbound_accepted",
            message.correlation_id,
            attempt_number=message.attempt_count,
            outcome="success",
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
                correlation_id=self._correlation_id(message.dedup_key, message.id),
            )

    async def _retry(self, claim: OutboundMessageClaim, attempt_count: int) -> datetime:
        next_attempt_at = datetime.now(UTC) + timedelta(
            seconds=self._retry_base_seconds * (2 ** (attempt_count - 1))
        )
        await self._transition(
            claim,
            OutboundMessageStatus.PENDING,
            "SEND_UNAVAILABLE",
            available_at=next_attempt_at,
        )
        return next_attempt_at

    @staticmethod
    def _correlation_id(dedup_key: str, fallback: UUID) -> UUID:
        parts = dedup_key.split(":", 2)
        if len(parts) == 3 and parts[0] == "processed-message":
            try:
                return UUID(parts[1])
            except ValueError:
                pass
        return fallback

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
