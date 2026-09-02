import asyncio
import random
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import Select, select
from sqlalchemy.exc import DataError, DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from oink_finai.database.models import (
    Category,
    ConversationState,
    Expense,
    OutboundMessage,
    ProcessedMessage,
    User,
)
from oink_finai.domain.enums import (
    ConversationStatus,
    ExpenseIntent,
    OutboundMessageKind,
    OutboundMessageStatus,
    ProcessedMessageStatus,
)
from oink_finai.domain.expense_limits import (
    EXPENSE_AMOUNT_MAX,
    EXPENSE_AMOUNT_SCALE,
    EXPENSE_DESCRIPTION_MAX_LENGTH,
    EXPENSE_MERCHANT_MAX_LENGTH,
    EXPENSE_PAYMENT_METHOD_MAX_LENGTH,
)
from oink_finai.schemas.expense_interpretation import ExpenseInterpretation
from oink_finai.services.expense_interpreter import ExpenseInterpreter
from oink_finai.services.gemini_errors import (
    GeminiAuthenticationError,
    GeminiConfigurationError,
    GeminiInterpreterError,
    GeminiModelUnavailableError,
    GeminiPermissionError,
    GeminiRateLimitError,
    GeminiRequestError,
    GeminiSchemaError,
    GeminiTimeoutError,
    GeminiUnavailableError,
)

CLARIFICATION_TEXT = "Não encontrei o valor. Envie novamente incluindo o valor do gasto."
PROCESSING_FAILURE_TEXT = (
    "⚠️ Não consegui registrar esse gasto agora. Envie a mensagem novamente em alguns minutos."
)


def utc_now() -> datetime:
    return datetime.now(UTC)


class InterpretationLimitError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class ExpenseProcessingService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        interpreter_factory: Callable[[str], ExpenseInterpreter],
        *,
        max_attempts: int = 3,
        retry_base_seconds: float = 0.5,
        retry_max_seconds: float = 5.0,
        jitter: Callable[[], float] = random.random,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._session_factory = session_factory
        self._interpreter_factory = interpreter_factory
        self._max_attempts = max_attempts
        self._retry_base_seconds = retry_base_seconds
        self._retry_max_seconds = retry_max_seconds
        self._jitter = jitter
        self._clock = clock

    async def recover_stale(self, older_than: datetime) -> int:
        now = self._now()
        async with self._session_factory() as session, session.begin():
            messages = list(
                await session.scalars(
                    select(ProcessedMessage)
                    .where(
                        ProcessedMessage.status == ProcessedMessageStatus.PROCESSING,
                        ProcessedMessage.locked_at < older_than,
                    )
                    .with_for_update(of=ProcessedMessage, skip_locked=True)
                )
            )
            for message in messages:
                message.locked_at = None
                if message.processing_attempts >= self._max_attempts:
                    await self._mark_attempts_exhausted(session, message)
                    continue
                message.status = ProcessedMessageStatus.PENDING
                message.error_code = "PROCESSING_INTERRUPTED"
                message.last_error_code = "PROCESSING_INTERRUPTED"
                message.next_attempt_at = now + self._retry_delay(
                    max(1, message.processing_attempts)
                )
            return len(messages)

    async def claim(self, batch_size: int) -> list[UUID]:
        now = self._now()
        async with self._session_factory() as session, session.begin():
            await self._reconcile_exhausted_pending(session, batch_size)
            messages = list(
                await session.scalars(
                    select(ProcessedMessage)
                    .where(
                        ProcessedMessage.status == ProcessedMessageStatus.PENDING,
                        ProcessedMessage.available_at <= now,
                        (
                            ProcessedMessage.next_attempt_at.is_(None)
                            | (ProcessedMessage.next_attempt_at <= now)
                        ),
                        ProcessedMessage.processing_attempts < self._max_attempts,
                    )
                    .order_by(ProcessedMessage.created_at)
                    .limit(batch_size)
                    .with_for_update(of=ProcessedMessage, skip_locked=True)
                )
            )
            for message in messages:
                message.status = ProcessedMessageStatus.PROCESSING
                message.locked_at = now
                message.processing_attempts += 1
            return [message.id for message in messages]

    async def _reconcile_exhausted_pending(self, session: AsyncSession, batch_size: int) -> None:
        messages = list(
            await session.scalars(
                select(ProcessedMessage)
                .where(
                    ProcessedMessage.status == ProcessedMessageStatus.PENDING,
                    ProcessedMessage.processing_attempts >= self._max_attempts,
                )
                .order_by(ProcessedMessage.created_at)
                .limit(batch_size)
                .with_for_update(of=ProcessedMessage, skip_locked=True)
            )
        )
        for message in messages:
            await self._mark_attempts_exhausted(session, message)

    async def _mark_attempts_exhausted(
        self, session: AsyncSession, message: ProcessedMessage
    ) -> None:
        message.status = ProcessedMessageStatus.FAILED
        message.error_code = "PROCESSING_ATTEMPTS_EXHAUSTED"
        message.last_error_code = "PROCESSING_ATTEMPTS_EXHAUSTED"
        message.locked_at = None
        message.next_attempt_at = None
        user = await session.get(User, message.user_id)
        await self._create_failure_notification(session, message, user)

    async def process(self, message_id: UUID) -> None:
        async with self._session_factory() as session:
            message = await session.get(ProcessedMessage, message_id)
            if message is None or message.status != ProcessedMessageStatus.PROCESSING:
                return
            user = await session.get(User, message.user_id)
            if user is None:
                await self._mark_failed(message_id, "USER_NOT_FOUND")
                return
            text = message.accepted_text
            timestamp = message.message_timestamp
            timezone = user.timezone

        try:
            interpreter = self._interpreter_factory(timezone)
            interpretation = await interpreter.interpret(text, reference_timestamp=timestamp)
            self._validate_interpretation(interpretation)
        except asyncio.CancelledError:
            await asyncio.shield(self._retry_or_fail(message_id, "GEMINI_TIMEOUT"))
            raise
        except InterpretationLimitError as exc:
            await self._mark_failed(message_id, exc.code)
            return
        except GeminiInterpreterError as exc:
            if self._is_transient(exc):
                await self._retry_or_fail(message_id, self._error_code(exc))
            else:
                await self._mark_failed(message_id, self._error_code(exc))
            return

        try:
            async with self._session_factory() as session, session.begin():
                message = await session.scalar(
                    self._locked_message_statement(message_id).options(
                        selectinload(ProcessedMessage.user)
                    )
                )
                if message is None or message.status != ProcessedMessageStatus.PROCESSING:
                    return
                if message.user is None or message.user_id is None:
                    message.status = ProcessedMessageStatus.FAILED
                    message.error_code = "USER_NOT_FOUND"
                    message.last_error_code = "USER_NOT_FOUND"
                    message.locked_at = None
                    message.next_attempt_at = None
                    return
                if interpretation.intent is ExpenseIntent.CREATE_EXPENSE:
                    await self._create_expense(session, message, interpretation)
                    status = ProcessedMessageStatus.PROCESSED
                elif interpretation.intent is ExpenseIntent.UNCLEAR:
                    self._create_outbox(
                        session,
                        message,
                        content=CLARIFICATION_TEXT,
                        kind=OutboundMessageKind.CLARIFICATION,
                        expense=None,
                    )
                    status = ProcessedMessageStatus.NEEDS_CLARIFICATION
                else:
                    status = ProcessedMessageStatus.NOT_EXPENSE
                self._complete_successfully(message, status)
        except GeminiInterpreterError as exc:
            await self._mark_failed(message_id, self._error_code(exc))
        except IntegrityError:
            await self._recover_unique_conflict(message_id)
        except DataError:
            await self._mark_failed(message_id, "PERSISTENCE_DATA_ERROR")
        except DBAPIError as exc:
            if not self._is_data_exception(exc):
                raise
            await self._mark_failed(message_id, "PERSISTENCE_DATA_ERROR")

    @staticmethod
    def _is_transient(exc: GeminiInterpreterError) -> bool:
        if isinstance(exc, (GeminiTimeoutError, GeminiRateLimitError)):
            return True
        if not isinstance(exc, GeminiUnavailableError):
            return False
        if exc.metadata is None or exc.metadata.http_status is None:
            return True
        return 500 <= exc.metadata.http_status < 600

    @staticmethod
    def _validate_interpretation(result: ExpenseInterpretation) -> None:
        if result.intent is not ExpenseIntent.CREATE_EXPENSE:
            if result.amount is not None:
                raise GeminiSchemaError("Non-expense result must not contain amount")
            return
        if not isinstance(result.amount, Decimal) or not result.amount.is_finite():
            raise GeminiSchemaError("Expense result failed deterministic validation")
        if result.amount <= 0 or result.amount > EXPENSE_AMOUNT_MAX:
            raise InterpretationLimitError("AMOUNT_OUT_OF_RANGE")
        if result.amount.as_tuple().exponent < -EXPENSE_AMOUNT_SCALE:
            raise InterpretationLimitError("AMOUNT_SCALE_EXCEEDED")
        if result.description is None or not result.description.strip():
            raise GeminiSchemaError("Expense result failed deterministic validation")
        if len(result.description.strip()) > EXPENSE_DESCRIPTION_MAX_LENGTH:
            raise InterpretationLimitError("DESCRIPTION_TOO_LONG")
        if result.merchant is not None and len(result.merchant) > EXPENSE_MERCHANT_MAX_LENGTH:
            raise InterpretationLimitError("MERCHANT_TOO_LONG")
        payment_method = result.payment_method.value if result.payment_method else None
        if payment_method is not None and len(payment_method) > EXPENSE_PAYMENT_METHOD_MAX_LENGTH:
            raise InterpretationLimitError("PAYMENT_METHOD_TOO_LONG")

    async def _create_expense(
        self,
        session: AsyncSession,
        message: ProcessedMessage,
        result: ExpenseInterpretation,
    ) -> None:
        assert result.amount is not None and result.description is not None
        category = await session.scalar(
            select(Category).where(
                Category.name == result.category.value, Category.is_active.is_(True)
            )
        )
        if category is None:
            raise GeminiSchemaError("Canonical category is unavailable")
        timezone = self._timezone(message.user.timezone)
        timestamp = message.message_timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        expense_date = result.expense_date or timestamp.astimezone(timezone).date()
        expense = Expense(
            user_id=message.user_id,
            processed_message_id=message.id,
            category_id=category.id,
            amount=result.amount,
            description=result.description.strip(),
            expense_date=expense_date,
            merchant=result.merchant,
            payment_method=result.payment_method.value if result.payment_method else None,
        )
        session.add(expense)
        await session.flush()
        state = await session.scalar(
            select(ConversationState).where(ConversationState.user_id == message.user_id)
        )
        if state is None:
            session.add(ConversationState(user_id=message.user_id))
        else:
            state.status = ConversationStatus.IDLE
            state.active_expense_id = None
        self._create_outbox(
            session,
            message,
            content=format_expense_confirmation(expense, category.name),
            kind=OutboundMessageKind.EXPENSE_CONFIRMATION,
            expense=expense,
        )

    @staticmethod
    def _complete_successfully(message: ProcessedMessage, status: ProcessedMessageStatus) -> None:
        message.status = status
        message.error_code = None
        message.next_attempt_at = None
        message.locked_at = None

    @staticmethod
    def _create_outbox(
        session: AsyncSession,
        message: ProcessedMessage,
        *,
        content: str,
        kind: OutboundMessageKind,
        expense: Expense | None,
    ) -> None:
        session.add(
            OutboundMessage(
                user_id=message.user_id,
                expense_id=expense.id if expense else None,
                destination=message.user.phone_number,
                content=content,
                kind=kind,
                dedup_key=f"processed-message:{message.id}:{kind.value}",
                status=OutboundMessageStatus.PENDING,
                available_at=datetime.now(UTC),
            )
        )

    async def _mark_failed(self, message_id: UUID, code: str) -> None:
        async with self._session_factory() as session, session.begin():
            message = await session.scalar(self._locked_message_statement(message_id))
            if message is not None and message.status == ProcessedMessageStatus.PROCESSING:
                message.status = ProcessedMessageStatus.FAILED
                message.error_code = code
                message.last_error_code = code
                message.locked_at = None
                message.next_attempt_at = None
                message.attempt_count += 1

    async def _retry_or_fail(self, message_id: UUID, code: str) -> None:
        async with self._session_factory() as session, session.begin():
            message = await session.scalar(self._locked_message_statement(message_id))
            if message is None or message.status != ProcessedMessageStatus.PROCESSING:
                return
            message.error_code = code
            message.last_error_code = code
            message.locked_at = None
            message.attempt_count += 1
            if message.processing_attempts >= self._max_attempts:
                message.status = ProcessedMessageStatus.FAILED
                message.next_attempt_at = None
                user = await session.get(User, message.user_id)
                await self._create_failure_notification(session, message, user)
                return
            message.status = ProcessedMessageStatus.PENDING
            message.next_attempt_at = self._now() + self._retry_delay(message.processing_attempts)

    @staticmethod
    def _locked_message_statement(message_id: UUID) -> Select[tuple[ProcessedMessage]]:
        return (
            select(ProcessedMessage)
            .where(ProcessedMessage.id == message_id)
            .with_for_update(of=ProcessedMessage)
        )

    async def _create_failure_notification(
        self, session: AsyncSession, message: ProcessedMessage, user: User | None
    ) -> None:
        if user is None or message.user_id is None:
            return
        dedup_key = f"processed-message:{message.id}:terminal"
        existing = await session.scalar(
            select(OutboundMessage.id).where(OutboundMessage.dedup_key == dedup_key)
        )
        if existing is not None:
            return
        session.add(
            OutboundMessage(
                user_id=message.user_id,
                expense_id=None,
                destination=user.phone_number,
                content=PROCESSING_FAILURE_TEXT,
                kind=OutboundMessageKind.PROCESSING_FAILURE,
                dedup_key=dedup_key,
                status=OutboundMessageStatus.PENDING,
                available_at=self._now(),
            )
        )

    def _retry_delay(self, attempt: int) -> timedelta:
        base_delay = min(
            self._retry_max_seconds,
            self._retry_base_seconds * (2 ** (attempt - 1)),
        )
        jitter = min(1.0, max(0.0, self._jitter()))
        return timedelta(seconds=base_delay * (0.5 + jitter / 2))

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return now.astimezone(UTC)

    async def _recover_unique_conflict(self, message_id: UUID) -> None:
        async with self._session_factory() as session, session.begin():
            expense = await session.scalar(
                select(Expense).where(Expense.processed_message_id == message_id)
            )
            message = await session.get(ProcessedMessage, message_id, with_for_update=True)
            if expense is not None and message is not None:
                self._complete_successfully(message, ProcessedMessageStatus.PROCESSED)
                return
            if message is not None and message.status == ProcessedMessageStatus.PROCESSING:
                message.status = ProcessedMessageStatus.FAILED
                message.error_code = "IDEMPOTENCY_CONFLICT"
                message.locked_at = None

    @staticmethod
    def _error_code(exc: GeminiInterpreterError) -> str:
        names = {
            GeminiConfigurationError: "GEMINI_CONFIGURATION",
            GeminiRequestError: "GEMINI_REQUEST",
            GeminiAuthenticationError: "GEMINI_AUTHENTICATION",
            GeminiPermissionError: "GEMINI_PERMISSION",
            GeminiModelUnavailableError: "GEMINI_MODEL_UNAVAILABLE",
            GeminiSchemaError: "GEMINI_SCHEMA_INVALID",
            GeminiTimeoutError: "GEMINI_TIMEOUT",
            GeminiRateLimitError: "GEMINI_RATE_LIMIT",
            GeminiUnavailableError: "GEMINI_UNAVAILABLE",
        }
        return next((code for cls, code in names.items() if isinstance(exc, cls)), "GEMINI_ERROR")

    @staticmethod
    def _is_data_exception(exc: DBAPIError) -> bool:
        sqlstate = getattr(exc.orig, "sqlstate", None) or getattr(exc.orig, "pgcode", None)
        return isinstance(sqlstate, str) and sqlstate.startswith("22")

    @staticmethod
    def _timezone(name: str) -> ZoneInfo:
        try:
            return ZoneInfo(name)
        except ZoneInfoNotFoundError:
            return ZoneInfo("UTC")


def format_expense_confirmation(expense: Expense, category_name: str) -> str:
    amount = f"{expense.amount:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
    payment = expense.payment_method or "Não informado"
    return (
        "✅ Gasto registrado\n\n"
        f" Valor: R$ {amount}\n"
        f" Descrição: {expense.description}\n"
        f" Categoria: {category_name}\n"
        f" Data: {expense.expense_date.strftime('%d/%m/%Y')}\n"
        f" Pagamento: {payment}"
    )
