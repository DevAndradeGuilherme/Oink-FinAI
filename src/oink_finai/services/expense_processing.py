import asyncio
import random
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import select, update
from sqlalchemy.exc import DataError, DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import joinedload

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
    GeminiInterpreterError,
    GeminiRateLimitError,
    GeminiSchemaError,
    GeminiTimeoutError,
    GeminiUnavailableError,
)

CLARIFICATION_TEXT = "Não encontrei o valor. Envie novamente incluindo o valor do gasto."


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
        sleep: Callable[[float], object] = asyncio.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self._session_factory = session_factory
        self._interpreter_factory = interpreter_factory
        self._max_attempts = max_attempts
        self._retry_base_seconds = retry_base_seconds
        self._retry_max_seconds = retry_max_seconds
        self._sleep = sleep
        self._jitter = jitter

    async def recover_stale(self, older_than: datetime) -> int:
        async with self._session_factory() as session, session.begin():
            result = await session.execute(
                update(ProcessedMessage)
                .where(
                    ProcessedMessage.status == ProcessedMessageStatus.PROCESSING,
                    ProcessedMessage.locked_at < older_than,
                )
                .values(
                    status=ProcessedMessageStatus.PENDING,
                    locked_at=None,
                    available_at=datetime.now(UTC),
                )
            )
            return result.rowcount or 0

    async def claim(self, batch_size: int) -> list[UUID]:
        now = datetime.now(UTC)
        async with self._session_factory() as session, session.begin():
            messages = list(
                await session.scalars(
                    select(ProcessedMessage)
                    .where(
                        ProcessedMessage.status == ProcessedMessageStatus.PENDING,
                        ProcessedMessage.available_at <= now,
                    )
                    .order_by(ProcessedMessage.created_at)
                    .limit(batch_size)
                    .with_for_update(skip_locked=True)
                )
            )
            for message in messages:
                message.status = ProcessedMessageStatus.PROCESSING
                message.locked_at = now
            return [message.id for message in messages]

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
            interpretation = await self._interpret_with_retry(text, timestamp, timezone)
            self._validate_interpretation(interpretation)
        except InterpretationLimitError as exc:
            await self._mark_failed(message_id, exc.code)
            return
        except GeminiInterpreterError as exc:
            await self._mark_failed(message_id, self._error_code(exc))
            return

        try:
            async with self._session_factory() as session, session.begin():
                message = await session.scalar(
                    select(ProcessedMessage)
                    .where(ProcessedMessage.id == message_id)
                    .options(joinedload(ProcessedMessage.user))
                    .with_for_update()
                )
                if message is None or message.status != ProcessedMessageStatus.PROCESSING:
                    return
                if interpretation.intent is ExpenseIntent.CREATE_EXPENSE:
                    await self._create_expense(session, message, interpretation)
                elif interpretation.intent is ExpenseIntent.UNCLEAR:
                    self._create_outbox(
                        session,
                        message,
                        content=CLARIFICATION_TEXT,
                        kind=OutboundMessageKind.CLARIFICATION,
                        expense=None,
                    )
                    message.status = ProcessedMessageStatus.NEEDS_CLARIFICATION
                else:
                    message.status = ProcessedMessageStatus.NOT_EXPENSE
                message.locked_at = None
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

    async def _interpret_with_retry(
        self, text: str, timestamp: datetime, timezone: str
    ) -> ExpenseInterpretation:
        interpreter = self._interpreter_factory(timezone)
        for attempt in range(1, self._max_attempts + 1):
            try:
                return await interpreter.interpret(text, reference_timestamp=timestamp)
            except GeminiInterpreterError as exc:
                if not self._is_transient(exc) or attempt == self._max_attempts:
                    raise
                delay = min(
                    self._retry_max_seconds,
                    self._retry_base_seconds * (2 ** (attempt - 1)),
                )
                await self._sleep(delay * (0.5 + self._jitter() / 2))  # type: ignore[misc]
        raise RuntimeError("unreachable")

    @staticmethod
    def _is_transient(exc: GeminiInterpreterError) -> bool:
        if isinstance(exc, (GeminiTimeoutError, GeminiRateLimitError)):
            return True
        return (
            isinstance(exc, GeminiUnavailableError)
            and exc.metadata is not None
            and (exc.metadata.http_status in {500, 503})
        )

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
        message.status = ProcessedMessageStatus.PROCESSED

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
            message = await session.get(ProcessedMessage, message_id, with_for_update=True)
            if message is not None and message.status == ProcessedMessageStatus.PROCESSING:
                message.status = ProcessedMessageStatus.FAILED
                message.error_code = code
                message.locked_at = None
                message.attempt_count += 1

    async def _recover_unique_conflict(self, message_id: UUID) -> None:
        async with self._session_factory() as session, session.begin():
            expense = await session.scalar(
                select(Expense).where(Expense.processed_message_id == message_id)
            )
            message = await session.get(ProcessedMessage, message_id, with_for_update=True)
            if expense is not None and message is not None:
                message.status = ProcessedMessageStatus.PROCESSED
                message.locked_at = None
                return
            if message is not None and message.status == ProcessedMessageStatus.PROCESSING:
                message.status = ProcessedMessageStatus.FAILED
                message.error_code = "IDEMPOTENCY_CONFLICT"
                message.locked_at = None

    @staticmethod
    def _error_code(exc: GeminiInterpreterError) -> str:
        names = {
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
