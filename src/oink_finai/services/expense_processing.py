import asyncio
import random
import re
import unicodedata
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import ValidationError
from sqlalchemy import Select, select
from sqlalchemy.exc import DataError, DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from oink_finai.database.models import (
    Category,
    ConversationState,
    Expense,
    ExpenseHistory,
    OutboundMessage,
    ProcessedMessage,
    User,
)
from oink_finai.domain.enums import (
    ConversationStatus,
    ExpenseClarificationField,
    ExpenseHistoryAction,
    ExpenseIntent,
    MessageSourceType,
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
from oink_finai.domain.monetary_value import MonetaryValueError, parse_monetary_value
from oink_finai.providers.whatsapp.base import WhatsAppProvider
from oink_finai.providers.whatsapp.evolution import (
    EvolutionMediaReference,
    EvolutionWhatsAppProvider,
)
from oink_finai.providers.whatsapp.media_errors import MediaError, MediaErrorCode
from oink_finai.schemas.expense_clarification import ExpenseClarificationContext
from oink_finai.schemas.expense_interpretation import ExpenseInterpretation
from oink_finai.schemas.image_checkpoint import ImageAnalysisCheckpoint
from oink_finai.schemas.whatsapp import InboundMedia
from oink_finai.services.audio_transcriber import AudioTranscriber, ValidatedAudio
from oink_finai.services.expense_commands import (
    ExpenseCommand,
    ExpenseCommandType,
    encode_expense_action,
    parse_expense_command,
)
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
from oink_finai.services.image_analysis_errors import ImageAnalysisError
from oink_finai.services.image_analyzer import (
    ImageAnalyzer,
    ValidatedImage,
    normalize_image_caption,
)
from oink_finai.services.pipeline_timing import PipelineTiming
from oink_finai.services.transcription_errors import NoSpeechError, TranscriptionError

CLARIFICATION_QUESTIONS = {
    ExpenseClarificationField.AMOUNT: "Qual foi o valor total do gasto?",
    ExpenseClarificationField.EXPENSE_DATE: (
        "Em qual data o gasto aconteceu? Responda no formato DD/MM/AAAA."
    ),
    ExpenseClarificationField.DESCRIPTION: "O que foi comprado ou pago?",
    ExpenseClarificationField.MERCHANT: "Em qual estabelecimento o gasto foi feito?",
    ExpenseClarificationField.CATEGORY: "Qual é a categoria do gasto?",
    ExpenseClarificationField.PAYMENT_METHOD: (
        "Como o gasto foi pago: Pix, débito, crédito, dinheiro, transferência ou boleto?"
    ),
    ExpenseClarificationField.INTENT: "Você quer registrar isso como gasto? Responda sim ou não.",
}
CLARIFICATION_TEXT = CLARIFICATION_QUESTIONS[ExpenseClarificationField.AMOUNT]
_NEW_EXPENSE_ACTION = re.compile(
    r"\b(?:gastei|paguei|comprei|abasteci|custou|desembolsei)\b", re.IGNORECASE
)
_MONETARY_REFERENCE = re.compile(r"(?:R\$|\d|\breais?\b)", re.IGNORECASE)
_EXPLICIT_MONETARY_VALUE = re.compile(
    r"(?:R\$[ \t]*)?[0-9][0-9.,]*(?:[ \t]*reais?)?", re.IGNORECASE
)
PROCESSING_FAILURE_TEXT = (
    "⚠️ Não consegui registrar esse gasto agora. Envie a mensagem novamente em alguns minutos."
)
AUDIO_NO_SPEECH_TEXT = (
    "Não consegui identificar uma fala nesse áudio. Envie outro áudio ou escreva o gasto."
)
AUDIO_INVALID_TEXT = (
    "Não consegui processar esse áudio. Envie novamente como mensagem de voz ou escreva o gasto."
)
IMAGE_INVALID_TEXT = (
    "Não consegui processar essa imagem. Envie outra foto legível ou escreva o gasto."
)
EXPENSE_NOT_FOUND_TEXT = "Gasto não encontrado ou indisponível."
INVALID_COMMAND_TEXT = "Comando inválido. Use o UUID completo, sem texto adicional."
EDIT_NOT_AVAILABLE_TEXT = "O fluxo de edição será disponibilizado na próxima etapa."
ACTION_CANCELLED_TEXT = "Operação cancelada."
NOTHING_TO_CANCEL_TEXT = "Nenhuma operação pendente."
DELETE_EXPIRED_TEXT = "A confirmação expirou. Envie um novo comando remover com o UUID do gasto."
EXPENSE_DELETED_TEXT = (
    "🗑️ Gasto removido.\n\nVocê já pode enviar outra mensagem para registrar um novo gasto."
)


def utc_now() -> datetime:
    return datetime.now(UTC)


def source_type_value(value: MessageSourceType | str) -> str:
    return value.value if isinstance(value, MessageSourceType) else value


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
        delete_confirmation_ttl_seconds: float = 600.0,
        clarification_ttl_seconds: float = 900.0,
        clarification_min_confidence: float = 0.75,
        media_provider: WhatsAppProvider | None = None,
        audio_transcriber_factory: Callable[[], AudioTranscriber] | None = None,
        image_analyzer_factory: Callable[[], ImageAnalyzer] | None = None,
        timing: PipelineTiming | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._interpreter_factory = interpreter_factory
        self._max_attempts = max_attempts
        self._retry_base_seconds = retry_base_seconds
        self._retry_max_seconds = retry_max_seconds
        self._jitter = jitter
        self._clock = clock
        self._delete_confirmation_ttl = timedelta(seconds=delete_confirmation_ttl_seconds)
        if clarification_ttl_seconds <= 0 or not 0 <= clarification_min_confidence <= 1:
            raise ValueError("invalid clarification configuration")
        self._clarification_ttl = timedelta(seconds=clarification_ttl_seconds)
        self._clarification_min_confidence = clarification_min_confidence
        self._media_provider = media_provider
        self._audio_transcriber_factory = audio_transcriber_factory
        self._image_analyzer_factory = image_analyzer_factory
        self._timing = timing or PipelineTiming(False)

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
            claimed = [
                (message.id, message.created_at, message.source_type, message.processing_attempts)
                for message in messages
            ]
        for message_id, created_at, source_type, attempt_number in claimed:
            fields = {
                "attempt_number": attempt_number,
                "source_type": source_type_value(source_type),
            }
            self._timing.event("processing_claimed", message_id, **fields)
            self._timing.event(
                "queue_wait_completed",
                message_id,
                duration_ms=self._timing.elapsed_ms(created_at, now),
                **fields,
            )
        return [message_id for message_id, *_ in claimed]

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
        message.media_remote_jid = None
        user = await session.get(User, message.user_id)
        await self._create_failure_notification(session, message, user)

    async def process(self, message_id: UUID) -> None:
        async with self._session_factory() as session:
            message = await session.get(ProcessedMessage, message_id)
            if message is None:
                return
            attempt_number = message.processing_attempts
            source_type = source_type_value(message.source_type)
        async with self._timing.span(
            None,
            "processing_completed",
            message_id,
            attempt_number=attempt_number,
            source_type=source_type,
        ) as span:
            try:
                await self._process(message_id)
            finally:
                async with self._session_factory() as session:
                    saved = await session.get(ProcessedMessage, message_id)
                    if saved is not None:
                        if saved.status is ProcessedMessageStatus.PENDING:
                            outcome = "transient_failure"
                        elif saved.status is ProcessedMessageStatus.FAILED:
                            outcome = "terminal_failure"
                        else:
                            outcome = "success"
                        span.result(
                            outcome=outcome,
                            stage=self._timing_stage(saved.last_error_code),
                            error_code=saved.last_error_code if outcome != "success" else None,
                            next_attempt_at=saved.next_attempt_at,
                        )

    async def _process(self, message_id: UUID) -> None:
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
            source_type = message.source_type
            transcribed_at = message.transcribed_at
            image_analyzed_at = message.image_analyzed_at
            image_analysis_payload = message.image_analysis
            media_caption = message.media_caption
            clarification_origin_id = message.clarification_origin_message_id

        if source_type == MessageSourceType.AUDIO and transcribed_at is None:
            text = await self._transcribe_audio(message_id)
            if text is None:
                return
        image_checkpoint: ImageAnalysisCheckpoint | None = None
        if source_type == MessageSourceType.IMAGE:
            try:
                media_caption = normalize_image_caption(media_caption)
            except ValueError:
                await self._mark_image_failed(
                    message_id, "IMAGE_ANALYSIS_INVALID_CHECKPOINT", IMAGE_INVALID_TEXT
                )
                return
            if image_analyzed_at is None:
                image_checkpoint = await self._analyze_image(message_id)
                if image_checkpoint is None:
                    return
            else:
                try:
                    image_checkpoint = ImageAnalysisCheckpoint.model_validate(
                        image_analysis_payload
                    )
                except (ValidationError, TypeError, ValueError):
                    await self._mark_image_failed(
                        message_id, "IMAGE_ANALYSIS_INVALID_CHECKPOINT", IMAGE_INVALID_TEXT
                    )
                    return
            text = image_checkpoint.interpreter_input(media_caption)
        if not text:
            await self._mark_audio_failed(
                message_id,
                "TRANSCRIPTION_INVALID_RESPONSE",
                AUDIO_INVALID_TEXT,
            )
            return

        command = parse_expense_command(text)
        if command is not None:
            try:
                await self._process_command(message_id, command)
            except IntegrityError:
                await self._recover_unique_conflict(message_id)
            return

        try:
            interpreter = self._interpreter_factory(timezone)
            clarification, clarification_expired = await self._load_clarification(message.user_id)
            is_new_intent = source_type == MessageSourceType.IMAGE or self._looks_like_new_expense(
                text
            )
            if (
                clarification_origin_id is not None
                and not is_new_intent
                and (
                    clarification is None
                    or clarification.origin_message_id != clarification_origin_id
                )
            ):
                await self._discard_stale_clarification_reply(message_id)
                return
            if clarification is not None and not is_new_intent:
                if clarification_expired:
                    await self._expire_clarification(message_id, clarification)
                    return
                interpretation = await self._interpret(
                    interpreter,
                    clarification.interpreter_input(text),
                    timestamp=clarification.reference_timestamp,
                    message=message,
                )
                self._validate_partial_interpretation(interpretation)
                await self._persist_clarification_answer(message_id, clarification, interpretation)
                return

            interpretation = await self._interpret(
                interpreter,
                text,
                timestamp=timestamp,
                message=message,
            )
            self._validate_partial_interpretation(interpretation)
            draft_interpretation = interpretation
            forced_fields = list(self._image_clarification_fields(image_checkpoint))
            if (
                source_type != MessageSourceType.IMAGE
                and len(self._explicit_monetary_values(text)) > 1
            ):
                forced_fields.append(ExpenseClarificationField.AMOUNT)
            if image_checkpoint is not None:
                interpretation = self._constrain_image_interpretation(
                    image_checkpoint, interpretation
                )
            clarification_fields = self._clarification_fields(
                interpretation, forced_fields=self._ordered_fields(forced_fields)
            )
            if not clarification_fields:
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
            async with (
                self._timing.span(
                    "expense_persistence_started",
                    "expense_persistence_completed",
                    message_id,
                    attempt_number=message.processing_attempts,
                    source_type=source_type_value(source_type),
                ),
                self._session_factory() as session,
                session.begin(),
            ):
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
                state = await self._locked_state(session, message.user_id)
                if clarification_fields:
                    context = self._new_clarification_context(
                        message,
                        draft_interpretation,
                        clarification_fields,
                        image_checkpoint=image_checkpoint,
                    )
                    self._set_clarification_state(state, context)
                    self._create_clarification_outbox(session, message, context.requested_field)
                    status = ProcessedMessageStatus.NEEDS_CLARIFICATION
                elif interpretation.intent is ExpenseIntent.CREATE_EXPENSE:
                    await self._create_expense(session, message, interpretation)
                    status = ProcessedMessageStatus.PROCESSED
                else:
                    if state.status in {
                        ConversationStatus.WAITING_EXPENSE_CLARIFICATION,
                        ConversationStatus.WAITING_EXPENSE_DELETE_CONFIRM,
                    }:
                        self._reset_state(state)
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

    async def _interpret(
        self,
        interpreter: ExpenseInterpreter,
        text: str,
        *,
        timestamp: datetime,
        message: ProcessedMessage,
    ) -> ExpenseInterpretation:
        async with self._timing.span(
            "interpretation_started",
            "interpretation_completed",
            message.id,
            attempt_number=message.processing_attempts,
            source_type=source_type_value(message.source_type),
        ):
            return await interpreter.interpret(text, reference_timestamp=timestamp)

    async def _load_clarification(
        self, user_id: UUID | None
    ) -> tuple[ExpenseClarificationContext | None, bool]:
        if user_id is None:
            return None, False
        async with self._session_factory() as session, session.begin():
            state = await session.scalar(
                select(ConversationState)
                .where(ConversationState.user_id == user_id)
                .with_for_update(of=ConversationState)
            )
            if (
                state is None
                or state.status is not ConversationStatus.WAITING_EXPENSE_CLARIFICATION
            ):
                return None, False
            try:
                context = ExpenseClarificationContext.model_validate(state.context)
            except (ValidationError, TypeError, ValueError):
                self._reset_state(state)
                return None, False
            return context, self._state_expired(state)

    async def _expire_clarification(
        self, message_id: UUID, expected: ExpenseClarificationContext | None
    ) -> None:
        async with self._session_factory() as session, session.begin():
            message = await session.scalar(self._locked_message_statement(message_id))
            if message is None or message.status is not ProcessedMessageStatus.PROCESSING:
                return
            if message.user_id is None:
                self._fail_locked_message(message, "USER_NOT_FOUND")
                return
            state = await self._locked_state(session, message.user_id)
            if state.status is ConversationStatus.WAITING_EXPENSE_CLARIFICATION:
                if (
                    expected is None
                    or state.context == expected.payload()
                    or self._state_expired(state)
                ):
                    self._reset_state(state)
            self._complete_successfully(message, ProcessedMessageStatus.NOT_EXPENSE)

    async def _discard_stale_clarification_reply(self, message_id: UUID) -> None:
        async with self._session_factory() as session, session.begin():
            message = await session.scalar(self._locked_message_statement(message_id))
            if message is None or message.status is not ProcessedMessageStatus.PROCESSING:
                return
            self._complete_successfully(message, ProcessedMessageStatus.NOT_EXPENSE)

    async def _persist_clarification_answer(
        self,
        message_id: UUID,
        expected: ExpenseClarificationContext,
        result: ExpenseInterpretation,
    ) -> None:
        async with self._session_factory() as session, session.begin():
            message = await session.scalar(
                self._locked_message_statement(message_id).options(
                    selectinload(ProcessedMessage.user)
                )
            )
            if (
                message is None
                or message.status is not ProcessedMessageStatus.PROCESSING
                or message.user_id is None
                or message.user is None
            ):
                return
            state = await self._locked_state(session, message.user_id)
            if (
                state.status is not ConversationStatus.WAITING_EXPENSE_CLARIFICATION
                or state.context != expected.payload()
            ):
                self._complete_successfully(message, ProcessedMessageStatus.NOT_EXPENSE)
                return
            if self._state_expired(state):
                self._reset_state(state)
                self._complete_successfully(message, ProcessedMessageStatus.NOT_EXPENSE)
                return

            if (
                expected.requested_field is ExpenseClarificationField.INTENT
                and result.intent is ExpenseIntent.NOT_EXPENSE
                and result.confidence >= self._clarification_min_confidence
            ):
                origin = await self._locked_clarification_origin(session, message, expected)
                if origin is not None:
                    self._complete_successfully(origin, ProcessedMessageStatus.NOT_EXPENSE)
                self._reset_state(state)
                self._complete_successfully(message, ProcessedMessageStatus.NOT_EXPENSE)
                return

            updated = self._merge_clarification(expected, result)
            if updated is None:
                self._create_clarification_outbox(session, message, expected.requested_field)
                self._complete_successfully(message, ProcessedMessageStatus.NEEDS_CLARIFICATION)
                return

            if updated.remaining_fields:
                next_context = ExpenseClarificationContext(
                    **updated.model_dump(exclude={"requested_field", "remaining_fields"}),
                    requested_field=updated.remaining_fields[0],
                    remaining_fields=updated.remaining_fields,
                )
                state.context = next_context.payload()
                self._create_clarification_outbox(session, message, next_context.requested_field)
                self._complete_successfully(message, ProcessedMessageStatus.NEEDS_CLARIFICATION)
                return

            final_result = self._clarification_result(updated, result.confidence)
            self._validate_interpretation(final_result)
            origin = await self._locked_clarification_origin(session, message, expected)
            if origin is None:
                self._reset_state(state)
                self._fail_locked_message(message, "CLARIFICATION_ORIGIN_INVALID")
                return
            await self._create_expense(
                session, message, final_result, origin_message=origin, reset_state=False
            )
            self._complete_successfully(origin, ProcessedMessageStatus.PROCESSED)
            self._reset_state(state)
            self._complete_successfully(message, ProcessedMessageStatus.PROCESSED)

    async def _locked_clarification_origin(
        self,
        session: AsyncSession,
        response: ProcessedMessage,
        context: ExpenseClarificationContext,
    ) -> ProcessedMessage | None:
        origin = await session.scalar(
            self._locked_message_statement(context.origin_message_id).options(
                selectinload(ProcessedMessage.user)
            )
        )
        if (
            origin is None
            or origin.user_id != response.user_id
            or origin.status is not ProcessedMessageStatus.NEEDS_CLARIFICATION
        ):
            return None
        return origin

    @staticmethod
    def _looks_like_new_expense(text: str) -> bool:
        return bool(_NEW_EXPENSE_ACTION.search(text) and _MONETARY_REFERENCE.search(text))

    @staticmethod
    def _explicit_monetary_values(text: str) -> set[Decimal]:
        values: set[Decimal] = set()
        for match in _EXPLICIT_MONETARY_VALUE.finditer(text):
            candidate = match.group().strip()
            if not candidate.lower().endswith(("real", "reais")) and not candidate.startswith("R$"):
                continue
            candidate = re.sub(r"[ \t]*reais?$", "", candidate, flags=re.IGNORECASE).strip()
            try:
                values.add(parse_monetary_value(candidate, maximum=EXPENSE_AMOUNT_MAX))
            except MonetaryValueError:
                continue
        return values

    @staticmethod
    def _normalized_missing_field(value: str) -> ExpenseClarificationField | None:
        normalized = "".join(
            character
            for character in unicodedata.normalize("NFKD", value.casefold())
            if not unicodedata.combining(character)
        )
        normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
        aliases = {
            "amount": ExpenseClarificationField.AMOUNT,
            "value": ExpenseClarificationField.AMOUNT,
            "valor": ExpenseClarificationField.AMOUNT,
            "total": ExpenseClarificationField.AMOUNT,
            "date": ExpenseClarificationField.EXPENSE_DATE,
            "expense_date": ExpenseClarificationField.EXPENSE_DATE,
            "data": ExpenseClarificationField.EXPENSE_DATE,
            "description": ExpenseClarificationField.DESCRIPTION,
            "descricao": ExpenseClarificationField.DESCRIPTION,
            "item": ExpenseClarificationField.DESCRIPTION,
            "merchant": ExpenseClarificationField.MERCHANT,
            "estabelecimento": ExpenseClarificationField.MERCHANT,
            "loja": ExpenseClarificationField.MERCHANT,
            "category": ExpenseClarificationField.CATEGORY,
            "categoria": ExpenseClarificationField.CATEGORY,
            "payment": ExpenseClarificationField.PAYMENT_METHOD,
            "payment_method": ExpenseClarificationField.PAYMENT_METHOD,
            "forma_de_pagamento": ExpenseClarificationField.PAYMENT_METHOD,
            "pagamento": ExpenseClarificationField.PAYMENT_METHOD,
            "intent": ExpenseClarificationField.INTENT,
            "intencao": ExpenseClarificationField.INTENT,
        }
        return aliases.get(normalized)

    @staticmethod
    def _ordered_fields(
        fields: list[ExpenseClarificationField] | tuple[ExpenseClarificationField, ...],
    ) -> tuple[ExpenseClarificationField, ...]:
        priority = (
            ExpenseClarificationField.AMOUNT,
            ExpenseClarificationField.EXPENSE_DATE,
            ExpenseClarificationField.DESCRIPTION,
            ExpenseClarificationField.MERCHANT,
            ExpenseClarificationField.CATEGORY,
            ExpenseClarificationField.PAYMENT_METHOD,
            ExpenseClarificationField.INTENT,
        )
        unique = set(fields)
        return tuple(field for field in priority if field in unique)

    def _clarification_fields(
        self,
        result: ExpenseInterpretation,
        *,
        forced_fields: tuple[ExpenseClarificationField, ...] = (),
    ) -> tuple[ExpenseClarificationField, ...]:
        if (
            result.intent is ExpenseIntent.NOT_EXPENSE
            and result.confidence >= self._clarification_min_confidence
            and not forced_fields
        ):
            return ()
        fields = list(forced_fields)
        for value in result.missing_fields:
            field = self._normalized_missing_field(value)
            if field is not None:
                fields.append(field)
        if result.intent in {ExpenseIntent.CREATE_EXPENSE, ExpenseIntent.UNCLEAR}:
            if result.amount is None:
                fields.append(ExpenseClarificationField.AMOUNT)
            if result.description is None or not result.description.strip():
                fields.append(ExpenseClarificationField.DESCRIPTION)
        if result.confidence < self._clarification_min_confidence:
            fields.append(ExpenseClarificationField.INTENT)
        if result.intent is ExpenseIntent.UNCLEAR and not fields:
            fields.append(ExpenseClarificationField.INTENT)
        return self._ordered_fields(fields)

    @staticmethod
    def _image_clarification_fields(
        checkpoint: ImageAnalysisCheckpoint | None,
    ) -> tuple[ExpenseClarificationField, ...]:
        if checkpoint is None or not checkpoint.is_financial_document:
            return ()
        fields: list[ExpenseClarificationField] = []
        if not checkpoint.is_legible:
            fields.extend([ExpenseClarificationField.AMOUNT, ExpenseClarificationField.DESCRIPTION])
        else:
            if len(checkpoint.distinct_amounts()) != 1:
                fields.append(ExpenseClarificationField.AMOUNT)
            if len(checkpoint.distinct_dates()) > 1:
                fields.append(ExpenseClarificationField.EXPENSE_DATE)
            if len(checkpoint.merchant_candidates) > 1:
                fields.append(ExpenseClarificationField.MERCHANT)
            if len(checkpoint.payment_method_candidates) > 1:
                fields.append(ExpenseClarificationField.PAYMENT_METHOD)
        return ExpenseProcessingService._ordered_fields(fields)

    def _new_clarification_context(
        self,
        message: ProcessedMessage,
        result: ExpenseInterpretation,
        fields: tuple[ExpenseClarificationField, ...],
        *,
        image_checkpoint: ImageAnalysisCheckpoint | None,
    ) -> ExpenseClarificationContext:
        values: dict[str, object] = {
            "amount": result.amount,
            "description": result.description,
            "merchant": result.merchant,
            "category": result.category,
            "payment_method": result.payment_method,
            "expense_date": result.expense_date,
        }
        if image_checkpoint is not None and image_checkpoint.is_legible:
            amounts = image_checkpoint.distinct_amounts()
            if len(amounts) == 1 and ExpenseClarificationField.AMOUNT not in fields:
                values["amount"] = next(iter(amounts))
            dates = image_checkpoint.distinct_dates()
            if len(dates) == 1 and ExpenseClarificationField.EXPENSE_DATE not in fields:
                candidate = next(iter(dates))
                try:
                    values["expense_date"] = date.fromisoformat(candidate)
                except ValueError:
                    values["expense_date"] = None
        for field in fields:
            if field is not ExpenseClarificationField.INTENT:
                values[field.value] = None
        reference_timestamp = message.message_timestamp
        if reference_timestamp.tzinfo is None or reference_timestamp.utcoffset() is None:
            reference_timestamp = reference_timestamp.replace(tzinfo=UTC)
        return ExpenseClarificationContext(
            origin_message_id=message.id,
            source_type=message.source_type,
            reference_timestamp=reference_timestamp,
            requested_field=fields[0],
            remaining_fields=fields,
            **values,
        )

    def _merge_clarification(
        self,
        context: ExpenseClarificationContext,
        result: ExpenseInterpretation,
    ) -> ExpenseClarificationContext | None:
        if (
            result.intent is not ExpenseIntent.CREATE_EXPENSE
            or result.confidence < self._clarification_min_confidence
        ):
            return None
        field = context.requested_field
        resolved: object | None
        if field is ExpenseClarificationField.INTENT:
            resolved = True
        else:
            resolved = getattr(result, field.value)
            if isinstance(resolved, str):
                resolved = resolved.strip() or None
        if resolved is None:
            return None

        values = context.model_dump(exclude={"requested_field", "remaining_fields"})
        if field is not ExpenseClarificationField.INTENT:
            values[field.value] = resolved
        remaining = [item for item in context.remaining_fields if item is not field]
        if values.get("amount") is None:
            remaining.append(ExpenseClarificationField.AMOUNT)
        if values.get("description") is None:
            remaining.append(ExpenseClarificationField.DESCRIPTION)
        if values.get("category") is None:
            remaining.append(ExpenseClarificationField.CATEGORY)
        ordered = self._ordered_fields(remaining)
        validation_fields = ordered or (field,)
        validated = ExpenseClarificationContext(
            **values,
            requested_field=validation_fields[0],
            remaining_fields=validation_fields,
        )
        if ordered:
            return validated
        return validated.model_copy(update={"remaining_fields": ()})

    @staticmethod
    def _clarification_result(
        context: ExpenseClarificationContext, confidence: float
    ) -> ExpenseInterpretation:
        assert context.amount is not None
        assert context.description is not None
        assert context.category is not None
        return ExpenseInterpretation(
            intent=ExpenseIntent.CREATE_EXPENSE,
            amount=context.amount,
            amount_evidence=None,
            description=context.description,
            merchant=context.merchant,
            category=context.category,
            payment_method=context.payment_method,
            expense_date=context.expense_date,
            confidence=confidence,
            missing_fields=[],
            reasoning_summary="clarification completed",
        )

    def _set_clarification_state(
        self, state: ConversationState, context: ExpenseClarificationContext
    ) -> None:
        state.status = ConversationStatus.WAITING_EXPENSE_CLARIFICATION
        state.active_expense_id = None
        state.context = context.payload()
        state.expires_at = self._now() + self._clarification_ttl

    def _create_clarification_outbox(
        self,
        session: AsyncSession,
        message: ProcessedMessage,
        field: ExpenseClarificationField,
    ) -> None:
        self._create_outbox(
            session,
            message,
            content=CLARIFICATION_QUESTIONS[field],
            kind=OutboundMessageKind.CLARIFICATION,
            expense=None,
        )

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
        ExpenseProcessingService._validate_partial_interpretation(result)
        if result.intent is not ExpenseIntent.CREATE_EXPENSE:
            return
        if not isinstance(result.amount, Decimal) or not result.amount.is_finite():
            raise GeminiSchemaError("Expense result failed deterministic validation")
        if result.description is None or not result.description.strip():
            raise GeminiSchemaError("Expense result failed deterministic validation")

    @staticmethod
    def _validate_partial_interpretation(result: ExpenseInterpretation) -> None:
        if result.intent is not ExpenseIntent.CREATE_EXPENSE and result.amount is not None:
            raise GeminiSchemaError("Non-expense result must not contain amount")
        if result.amount is not None:
            if not isinstance(result.amount, Decimal) or not result.amount.is_finite():
                raise GeminiSchemaError("Expense result failed deterministic validation")
            if result.amount <= 0 or result.amount > EXPENSE_AMOUNT_MAX:
                raise InterpretationLimitError("AMOUNT_OUT_OF_RANGE")
            if result.amount.as_tuple().exponent < -EXPENSE_AMOUNT_SCALE:
                raise InterpretationLimitError("AMOUNT_SCALE_EXCEEDED")
        if (
            result.description is not None
            and len(result.description.strip()) > EXPENSE_DESCRIPTION_MAX_LENGTH
        ):
            raise InterpretationLimitError("DESCRIPTION_TOO_LONG")
        if result.merchant is not None and len(result.merchant) > EXPENSE_MERCHANT_MAX_LENGTH:
            raise InterpretationLimitError("MERCHANT_TOO_LONG")
        payment_method = result.payment_method.value if result.payment_method else None
        if payment_method is not None and len(payment_method) > EXPENSE_PAYMENT_METHOD_MAX_LENGTH:
            raise InterpretationLimitError("PAYMENT_METHOD_TOO_LONG")

    @staticmethod
    def _constrain_image_interpretation(
        checkpoint: ImageAnalysisCheckpoint,
        result: ExpenseInterpretation,
    ) -> ExpenseInterpretation:
        if not checkpoint.is_financial_document:
            intent = ExpenseIntent.NOT_EXPENSE
        elif not checkpoint.is_legible:
            intent = ExpenseIntent.UNCLEAR
        elif len(checkpoint.distinct_amounts()) != 1 or len(checkpoint.distinct_dates()) > 1:
            intent = ExpenseIntent.UNCLEAR
        elif result.intent is ExpenseIntent.CREATE_EXPENSE:
            visual_amount = next(iter(checkpoint.distinct_amounts()))
            valid_evidence = {
                candidate.evidence
                for candidate in checkpoint.amount_candidates
                if Decimal(candidate.value) == visual_amount
            }
            intent = (
                ExpenseIntent.CREATE_EXPENSE
                if result.amount == visual_amount and result.amount_evidence in valid_evidence
                else ExpenseIntent.UNCLEAR
            )
        else:
            intent = result.intent
        if intent is result.intent:
            return result
        return result.model_copy(
            update={
                "intent": intent,
                "amount": None,
                "amount_evidence": None,
                "description": None,
            }
        )

    async def _create_expense(
        self,
        session: AsyncSession,
        message: ProcessedMessage,
        result: ExpenseInterpretation,
        *,
        origin_message: ProcessedMessage | None = None,
        reset_state: bool = True,
    ) -> None:
        assert result.amount is not None and result.description is not None
        origin = origin_message or message
        if origin.user_id != message.user_id:
            raise GeminiSchemaError("Clarification user does not match expense origin")
        category = await session.scalar(
            select(Category).where(
                Category.name == result.category.value, Category.is_active.is_(True)
            )
        )
        if category is None:
            raise GeminiSchemaError("Canonical category is unavailable")
        timezone = self._timezone(message.user.timezone)
        timestamp = origin.message_timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        expense_date = result.expense_date or timestamp.astimezone(timezone).date()
        expense = Expense(
            user_id=message.user_id,
            processed_message_id=origin.id,
            category_id=category.id,
            amount=result.amount,
            description=result.description.strip(),
            expense_date=expense_date,
            merchant=result.merchant,
            payment_method=result.payment_method.value if result.payment_method else None,
            source_type=origin.source_type,
        )
        session.add(expense)
        await session.flush()
        if reset_state:
            state = await session.scalar(
                select(ConversationState).where(ConversationState.user_id == message.user_id)
            )
            if state is None:
                session.add(ConversationState(user_id=message.user_id))
            else:
                self._reset_state(state)
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
        actions: list[dict[str, str]] | None = None,
        fallback_content: str | None = None,
    ) -> None:
        session.add(
            OutboundMessage(
                user_id=message.user_id,
                expense_id=expense.id if expense else None,
                destination=message.user.phone_number,
                content=content,
                content_type="BUTTONS" if actions else "TEXT",
                actions=actions,
                fallback_content=fallback_content,
                kind=kind,
                dedup_key=f"processed-message:{message.id}:{kind.value}",
                status=OutboundMessageStatus.PENDING,
                available_at=datetime.now(UTC),
            )
        )

    async def _mark_failed(self, message_id: UUID, code: str) -> None:
        async with self._session_factory() as session, session.begin():
            message = await session.scalar(
                self._locked_message_statement(message_id).options(
                    selectinload(ProcessedMessage.user)
                )
            )
            if message is not None and message.status == ProcessedMessageStatus.PROCESSING:
                message.status = ProcessedMessageStatus.FAILED
                message.error_code = code
                message.last_error_code = code
                message.locked_at = None
                message.next_attempt_at = None
                message.media_remote_jid = None
                message.attempt_count += 1
                if message.source_type == MessageSourceType.IMAGE:
                    await self._create_failure_notification(session, message, message.user)

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
                message.media_remote_jid = None
                user = await session.get(User, message.user_id)
                await self._create_failure_notification(session, message, user)
                return
            message.status = ProcessedMessageStatus.PENDING
            message.next_attempt_at = self._now() + self._retry_delay(message.processing_attempts)

    async def _analyze_image(self, message_id: UUID) -> ImageAnalysisCheckpoint | None:
        async with self._session_factory() as session:
            message = await session.get(ProcessedMessage, message_id)
            if message is None or message.status != ProcessedMessageStatus.PROCESSING:
                return None
            if message.image_analyzed_at is not None:
                try:
                    return ImageAnalysisCheckpoint.model_validate(message.image_analysis)
                except (ValidationError, TypeError, ValueError):
                    await self._mark_image_failed(
                        message_id, "IMAGE_ANALYSIS_INVALID_CHECKPOINT", IMAGE_INVALID_TEXT
                    )
                    return None
            if (
                self._media_provider is None
                or self._image_analyzer_factory is None
                or not message.media_remote_jid
                or not message.media_mime_type
            ):
                await self._mark_image_failed(
                    message_id, "IMAGE_ANALYSIS_CONFIGURATION_ERROR", IMAGE_INVALID_TEXT
                )
                return None
            attempt_number = message.processing_attempts
            caption = message.media_caption
            reference = EvolutionMediaReference(
                message.external_message_id,
                message.media_remote_jid,
                False,
            )
            media = InboundMedia(
                media_type="image",
                declared_mime_type=message.media_mime_type,
                caption=caption,
                reference=reference,
            )

        try:
            async with self._timing.span(
                "image_download_started",
                "image_download_completed",
                message_id,
                attempt_number=attempt_number,
                source_type=MessageSourceType.IMAGE.value,
                mime_type=media.declared_mime_type,
            ) as download_span:
                try:
                    content = await self._media_provider.download_media(media)
                except MediaError as exc:
                    download_span.result(
                        outcome=("transient_failure" if exc.transient else "terminal_failure"),
                        error_code=exc.code.value,
                    )
                    raise
                else:
                    download_span.result(outcome="success", size_bytes=len(content))
            mime_type = media.declared_mime_type.partition(";")[0].strip().lower()
            width, height = EvolutionWhatsAppProvider._image_dimensions(mime_type, content)
            detected_format = {
                "image/jpeg": "JPEG",
                "image/png": "PNG",
                "image/webp": "WEBP",
            }.get(mime_type)
            if detected_format is None:
                raise MediaError(MediaErrorCode.UNSUPPORTED_TYPE, transient=False)
            image = ValidatedImage(
                content=content,
                mime_type=mime_type,
                width=width,
                height=height,
                detected_format=detected_format,
            )
            async with self._timing.span(
                "image_analysis_started",
                "image_analysis_completed",
                message_id,
                attempt_number=attempt_number,
                source_type=MessageSourceType.IMAGE.value,
            ) as analysis_span:
                try:
                    analysis = await self._image_analyzer_factory().analyze(image, caption)
                except ImageAnalysisError as exc:
                    analysis_span.result(
                        outcome=("transient_failure" if exc.transient else "terminal_failure"),
                        error_code=exc.code.value,
                        http_status=(exc.metadata.http_status if exc.metadata else None),
                    )
                    raise
                else:
                    analysis_span.result(outcome="success")
            checkpoint = ImageAnalysisCheckpoint.from_analysis(analysis)
        except asyncio.CancelledError:
            await asyncio.shield(self._retry_or_fail(message_id, "IMAGE_ANALYSIS_UNAVAILABLE"))
            raise
        except MediaError as exc:
            if exc.transient:
                await self._retry_or_fail(message_id, exc.code.value)
            else:
                await self._mark_image_failed(message_id, exc.code.value, IMAGE_INVALID_TEXT)
            return None
        except ImageAnalysisError as exc:
            if exc.transient:
                await self._retry_or_fail(message_id, exc.code.value)
            else:
                await self._mark_image_failed(message_id, exc.code.value, IMAGE_INVALID_TEXT)
            return None
        except (ValidationError, TypeError, ValueError):
            await self._mark_image_failed(
                message_id, "IMAGE_ANALYSIS_INVALID_CHECKPOINT", IMAGE_INVALID_TEXT
            )
            return None

        async with (
            self._timing.span(
                "image_checkpoint_started",
                "image_checkpoint_completed",
                message_id,
                attempt_number=attempt_number,
                source_type=MessageSourceType.IMAGE.value,
            ) as checkpoint_span,
            self._session_factory() as session,
            session.begin(),
        ):
            message = await session.scalar(self._locked_message_statement(message_id))
            if (
                message is None
                or message.status != ProcessedMessageStatus.PROCESSING
                or message.processing_attempts != attempt_number
            ):
                checkpoint_span.result(outcome="terminal_failure", error_code="STALE_ATTEMPT")
                return None
            if message.image_analyzed_at is None:
                message.image_analysis = checkpoint.payload()
                message.image_analyzed_at = self._now()
                message.media_remote_jid = None
            checkpoint_span.result(outcome="success")
            return checkpoint

    async def _mark_image_failed(self, message_id: UUID, code: str, content: str) -> None:
        async with self._session_factory() as session, session.begin():
            message = await session.scalar(
                self._locked_message_statement(message_id).options(
                    selectinload(ProcessedMessage.user)
                )
            )
            if message is None or message.status != ProcessedMessageStatus.PROCESSING:
                return
            message.status = ProcessedMessageStatus.FAILED
            message.error_code = code
            message.last_error_code = code
            message.locked_at = None
            message.next_attempt_at = None
            message.media_remote_jid = None
            message.attempt_count += 1
            await self._create_failure_notification(session, message, message.user, content=content)

    async def _transcribe_audio(self, message_id: UUID) -> str | None:
        async with self._session_factory() as session:
            message = await session.get(ProcessedMessage, message_id)
            if message is None or message.status != ProcessedMessageStatus.PROCESSING:
                return None
            if message.transcribed_at is not None:
                return message.accepted_text
            if (
                self._media_provider is None
                or self._audio_transcriber_factory is None
                or not message.media_remote_jid
                or not message.media_mime_type
            ):
                await self._mark_audio_failed(
                    message_id, "MEDIA_CONFIGURATION_ERROR", AUDIO_INVALID_TEXT
                )
                return None
            reference = EvolutionMediaReference(
                message.external_message_id,
                message.media_remote_jid,
                False,
            )
            media = InboundMedia(
                media_type="audio",
                declared_mime_type=message.media_mime_type,
                declared_duration_seconds=message.media_duration_seconds,
                is_voice_note=bool(message.media_is_voice_note),
                reference=reference,
            )

        try:
            async with self._timing.span(
                "media_download_started",
                "media_download_completed",
                message_id,
                attempt_number=message.processing_attempts,
                source_type=source_type_value(message.source_type),
                mime_type=media.declared_mime_type,
                audio_duration_seconds=media.declared_duration_seconds,
            ) as download_span:
                content = await self._media_provider.download_media(media)
                download_span.result(size_bytes=len(content))
            async with self._timing.span(
                "transcription_started",
                "transcription_completed",
                message_id,
                attempt_number=message.processing_attempts,
                source_type=source_type_value(message.source_type),
                size_bytes=len(content),
                mime_type=media.declared_mime_type,
                audio_duration_seconds=media.declared_duration_seconds,
            ):
                transcription = await self._audio_transcriber_factory().transcribe(
                    ValidatedAudio(
                        content=content,
                        mime_type=media.declared_mime_type,
                        declared_duration_seconds=media.declared_duration_seconds,
                        is_voice_note=media.is_voice_note,
                    )
                )
        except asyncio.CancelledError:
            await asyncio.shield(self._retry_or_fail(message_id, "AUDIO_PROCESSING_INTERRUPTED"))
            raise
        except MediaError as exc:
            if exc.transient:
                await self._retry_or_fail(message_id, exc.code.value)
            else:
                await self._mark_audio_failed(message_id, exc.code.value, AUDIO_INVALID_TEXT)
            return None
        except NoSpeechError as exc:
            await self._mark_audio_failed(message_id, exc.code.value, AUDIO_NO_SPEECH_TEXT)
            return None
        except TranscriptionError as exc:
            if exc.transient:
                await self._retry_or_fail(message_id, exc.code.value)
            else:
                await self._mark_audio_failed(message_id, exc.code.value, AUDIO_INVALID_TEXT)
            return None

        async with (
            self._timing.span(
                "transcript_checkpoint_started",
                "transcript_checkpoint_completed",
                message_id,
                attempt_number=message.processing_attempts,
                source_type=source_type_value(message.source_type),
            ),
            self._session_factory() as session,
            session.begin(),
        ):
            message = await session.scalar(self._locked_message_statement(message_id))
            if message is None or message.status != ProcessedMessageStatus.PROCESSING:
                return None
            if message.transcribed_at is None:
                message.accepted_text = transcription.transcript
                message.transcribed_at = self._now()
                message.media_remote_jid = None
            return message.accepted_text

    async def _mark_audio_failed(self, message_id: UUID, code: str, content: str) -> None:
        async with self._session_factory() as session, session.begin():
            message = await session.scalar(
                self._locked_message_statement(message_id).options(
                    selectinload(ProcessedMessage.user)
                )
            )
            if message is None or message.status != ProcessedMessageStatus.PROCESSING:
                return
            message.status = ProcessedMessageStatus.FAILED
            message.error_code = code
            message.last_error_code = code
            message.locked_at = None
            message.next_attempt_at = None
            message.media_remote_jid = None
            message.attempt_count += 1
            await self._create_failure_notification(session, message, message.user, content=content)

    @staticmethod
    def _locked_message_statement(message_id: UUID) -> Select[tuple[ProcessedMessage]]:
        return (
            select(ProcessedMessage)
            .where(ProcessedMessage.id == message_id)
            .with_for_update(of=ProcessedMessage)
        )

    async def _create_failure_notification(
        self,
        session: AsyncSession,
        message: ProcessedMessage,
        user: User | None,
        *,
        content: str = PROCESSING_FAILURE_TEXT,
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
                content=content,
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

    async def _process_command(self, message_id: UUID, command: ExpenseCommand) -> None:
        async with self._session_factory() as session, session.begin():
            message = await session.scalar(
                self._locked_message_statement(message_id).options(
                    selectinload(ProcessedMessage.user)
                )
            )
            if message is None or message.status != ProcessedMessageStatus.PROCESSING:
                return
            if message.user is None or message.user_id is None:
                self._fail_locked_message(message, "USER_NOT_FOUND")
                return
            state = await self._locked_state(session, message.user_id)
            if command.type is ExpenseCommandType.REMOVE:
                await self._request_delete(session, message, state, command)
            elif command.type is ExpenseCommandType.CONFIRM_REMOVE:
                await self._confirm_delete(session, message, state, command)
            elif command.type is ExpenseCommandType.CANCEL:
                pending = state.status in {
                    ConversationStatus.WAITING_EXPENSE_DELETE_CONFIRM,
                    ConversationStatus.WAITING_EXPENSE_CLARIFICATION,
                }
                self._reset_state(state)
                self._complete_command(
                    session,
                    message,
                    ACTION_CANCELLED_TEXT if pending else NOTHING_TO_CANCEL_TEXT,
                    OutboundMessageKind.ACTION_CANCELLED,
                )
            elif command.type is ExpenseCommandType.EDIT:
                self._complete_command(
                    session,
                    message,
                    EDIT_NOT_AVAILABLE_TEXT,
                    OutboundMessageKind.ACTION_ERROR,
                )
            else:
                self._complete_command(
                    session,
                    message,
                    INVALID_COMMAND_TEXT,
                    OutboundMessageKind.ACTION_ERROR,
                )

    async def _request_delete(
        self,
        session: AsyncSession,
        message: ProcessedMessage,
        state: ConversationState,
        command: ExpenseCommand,
    ) -> None:
        assert command.expense_id is not None
        expense = await session.scalar(
            select(Expense)
            .where(
                Expense.id == command.expense_id,
                Expense.user_id == message.user_id,
                Expense.deleted_at.is_(None),
            )
            .with_for_update(of=Expense)
        )
        if expense is None:
            self._complete_command(
                session,
                message,
                EXPENSE_NOT_FOUND_TEXT,
                OutboundMessageKind.ACTION_ERROR,
            )
            return
        state.status = ConversationStatus.WAITING_EXPENSE_DELETE_CONFIRM
        state.active_expense_id = expense.id
        state.context = {"action": "DELETE", "expense_id": str(expense.id)}
        state.expires_at = self._now() + self._delete_confirmation_ttl
        self._complete_command(
            session,
            message,
            format_delete_confirmation_request(expense),
            OutboundMessageKind.DELETE_CONFIRMATION_REQUEST,
            expense=expense,
            actions=delete_confirmation_actions(expense.id),
            fallback_content=format_delete_confirmation_fallback(expense),
        )

    async def _confirm_delete(
        self,
        session: AsyncSession,
        message: ProcessedMessage,
        state: ConversationState,
        command: ExpenseCommand,
    ) -> None:
        assert command.expense_id is not None
        if self._state_expired(state):
            self._reset_state(state)
            self._complete_command(
                session,
                message,
                DELETE_EXPIRED_TEXT,
                OutboundMessageKind.ACTION_ERROR,
            )
            return
        if (
            state.status is not ConversationStatus.WAITING_EXPENSE_DELETE_CONFIRM
            or state.active_expense_id != command.expense_id
        ):
            self._complete_command(
                session,
                message,
                EXPENSE_NOT_FOUND_TEXT,
                OutboundMessageKind.ACTION_ERROR,
            )
            return
        expense = await session.scalar(
            select(Expense)
            .where(
                Expense.id == command.expense_id,
                Expense.user_id == message.user_id,
                Expense.deleted_at.is_(None),
            )
            .with_for_update(of=Expense)
        )
        if expense is None:
            self._reset_state(state)
            self._complete_command(
                session,
                message,
                EXPENSE_NOT_FOUND_TEXT,
                OutboundMessageKind.ACTION_ERROR,
            )
            return
        old_data = expense_history_data(expense)
        now = self._now()
        expense.deleted_at = now
        expense.updated_at = now
        new_data = expense_history_data(expense)
        session.add(
            ExpenseHistory(
                expense_id=expense.id,
                action=ExpenseHistoryAction.DELETE,
                changes={"old_data": old_data, "new_data": new_data},
            )
        )
        self._reset_state(state)
        self._complete_command(
            session,
            message,
            EXPENSE_DELETED_TEXT,
            OutboundMessageKind.EXPENSE_DELETED,
            expense=expense,
        )

    @staticmethod
    async def _locked_state(session: AsyncSession, user_id: UUID) -> ConversationState:
        state = await session.scalar(
            select(ConversationState)
            .where(ConversationState.user_id == user_id)
            .with_for_update(of=ConversationState)
        )
        if state is None:
            state = ConversationState(user_id=user_id)
            session.add(state)
            await session.flush()
        return state

    def _complete_command(
        self,
        session: AsyncSession,
        message: ProcessedMessage,
        content: str,
        kind: OutboundMessageKind,
        *,
        expense: Expense | None = None,
        actions: list[dict[str, str]] | None = None,
        fallback_content: str | None = None,
    ) -> None:
        self._create_outbox(
            session,
            message,
            content=content,
            kind=kind,
            expense=expense,
            actions=actions,
            fallback_content=fallback_content,
        )
        self._complete_successfully(message, ProcessedMessageStatus.PROCESSED)

    @staticmethod
    def _reset_state(state: ConversationState) -> None:
        state.status = ConversationStatus.IDLE
        state.active_expense_id = None
        state.context = None
        state.expires_at = None

    def _state_expired(self, state: ConversationState) -> bool:
        if state.status not in {
            ConversationStatus.WAITING_EXPENSE_DELETE_CONFIRM,
            ConversationStatus.WAITING_EXPENSE_CLARIFICATION,
        }:
            return False
        if state.expires_at is None:
            return True
        expires_at = state.expires_at
        if expires_at.tzinfo is None or expires_at.utcoffset() is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        return expires_at <= self._now()

    @staticmethod
    def _fail_locked_message(message: ProcessedMessage, code: str) -> None:
        message.status = ProcessedMessageStatus.FAILED
        message.error_code = code
        message.last_error_code = code
        message.locked_at = None
        message.next_attempt_at = None

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
    def _timing_stage(error_code: str | None) -> str:
        if error_code is None:
            return "processing"
        if error_code.startswith("MEDIA_"):
            return "media_download"
        if error_code.startswith(("TRANSCRIPTION_", "AUDIO_")):
            return "transcription"
        if error_code.startswith("IMAGE_ANALYSIS_"):
            return "image_analysis"
        if error_code.startswith("GEMINI_"):
            return "interpretation"
        if error_code.startswith("PERSISTENCE_"):
            return "expense_persistence"
        return "processing"

    @staticmethod
    def _timezone(name: str) -> ZoneInfo:
        try:
            return ZoneInfo(name)
        except ZoneInfoNotFoundError:
            return ZoneInfo("UTC")


def format_expense_confirmation(expense: Expense, category_name: str) -> str:
    amount = f"{expense.amount:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
    return (
        "✅ Novo Gasto Registrado!\n\n"
        f"📝 Descrição: {expense.description}\n"
        f"🛍️ Categoria: {category_name}\n"
        f"💵 Valor: R$ {amount}\n\n"
        f"📅 Data: {expense.expense_date.strftime('%d/%m/%Y')}\n"
        f"⚙️ ID: {str(expense.id)[:8]}"
    )


def format_delete_confirmation_request(expense: Expense) -> str:
    amount = f"{expense.amount:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
    return (
        "⚠️ Confirmar remoção?\n\n"
        f"Valor: R$ {amount}\n"
        f"Descrição: {expense.description}\n"
        f"Data: {expense.expense_date.strftime('%d/%m/%Y')}\n\n"
        "Use os botões abaixo para confirmar ou cancelar."
    )


def format_delete_confirmation_fallback(expense: Expense) -> str:
    amount = f"{expense.amount:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
    return (
        "⚠️ Confirmar remoção?\n\n"
        f"Valor: R$ {amount}\n"
        f"Descrição: {expense.description}\n"
        f"Data: {expense.expense_date.strftime('%d/%m/%Y')}\n\n"
        "Para confirmar:\n"
        f"confirmar-remocao {expense.id}\n\n"
        "Para cancelar:\n"
        "cancelar"
    )


def format_expense_confirmation_fallback(expense: Expense, category_name: str) -> str:
    return (
        f"{format_expense_confirmation(expense, category_name)}\n\n"
        "Para editar:\n"
        f"editar {expense.id}\n\n"
        "Para remover:\n"
        f"remover {expense.id}"
    )


def expense_confirmation_actions(expense_id: UUID) -> list[dict[str, str]]:
    return [
        {
            "id": encode_expense_action(ExpenseCommand(ExpenseCommandType.EDIT, expense_id)),
            "label": "✏️ Editar",
        },
        {
            "id": encode_expense_action(ExpenseCommand(ExpenseCommandType.REMOVE, expense_id)),
            "label": "↩️ Excluir",
        },
    ]


def delete_confirmation_actions(expense_id: UUID) -> list[dict[str, str]]:
    return [
        {
            "id": encode_expense_action(
                ExpenseCommand(ExpenseCommandType.CONFIRM_REMOVE, expense_id)
            ),
            "label": "️ Confirmar exclusão",
        },
        {
            "id": encode_expense_action(ExpenseCommand(ExpenseCommandType.CANCEL)),
            "label": "Cancelar",
        },
    ]


def expense_history_data(expense: Expense) -> dict[str, object]:
    return {
        "id": str(expense.id),
        "user_id": str(expense.user_id),
        "category_id": str(expense.category_id),
        "amount": str(expense.amount),
        "description": expense.description,
        "expense_date": expense.expense_date.isoformat(),
        "merchant": expense.merchant,
        "payment_method": expense.payment_method,
        "deleted_at": expense.deleted_at.isoformat() if expense.deleted_at else None,
    }
