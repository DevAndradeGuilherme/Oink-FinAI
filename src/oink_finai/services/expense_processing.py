import asyncio
import random
import re
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
    ExpenseCategory,
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
from oink_finai.domain.expense_query import ExpenseQueryIntent
from oink_finai.domain.monetary_value import MonetaryValueError, parse_monetary_value
from oink_finai.providers.whatsapp.base import WhatsAppProvider
from oink_finai.providers.whatsapp.evolution import (
    EvolutionMediaReference,
    EvolutionWhatsAppProvider,
)
from oink_finai.providers.whatsapp.media_errors import MediaError, MediaErrorCode
from oink_finai.schemas.expense_interpretation import ExpenseInterpretation
from oink_finai.schemas.expense_query import ExpenseQueryPlan
from oink_finai.schemas.expense_query_checkpoint import (
    ExpenseClassificationCheckpoint,
    ExpenseQueryPlanCheckpoint,
    ExpenseQueryResultCheckpoint,
)
from oink_finai.schemas.expense_query_messages import ExpenseQueryFormattingContext
from oink_finai.schemas.expense_query_result import (
    ExpenseAggregateResult,
    ExpenseComparisonResult,
    ExpenseGroupResult,
    ExpenseListResult,
    ExpenseQueryResult,
)
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
from oink_finai.services.expense_query_executor import (
    ExpenseQueryExecutionError,
    ExpenseQueryExecutor,
)
from oink_finai.services.expense_query_interpreter import ExpenseQueryInterpreter
from oink_finai.services.expense_query_result_formatter import ExpenseQueryResultFormatter
from oink_finai.services.image_analysis_errors import ImageAnalysisError
from oink_finai.services.image_analyzer import (
    ImageAnalyzer,
    ValidatedImage,
    normalize_image_caption,
)
from oink_finai.services.interpretation_errors import (
    InterpretationError,
    InterpretationErrorCode,
)
from oink_finai.services.pipeline_timing import PipelineTiming
from oink_finai.services.transcription_errors import NoSpeechError, TranscriptionError

_EXPLICIT_MONETARY_VALUE = re.compile(
    r"(?:R\$[ \t]*)?[0-9][0-9.,]*(?:[ \t]*reais?)?", re.IGNORECASE
)
_LITERAL_EXPENSE_DESCRIPTION = re.compile(
    r"\b(?:gastei|paguei)\b.{0,80}?\b(?:com|de)\s+"
    r"(?P<description>[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9 -]{0,99}?)"
    r"(?=\s+(?:hoje|ontem|anteontem|via|no|na|pelo|pela)\b|[.,!?]|$)",
    re.IGNORECASE,
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
INCOMPLETE_EXPENSE_TEMPLATE = (
    "⚠️ Não foi possível registrar o gasto.\n\n"
    "Está faltando: {missing_fields}.\n\n"
    "Envie novamente informando o que foi comprado ou pago e o valor.\n\n"
    "Exemplo: Gastei R$ 32,90 com gasolina."
)
QUERY_UNCLEAR_TEXT = (
    "Não consegui entender a consulta. Reenvie uma pergunta completa, por exemplo: "
    "Quanto gastei com alimentação este mês?"
)
QUERY_FAILURE_TEXT = (
    "Não consegui consultar seus gastos agora. Reenvie a consulta em alguns minutos."
)
QUERY_INVALID_TEXT = (
    "Não consegui processar essa consulta. Reenvie informando claramente o período e os filtros."
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
        media_provider: WhatsAppProvider | None = None,
        audio_transcriber_factory: Callable[[], AudioTranscriber] | None = None,
        image_analyzer_factory: Callable[[], ImageAnalyzer] | None = None,
        query_interpreter_factory: Callable[[str], ExpenseQueryInterpreter] | None = None,
        query_executor: ExpenseQueryExecutor | None = None,
        query_formatter: ExpenseQueryResultFormatter | None = None,
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
        self._media_provider = media_provider
        self._audio_transcriber_factory = audio_transcriber_factory
        self._image_analyzer_factory = image_analyzer_factory
        self._query_interpreter_factory = query_interpreter_factory
        self._query_executor = query_executor
        self._query_formatter = query_formatter
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
        if message.classification_intent is ExpenseIntent.QUERY and user is not None:
            self._create_query_outbox_pages(
                session,
                message,
                (QUERY_FAILURE_TEXT,),
                kind=OutboundMessageKind.QUERY_GUIDANCE,
                destination=user.phone_number,
            )
            message.query_page_count = 1
            message.query_result_checkpoint = None
        else:
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
            query_reference_timestamp = message.created_at
            timezone = user.timezone
            source_type = message.source_type
            transcribed_at = message.transcribed_at
            image_analyzed_at = message.image_analyzed_at
            image_analysis_payload = message.image_analysis
            media_caption = message.media_caption
            classified_at = message.classified_at
            classification_payload = message.classification_checkpoint
            classification_intent = message.classification_intent

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
            if classified_at is not None:
                classification = ExpenseClassificationCheckpoint.from_payload(
                    classification_payload
                )
                interpretation = classification.to_interpretation()
                missing_required = list(classification.missing_fields)
            else:
                interpreter = self._interpreter_factory(timezone)
                interpretation = await self._interpret(
                    interpreter,
                    text,
                    timestamp=timestamp,
                    message=message,
                )
                self._validate_partial_interpretation(interpretation)
                if image_checkpoint is not None:
                    interpretation = self._constrain_image_interpretation(
                        image_checkpoint, interpretation
                    )
                    if interpretation.intent is ExpenseIntent.QUERY:
                        interpretation = interpretation.model_copy(
                            update={"intent": ExpenseIntent.UNCLEAR}
                        )
                elif (
                    interpretation.intent is ExpenseIntent.CREATE_EXPENSE
                    and (
                        interpretation.description is None or not interpretation.description.strip()
                    )
                    and (description := self._literal_description(text)) is not None
                ):
                    interpretation = interpretation.model_copy(update={"description": description})
                missing_required = self._missing_required_fields(
                    interpretation,
                    force_amount=(
                        source_type != MessageSourceType.IMAGE
                        and len(self._explicit_monetary_values(text)) > 1
                    ),
                )
                interpretation = interpretation.model_copy(
                    update={"missing_fields": missing_required}
                )
                if not missing_required:
                    self._validate_interpretation(interpretation)
                checkpointed = await self._checkpoint_classification(
                    message_id, interpretation, message.processing_attempts
                )
                if checkpointed is None:
                    return
                interpretation = checkpointed
                missing_required = list(interpretation.missing_fields)
        except asyncio.CancelledError:
            await asyncio.shield(
                self._retry_or_fail(message_id, InterpretationErrorCode.TIMEOUT.value)
            )
            raise
        except InterpretationLimitError as exc:
            await self._mark_failed(message_id, exc.code)
            return
        except InterpretationError as exc:
            if self._is_transient(exc):
                await self._retry_or_fail(message_id, self._error_code(exc))
            else:
                await self._mark_failed(message_id, self._error_code(exc))
            return
        except (ValidationError, TypeError, ValueError):
            if classification_intent is ExpenseIntent.QUERY:
                await self._mark_query_failed(
                    message_id,
                    "CLASSIFICATION_INVALID_CHECKPOINT",
                    attempt_number=message.processing_attempts,
                )
            else:
                await self._mark_failed(message_id, "CLASSIFICATION_INVALID_CHECKPOINT")
            return

        if interpretation.intent is ExpenseIntent.QUERY:
            await self._process_query(
                message_id,
                text,
                timestamp=query_reference_timestamp,
                timezone=timezone,
                attempt_number=message.processing_attempts,
            )
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
                state = await self._existing_locked_state(session, message.user_id)
                if (
                    state is not None
                    and state.status is ConversationStatus.WAITING_EXPENSE_CLARIFICATION
                ):
                    self._reset_state(state)
                if missing_required:
                    self._create_incomplete_expense_outbox(session, message, missing_required)
                    status = ProcessedMessageStatus.PROCESSED
                elif interpretation.intent is ExpenseIntent.CREATE_EXPENSE:
                    await self._create_expense(session, message, interpretation)
                    status = ProcessedMessageStatus.PROCESSED
                else:
                    if state is not None and state.status in {
                        ConversationStatus.WAITING_EXPENSE_CLARIFICATION,
                        ConversationStatus.WAITING_EXPENSE_DELETE_CONFIRM,
                    }:
                        self._reset_state(state)
                    status = ProcessedMessageStatus.NOT_EXPENSE
                self._complete_successfully(message, status)
        except InterpretationError as exc:
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

    async def _checkpoint_classification(
        self,
        message_id: UUID,
        interpretation: ExpenseInterpretation,
        attempt_number: int,
    ) -> ExpenseInterpretation | None:
        checkpoint = ExpenseClassificationCheckpoint.from_interpretation(interpretation)
        async with self._session_factory() as session, session.begin():
            message = await session.scalar(self._locked_message_statement(message_id))
            if (
                message is None
                or message.status is not ProcessedMessageStatus.PROCESSING
                or message.processing_attempts != attempt_number
            ):
                return None
            if message.classified_at is not None:
                return ExpenseClassificationCheckpoint.from_payload(
                    message.classification_checkpoint
                ).to_interpretation()
            message.classification_intent = checkpoint.intent
            message.classification_checkpoint = checkpoint.payload()
            message.classified_at = self._now()
            return checkpoint.to_interpretation()

    async def _process_query(
        self,
        message_id: UUID,
        text: str,
        *,
        timestamp: datetime,
        timezone: str,
        attempt_number: int,
    ) -> None:
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        formatting_timezone = self._timezone(timezone)
        formatting_context = ExpenseQueryFormattingContext(
            reference_date=timestamp.astimezone(formatting_timezone).date(),
            timezone=formatting_timezone.key,
        )
        try:
            plan = await self._query_plan(
                message_id,
                text,
                timestamp=timestamp,
                timezone=timezone,
                attempt_number=attempt_number,
            )
        except asyncio.CancelledError:
            await asyncio.shield(self._retry_or_fail(message_id, "QUERY_INTERPRETATION_TIMEOUT"))
            raise
        except InterpretationError as exc:
            if self._is_transient(exc):
                await self._retry_or_fail(message_id, self._query_error_code(exc))
            else:
                await self._mark_query_failed(
                    message_id, self._query_error_code(exc), attempt_number=attempt_number
                )
            return
        except (ValidationError, TypeError, ValueError):
            await self._mark_query_failed(
                message_id, "QUERY_PLAN_INVALID_CHECKPOINT", attempt_number=attempt_number
            )
            return

        if plan.intent in {ExpenseQueryIntent.NOT_QUERY, ExpenseQueryIntent.QUERY_UNCLEAR}:
            page_count = await self._complete_query_guidance(
                message_id,
                QUERY_UNCLEAR_TEXT,
                status=ProcessedMessageStatus.PROCESSED,
            )
            if page_count is not None:
                self._emit_query_completed(
                    message_id,
                    attempt_number=attempt_number,
                    intent=plan.intent.value,
                    page_count=page_count,
                    outcome="success",
                )
            return
        if self._query_executor is None or self._query_formatter is None:
            await self._mark_query_failed(
                message_id, "QUERY_CONFIGURATION_ERROR", attempt_number=attempt_number
            )
            return

        try:
            result = await self._checkpointed_query_execution(
                message_id,
                plan,
                attempt_number=attempt_number,
            )
        except asyncio.CancelledError:
            await asyncio.shield(self._retry_or_fail(message_id, "QUERY_EXECUTION_INTERRUPTED"))
            raise
        except ExpenseQueryExecutionError:
            await self._mark_query_failed(
                message_id, "QUERY_PLAN_INVALID", attempt_number=attempt_number
            )
            return
        except DBAPIError:
            await self._retry_or_fail(message_id, "QUERY_DATABASE_UNAVAILABLE")
            return
        except (ConnectionError, TimeoutError):
            await self._retry_or_fail(message_id, "QUERY_DATABASE_UNAVAILABLE")
            return
        except (ValidationError, TypeError, ValueError):
            await self._mark_query_failed(
                message_id, "QUERY_RESULT_INVALID_CHECKPOINT", attempt_number=attempt_number
            )
            return

        try:
            with self._timing.span(
                "query_formatting_started",
                "query_formatting_completed",
                message_id,
                attempt_number=attempt_number,
                stage="query_formatting",
                intent=plan.intent.value,
            ) as formatting_span:
                formatted = self._query_formatter.format(result, context=formatting_context)
                formatting_span.result(outcome="success", page_count=len(formatted.messages))
        except (TypeError, ValueError):
            await self._mark_query_failed(
                message_id, "QUERY_FORMATTING_FAILED", attempt_number=attempt_number
            )
            return

        try:
            page_count = await self._complete_query_result(
                message_id,
                formatted.messages,
                attempt_number=attempt_number,
            )
        except DBAPIError:
            await self._retry_or_fail(message_id, "QUERY_DATABASE_UNAVAILABLE")
            return
        except (ConnectionError, TimeoutError):
            await self._retry_or_fail(message_id, "QUERY_DATABASE_UNAVAILABLE")
            return
        if page_count is not None:
            self._emit_query_completed(
                message_id,
                attempt_number=attempt_number,
                intent=plan.intent.value,
                page_count=page_count,
                outcome="success",
            )

    async def _query_plan(
        self,
        message_id: UUID,
        text: str,
        *,
        timestamp: datetime,
        timezone: str,
        attempt_number: int,
    ) -> ExpenseQueryPlan:
        async with self._session_factory() as session:
            message = await session.get(ProcessedMessage, message_id)
            if message is None or message.status is not ProcessedMessageStatus.PROCESSING:
                raise ValueError("query message is not processable")
            if message.query_plan_created_at is not None:
                return ExpenseQueryPlanCheckpoint.from_payload(message.query_plan_checkpoint).plan

        if self._query_interpreter_factory is None:
            raise ValueError("query interpreter is not configured")
        interpreter = self._query_interpreter_factory(timezone)
        async with self._timing.span(
            "query_interpretation_started",
            "query_interpretation_completed",
            message_id,
            attempt_number=attempt_number,
            stage="query_interpretation",
        ) as interpretation_span:
            plan = await interpreter.interpret(text, reference_timestamp=timestamp)
            interpretation_span.result(
                outcome="success",
                intent=plan.intent.value,
                metric=plan.metric.value if plan.metric else None,
                group=plan.group_by.value if plan.group_by else None,
            )
        return await self._checkpoint_query_plan(message_id, plan, attempt_number)

    async def _checkpoint_query_plan(
        self, message_id: UUID, plan: ExpenseQueryPlan, attempt_number: int
    ) -> ExpenseQueryPlan:
        checkpoint = ExpenseQueryPlanCheckpoint(plan=plan)
        async with (
            self._timing.span(
                "query_plan_checkpoint_started",
                "query_plan_checkpoint_completed",
                message_id,
                attempt_number=attempt_number,
                stage="query_plan_checkpoint",
                intent=plan.intent.value,
            ) as checkpoint_span,
            self._session_factory() as session,
            session.begin(),
        ):
            message = await session.scalar(self._locked_message_statement(message_id))
            if (
                message is None
                or message.status is not ProcessedMessageStatus.PROCESSING
                or message.processing_attempts != attempt_number
            ):
                checkpoint_span.result(outcome="terminal_failure", error_code="STALE_ATTEMPT")
                raise ValueError("query checkpoint attempt is stale")
            if message.query_plan_created_at is None:
                message.query_plan_checkpoint = checkpoint.payload()
                message.query_plan_created_at = self._now()
            stored = ExpenseQueryPlanCheckpoint.from_payload(message.query_plan_checkpoint).plan
            checkpoint_span.result(outcome="success")
            return stored

    async def _trusted_query_user_id(self, message_id: UUID) -> UUID:
        async with self._session_factory() as session:
            message = await session.get(ProcessedMessage, message_id)
            if message is None or message.user_id is None:
                raise ValueError("query user is unavailable")
            return message.user_id

    async def _checkpointed_query_execution(
        self,
        message_id: UUID,
        plan: ExpenseQueryPlan,
        *,
        attempt_number: int,
    ) -> ExpenseQueryResult:
        async with self._session_factory() as session:
            message = await session.get(ProcessedMessage, message_id)
            if message is None or message.status is not ProcessedMessageStatus.PROCESSING:
                raise ValueError("query message is not processable")
            if message.query_executed_at is not None:
                return ExpenseQueryResultCheckpoint.from_payload(
                    message.query_result_checkpoint
                ).result

        assert self._query_executor is not None
        async with self._timing.span(
            "query_execution_started",
            "query_execution_completed",
            message_id,
            attempt_number=attempt_number,
            stage="query_execution",
            intent=plan.intent.value,
            metric=plan.metric.value if plan.metric else None,
            group=plan.group_by.value if plan.group_by else None,
        ) as execution_span:
            result = await self._query_executor.execute(
                user_id=await self._trusted_query_user_id(message_id),
                plan=plan,
            )
            stored = await self._checkpoint_query_result(message_id, result, attempt_number)
            execution_span.result(outcome="success", **self._query_result_counts(stored))
            return stored

    async def _checkpoint_query_result(
        self,
        message_id: UUID,
        result: ExpenseQueryResult,
        attempt_number: int,
    ) -> ExpenseQueryResult:
        checkpoint = ExpenseQueryResultCheckpoint(result=result)
        async with self._session_factory() as session, session.begin():
            message = await session.scalar(self._locked_message_statement(message_id))
            if (
                message is None
                or message.status is not ProcessedMessageStatus.PROCESSING
                or message.processing_attempts != attempt_number
            ):
                raise ValueError("query execution checkpoint attempt is stale")
            if message.query_executed_at is None:
                message.query_result_checkpoint = checkpoint.payload()
                message.query_executed_at = self._now()
            return ExpenseQueryResultCheckpoint.from_payload(message.query_result_checkpoint).result

    async def _complete_query_result(
        self,
        message_id: UUID,
        pages: tuple[str, ...],
        *,
        attempt_number: int,
    ) -> int | None:
        if not pages:
            raise ValueError("query formatter returned no pages")
        async with self._session_factory() as session, session.begin():
            message = await session.scalar(
                self._locked_message_statement(message_id).options(
                    selectinload(ProcessedMessage.user)
                )
            )
            if (
                message is None
                or message.status is not ProcessedMessageStatus.PROCESSING
                or message.processing_attempts != attempt_number
                or message.user is None
                or message.user_id is None
            ):
                return None
            existing = list(
                await session.scalars(
                    select(OutboundMessage).where(
                        OutboundMessage.processed_message_id == message.id,
                        OutboundMessage.kind == OutboundMessageKind.QUERY_RESULT,
                    )
                )
            )
            if not existing:
                self._create_query_outbox_pages(
                    session,
                    message,
                    pages,
                    kind=OutboundMessageKind.QUERY_RESULT,
                    destination=message.user.phone_number,
                )
            message.query_result_checkpoint = None
            message.query_page_count = len(existing) if existing else len(pages)
            self._complete_successfully(message, ProcessedMessageStatus.PROCESSED)
            return message.query_page_count

    async def _complete_query_guidance(
        self,
        message_id: UUID,
        content: str,
        *,
        status: ProcessedMessageStatus,
        error_code: str | None = None,
    ) -> int | None:
        async with self._session_factory() as session, session.begin():
            message = await session.scalar(
                self._locked_message_statement(message_id).options(
                    selectinload(ProcessedMessage.user)
                )
            )
            if (
                message is None
                or message.status is not ProcessedMessageStatus.PROCESSING
                or message.user is None
                or message.user_id is None
            ):
                return None
            existing = await session.scalar(
                select(OutboundMessage.id).where(
                    OutboundMessage.processed_message_id == message.id,
                    OutboundMessage.kind == OutboundMessageKind.QUERY_GUIDANCE,
                )
            )
            if existing is None:
                self._create_query_outbox_pages(
                    session,
                    message,
                    (content,),
                    kind=OutboundMessageKind.QUERY_GUIDANCE,
                    destination=message.user.phone_number,
                )
            message.query_page_count = 1
            message.query_result_checkpoint = None
            self._complete_successfully(message, status)
            if error_code is not None:
                message.error_code = error_code
                message.last_error_code = error_code
            return 1

    async def _mark_query_failed(self, message_id: UUID, code: str, *, attempt_number: int) -> None:
        page_count = await self._complete_query_guidance(
            message_id,
            QUERY_INVALID_TEXT,
            status=ProcessedMessageStatus.FAILED,
            error_code=code,
        )
        if page_count is not None:
            self._emit_query_completed(
                message_id,
                attempt_number=attempt_number,
                intent=ExpenseIntent.QUERY.value,
                page_count=page_count,
                outcome="terminal_failure",
                error_code=code,
            )

    def _emit_query_completed(
        self,
        message_id: UUID,
        *,
        attempt_number: int,
        intent: str,
        page_count: int,
        outcome: str,
        error_code: str | None = None,
    ) -> None:
        self._timing.event(
            "query_outbox_created",
            message_id,
            attempt_number=attempt_number,
            outcome=outcome,
            page_count=page_count,
        )
        self._timing.event(
            "query_processing_completed",
            message_id,
            attempt_number=attempt_number,
            outcome=outcome,
            intent=intent,
            page_count=page_count,
            error_code=error_code,
        )

    @staticmethod
    def _create_query_outbox_pages(
        session: AsyncSession,
        message: ProcessedMessage,
        pages: tuple[str, ...],
        *,
        kind: OutboundMessageKind,
        destination: str,
    ) -> None:
        page_count = len(pages)
        now = datetime.now(UTC)
        for position, content in enumerate(pages, start=1):
            session.add(
                OutboundMessage(
                    user_id=message.user_id,
                    processed_message_id=message.id,
                    expense_id=None,
                    destination=destination,
                    content=content,
                    kind=kind,
                    dedup_key=(f"processed-message:{message.id}:{kind.value}:page:{position}"),
                    sequence_no=position,
                    sequence_count=page_count,
                    status=OutboundMessageStatus.PENDING,
                    available_at=now,
                )
            )

    @staticmethod
    def _query_result_counts(result) -> dict[str, int]:
        if isinstance(result, ExpenseListResult):
            return {"row_count": len(result.items)}
        if isinstance(result, ExpenseGroupResult):
            return {"group_count": len(result.items)}
        if isinstance(result, ExpenseAggregateResult):
            return {"row_count": result.record_count}
        if isinstance(result, ExpenseComparisonResult):
            return {"row_count": result.record_count + result.comparison_record_count}
        return {}

    @staticmethod
    def _query_error_code(exc: InterpretationError) -> str:
        return f"QUERY_{exc.code.value.removeprefix('GEMINI_')}"

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
    def _literal_description(text: str) -> str | None:
        match = _LITERAL_EXPENSE_DESCRIPTION.search(text)
        if match is None:
            return None
        description = match.group("description").strip()
        return description or None

    @staticmethod
    def _missing_required_fields(
        result: ExpenseInterpretation, *, force_amount: bool = False
    ) -> list[str]:
        if result.intent in {ExpenseIntent.NOT_EXPENSE, ExpenseIntent.QUERY}:
            return []
        missing: list[str] = []
        if force_amount or result.amount is None:
            missing.append("amount")
        if result.description is None or not result.description.strip():
            missing.append("description")
        return missing

    @staticmethod
    def _create_incomplete_expense_outbox(
        session: AsyncSession,
        message: ProcessedMessage,
        missing_fields: list[str],
    ) -> None:
        labels = {"amount": "valor", "description": "descrição"}
        rendered = " e ".join(labels[field] for field in missing_fields)
        ExpenseProcessingService._create_outbox(
            session,
            message,
            content=INCOMPLETE_EXPENSE_TEMPLATE.format(missing_fields=rendered),
            kind=OutboundMessageKind.INCOMPLETE_EXPENSE,
            expense=None,
        )

    @staticmethod
    def _is_transient(exc: InterpretationError) -> bool:
        return exc.transient

    @staticmethod
    def _validate_interpretation(result: ExpenseInterpretation) -> None:
        ExpenseProcessingService._validate_partial_interpretation(result)
        if result.intent is not ExpenseIntent.CREATE_EXPENSE:
            return
        if not isinstance(result.amount, Decimal) or not result.amount.is_finite():
            raise InterpretationError(InterpretationErrorCode.INVALID_RESPONSE, transient=False)
        if result.description is None or not result.description.strip():
            raise InterpretationError(InterpretationErrorCode.INVALID_RESPONSE, transient=False)

    @staticmethod
    def _validate_partial_interpretation(result: ExpenseInterpretation) -> None:
        if result.intent is not ExpenseIntent.CREATE_EXPENSE and result.amount is not None:
            raise InterpretationError(InterpretationErrorCode.INVALID_RESPONSE, transient=False)
        if result.amount is not None:
            if not isinstance(result.amount, Decimal) or not result.amount.is_finite():
                raise InterpretationError(InterpretationErrorCode.INVALID_RESPONSE, transient=False)
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
            return result.model_copy(
                update={
                    "intent": ExpenseIntent.NOT_EXPENSE,
                    "amount": None,
                    "amount_evidence": None,
                    "description": None,
                }
            )
        if not checkpoint.is_legible:
            return result.model_copy(
                update={
                    "intent": ExpenseIntent.UNCLEAR,
                    "amount": None,
                    "amount_evidence": None,
                    "description": None,
                }
            )

        dates = checkpoint.distinct_dates()
        expense_date = date.fromisoformat(next(iter(dates))) if len(dates) == 1 else None
        amounts = checkpoint.distinct_amounts()
        if len(amounts) != 1:
            return result.model_copy(
                update={
                    "intent": ExpenseIntent.UNCLEAR,
                    "amount": None,
                    "amount_evidence": None,
                    "expense_date": expense_date,
                }
            )
        if result.intent is not ExpenseIntent.CREATE_EXPENSE:
            return result.model_copy(update={"expense_date": expense_date})

        visual_amount = next(iter(amounts))
        valid_evidence = {
            candidate.evidence
            for candidate in checkpoint.amount_candidates
            if Decimal(candidate.value) == visual_amount
        }
        if result.amount == visual_amount and result.amount_evidence in valid_evidence:
            return result.model_copy(update={"expense_date": expense_date})
        return result.model_copy(
            update={
                "intent": ExpenseIntent.UNCLEAR,
                "amount": None,
                "amount_evidence": None,
                "expense_date": expense_date,
            }
        )

    async def _create_expense(
        self,
        session: AsyncSession,
        message: ProcessedMessage,
        result: ExpenseInterpretation,
    ) -> None:
        assert result.amount is not None and result.description is not None
        category_value = (result.category or ExpenseCategory.OTHER).value
        category = await session.scalar(
            select(Category).where(Category.name == category_value, Category.is_active.is_(True))
        )
        if category is None:
            raise InterpretationError(InterpretationErrorCode.INVALID_RESPONSE, transient=False)
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
            source_type=message.source_type,
        )
        session.add(expense)
        await session.flush()
        state = await self._existing_locked_state(session, message.user_id)
        if state is not None:
            self._reset_state(state)
        self._create_outbox(
            session,
            message,
            content=format_expense_confirmation(expense, category_value),
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
            message = await session.scalar(
                self._locked_message_statement(message_id).options(
                    selectinload(ProcessedMessage.user)
                )
            )
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
                if (
                    message.classification_intent is ExpenseIntent.QUERY
                    and message.user is not None
                ):
                    self._create_query_outbox_pages(
                        session,
                        message,
                        (QUERY_FAILURE_TEXT,),
                        kind=OutboundMessageKind.QUERY_GUIDANCE,
                        destination=message.user.phone_number,
                    )
                    message.query_page_count = 1
                    message.query_result_checkpoint = None
                else:
                    await self._create_failure_notification(session, message, message.user)
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
                pending = state.status is ConversationStatus.WAITING_EXPENSE_DELETE_CONFIRM
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

    @staticmethod
    async def _existing_locked_state(
        session: AsyncSession, user_id: UUID
    ) -> ConversationState | None:
        return await session.scalar(
            select(ConversationState)
            .where(ConversationState.user_id == user_id)
            .with_for_update(of=ConversationState)
        )

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
        if state.status is not ConversationStatus.WAITING_EXPENSE_DELETE_CONFIRM:
            return False
        if state.expires_at is None:
            return True
        return self._deadline_expired(state.expires_at)

    def _deadline_expired(self, expires_at: datetime) -> bool:
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
    def _error_code(exc: InterpretationError) -> str:
        return exc.code.value

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
        if error_code.startswith("QUERY_"):
            return "query_execution"
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
