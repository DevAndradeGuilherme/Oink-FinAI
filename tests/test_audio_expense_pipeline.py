from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from oink_finai.database.base import Base
from oink_finai.database.models import Category, Expense, OutboundMessage, ProcessedMessage, User
from oink_finai.domain.enums import (
    ExpenseCategory,
    ExpenseIntent,
    MessageSourceType,
    PaymentMethod,
    ProcessedMessageStatus,
)
from oink_finai.providers.whatsapp import WhatsAppProvider
from oink_finai.providers.whatsapp.media_errors import MediaError, MediaErrorCode
from oink_finai.schemas.audio import AudioTranscription
from oink_finai.schemas.expense_interpretation import ExpenseInterpretation
from oink_finai.schemas.whatsapp import InboundMedia, InboundWhatsAppMessage
from oink_finai.services.audio_transcriber import AudioTranscriber, ValidatedAudio
from oink_finai.services.expense_interpreter import ExpenseInterpreter
from oink_finai.services.expense_processing import (
    AUDIO_INVALID_TEXT,
    AUDIO_NO_SPEECH_TEXT,
    ExpenseProcessingService,
)
from oink_finai.services.gemini_errors import GeminiRateLimitError
from oink_finai.services.transcription_errors import (
    NoSpeechError,
    TranscriptionError,
    TranscriptionErrorCode,
)


@pytest_asyncio.fixture
async def audio_factory(tmp_path: Path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'audio.db').as_posix()}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


class FakeProvider(WhatsAppProvider):
    def __init__(self, results: list[bytes | Exception]) -> None:
        self.results = results
        self.download_calls = 0
        self.send_text_calls = 0
        self.send_interactive_calls = 0

    async def parse_webhook(self, payload: dict[str, object]) -> InboundWhatsAppMessage | None:
        raise AssertionError("webhook parsing is outside this worker fake")

    async def send_text(self, phone_number: str, text: str) -> str | None:
        self.send_text_calls += 1
        return None

    async def send_interactive(self, phone_number: str, message: object) -> str | None:
        self.send_interactive_calls += 1
        return None

    async def download_media(self, media: InboundMedia) -> bytes:
        result = self.results[min(self.download_calls, len(self.results) - 1)]
        self.download_calls += 1
        if isinstance(result, Exception):
            raise result
        return result


class FakeTranscriber(AudioTranscriber):
    def __init__(self, results: list[AudioTranscription | Exception]) -> None:
        self.results = results
        self.calls = 0

    async def transcribe(self, audio: ValidatedAudio) -> AudioTranscription:
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        if isinstance(result, Exception):
            raise result
        return result


class FakeAudioInterpreter(ExpenseInterpreter):
    def __init__(self, results: list[ExpenseInterpretation | Exception]) -> None:
        self.results = results
        self.calls = 0
        self.messages: list[str] = []

    async def interpret(
        self, message: str, *, reference_timestamp: datetime
    ) -> ExpenseInterpretation:
        self.messages.append(message)
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        if isinstance(result, Exception):
            raise result
        return result


def expense_result() -> ExpenseInterpretation:
    return ExpenseInterpretation(
        intent=ExpenseIntent.CREATE_EXPENSE,
        amount=Decimal("42.50"),
        amount_evidence="quarenta e dois e cinquenta",
        description="Mercado",
        merchant="Mercado",
        category=ExpenseCategory.FOOD,
        payment_method=PaymentMethod.PIX,
        expense_date=None,
        confidence=0.99,
        missing_fields=[],
        reasoning_summary="valid",
    )


async def seed_audio(
    factory: async_sessionmaker[AsyncSession],
    *,
    transcript: str = "",
    transcribed_at: datetime | None = None,
    processing_attempts: int = 0,
) -> ProcessedMessage:
    async with factory() as session, session.begin():
        user = User(phone_number="5511999999999", timezone="America/Sao_Paulo")
        session.add_all([user, Category(name=ExpenseCategory.FOOD.value, slug="alimentacao")])
        await session.flush()
        message = ProcessedMessage(
            provider="evolution",
            instance_id="finance-instance",
            external_message_id="opaque-message-id",
            user_id=user.id,
            accepted_text=transcript,
            source_type=MessageSourceType.AUDIO,
            media_remote_jid=(None if transcribed_at else "opaque-retrieval-jid"),
            media_mime_type="audio/ogg",
            media_duration_seconds=12,
            media_is_voice_note=True,
            transcribed_at=transcribed_at,
            message_timestamp=datetime(2026, 9, 2, 12, tzinfo=UTC),
            status=ProcessedMessageStatus.PENDING,
            available_at=datetime(2026, 9, 2, 12, tzinfo=UTC),
            processing_attempts=processing_attempts,
        )
        session.add(message)
        await session.flush()
        return message


def processor(
    factory: async_sessionmaker[AsyncSession],
    provider: FakeProvider,
    transcriber: FakeTranscriber,
    interpreter: FakeAudioInterpreter,
    *,
    max_attempts: int = 3,
) -> ExpenseProcessingService:
    return ExpenseProcessingService(
        factory,
        lambda _timezone: interpreter,
        media_provider=provider,
        audio_transcriber_factory=lambda: transcriber,
        max_attempts=max_attempts,
        retry_base_seconds=0,
        retry_max_seconds=0,
        jitter=lambda: 0,
        clock=lambda: datetime(2026, 9, 2, 12, 1, tzinfo=UTC),
    )


async def test_audio_success_checkpoints_transcript_and_creates_text_confirmation(
    audio_factory: async_sessionmaker[AsyncSession],
) -> None:
    message = await seed_audio(audio_factory)
    provider = FakeProvider([b"OggSvalid"])
    transcriber = FakeTranscriber(
        [AudioTranscription(transcript="Mercado quarenta e dois e cinquenta", has_speech=True)]
    )
    interpreter = FakeAudioInterpreter([expense_result()])
    service = processor(audio_factory, provider, transcriber, interpreter)

    assert await service.claim(1) == [message.id]
    await service.process(message.id)

    async with audio_factory() as session:
        saved = await session.get(ProcessedMessage, message.id)
        expense = await session.scalar(select(Expense))
        outbox = await session.scalar(select(OutboundMessage))
        assert saved is not None and saved.status is ProcessedMessageStatus.PROCESSED
        assert saved.accepted_text == "Mercado quarenta e dois e cinquenta"
        assert saved.transcribed_at is not None and saved.media_remote_jid is None
        assert expense is not None and expense.source_type == MessageSourceType.AUDIO
        assert outbox is not None and outbox.content_type == "TEXT" and outbox.actions is None
        assert outbox.content.startswith("✅ Novo Gasto Registrado!")
    assert (provider.download_calls, transcriber.calls, interpreter.calls) == (1, 1, 1)
    assert provider.send_text_calls == 0 and provider.send_interactive_calls == 0


async def test_retry_after_transcript_checkpoint_skips_download_and_transcription(
    audio_factory: async_sessionmaker[AsyncSession],
) -> None:
    message = await seed_audio(
        audio_factory,
        transcript="Mercado quarenta e dois e cinquenta",
        transcribed_at=datetime(2026, 9, 2, 12, tzinfo=UTC),
    )
    provider = FakeProvider([AssertionError("download must not run")])
    transcriber = FakeTranscriber([AssertionError("transcription must not run")])
    interpreter = FakeAudioInterpreter([GeminiRateLimitError("rate limited"), expense_result()])
    service = processor(audio_factory, provider, transcriber, interpreter)

    await service.claim(1)
    await service.process(message.id)
    await service.claim(1)
    await service.process(message.id)

    assert provider.download_calls == 0 and transcriber.calls == 0 and interpreter.calls == 2
    async with audio_factory() as session:
        assert await session.scalar(select(func.count()).select_from(Expense)) == 1
        assert await session.scalar(select(func.count()).select_from(OutboundMessage)) == 1


@pytest.mark.parametrize(
    ("failure", "expected_code", "expected_content"),
    [
        (
            MediaError(MediaErrorCode.INVALID_BASE64, transient=False),
            MediaErrorCode.INVALID_BASE64.value,
            AUDIO_INVALID_TEXT,
        ),
        (NoSpeechError(), TranscriptionErrorCode.NO_SPEECH.value, AUDIO_NO_SPEECH_TEXT),
    ],
)
async def test_terminal_audio_failure_clears_reference_and_notifies_once(
    audio_factory: async_sessionmaker[AsyncSession],
    failure: Exception,
    expected_code: str,
    expected_content: str,
) -> None:
    message = await seed_audio(audio_factory)
    provider = FakeProvider([failure] if isinstance(failure, MediaError) else [b"OggSvalid"])
    transcriber = FakeTranscriber([failure])
    interpreter = FakeAudioInterpreter([expense_result()])
    service = processor(audio_factory, provider, transcriber, interpreter)
    await service.claim(1)
    await service.process(message.id)
    await service.process(message.id)

    async with audio_factory() as session:
        saved = await session.get(ProcessedMessage, message.id)
        outbox = list(await session.scalars(select(OutboundMessage)))
        assert saved is not None and saved.status is ProcessedMessageStatus.FAILED
        assert saved.error_code == expected_code and saved.media_remote_jid is None
        assert len(outbox) == 1 and outbox[0].content == expected_content
        assert await session.scalar(select(func.count()).select_from(Expense)) == 0
    assert interpreter.calls == 0


@pytest.mark.parametrize(
    "failure",
    [
        MediaError(MediaErrorCode.TIMEOUT, transient=True),
        TranscriptionError(TranscriptionErrorCode.UNAVAILABLE, transient=True),
    ],
)
async def test_transient_audio_failure_keeps_reference_for_durable_retry(
    audio_factory: async_sessionmaker[AsyncSession], failure: Exception
) -> None:
    message = await seed_audio(audio_factory)
    provider = FakeProvider([failure] if isinstance(failure, MediaError) else [b"OggSvalid"])
    transcriber = FakeTranscriber([failure])
    interpreter = FakeAudioInterpreter([expense_result()])
    service = processor(audio_factory, provider, transcriber, interpreter)
    await service.claim(1)
    await service.process(message.id)

    async with audio_factory() as session:
        saved = await session.get(ProcessedMessage, message.id)
        assert saved is not None and saved.status is ProcessedMessageStatus.PENDING
        assert saved.media_remote_jid == "opaque-retrieval-jid"
        assert saved.next_attempt_at is not None
        assert await session.scalar(select(func.count()).select_from(Expense)) == 0
        assert await session.scalar(select(func.count()).select_from(OutboundMessage)) == 0


async def test_audio_attempt_exhaustion_clears_reference_and_notifies_once(
    audio_factory: async_sessionmaker[AsyncSession],
) -> None:
    message = await seed_audio(audio_factory, processing_attempts=1)
    provider = FakeProvider([MediaError(MediaErrorCode.TIMEOUT, transient=True)])
    transcriber = FakeTranscriber([AssertionError("not reached")])
    interpreter = FakeAudioInterpreter([expense_result()])
    service = processor(audio_factory, provider, transcriber, interpreter, max_attempts=2)
    await service.claim(1)
    await service.process(message.id)

    async with audio_factory() as session:
        saved = await session.get(ProcessedMessage, message.id)
        assert saved is not None and saved.status is ProcessedMessageStatus.FAILED
        assert saved.media_remote_jid is None and saved.next_attempt_at is None
        assert await session.scalar(select(func.count()).select_from(OutboundMessage)) == 1
