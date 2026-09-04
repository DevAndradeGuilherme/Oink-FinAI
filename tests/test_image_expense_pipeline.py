import io
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
import pytest_asyncio
from PIL import Image
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
from oink_finai.schemas.expense_interpretation import ExpenseInterpretation
from oink_finai.schemas.image_analysis import (
    AmountCandidate,
    DateCandidate,
    EvidenceCandidate,
    ImageAnalysis,
    ImageAnalysisWarning,
    ImageDocumentType,
)
from oink_finai.schemas.image_checkpoint import ImageAnalysisCheckpoint
from oink_finai.schemas.whatsapp import InboundMedia, InboundWhatsAppMessage
from oink_finai.services.expense_interpreter import ExpenseInterpreter
from oink_finai.services.expense_processing import ExpenseProcessingService
from oink_finai.services.gemini_errors import GeminiRateLimitError
from oink_finai.services.image_analysis_errors import ImageAnalysisError, ImageAnalysisErrorCode
from oink_finai.services.image_analyzer import ImageAnalyzer, ValidatedImage


@pytest_asyncio.fixture
async def image_factory(tmp_path: Path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'image.db').as_posix()}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


def png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (3, 2), "white").save(output, format="PNG")
    return output.getvalue()


class FakeProvider(WhatsAppProvider):
    def __init__(self, results: list[bytes | Exception]) -> None:
        self.results = results
        self.download_calls = 0

    async def parse_webhook(self, payload: dict[str, object]) -> InboundWhatsAppMessage | None:
        raise AssertionError("not used")

    async def send_text(self, phone_number: str, text: str) -> str | None:
        raise AssertionError("outbox delivery is separate")

    async def send_interactive(self, phone_number: str, message: object) -> str | None:
        raise AssertionError("interactive send is forbidden")

    async def download_media(self, media: InboundMedia) -> bytes:
        result = self.results[min(self.download_calls, len(self.results) - 1)]
        self.download_calls += 1
        if isinstance(result, Exception):
            raise result
        return result


class FakeAnalyzer(ImageAnalyzer):
    def __init__(self, results: list[ImageAnalysis | Exception]) -> None:
        self.results = results
        self.calls = 0
        self.captions: list[str | None] = []

    async def analyze(self, image: ValidatedImage, caption: str | None = None) -> ImageAnalysis:
        assert image.mime_type == "image/png" and image.detected_format == "PNG"
        assert (image.width, image.height) == (3, 2)
        self.captions.append(caption)
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        if isinstance(result, Exception):
            raise result
        return result


class FakeInterpreter(ExpenseInterpreter):
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


def analysis(*, amounts: tuple[str, ...] = ("42.50",), caption: str | None = None) -> ImageAnalysis:
    return ImageAnalysis(
        document_type=ImageDocumentType.RECEIPT,
        visible_text="Mercado\nTotal R$ 42,50\n03/09/2026",
        amount_candidates=[
            AmountCandidate(
                value=Decimal(value),
                evidence=f"R$ {value.replace('.', ',')}",
                label="TOTAL",
            )
            for value in amounts
        ],
        date_candidates=[DateCandidate(value="2026-09-03", evidence="03/09/2026", label="DATA")],
        merchant_candidates=[EvidenceCandidate(value="Mercado", evidence="Mercado")],
        payment_method_candidates=[EvidenceCandidate(value="Pix", evidence="Pix")],
        caption=caption,
        is_financial_document=True,
        is_legible=True,
        confidence=0.95,
        warnings=(
            [ImageAnalysisWarning.MULTIPLE_AMOUNTS]
            if len(amounts) > 1
            else [ImageAnalysisWarning.NONE]
        ),
    )


def expense_result(amount: str = "42.50", evidence: str = "R$ 42,50") -> ExpenseInterpretation:
    return ExpenseInterpretation(
        intent=ExpenseIntent.CREATE_EXPENSE,
        amount=Decimal(amount),
        amount_evidence=evidence,
        description="Mercado",
        merchant="Mercado",
        category=ExpenseCategory.FOOD,
        payment_method=PaymentMethod.PIX,
        expense_date=datetime(2026, 9, 3, tzinfo=UTC).date(),
        confidence=0.95,
        missing_fields=[],
        reasoning_summary="valid",
    )


async def seed_image(
    factory: async_sessionmaker[AsyncSession],
    *,
    checkpoint: dict[str, object] | None = None,
    processing_attempts: int = 0,
    caption: str | None = " contexto ",
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
            accepted_text="",
            source_type=MessageSourceType.IMAGE,
            media_remote_jid=None if checkpoint else "opaque-retrieval-jid",
            media_mime_type="image/png",
            media_caption=caption.strip() if caption else None,
            image_analysis=checkpoint,
            image_analyzed_at=(datetime(2026, 9, 3, 12, tzinfo=UTC) if checkpoint else None),
            message_timestamp=datetime(2026, 9, 3, 12, tzinfo=UTC),
            status=ProcessedMessageStatus.PENDING,
            available_at=datetime(2026, 9, 3, 12, tzinfo=UTC),
            processing_attempts=processing_attempts,
        )
        session.add(message)
        await session.flush()
        return message


def processor(
    factory: async_sessionmaker[AsyncSession],
    provider: FakeProvider,
    analyzer: FakeAnalyzer,
    interpreter: FakeInterpreter,
    *,
    max_attempts: int = 3,
) -> ExpenseProcessingService:
    return ExpenseProcessingService(
        factory,
        lambda _timezone: interpreter,
        media_provider=provider,
        image_analyzer_factory=lambda: analyzer,
        max_attempts=max_attempts,
        retry_base_seconds=0,
        retry_max_seconds=0,
        jitter=lambda: 0,
        clock=lambda: datetime(2026, 9, 3, 12, 1, tzinfo=UTC),
    )


@pytest.mark.parametrize("caption", ["contexto", None])
async def test_image_success_checkpoints_before_interpretation_and_creates_text_outbox(
    image_factory: async_sessionmaker[AsyncSession],
    caption: str | None,
) -> None:
    message = await seed_image(image_factory, caption=caption)
    provider = FakeProvider([png_bytes()])
    analyzer = FakeAnalyzer([analysis(caption=caption)])
    interpreter = FakeInterpreter([expense_result()])
    service = processor(image_factory, provider, analyzer, interpreter)

    await service.claim(1)
    await service.process(message.id)

    async with image_factory() as session:
        saved = await session.get(ProcessedMessage, message.id)
        expense = await session.scalar(select(Expense))
        outbox = await session.scalar(select(OutboundMessage))
        assert saved is not None and saved.status is ProcessedMessageStatus.PROCESSED
        assert saved.image_analyzed_at is not None and saved.media_remote_jid is None
        assert saved.image_analysis is not None and "visible_text" not in saved.image_analysis
        assert "caption" not in saved.image_analysis
        assert expense is not None and expense.source_type == MessageSourceType.IMAGE
        assert outbox is not None and outbox.content_type == "TEXT" and outbox.actions is None
    assert (provider.download_calls, analyzer.calls, interpreter.calls) == (1, 1, 1)
    assert analyzer.captions == [caption]
    assert "user_caption_context" in interpreter.messages[0]


async def test_retry_after_checkpoint_never_downloads_or_analyzes_again(
    image_factory: async_sessionmaker[AsyncSession],
) -> None:
    checkpoint = ImageAnalysisCheckpoint.from_analysis(analysis()).payload()
    message = await seed_image(image_factory, checkpoint=checkpoint)
    provider = FakeProvider([AssertionError("download forbidden")])
    analyzer = FakeAnalyzer([AssertionError("analysis forbidden")])
    interpreter = FakeInterpreter([GeminiRateLimitError("limited"), expense_result()])
    service = processor(image_factory, provider, analyzer, interpreter)

    await service.claim(1)
    await service.process(message.id)
    await service.claim(1)
    await service.process(message.id)

    assert provider.download_calls == 0 and analyzer.calls == 0 and interpreter.calls == 2
    async with image_factory() as session:
        assert await session.scalar(select(func.count()).select_from(Expense)) == 1
        assert await session.scalar(select(func.count()).select_from(OutboundMessage)) == 1


@pytest.mark.parametrize(
    "failure",
    [
        MediaError(MediaErrorCode.TIMEOUT, transient=True),
        ImageAnalysisError(ImageAnalysisErrorCode.UNAVAILABLE, transient=True),
    ],
)
async def test_transient_precheckpoint_failure_keeps_reference_and_retries_stage(
    image_factory: async_sessionmaker[AsyncSession], failure: Exception
) -> None:
    message = await seed_image(image_factory)
    provider_results = [failure, png_bytes()] if isinstance(failure, MediaError) else [png_bytes()]
    analyzer_results = (
        [failure, analysis()] if isinstance(failure, ImageAnalysisError) else [analysis()]
    )
    provider = FakeProvider(provider_results)
    analyzer = FakeAnalyzer(analyzer_results)
    interpreter = FakeInterpreter([expense_result()])
    service = processor(image_factory, provider, analyzer, interpreter)

    await service.claim(1)
    await service.process(message.id)
    async with image_factory() as session:
        saved = await session.get(ProcessedMessage, message.id)
        assert saved is not None and saved.status is ProcessedMessageStatus.PENDING
        assert saved.media_remote_jid is not None and saved.image_analysis is None
    await service.claim(1)
    await service.process(message.id)

    assert interpreter.calls == 1
    if isinstance(failure, MediaError):
        assert provider.download_calls == 2 and analyzer.calls == 1
    else:
        assert provider.download_calls == 2 and analyzer.calls == 2


async def test_invalid_checkpoint_is_terminal_without_download_or_analysis(
    image_factory: async_sessionmaker[AsyncSession],
) -> None:
    message = await seed_image(image_factory, checkpoint={"version": 99})
    provider = FakeProvider([AssertionError("download forbidden")])
    analyzer = FakeAnalyzer([AssertionError("analysis forbidden")])
    interpreter = FakeInterpreter([expense_result()])
    service = processor(image_factory, provider, analyzer, interpreter)

    await service.claim(1)
    await service.process(message.id)

    async with image_factory() as session:
        saved = await session.get(ProcessedMessage, message.id)
        assert saved is not None and saved.status is ProcessedMessageStatus.FAILED
        assert saved.error_code == "IMAGE_ANALYSIS_INVALID_CHECKPOINT"
        assert await session.scalar(select(func.count()).select_from(OutboundMessage)) == 1
    assert provider.download_calls == analyzer.calls == interpreter.calls == 0


async def test_multiple_visual_amounts_force_clarification_without_expense(
    image_factory: async_sessionmaker[AsyncSession],
) -> None:
    message = await seed_image(image_factory)
    provider = FakeProvider([png_bytes()])
    analyzer = FakeAnalyzer([analysis(amounts=("42.50", "100.00"))])
    interpreter = FakeInterpreter([expense_result()])
    service = processor(image_factory, provider, analyzer, interpreter)

    await service.claim(1)
    await service.process(message.id)

    async with image_factory() as session:
        saved = await session.get(ProcessedMessage, message.id)
        assert saved is not None and saved.status is ProcessedMessageStatus.NEEDS_CLARIFICATION
        assert await session.scalar(select(func.count()).select_from(Expense)) == 0
        assert await session.scalar(select(func.count()).select_from(OutboundMessage)) == 1


async def test_multiple_visual_dates_force_clarification_without_expense(
    image_factory: async_sessionmaker[AsyncSession],
) -> None:
    message = await seed_image(image_factory)
    ambiguous = analysis().model_copy(
        update={
            "date_candidates": [
                DateCandidate(value="2026-09-03", evidence="03/09/2026", label="DATA"),
                DateCandidate(value="2026-09-04", evidence="04/09/2026", label="DATA"),
            ],
            "warnings": [ImageAnalysisWarning.MULTIPLE_DATES],
        }
    )
    service = processor(
        image_factory,
        FakeProvider([png_bytes()]),
        FakeAnalyzer([ambiguous]),
        FakeInterpreter([expense_result()]),
    )

    await service.claim(1)
    await service.process(message.id)

    async with image_factory() as session:
        saved = await session.get(ProcessedMessage, message.id)
        assert saved is not None and saved.status is ProcessedMessageStatus.NEEDS_CLARIFICATION
        assert await session.scalar(select(func.count()).select_from(Expense)) == 0


@pytest.mark.parametrize(
    "failure",
    [
        MediaError(MediaErrorCode.CONTENT_MISMATCH, transient=False),
        ImageAnalysisError(ImageAnalysisErrorCode.GROUNDING, transient=False),
    ],
)
async def test_terminal_image_failure_clears_reference_and_notifies_once(
    image_factory: async_sessionmaker[AsyncSession], failure: Exception
) -> None:
    message = await seed_image(image_factory)
    provider = FakeProvider([failure] if isinstance(failure, MediaError) else [png_bytes()])
    analyzer = FakeAnalyzer([failure] if isinstance(failure, ImageAnalysisError) else [analysis()])
    interpreter = FakeInterpreter([expense_result()])
    service = processor(image_factory, provider, analyzer, interpreter)

    await service.claim(1)
    await service.process(message.id)
    await service.process(message.id)

    async with image_factory() as session:
        saved = await session.get(ProcessedMessage, message.id)
        assert saved is not None and saved.status is ProcessedMessageStatus.FAILED
        assert saved.media_remote_jid is None
        assert await session.scalar(select(func.count()).select_from(OutboundMessage)) == 1
        assert await session.scalar(select(func.count()).select_from(Expense)) == 0
