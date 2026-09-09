import asyncio
import logging
import signal
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime, timedelta

from oink_finai.config.settings import get_settings
from oink_finai.database.session import SessionFactory, engine
from oink_finai.providers.whatsapp import EvolutionWhatsAppProvider
from oink_finai.repositories import SQLAlchemyExpenseQueryExecutor
from oink_finai.services.audio_transcriber_factory import create_audio_transcriber
from oink_finai.services.expense_processing import ExpenseProcessingService
from oink_finai.services.openai_client import create_openai_client
from oink_finai.services.openai_expense_interpreter import OpenAIExpenseInterpreter
from oink_finai.services.openai_expense_query_interpreter import (
    OpenAIExpenseQueryInterpreter,
)
from oink_finai.services.openai_image_analyzer import OpenAIImageAnalyzer
from oink_finai.services.openai_privacy import openai_private_operation
from oink_finai.services.outbox_delivery import OutboxDeliveryService
from oink_finai.services.pipeline_timing import PipelineTiming
from oink_finai.services.usage_control import UsageControl
from oink_finai.services.whatsapp_expense_query_result_formatter import (
    WhatsAppExpenseQueryResultFormatter,
)

logger = logging.getLogger(__name__)


async def _run_claim_batch[ClaimT](
    stop: asyncio.Event,
    batch_size: int,
    claim: Callable[[int], Awaitable[Sequence[ClaimT]]],
    handle: Callable[[ClaimT], Awaitable[None]],
) -> None:
    """Claim one item at a time so shutdown never strands unstarted work."""
    for _ in range(batch_size):
        if stop.is_set():
            return
        claimed = await claim(1)
        if not claimed:
            return
        # A signal received while claim() is in flight does not abandon its result.
        await handle(claimed[0])


async def _close_safely(name: str, close: Callable[[], Awaitable[None]]) -> None:
    try:
        await close()
    except Exception as exc:
        logger.warning(
            "Worker resource close failed",
            extra={"resource": name, "error_type": type(exc).__name__},
        )


async def run_worker() -> None:
    settings = get_settings()
    openai_api_key = settings.openai_api_key_value
    evolution_api_key = settings.evolution_api_key_value
    if not all(
        (
            openai_api_key,
            settings.openai_expense_model,
            settings.openai_image_model,
            settings.evolution_base_url,
            evolution_api_key,
            settings.evolution_instance,
        )
    ):
        raise RuntimeError("Worker provider configuration is incomplete")

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signal_name in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signal_name, stop.set)
        except NotImplementedError:
            signal.signal(signal_name, lambda *_: loop.call_soon_threadsafe(stop.set))

    openai_client = create_openai_client(
        api_key=openai_api_key,
        timeout_seconds=max(
            settings.openai_expense_timeout_seconds,
            settings.openai_query_timeout_seconds,
            settings.openai_image_timeout_seconds,
            settings.openai_audio_transcription_timeout_seconds,
        ),
    )
    provider = EvolutionWhatsAppProvider(
        settings.evolution_base_url,
        evolution_api_key,
        settings.evolution_instance,
        timeout_seconds=settings.evolution_timeout_seconds,
        media_timeout_seconds=settings.evolution_media_timeout_seconds,
        media_max_bytes=settings.media_max_bytes,
        media_max_duration_seconds=settings.media_max_duration_seconds,
        image_max_width=settings.image_max_width,
        image_max_height=settings.image_max_height,
        image_max_pixels=settings.image_max_pixels,
        max_retries=0,
    )
    audio_transcriber = create_audio_transcriber(settings, client=openai_client)
    image_analyzer = OpenAIImageAnalyzer(
        api_key=openai_api_key,
        model=settings.openai_image_model,
        timeout_seconds=settings.openai_image_timeout_seconds,
        max_output_tokens=settings.openai_image_max_output_tokens,
        max_image_bytes=settings.media_max_bytes,
        client=openai_client,
    )
    timing = PipelineTiming(settings.pipeline_timing_enabled)
    usage_control = UsageControl(settings, SessionFactory)
    processing = ExpenseProcessingService(
        SessionFactory,
        lambda timezone: OpenAIExpenseInterpreter(
            api_key=openai_api_key,
            model=settings.openai_expense_model,
            timeout_seconds=settings.openai_expense_timeout_seconds,
            max_output_tokens=settings.openai_expense_max_output_tokens,
            timezone=timezone,
            client=openai_client,
        ),
        max_attempts=settings.expense_processing_max_attempts,
        retry_base_seconds=settings.expense_retry_base_seconds,
        retry_max_seconds=settings.expense_retry_max_seconds,
        delete_confirmation_ttl_seconds=settings.expense_delete_confirmation_ttl_seconds,
        media_provider=provider,
        audio_transcriber_factory=lambda: audio_transcriber,
        image_analyzer_factory=lambda: image_analyzer,
        query_interpreter_factory=lambda timezone: OpenAIExpenseQueryInterpreter(
            api_key=openai_api_key,
            timeout_seconds=settings.openai_query_timeout_seconds,
            timezone=timezone,
            model=settings.openai_query_model,
            max_output_tokens=settings.openai_query_max_output_tokens,
            client=openai_client,
        ),
        query_executor=SQLAlchemyExpenseQueryExecutor(SessionFactory),
        query_formatter=WhatsAppExpenseQueryResultFormatter(
            max_message_chars=settings.whatsapp_query_message_max_chars,
            max_pages=settings.whatsapp_query_max_pages,
        ),
        timing=timing,
        usage_control=usage_control,
        expense_model=settings.openai_expense_model,
        query_model=settings.openai_query_model,
        image_model=settings.openai_image_model,
        audio_model=settings.openai_audio_transcription_model,
    )
    delivery = OutboxDeliveryService(
        SessionFactory,
        provider,
        max_attempts=settings.outbox_max_attempts,
        retry_base_seconds=settings.outbox_retry_base_seconds,
        timing=timing,
    )
    try:
        outbox_cutoff = datetime.now(UTC) - timedelta(seconds=settings.outbox_state_timeout_seconds)
        await delivery.recover_stale(outbox_cutoff)
        while not stop.is_set():
            try:
                cutoff = datetime.now(UTC) - timedelta(
                    seconds=settings.worker_processing_lock_timeout_seconds
                )
                await processing.recover_stale(cutoff)
                outbox_cutoff = datetime.now(UTC) - timedelta(
                    seconds=settings.outbox_state_timeout_seconds
                )
                await delivery.recover_stale(outbox_cutoff)
                await _run_claim_batch(
                    stop,
                    settings.worker_batch_size,
                    processing.claim,
                    processing.process,
                )
                await _run_claim_batch(
                    stop,
                    settings.worker_batch_size,
                    delivery.claim,
                    delivery.send,
                )
            except Exception as exc:
                logger.error("Worker iteration failed", extra={"error_type": type(exc).__name__})
            try:
                await asyncio.wait_for(stop.wait(), timeout=settings.worker_poll_interval_seconds)
            except TimeoutError:
                pass
    finally:
        await _close_safely("audio_transcriber", audio_transcriber.aclose)
        await _close_safely("image_analyzer", image_analyzer.aclose)
        await _close_safely("whatsapp_provider", provider.aclose)
        with openai_private_operation():
            await _close_safely("openai_client", openai_client.close)
        await _close_safely("database_engine", engine.dispose)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
