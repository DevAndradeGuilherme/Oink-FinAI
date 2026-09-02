import asyncio
import logging
import signal
from datetime import UTC, datetime, timedelta

from oink_finai.config.settings import get_settings
from oink_finai.database.session import SessionFactory, engine
from oink_finai.providers.whatsapp import EvolutionWhatsAppProvider
from oink_finai.services.expense_processing import ExpenseProcessingService
from oink_finai.services.gemini_expense_interpreter import GeminiExpenseInterpreter
from oink_finai.services.outbox_delivery import OutboxDeliveryService

logger = logging.getLogger(__name__)


async def run_worker() -> None:
    settings = get_settings()
    if not all(
        (
            settings.gemini_api_key,
            settings.gemini_model,
            settings.evolution_base_url,
            settings.evolution_api_key,
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

    processing = ExpenseProcessingService(
        SessionFactory,
        lambda timezone: GeminiExpenseInterpreter(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            timeout_seconds=settings.gemini_timeout_seconds,
            timezone=timezone,
        ),
        max_attempts=settings.expense_processing_max_attempts,
        retry_base_seconds=settings.expense_retry_base_seconds,
        retry_max_seconds=settings.expense_retry_max_seconds,
    )
    provider = EvolutionWhatsAppProvider(
        settings.evolution_base_url,
        settings.evolution_api_key,
        settings.evolution_instance,
        timeout_seconds=settings.evolution_timeout_seconds,
        max_retries=0,
    )
    delivery = OutboxDeliveryService(
        SessionFactory,
        provider,
        max_attempts=settings.outbox_max_attempts,
        retry_base_seconds=settings.outbox_retry_base_seconds,
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
                for message_id in await processing.claim(settings.worker_batch_size):
                    await processing.process(message_id)
                for outbound_claim in await delivery.claim(settings.worker_batch_size):
                    await delivery.send(outbound_claim)
            except Exception as exc:
                logger.error("Worker iteration failed", extra={"error_type": type(exc).__name__})
            try:
                await asyncio.wait_for(stop.wait(), timeout=settings.worker_poll_interval_seconds)
            except TimeoutError:
                pass
    finally:
        await provider.aclose()
        await engine.dispose()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
