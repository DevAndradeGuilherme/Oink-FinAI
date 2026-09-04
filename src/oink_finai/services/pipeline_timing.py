import logging
import math
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from types import TracebackType
from typing import Self
from uuid import UUID

logger = logging.getLogger("oink_finai.pipeline_timing")

ALLOWED_FIELDS = frozenset(
    {
        "event",
        "correlation_id",
        "timestamp",
        "duration_ms",
        "attempt_number",
        "source_type",
        "outcome",
        "error_code",
        "http_status",
        "size_bytes",
        "mime_type",
        "audio_duration_seconds",
        "next_attempt_at",
        "stage",
    }
)
TIMING_EVENTS = frozenset(
    {
        "webhook_received",
        "access_filter_completed",
        "inbound_persisted",
        "webhook_completed",
        "processing_claimed",
        "queue_wait_completed",
        "media_download_started",
        "media_download_completed",
        "transcription_started",
        "transcription_completed",
        "transcript_checkpoint_started",
        "transcript_checkpoint_completed",
        "interpretation_started",
        "interpretation_completed",
        "expense_persistence_started",
        "expense_persistence_completed",
        "processing_completed",
        "outbox_claimed",
        "outbox_queue_wait_completed",
        "outbound_send_started",
        "outbound_send_completed",
        "outbound_accepted",
    }
)
_SAFE_CODE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_SAFE_MIME = re.compile(r"^[a-z0-9][a-z0-9.+-]*/[a-z0-9][a-z0-9.+-]*$")
_STAGES = frozenset(
    {
        "processing",
        "media_download",
        "transcription",
        "interpretation",
        "expense_persistence",
        "send",
    }
)


def utc_now() -> datetime:
    return datetime.now(UTC)


class PipelineTiming:
    """Opt-in timing events containing only explicitly allowlisted metadata."""

    def __init__(
        self,
        enabled: bool,
        *,
        monotonic_clock: Callable[[], float] = time.perf_counter,
        utc_clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.enabled = enabled
        self._monotonic_clock = monotonic_clock
        self._utc_clock = utc_clock

    def event(self, event: str, correlation_id: UUID, **fields: object) -> None:
        if not self.enabled:
            return
        if event not in TIMING_EVENTS:
            raise ValueError("Timing event is not allowlisted")
        unknown = fields.keys() - ALLOWED_FIELDS
        if unknown:
            raise ValueError(f"Timing fields are not allowlisted: {sorted(unknown)}")
        payload: dict[str, object] = {
            "event": event,
            "correlation_id": str(UUID(str(correlation_id))),
            "timestamp": self._iso_timestamp(self._utc_clock()),
        }
        for name, value in fields.items():
            if value is None:
                continue
            if name == "error_code":
                value = (
                    value if isinstance(value, str) and _SAFE_CODE.fullmatch(value) else "UNKNOWN"
                )
            elif name == "outcome" and value not in {
                "success",
                "transient_failure",
                "terminal_failure",
            }:
                continue
            elif name == "source_type" and value not in {"TEXT", "AUDIO"}:
                continue
            elif name == "stage" and value not in _STAGES:
                continue
            elif name == "mime_type":
                if not isinstance(value, str):
                    continue
                value = value.partition(";")[0].strip().lower()
                if not _SAFE_MIME.fullmatch(value):
                    continue
            elif name == "http_status":
                if isinstance(value, bool) or not isinstance(value, int):
                    continue
            elif name in {"duration_ms", "audio_duration_seconds"}:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                value = round(max(0.0, value), 3)
                if not math.isfinite(value):
                    continue
            elif name in {"attempt_number", "size_bytes"}:
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    continue
            elif name == "next_attempt_at" and isinstance(value, datetime):
                value = self._iso_timestamp(value)
            payload[name] = value
        logger.info("pipeline_timing", extra=payload)

    def span(
        self,
        started_event: str | None,
        completed_event: str,
        correlation_id: UUID,
        **fields: object,
    ) -> "TimingSpan":
        return TimingSpan(self, started_event, completed_event, correlation_id, fields)

    @staticmethod
    def elapsed_ms(started_at: datetime, completed_at: datetime) -> float:
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=UTC)
        if completed_at.tzinfo is None:
            completed_at = completed_at.replace(tzinfo=UTC)
        return round(max(0.0, (completed_at - started_at).total_seconds()) * 1000, 3)

    @staticmethod
    def _iso_timestamp(value: datetime) -> str:
        if value.tzinfo is None or value.utcoffset() is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


class TimingSpan:
    def __init__(
        self,
        timing: PipelineTiming,
        started_event: str | None,
        completed_event: str,
        correlation_id: UUID,
        fields: dict[str, object],
    ) -> None:
        self._timing = timing
        self._started_event = started_event
        self._completed_event = completed_event
        self._correlation_id = correlation_id
        self._fields = fields
        self._started: float | None = None
        self._result_fields: dict[str, object] = {}

    def result(self, **fields: object) -> None:
        self._result_fields.update(fields)

    def __enter__(self) -> Self:
        if self._timing.enabled:
            self._started = self._timing._monotonic_clock()
            if self._started_event is not None:
                self._timing.event(self._started_event, self._correlation_id, **self._fields)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self._complete()
        return False

    async def __aenter__(self) -> Self:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        return self.__exit__(exc_type, exc, traceback)

    def _complete(self) -> None:
        if self._started is None:
            return
        elapsed = max(0.0, self._timing._monotonic_clock() - self._started)
        self._timing.event(
            self._completed_event,
            self._correlation_id,
            duration_ms=round(elapsed * 1000, 3),
            **self._fields,
            **self._result_fields,
        )
