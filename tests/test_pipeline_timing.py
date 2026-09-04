import asyncio
import logging
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from oink_finai.config.settings import Settings
from oink_finai.services.pipeline_timing import ALLOWED_FIELDS, PipelineTiming


class FakeClocks:
    def __init__(self) -> None:
        self.monotonic_value = 10.0
        self.utc_value = datetime(2026, 9, 4, 12, tzinfo=UTC)

    def monotonic(self) -> float:
        value = self.monotonic_value
        self.monotonic_value += 0.125
        return value

    def utc(self) -> datetime:
        value = self.utc_value
        self.utc_value += timedelta(seconds=1)
        return value


def records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == "oink_finai.pipeline_timing"]


def test_timing_disabled_emits_nothing(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        with PipelineTiming(False).span(
            "transcription_started", "transcription_completed", uuid4()
        ):
            pass
    assert records(caplog) == []
    assert Settings(_env_file=None).pipeline_timing_enabled is False


def test_sync_span_uses_fake_clocks_and_allowlisted_fields(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clocks = FakeClocks()
    correlation_id = uuid4()
    timing = PipelineTiming(True, monotonic_clock=clocks.monotonic, utc_clock=clocks.utc)

    with caplog.at_level(logging.INFO):
        with timing.span(
            "transcription_started",
            "transcription_completed",
            correlation_id,
            attempt_number=1,
            source_type="AUDIO",
            mime_type="audio/ogg",
        ) as span:
            span.result(outcome="success", size_bytes=12)

    captured = records(caplog)
    assert [record.event for record in captured] == [
        "transcription_started",
        "transcription_completed",
    ]
    assert captured[1].correlation_id == str(correlation_id)
    assert captured[1].duration_ms == 125.0
    assert captured[1].timestamp.endswith("Z")
    assert captured[1].duration_ms >= 0
    assert set(captured[1].__dict__) & ALLOWED_FIELDS <= ALLOWED_FIELDS


@pytest.mark.parametrize("failure", [RuntimeError("private failure"), asyncio.CancelledError()])
async def test_async_span_records_duration_and_preserves_failure(
    caplog: pytest.LogCaptureFixture, failure: BaseException
) -> None:
    clocks = FakeClocks()
    timing = PipelineTiming(True, monotonic_clock=clocks.monotonic, utc_clock=clocks.utc)

    with caplog.at_level(logging.INFO), pytest.raises(type(failure)):
        async with timing.span("transcription_started", "transcription_completed", uuid4()):
            raise failure

    assert records(caplog)[-1].event == "transcription_completed"
    assert records(caplog)[-1].duration_ms == 125.0


def test_two_attempts_keep_correlation_and_sanitize_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clocks = FakeClocks()
    correlation_id = uuid4()
    timing = PipelineTiming(True, monotonic_clock=clocks.monotonic, utc_clock=clocks.utc)

    with caplog.at_level(logging.INFO):
        for attempt in (1, 2):
            with timing.span(
                "media_download_started",
                "media_download_completed",
                correlation_id,
                attempt_number=attempt,
                stage="media_download",
            ) as span:
                span.result(
                    outcome="transient_failure" if attempt == 1 else "success",
                    error_code="secret value from provider" if attempt == 1 else None,
                )

    completed = [record for record in records(caplog) if record.event.endswith("completed")]
    assert [record.attempt_number for record in completed] == [1, 2]
    assert {record.correlation_id for record in completed} == {str(correlation_id)}
    assert completed[0].error_code == "UNKNOWN"
    assert "secret value from provider" not in caplog.text


def test_negative_clock_delta_is_clamped() -> None:
    now = datetime(2026, 9, 4, 12, tzinfo=UTC)
    assert PipelineTiming.elapsed_ms(now, now - timedelta(seconds=1)) == 0


def test_image_timing_events_accept_only_sanitized_metadata(
    caplog: pytest.LogCaptureFixture,
) -> None:
    timing = PipelineTiming(True)
    correlation_id = uuid4()

    with caplog.at_level(logging.INFO):
        with timing.span(
            "image_analysis_started",
            "image_analysis_completed",
            correlation_id,
            attempt_number=2,
            source_type="IMAGE",
        ) as span:
            span.result(outcome="terminal_failure", error_code="IMAGE_ANALYSIS_INVALID_RESPONSE")

    completed = records(caplog)[-1]
    assert completed.event == "image_analysis_completed"
    assert completed.source_type == "IMAGE" and completed.attempt_number == 2
    assert completed.error_code == "IMAGE_ANALYSIS_INVALID_RESPONSE"


def test_unknown_or_sensitive_fields_never_enter_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinel = "SECRET-SENTINEL-5511999999999"
    timing = PipelineTiming(True)

    with caplog.at_level(logging.INFO), pytest.raises(ValueError):
        timing.event("webhook_received", uuid4(), payload=sentinel)

    assert sentinel not in caplog.text
    assert records(caplog) == []
