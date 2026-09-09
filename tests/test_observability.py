import io
import json
import logging
import sys
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from oink_finai.observability import (
    ALLOWED_LOG_EVENTS,
    ALLOWED_LOG_FIELDS,
    SafeJsonFormatter,
    configure_application_logging,
)
from oink_finai.services.pipeline_timing import PipelineTiming

PRIVATE_VALUES = (
    "+5511999999999",
    "5511999999999@s.whatsapp.net",
    "private message text",
    "private transcript",
    "R$ 123,45",
    "private prompt",
    "private OpenAI response",
    "cHJpdmF0ZS1iYXNlNjQ=",
    "https://private.invalid/path?token=secret",
    "Authorization: Bearer secret",
    "postgresql+asyncpg://private:secret@db/private",
)


def make_record(name: str = "oink_finai.test", **extra: object) -> logging.LogRecord:
    record = logging.LogRecord(
        name=name,
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=" ".join(PRIVATE_VALUES),
        args=(),
        exc_info=None,
    )
    record.created = 1_788_955_200.0
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_is_valid_deterministic_and_strictly_allowlisted() -> None:
    correlation_id = uuid4()
    record = make_record(
        event="interpretation_completed",
        timestamp=datetime(2026, 9, 10, 12, tzinfo=UTC),
        correlation_id=correlation_id,
        processed_message_id=correlation_id,
        operation="TEXT_INTERPRETATION",
        stage="interpretation",
        outcome="success",
        attempt_number=2,
        duration_ms=12.3456,
        unknown_private_field=PRIVATE_VALUES[0],
    )
    formatter = SafeJsonFormatter("worker")

    first = formatter.format(record)
    second = formatter.format(record)
    payload = json.loads(first)

    assert first == second
    assert set(payload) <= ALLOWED_LOG_FIELDS
    assert payload["event"] in ALLOWED_LOG_EVENTS
    assert payload["correlation_id"] == str(correlation_id)
    assert payload["duration_ms"] == 12.346
    assert "unknown_private_field" not in payload
    assert all(value not in first for value in PRIVATE_VALUES)


def test_unknown_event_fields_and_invalid_identifiers_are_discarded_or_sanitized() -> None:
    rendered = SafeJsonFormatter("api").format(
        make_record(
            event="private-event-name",
            correlation_id="not-a-uuid",
            error_code="provider said secret value",
            status_code="200",
            operation="contains spaces",
            headers=PRIVATE_VALUES[-2],
        )
    )
    payload = json.loads(rendered)

    assert payload["event"] == "application_log"
    assert payload["error_code"] == "UNKNOWN"
    assert "correlation_id" not in payload
    assert "status_code" not in payload
    assert "operation" not in payload
    assert all(value not in rendered for value in PRIVATE_VALUES)


def test_explicitly_forbidden_domain_and_transport_fields_are_never_serialized() -> None:
    forbidden_fields = {
        "phone": PRIVATE_VALUES[0],
        "remoteJid": PRIVATE_VALUES[1],
        "accepted_text": PRIVATE_VALUES[2],
        "transcript": PRIVATE_VALUES[3],
        "amount": PRIVATE_VALUES[4],
        "prompt": PRIVATE_VALUES[5],
        "response": PRIVATE_VALUES[6],
        "base64": PRIVATE_VALUES[7],
        "url": PRIVATE_VALUES[8],
        "headers": PRIVATE_VALUES[9],
        "api_key": "private-api-key",
        "database_url": PRIVATE_VALUES[10],
    }
    rendered = SafeJsonFormatter("api").format(
        make_record(event="application_log", **forbidden_fields)
    )
    payload = json.loads(rendered)

    assert forbidden_fields.keys().isdisjoint(payload)
    assert all(value not in rendered for value in forbidden_fields.values())


def test_exception_keeps_only_sanitized_class_without_message_or_traceback() -> None:
    try:
        raise RuntimeError(PRIVATE_VALUES[2])
    except RuntimeError:
        exc_info = sys.exc_info()
    record = make_record(event="worker_iteration_failed")
    record.exc_info = exc_info
    rendered = SafeJsonFormatter("worker").format(record)
    payload = json.loads(rendered)

    assert payload["exception_class"] == "RuntimeError"
    assert PRIVATE_VALUES[2] not in rendered
    assert "Traceback" not in rendered


@pytest.mark.parametrize("logger_name", ["uvicorn.access", "httpx", "httpcore.http11", "openai"])
def test_library_records_never_render_request_response_or_url(logger_name: str) -> None:
    rendered = SafeJsonFormatter("api").format(
        make_record(logger_name, event="unknown", headers=PRIVATE_VALUES[-2])
    )
    payload = json.loads(rendered)

    assert payload["event"] in {"server_lifecycle", "library_log"}
    assert all(value not in rendered for value in PRIVATE_VALUES)


def test_production_policy_disables_access_and_private_sdk_loggers() -> None:
    stream = io.StringIO()
    root = logging.getLogger()
    previous_handlers = list(root.handlers)
    previous_level = root.level
    tracked_names = {
        name
        for name, candidate in logging.root.manager.loggerDict.items()
        if isinstance(candidate, logging.Logger)
    } | {
        "uvicorn",
        "uvicorn.access",
        "uvicorn.error",
        "fastapi",
        "sqlalchemy",
        "alembic",
        "asyncio",
        "httpx",
        "httpcore",
        "openai",
    }
    states = {}
    for name in tracked_names:
        candidate = logging.getLogger(name)
        states[name] = (
            list(candidate.handlers),
            candidate.level,
            candidate.disabled,
            candidate.propagate,
        )
    try:
        configure_application_logging(log_format="json", level="INFO", service="api", stream=stream)
        logging.getLogger("uvicorn.access").info("POST /private?secret=yes")
        logging.getLogger("httpx").error(PRIVATE_VALUES[6])
        logging.getLogger("openai").error(PRIVATE_VALUES[5])
        logging.getLogger("oink_finai.test").info(
            "ignored raw message", extra={"event": "webhook_received", "correlation_id": uuid4()}
        )
        lines = stream.getvalue().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["event"] == "webhook_received"
        assert logging.getLogger("uvicorn.access").disabled
        assert logging.getLogger("httpx").disabled
        assert logging.getLogger("openai").disabled
    finally:
        root.handlers[:] = previous_handlers
        root.setLevel(previous_level)
        for name, (handlers, level, disabled, propagate) in states.items():
            target = logging.getLogger(name)
            target.handlers[:] = handlers
            target.setLevel(level)
            target.disabled = disabled
            target.propagate = propagate


def test_pipeline_correlation_survives_webhook_worker_openai_and_outbox() -> None:
    stream = io.StringIO()
    logger = logging.getLogger("oink_finai.pipeline_timing")
    handler = logging.StreamHandler(stream)
    handler.setFormatter(SafeJsonFormatter("worker"))
    previous_handlers = list(logger.handlers)
    previous_level = logger.level
    previous_propagate = logger.propagate
    correlation_id = uuid4()
    outbound_id = uuid4()
    try:
        logger.handlers[:] = [handler]
        logger.setLevel(logging.INFO)
        logger.propagate = False
        timing = PipelineTiming(True)
        timing.event(
            "webhook_received",
            correlation_id,
            processed_message_id=correlation_id,
            operation="INBOUND_MESSAGE",
        )
        timing.event(
            "interpretation_completed",
            correlation_id,
            processed_message_id=correlation_id,
            operation="TEXT_INTERPRETATION",
            outcome="success",
        )
        timing.event(
            "outbound_accepted",
            correlation_id,
            processed_message_id=correlation_id,
            outbound_message_id=outbound_id,
            operation="OUTBOUND_DELIVERY",
            outcome="success",
        )
    finally:
        logger.handlers[:] = previous_handlers
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate

    payloads = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert {item["correlation_id"] for item in payloads} == {str(correlation_id)}
    assert payloads[-1]["outbound_message_id"] == str(outbound_id)
    assert [item["operation"] for item in payloads] == [
        "INBOUND_MESSAGE",
        "TEXT_INTERPRETATION",
        "OUTBOUND_DELIVERY",
    ]
