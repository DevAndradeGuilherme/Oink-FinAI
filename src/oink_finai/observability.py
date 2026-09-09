import json
import logging
import math
import re
import sys
from datetime import UTC, datetime
from typing import IO, Literal
from uuid import UUID

LogFormat = Literal["json", "console"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

ALLOWED_LOG_FIELDS = frozenset(
    {
        "timestamp",
        "level",
        "event",
        "service",
        "correlation_id",
        "processed_message_id",
        "outbound_message_id",
        "operation",
        "stage",
        "outcome",
        "attempt_number",
        "duration_ms",
        "status_code",
        "error_code",
        "next_attempt_at",
        "sequence_no",
        "sequence_count",
        "source_type",
        "size_bytes",
        "mime_type",
        "audio_duration_seconds",
        "row_count",
        "group_count",
        "page_count",
        "intent",
        "metric",
        "group",
        "method",
        "route",
        "exception_class",
    }
)

PIPELINE_EVENTS = frozenset(
    {
        "webhook_received",
        "access_filter_completed",
        "inbound_persisted",
        "webhook_completed",
        "processing_claimed",
        "queue_wait_completed",
        "media_download_started",
        "media_download_completed",
        "image_download_started",
        "image_download_completed",
        "image_analysis_started",
        "image_analysis_completed",
        "image_checkpoint_started",
        "image_checkpoint_completed",
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
        "query_interpretation_started",
        "query_interpretation_completed",
        "query_plan_checkpoint_started",
        "query_plan_checkpoint_completed",
        "query_execution_started",
        "query_execution_completed",
        "query_formatting_started",
        "query_formatting_completed",
        "query_outbox_created",
        "query_processing_completed",
    }
)

ALLOWED_LOG_EVENTS = PIPELINE_EVENTS | frozenset(
    {
        "application_log",
        "asyncio_runtime",
        "database_library",
        "library_log",
        "migration_runtime",
        "openai_operation_failed",
        "server_lifecycle",
        "webhook_http_completed",
        "worker_heartbeat_failed",
        "worker_iteration_failed",
        "worker_resource_close_failed",
    }
)

_SAFE_CODE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_SAFE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,127}$")
_SAFE_OPERATION = re.compile(r"^(?:[A-Z][A-Z0-9_]{0,63}|[a-z][a-z0-9_]{0,63})$")
_SAFE_MIME = re.compile(r"^[a-z0-9][a-z0-9.+-]*/[a-z0-9][a-z0-9.+-]*$")
_SAFE_METHODS = frozenset({"POST"})
_SAFE_ROUTES = frozenset({"/api/v1/webhooks/evolution"})
_SAFE_OUTCOMES = frozenset(
    {"success", "transient_failure", "terminal_failure", "rejected", "unavailable"}
)
_SAFE_STAGES = frozenset(
    {
        "processing",
        "media_download",
        "transcription",
        "image_download",
        "image_analysis",
        "image_checkpoint",
        "interpretation",
        "expense_persistence",
        "send",
        "query_interpretation",
        "query_plan_checkpoint",
        "query_execution",
        "query_formatting",
    }
)
_SAFE_SOURCE_TYPES = frozenset({"TEXT", "AUDIO", "IMAGE"})
_SAFE_INTENTS = frozenset(
    {"QUERY", "LIST", "AGGREGATE", "GROUP", "RANK", "COMPARE", "NOT_QUERY", "QUERY_UNCLEAR"}
)
_SAFE_METRICS = frozenset({"TOTAL", "COUNT", "AVERAGE", "MINIMUM", "MAXIMUM"})
_SAFE_GROUPS = frozenset(
    {"DAY", "WEEK", "MONTH", "CATEGORY", "MERCHANT", "PAYMENT_METHOD", "SOURCE_TYPE"}
)
_INTEGER_FIELDS = frozenset(
    {
        "attempt_number",
        "sequence_no",
        "sequence_count",
        "size_bytes",
        "row_count",
        "group_count",
        "page_count",
    }
)
_DURATION_FIELDS = frozenset({"duration_ms", "audio_duration_seconds"})
_UUID_FIELDS = frozenset({"correlation_id", "processed_message_id", "outbound_message_id"})
_PRIVATE_LIBRARY_PREFIXES = ("httpx", "httpcore", "openai")
_SAFE_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
_CONTROLLED_LIBRARY_LOGGERS = (
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
    "fastapi",
    "sqlalchemy",
    "alembic",
    "asyncio",
    *_PRIVATE_LIBRARY_PREFIXES,
)


def _utc_timestamp(value: object) -> str | None:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    elif isinstance(value, datetime):
        parsed = value
    else:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _event_for_record(record: logging.LogRecord) -> str:
    candidate = getattr(record, "event", None)
    if candidate in ALLOWED_LOG_EVENTS:
        return candidate
    if record.name.startswith("uvicorn"):
        return "server_lifecycle"
    if record.name.startswith(("sqlalchemy", "alembic")):
        return "database_library" if record.name.startswith("sqlalchemy") else "migration_runtime"
    if record.name.startswith("asyncio"):
        return "asyncio_runtime"
    if record.name.startswith("oink_finai.services.openai"):
        return "openai_operation_failed"
    if record.name.startswith("oink_finai"):
        return "application_log"
    return "library_log"


def _safe_uuid(value: object) -> str | None:
    try:
        return str(UUID(str(value)))
    except (AttributeError, TypeError, ValueError):
        return None


def _safe_number(value: object, *, maximum: float) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0 or number > maximum:
        return None
    return int(value) if isinstance(value, int) else round(number, 3)


class SafeJsonFormatter(logging.Formatter):
    """Serialize only a closed set of metadata; never serialize messages or arguments."""

    def __init__(self, service: str) -> None:
        super().__init__()
        if not _SAFE_NAME.fullmatch(service):
            raise ValueError("logging service name must be opaque")
        self._service = service

    def format(self, record: logging.LogRecord) -> str:
        try:
            payload = self._payload(record)
            return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        except Exception:
            return (
                '{"event":"application_log","level":"ERROR","service":"logging",'
                '"timestamp":"1970-01-01T00:00:00.000Z"}'
            )

    def _payload(self, record: logging.LogRecord) -> dict[str, object]:
        timestamp = _utc_timestamp(getattr(record, "timestamp", None))
        if timestamp is None:
            timestamp = (
                datetime.fromtimestamp(record.created, UTC)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z")
            )
        payload: dict[str, object] = {
            "timestamp": timestamp,
            "level": record.levelname if record.levelname in _SAFE_LEVELS else "INFO",
            "event": _event_for_record(record),
            "service": self._service,
        }
        aliases = {
            "status_code": getattr(
                record,
                "status_code",
                getattr(record, "http_status", getattr(record, "openai_status", None)),
            ),
            "error_code": getattr(
                record,
                "error_code",
                getattr(record, "openai_code", getattr(record, "image_analysis_code", None)),
            ),
            "exception_class": getattr(
                record,
                "exception_class",
                getattr(record, "error_type", getattr(record, "openai_exception_class", None)),
            ),
        }
        for name in ALLOWED_LOG_FIELDS - {"timestamp", "level", "event", "service", *aliases}:
            value = getattr(record, name, None)
            sanitized = self._sanitize(name, value)
            if sanitized is not None:
                payload[name] = sanitized
        for name, value in aliases.items():
            sanitized = self._sanitize(name, value)
            if sanitized is not None:
                payload[name] = sanitized
        if record.exc_info and "exception_class" not in payload:
            sanitized = self._sanitize("exception_class", record.exc_info[0].__name__)
            if sanitized is not None:
                payload["exception_class"] = sanitized
        return payload

    @staticmethod
    def _sanitize(name: str, value: object) -> object | None:
        if value is None:
            return None
        if name in _UUID_FIELDS:
            return _safe_uuid(value)
        if name in _INTEGER_FIELDS:
            return _safe_number(value, maximum=1_000_000_000)
        if name in _DURATION_FIELDS:
            return _safe_number(value, maximum=86_400_000)
        if name == "status_code":
            return (
                value
                if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599
                else None
            )
        if name == "error_code":
            return value if isinstance(value, str) and _SAFE_CODE.fullmatch(value) else "UNKNOWN"
        if name == "next_attempt_at":
            return _utc_timestamp(value)
        if name == "operation":
            return value if isinstance(value, str) and _SAFE_OPERATION.fullmatch(value) else None
        if name == "stage":
            return value if value in _SAFE_STAGES else None
        if name == "outcome":
            return value if value in _SAFE_OUTCOMES else None
        if name == "source_type":
            return value if value in _SAFE_SOURCE_TYPES else None
        if name == "intent":
            return value if value in _SAFE_INTENTS else None
        if name == "metric":
            return value if value in _SAFE_METRICS else None
        if name == "group":
            return value if value in _SAFE_GROUPS else None
        if name == "mime_type" and isinstance(value, str):
            normalized = value.partition(";")[0].strip().lower()
            return normalized if _SAFE_MIME.fullmatch(normalized) else None
        if name == "method":
            return value if value in _SAFE_METHODS else None
        if name == "route":
            return value if value in _SAFE_ROUTES else None
        if name == "exception_class":
            return value if isinstance(value, str) and _SAFE_NAME.fullmatch(value) else "Exception"
        return None


def emit_event(
    logger: logging.Logger,
    level: int,
    event: str,
    **fields: object,
) -> None:
    safe_event = event if event in ALLOWED_LOG_EVENTS else "application_log"
    safe_fields = {name: value for name, value in fields.items() if name in ALLOWED_LOG_FIELDS}
    logger.log(level, safe_event, extra={"event": safe_event, **safe_fields})


def configure_application_logging(
    *,
    log_format: LogFormat,
    level: str,
    service: str,
    include_traceback: bool = False,
    stream: IO[str] | None = None,
) -> None:
    if level not in _SAFE_LEVELS:
        raise ValueError("logging level is invalid")
    numeric_level = logging._nameToLevel[level]
    if log_format == "console":
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
            force=True,
            stream=stream or sys.stderr,
        )
        return

    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(SafeJsonFormatter(service))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(numeric_level)

    for name in _CONTROLLED_LIBRARY_LOGGERS:
        library_logger = logging.getLogger(name)
        library_logger.handlers.clear()
        library_logger.propagate = True
        library_logger.disabled = False
        library_logger.setLevel(numeric_level)
    controlled_prefixes = ("oink_finai", *_CONTROLLED_LIBRARY_LOGGERS)
    for name, candidate in logging.root.manager.loggerDict.items():
        if isinstance(candidate, logging.Logger) and (
            name in controlled_prefixes
            or name.startswith(tuple(f"{prefix}." for prefix in controlled_prefixes))
        ):
            candidate.handlers.clear()
            candidate.propagate = True
            candidate.disabled = False

    logging.getLogger("uvicorn.access").disabled = True
    for name in _PRIVATE_LIBRARY_PREFIXES:
        private_logger = logging.getLogger(name)
        private_logger.handlers[:] = [logging.NullHandler()]
        private_logger.propagate = False
        private_logger.disabled = True
    for name, candidate in logging.root.manager.loggerDict.items():
        if isinstance(candidate, logging.Logger) and name.startswith(
            tuple(f"{prefix}." for prefix in _PRIVATE_LIBRARY_PREFIXES)
        ):
            candidate.handlers.clear()
            candidate.propagate = True
            candidate.disabled = False

    if include_traceback:
        # Production settings reject this. Development console logging owns traceback policy.
        logging.getLogger(__name__).warning("traceback output is unavailable in JSON mode")
