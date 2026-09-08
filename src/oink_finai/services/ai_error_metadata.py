from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AIErrorMetadata:
    """Sanitized provider diagnostics safe for logs and durable retry decisions."""

    exception_class: str
    category: str
    duration_ms: int
    http_status: int | None = None
    provider_code: str | None = None
    request_id_present: bool = False
