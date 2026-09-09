import asyncio
import logging
import time

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from oink_finai.observability import emit_event

EVOLUTION_WEBHOOK_PATH = "/api/v1/webhooks/evolution"
logger = logging.getLogger(__name__)


class _BodyTooLarge(Exception):
    pass


class _ClientDisconnected(Exception):
    pass


class EvolutionWebhookGuardMiddleware:
    """Bound raw webhook input before Starlette constructs Request or parses JSON."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_body_bytes: int,
        timeout_seconds: float,
        max_concurrency: int,
        log_requests: bool = False,
    ) -> None:
        self._app = app
        self._max_body_bytes = max_body_bytes
        self._timeout_seconds = timeout_seconds
        self._max_concurrency = max_concurrency
        self._log_requests = log_requests
        self._active_requests = 0
        self._capacity_lock = asyncio.Lock()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") != EVOLUTION_WEBHOOK_PATH:
            await self._app(scope, receive, send)
            return

        started_at = time.perf_counter()
        status_code = 500
        acquired = False
        received_bytes = 0

        async def tracked_send(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
            await send(message)

        async def limited_receive() -> Message:
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.disconnect":
                raise _ClientDisconnected
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self._max_body_bytes:
                    raise _BodyTooLarge
            return message

        try:
            content_length = self._content_length(scope)
            if content_length is None:
                await self._respond(tracked_send, 400, b'{"detail":"invalid request"}')
                return
            if content_length > self._max_body_bytes:
                await self._respond(tracked_send, 413, b'{"detail":"request too large"}')
                return
            if not await self._try_acquire_capacity():
                await self._respond(tracked_send, 503, b'{"detail":"temporarily unavailable"}')
                return
            acquired = True
            try:
                async with asyncio.timeout(self._timeout_seconds):
                    await self._app(scope, limited_receive, tracked_send)
            except _BodyTooLarge:
                await self._respond(tracked_send, 413, b'{"detail":"request too large"}')
            except _ClientDisconnected:
                await self._respond(tracked_send, 400, b'{"detail":"invalid request"}')
            except TimeoutError:
                await self._respond(tracked_send, 503, b'{"detail":"temporarily unavailable"}')
        finally:
            if acquired:
                await self._release_capacity()
            if self._log_requests:
                emit_event(
                    logger,
                    logging.INFO,
                    "webhook_http_completed",
                    method="POST",
                    route=EVOLUTION_WEBHOOK_PATH,
                    status_code=status_code,
                    duration_ms=round(max(0.0, time.perf_counter() - started_at) * 1000, 3),
                    outcome=(
                        "success"
                        if status_code < 400
                        else "unavailable"
                        if status_code == 503
                        else "rejected"
                    ),
                )

    async def _try_acquire_capacity(self) -> bool:
        async with self._capacity_lock:
            if self._active_requests >= self._max_concurrency:
                return False
            self._active_requests += 1
            return True

    async def _release_capacity(self) -> None:
        async with self._capacity_lock:
            self._active_requests -= 1

    @staticmethod
    def _content_length(scope: Scope) -> int | None:
        values = [
            value.strip()
            for name, value in scope.get("headers", ())
            if name.lower() == b"content-length"
        ]
        if not values:
            return 0
        if len(values) != 1 or not values[0].isdigit():
            return None
        try:
            return int(values[0])
        except ValueError:
            return None

    @staticmethod
    async def _respond(send: Send, status: int, body: bytes) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": (
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ),
            }
        )
        await send({"type": "http.response.body", "body": body})
