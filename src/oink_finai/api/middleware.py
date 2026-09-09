import asyncio

from starlette.types import ASGIApp, Message, Receive, Scope, Send

EVOLUTION_WEBHOOK_PATH = "/api/v1/webhooks/evolution"


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
    ) -> None:
        self._app = app
        self._max_body_bytes = max_body_bytes
        self._timeout_seconds = timeout_seconds
        self._max_concurrency = max_concurrency
        self._active_requests = 0
        self._capacity_lock = asyncio.Lock()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") != EVOLUTION_WEBHOOK_PATH:
            await self._app(scope, receive, send)
            return

        content_length = self._content_length(scope)
        if content_length is None:
            await self._respond(send, 400, b'{"detail":"invalid request"}')
            return
        if content_length > self._max_body_bytes:
            await self._respond(send, 413, b'{"detail":"request too large"}')
            return
        if not await self._try_acquire_capacity():
            await self._respond(send, 503, b'{"detail":"temporarily unavailable"}')
            return

        received_bytes = 0

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
            try:
                async with asyncio.timeout(self._timeout_seconds):
                    await self._app(scope, limited_receive, send)
            except _BodyTooLarge:
                await self._respond(send, 413, b'{"detail":"request too large"}')
            except _ClientDisconnected:
                await self._respond(send, 400, b'{"detail":"invalid request"}')
            except TimeoutError:
                await self._respond(send, 503, b'{"detail":"temporarily unavailable"}')
        finally:
            await self._release_capacity()

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
