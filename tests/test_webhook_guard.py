import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from oink_finai.api.middleware import EVOLUTION_WEBHOOK_PATH, EvolutionWebhookGuardMiddleware
from oink_finai.database.base import Base
from oink_finai.database.models import User


def scope(*, content_length: bytes | None = None) -> dict[str, object]:
    headers = [] if content_length is None else [(b"content-length", content_length)]
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": EVOLUTION_WEBHOOK_PATH,
        "raw_path": EVOLUTION_WEBHOOK_PATH.encode(),
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 1),
        "server": ("test", 80),
    }


async def invoke(
    middleware: EvolutionWebhookGuardMiddleware,
    messages: list[dict[str, object]],
    *,
    content_length: bytes | None = None,
) -> list[dict[str, object]]:
    sent: list[dict[str, object]] = []
    queue = list(messages)

    async def receive():
        return queue.pop(0)

    async def send(message):
        sent.append(message)

    await middleware(scope(content_length=content_length), receive, send)
    return sent


def status_of(messages: list[dict[str, object]]) -> int:
    return int(next(item["status"] for item in messages if item["type"] == "http.response.start"))


def draining_app(seen: list[bytes]):
    async def app(_scope, receive, send):
        while True:
            message = await receive()
            seen.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    return app


async def test_content_length_over_limit_rejected_without_reading() -> None:
    calls = 0

    async def app(*_args):
        nonlocal calls
        calls += 1

    guard = EvolutionWebhookGuardMiddleware(
        app, max_body_bytes=8, timeout_seconds=1, max_concurrency=1
    )
    sent = await invoke(
        guard,
        [{"type": "http.request", "body": b"ignored", "more_body": False}],
        content_length=b"9",
    )
    assert status_of(sent) == 413 and calls == 0


@pytest.mark.parametrize("content_length", [None, b"2"])
async def test_chunked_or_lying_content_length_cannot_bypass_limit(
    content_length: bytes | None,
) -> None:
    seen: list[bytes] = []
    guard = EvolutionWebhookGuardMiddleware(
        draining_app(seen), max_body_bytes=5, timeout_seconds=1, max_concurrency=1
    )
    sent = await invoke(
        guard,
        [
            {"type": "http.request", "body": b"123", "more_body": True},
            {"type": "http.request", "body": b"456", "more_body": False},
        ],
        content_length=content_length,
    )
    assert status_of(sent) == 413
    assert seen == [b"123"]


async def test_exact_chunked_limit_and_original_chunks_are_forwarded() -> None:
    chunks = [b"12", b"345"]
    seen: list[bytes] = []
    guard = EvolutionWebhookGuardMiddleware(
        draining_app(seen), max_body_bytes=5, timeout_seconds=1, max_concurrency=1
    )
    sent = await invoke(
        guard,
        [
            {"type": "http.request", "body": chunks[0], "more_body": True},
            {"type": "http.request", "body": chunks[1], "more_body": False},
        ],
    )
    assert status_of(sent) == 200
    assert seen[0] is chunks[0] and seen[1] is chunks[1]


async def test_client_disconnect_and_malformed_content_length_are_sanitized() -> None:
    guard = EvolutionWebhookGuardMiddleware(
        draining_app([]), max_body_bytes=8, timeout_seconds=1, max_concurrency=1
    )
    disconnected = await invoke(guard, [{"type": "http.disconnect"}])
    malformed = await invoke(
        guard,
        [{"type": "http.request", "body": b"", "more_body": False}],
        content_length=b"invalid",
    )
    assert status_of(disconnected) == 400
    assert status_of(malformed) == 400


def test_empty_and_invalid_json_reach_fastapi_validation_without_integrations() -> None:
    application = FastAPI()

    @application.post(EVOLUTION_WEBHOOK_PATH)
    async def endpoint(payload: dict[str, object]):
        return payload

    application.add_middleware(
        EvolutionWebhookGuardMiddleware,
        max_body_bytes=64,
        timeout_seconds=1,
        max_concurrency=1,
    )
    with TestClient(application) as client:
        empty = client.post(EVOLUTION_WEBHOOK_PATH, content=b"")
        invalid = client.post(
            EVOLUTION_WEBHOOK_PATH,
            content=b"{invalid",
            headers={"content-type": "application/json"},
        )
    assert empty.status_code == 422
    assert invalid.status_code == 422


async def test_capacity_is_fail_fast_at_the_exact_limit() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def app(_scope, receive, send):
        await receive()
        entered.set()
        await release.wait()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    guard = EvolutionWebhookGuardMiddleware(
        app, max_body_bytes=8, timeout_seconds=1, max_concurrency=1
    )
    first = asyncio.create_task(
        invoke(
            guard,
            [{"type": "http.request", "body": b"1", "more_body": False}],
            content_length=b"1",
        )
    )
    await entered.wait()
    rejected = await invoke(
        guard,
        [{"type": "http.request", "body": b"2", "more_body": False}],
        content_length=b"1",
    )
    release.set()
    accepted = await first
    assert status_of(accepted) == 200
    assert status_of(rejected) == 503


async def test_timeout_during_body_read_returns_sanitized_503() -> None:
    async def never_returns():
        await asyncio.sleep(10)

    sent: list[dict[str, object]] = []

    async def send(message):
        sent.append(message)

    guard = EvolutionWebhookGuardMiddleware(
        draining_app([]), max_body_bytes=8, timeout_seconds=0.01, max_concurrency=1
    )
    await guard(scope(), never_returns, send)
    assert status_of(sent) == 503


@pytest_asyncio.fixture
async def factory(tmp_path: Path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'rollback.db').as_posix()}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    result = async_sessionmaker(engine, expire_on_commit=False)
    yield result
    await engine.dispose()


async def test_timeout_during_persistence_cancels_and_rolls_back(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    cancelled = asyncio.Event()

    async def app(_scope, receive, _send):
        await receive()
        try:
            async with factory() as session, session.begin():
                session.add(User(phone_number="must-rollback", timezone="UTC"))
                await session.flush()
                await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    guard = EvolutionWebhookGuardMiddleware(
        app, max_body_bytes=8, timeout_seconds=0.01, max_concurrency=1
    )
    sent = await invoke(
        guard,
        [{"type": "http.request", "body": b"1", "more_body": False}],
        content_length=b"1",
    )
    async with factory() as session:
        users = await session.scalar(select(func.count()).select_from(User))
    assert status_of(sent) == 503
    assert cancelled.is_set()
    assert users == 0
