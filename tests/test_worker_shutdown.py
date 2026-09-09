import asyncio

from oink_finai.worker import _run_claim_batch


async def test_stopped_worker_does_not_claim_new_work() -> None:
    stop = asyncio.Event()
    stop.set()
    claim_calls = 0

    async def claim(_: int) -> list[str]:
        nonlocal claim_calls
        claim_calls += 1
        return ["unexpected"]

    async def handle(_: str) -> None:
        raise AssertionError("no item should be handled")

    await _run_claim_batch(stop, 10, claim, handle)

    assert claim_calls == 0


async def test_signal_between_batch_items_prevents_another_claim() -> None:
    stop = asyncio.Event()
    claimed_limits: list[int] = []
    handled: list[str] = []

    async def claim(limit: int) -> list[str]:
        claimed_limits.append(limit)
        return ["current"]

    async def handle(item: str) -> None:
        handled.append(item)
        stop.set()

    await _run_claim_batch(stop, 10, claim, handle)

    assert claimed_limits == [1]
    assert handled == ["current"]


async def test_signal_during_claim_finishes_the_claimed_item() -> None:
    stop = asyncio.Event()
    handled: list[str] = []

    async def claim(_: int) -> list[str]:
        stop.set()
        return ["already-claimed"]

    async def handle(item: str) -> None:
        handled.append(item)

    await _run_claim_batch(stop, 10, claim, handle)

    assert handled == ["already-claimed"]
