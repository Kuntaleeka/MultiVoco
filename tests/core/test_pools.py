import asyncio
import time

from app.core.pools import iter_blocking, run_blocking


async def test_blocking_work_does_not_stall_the_event_loop():
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(ticker())
    assert await run_blocking(lambda: time.sleep(0.2) or "done") == "done"
    task.cancel()
    assert ticks >= 10


async def test_iter_blocking_yields_every_item():
    assert [item async for item in iter_blocking(lambda: iter(range(5)))] == [0, 1, 2, 3, 4]


async def test_iter_blocking_stops_between_items_when_cancelled():
    produced = []
    closed = False

    def slow():
        nonlocal closed
        try:
            for i in range(100):
                time.sleep(0.02)
                produced.append(i)
                yield i
        finally:
            closed = True

    async def consume():
        async for _ in iter_blocking(slow):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.1)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0.1)
    assert closed
    assert len(produced) < 20
