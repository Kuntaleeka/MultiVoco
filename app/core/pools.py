"""The shared executor for blocking work (Silero, Piper, anything CPU-bound).

Nothing blocking may run on the event loop. Cancelling a task that awaits run_blocking
does not stop the thread, so keep each job short: process one frame or one audio chunk
per call. For a blocking generator, use iter_blocking, which hops to the pool once per
item and so can be cancelled between items.
"""

import asyncio
import functools
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import TypeVar

from app.core.config import get_settings

T = TypeVar("T")

_executor: ThreadPoolExecutor | None = None
_STOP = object()


def get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=get_settings().executor_workers, thread_name_prefix="multivoco"
        )
    return _executor


async def run_blocking(fn: Callable[..., T], /, *args, **kwargs) -> T:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(get_executor(), functools.partial(fn, *args, **kwargs))


async def iter_blocking(make_iterator: Callable[[], Iterator[T]]) -> AsyncIterator[T]:
    """Drive a blocking iterator from the pool, one item per hop."""
    executor = get_executor()
    iterator = await run_blocking(make_iterator)
    in_flight: Future | None = None
    try:
        while True:
            in_flight = executor.submit(next, iterator, _STOP)
            item = await asyncio.wrap_future(in_flight)
            if item is _STOP:
                return
            yield item
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            # On cancellation a next() may still be running in the pool, and a running
            # generator cannot be closed. Close it from the pool once that call returns.
            def close_when_idle(pending: Future | None = in_flight) -> None:
                if pending is not None:
                    wait([pending])
                close()

            executor.submit(close_when_idle)


def shutdown_pools() -> None:
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)
        _executor = None
