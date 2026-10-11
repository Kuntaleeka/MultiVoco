"""Benchmark Piper: memory per loaded voice, time to first audio, real-time factor.

    uv run python -m app.tts.bench

Run it on the host that matters. Render's free service has 512 MB and a shared CPU.
"""

import asyncio
import os
import statistics
import sys
import time

from app.core.languages import Lang
from app.core.pools import run_blocking, shutdown_pools
from app.tts.piper import PiperTTS

# About ten words each, with a figure and a date, like a real reply.
SENTENCES = {
    Lang.EN: "Your next EMI of 8450.00 rupees is due on 5 November 2026.",
    Lang.HI: "आपकी अगली EMI 8450.00 रुपये की है और 5 नवंबर 2026 को देय है।",
}
RUNS = 5


def memory_mb() -> float:
    """Resident memory of this process."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        return counters.WorkingSetSize / 2**20
    with open("/proc/self/statm") as statm:
        return int(statm.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20


async def one_run(tts: PiperTTS, lang: Lang) -> tuple[float, float, float]:
    """(ms to first audio, ms in total, seconds of audio produced)."""
    started = time.perf_counter()
    first, size = None, 0
    async for piece in tts.synthesize(SENTENCES[lang], lang):
        first = first or (time.perf_counter() - started) * 1000
        size += len(piece)
    total = (time.perf_counter() - started) * 1000
    return first or total, total, size / 2 / tts.sample_rate(lang)


async def main() -> None:
    tts = PiperTTS()
    print(f"cpus: {os.cpu_count()}   memory before any voice: {memory_mb():.0f} MB")
    for lang in (Lang.EN, Lang.HI):
        before = memory_mb()
        started = time.perf_counter()
        await run_blocking(tts.load, lang)
        load_ms = (time.perf_counter() - started) * 1000
        cold = await one_run(tts, lang)
        runs = [await one_run(tts, lang) for _ in range(RUNS)]
        first = statistics.median(run[0] for run in runs)
        rtf = statistics.median(run[1] / 1000 / run[2] for run in runs)
        print(
            f"{lang.value}: load {load_ms:.0f} ms, +{memory_mb() - before:.0f} MB, "
            f"first audio cold {cold[0]:.0f} ms, warm median {first:.0f} ms, "
            f"real-time factor {rtf:.2f}, {runs[0][2]:.1f} s of audio"
        )
    print(f"memory with both voices loaded: {memory_mb():.0f} MB")
    shutdown_pools()


if __name__ == "__main__":
    asyncio.run(main())
