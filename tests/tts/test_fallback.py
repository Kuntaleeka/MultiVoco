import asyncio

import pytest

import app.tts
from app.core import registry
from app.core.interfaces import TTS, ProviderError, QuotaExceeded
from app.core.languages import Lang
from app.tts.fallback import FallbackTTS
from app.tts.piper import PiperTTS


class Voice:
    """A scripted TTS: says its name in pieces, after a delay, or fails."""

    def __init__(self, name, *, rate=22_050, delay=0.0, fail=None, fail_after=None, langs=None):
        self.name, self.rate, self.delay = name, rate, delay
        self.fail, self.fail_after = fail, fail_after
        self.languages = set(langs or Lang)
        self.calls = 0

    def sample_rate(self, language):
        return self.rate

    async def synthesize(self, text, language):
        self.calls += 1
        await asyncio.sleep(self.delay)
        if self.fail and self.fail_after is None:
            raise self.fail
        for index in range(3):
            if self.fail_after == index:
                raise self.fail
            yield f"{self.name}{index}".encode()


async def collect(tts, lang=Lang.EN) -> list[bytes]:
    return [piece async for piece in tts.synthesize("Hello.", lang)]


async def test_the_primary_speaks_when_it_works():
    primary, backup = Voice("p"), Voice("b")
    tts = FallbackTTS(primary, backup, 1.0)
    assert isinstance(tts, TTS)
    assert await collect(tts) == [b"p0", b"p1", b"p2"]
    assert backup.calls == 0


@pytest.mark.parametrize("failure", [ProviderError("piper broke"), QuotaExceeded("no more")])
async def test_the_backup_speaks_when_the_primary_fails_before_any_audio(failure):
    tts = FallbackTTS(Voice("p", fail=failure), Voice("b"), 1.0)
    assert await collect(tts) == [b"b0", b"b1", b"b2"]


async def test_the_backup_speaks_when_the_primary_is_too_slow_to_start():
    primary, backup = Voice("p", delay=0.5), Voice("b")
    tts = FallbackTTS(primary, backup, 0.05)
    assert await collect(tts) == [b"b0", b"b1", b"b2"]


async def test_no_switch_once_audio_has_gone_out():
    """Changing voice mid-sentence would repeat the start of it."""
    primary = Voice("p", fail=ProviderError("died half-way"), fail_after=1)
    backup = Voice("b")
    heard = []
    with pytest.raises(ProviderError, match="half-way"):
        async for piece in FallbackTTS(primary, backup, 1.0).synthesize("Hello.", Lang.EN):
            heard.append(piece)
    assert heard == [b"p0"]
    assert backup.calls == 0


async def test_no_fallback_to_a_voice_with_another_sample_rate_or_without_the_language():
    """The rate is announced before the first piece, so it cannot change afterwards."""
    failing = Voice("p", fail=ProviderError("piper broke"))
    for backup in (Voice("b", rate=24_000), Voice("b", langs={Lang.KN})):
        with pytest.raises(ProviderError, match="piper broke"):
            await collect(FallbackTTS(failing, backup, 1.0))
        assert backup.calls == 0


async def test_both_failing_is_the_backups_error():
    tts = FallbackTTS(
        Voice("p", fail=ProviderError("piper broke")), Voice("b", fail=QuotaExceeded("cap")), 1.0
    )
    with pytest.raises(QuotaExceeded, match="cap"):
        await collect(tts)


async def test_cancelling_during_the_wait_for_the_primary_does_not_start_the_backup():
    primary, backup = Voice("p", delay=5.0), Voice("b")
    task = asyncio.create_task(collect(FallbackTTS(primary, backup, 10.0)))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert backup.calls == 0


def test_the_piper_route_has_sarvam_behind_it_only_when_there_is_a_key(settings_env, monkeypatch):
    settings_env(MOCK="vad,stt,langid,llm,agent,trace_sink")
    app.tts.piper_with_fallback.cache_clear()
    assert isinstance(registry.get_tts(Lang.EN), PiperTTS)

    monkeypatch.setenv("MULTIVOCO_SARVAM_API_KEY", "k")
    app.tts.get_tts_settings.cache_clear()
    app.tts.piper_with_fallback.cache_clear()
    try:
        assert isinstance(registry.get_tts(Lang.HI), FallbackTTS)
    finally:
        monkeypatch.undo()
        app.tts.get_tts_settings.cache_clear()
        app.tts.piper_with_fallback.cache_clear()
