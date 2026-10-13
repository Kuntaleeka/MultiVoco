import pytest

from app.core import registry
from app.core.interfaces import STT, ProviderError, QuotaExceeded
from app.core.languages import Lang
from app.core.mocks import MockSTT
from app.stt.fallback import FallbackSTT

from .conftest import FRAME, collect_until_final


class Scripted(MockSTT):
    """A mock STT that can be told to fail at a chosen point."""

    def __init__(self, text: str, fail_on: str | None = None, error=ProviderError) -> None:
        super().__init__([text])
        self.fail_on = fail_on
        self.error = error
        self.frames: list[bytes] = []
        self.finishes = 0
        self.closed = False

    def _maybe_fail(self, where: str) -> None:
        if self.fail_on == where:
            raise self.error(f"failed in {where}")

    async def start(self, language: Lang) -> None:
        self._maybe_fail("start")
        await super().start(language)

    async def send_audio(self, frame: bytes) -> None:
        self._maybe_fail("send_audio")
        self.frames.append(frame)
        await super().send_audio(frame)

    async def finish(self) -> None:
        self._maybe_fail("finish")
        self.finishes += 1
        if self.fail_on == "events":
            self._queue.put_nowait(self.error("failed in events"))
            return
        await super().finish()

    async def events(self):
        while (item := await self._queue.get()) is not None:
            if isinstance(item, Exception):
                raise item
            yield item

    async def close(self) -> None:
        self.closed = True
        await super().close()


def pair(fail_on: str | None, second_fails: str | None = None):
    primary = Scripted("from primary", fail_on)
    secondary = Scripted("from secondary", second_fails)
    return primary, secondary, FallbackSTT(primary, lambda: secondary)


async def utterance(stt: FallbackSTT, frames: int = 3) -> str:
    for _ in range(frames):
        await stt.send_audio(FRAME)
    await stt.finish()
    return (await collect_until_final(stt))[-1].text


async def test_the_primary_is_used_while_it_works():
    primary, secondary, stt = pair(None)
    assert isinstance(stt, STT)
    await stt.start(Lang.EN)
    assert await utterance(stt) == "from primary"
    assert await utterance(stt) == "from primary"
    assert not stt.on_fallback and secondary.frames == []
    await stt.close()
    assert primary.closed


@pytest.mark.parametrize("where", ["start", "send_audio", "finish", "events"])
async def test_a_failure_anywhere_moves_the_call_to_the_secondary(where):
    primary, secondary, stt = pair(where)
    await stt.start(Lang.HI)
    assert await utterance(stt) == "from secondary"
    assert stt.on_fallback and primary.closed
    assert secondary._language is Lang.HI
    # The caller does not repeat themselves: the whole utterance was replayed.
    assert len(secondary.frames) == 3 and secondary.finishes == 1
    # And the call stays on the secondary.
    assert await utterance(stt) == "from secondary"
    assert len(secondary.frames) == 6
    await stt.close()


async def test_partials_keep_flowing_after_a_switch():
    _, _, stt = pair("send_audio")
    await stt.start(Lang.EN)
    await stt.send_audio(FRAME)
    await stt.finish()
    transcripts = await collect_until_final(stt)
    assert [t.final for t in transcripts] == [False, True]
    await stt.close()


@pytest.mark.parametrize("error", [ProviderError, QuotaExceeded])
async def test_when_both_fail_the_error_surfaces(error):
    primary = Scripted("p", "send_audio")
    secondary = Scripted("s", "start", error)
    stt = FallbackSTT(primary, lambda: secondary)
    await stt.start(Lang.EN)
    await stt.send_audio(FRAME)
    with pytest.raises(error):
        await collect_until_final(stt)
    await stt.close()


async def test_a_failure_on_the_secondary_is_not_retried():
    _, secondary, stt = pair("start")
    await stt.start(Lang.EN)
    assert await utterance(stt) == "from secondary"
    secondary.fail_on = "events"
    await stt.send_audio(FRAME)
    await stt.finish()
    with pytest.raises(ProviderError, match="events"):
        await collect_until_final(stt)
    await stt.close()


def test_routes_resolve_to_the_right_providers(settings_env):
    import app.stt  # noqa: F401  (registers the providers)
    from app.stt.deepgram import DeepgramSTT
    from app.stt.groq_whisper import GroqWhisperSTT

    settings_env(MOCK="vad,langid,llm,tts,agent,trace_sink", GROQ_API_KEY="g")
    assert isinstance(registry.get_stt(Lang.EN), FallbackSTT)
    assert isinstance(registry.get_stt(Lang.HI), FallbackSTT)
    assert isinstance(registry.get_stt(Lang.KN), GroqWhisperSTT)

    settings_env(GROQ_API_KEY="")
    assert isinstance(registry.get_stt(Lang.EN), DeepgramSTT)  # nothing to fall back on
