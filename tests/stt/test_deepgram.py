import asyncio
import http

import pytest

from app.core.interfaces import STT, ProviderError, QuotaExceeded
from app.core.languages import Lang
from app.stt.deepgram import DeepgramSTT

from .conftest import FRAME, collect_until_final


async def speak(stt: DeepgramSTT, frames: int = 6) -> None:
    for _ in range(frames):
        await stt.send_audio(FRAME)
        await asyncio.sleep(0.01)
    await stt.finish()


@pytest.fixture
async def stt(stt_settings):
    stt = DeepgramSTT("test-key", stt_settings)
    yield stt
    await stt.close()


def test_it_is_an_stt(stt_settings):
    assert isinstance(DeepgramSTT("k", stt_settings), STT)


@pytest.mark.parametrize("lang, expected", [("en", "multi"), ("hi", "multi"), ("kn", "kn")])
async def test_connects_with_the_key_and_the_right_language(deepgram, stt, lang, expected):
    await stt.start(Lang(lang))
    assert deepgram.authorization == ["Token test-key"]
    query = deepgram.queries[0]
    assert query["language"] == expected
    assert (query["model"], query["encoding"], query["sample_rate"]) == (
        "nova-3",
        "linear16",
        "16000",
    )
    assert query["interim_results"] == "true"
    assert query["endpointing"] == "false"  # the default loses short opening words


async def test_segments_are_joined_into_one_final_per_utterance(deepgram, stt):
    await stt.start(Lang.EN)
    await speak(stt)
    transcripts = await collect_until_final(stt)

    assert [(t.text, t.final) for t in transcripts] == [
        ("When", False),  # an interim result
        ("When is my", False),  # Deepgram finalised a segment: still a partial for us
        ("When is my next EMI due?", True),  # the answer to Finalize closes the utterance
    ]
    assert all(t.language is Lang.EN for t in transcripts)
    assert transcripts[-1].confidence == pytest.approx(0.8)  # mean of 0.9 and 0.7
    assert deepgram.control == ["Finalize"]


async def test_the_next_utterance_starts_clean_on_the_same_connection(deepgram, stt):
    deepgram.utterances = [("When is my", "next EMI due?"), ("Thank", "you.")]
    await stt.start(Lang.EN)
    await speak(stt)
    await collect_until_final(stt)
    await speak(stt, frames=2)
    second = await collect_until_final(stt)
    assert second[-1].text == "you."  # only two frames: the first segment never finalised
    assert deepgram.connections == 1


async def test_finish_without_audio_gives_an_empty_final(deepgram, stt):
    await stt.start(Lang.EN)
    await stt.finish()
    transcripts = await collect_until_final(stt)
    assert [(t.text, t.final) for t in transcripts] == [("", True)]


async def test_an_unanswered_finalize_times_out_with_what_arrived(deepgram, stt_settings):
    deepgram.ignore_finalize = True
    stt = DeepgramSTT("k", stt_settings.model_copy(update={"finalize_timeout_seconds": 0.1}))
    await stt.start(Lang.EN)
    await speak(stt)
    transcripts = await collect_until_final(stt)
    assert transcripts[-1].text == "When is my"
    await stt.close()


async def test_keepalive_is_sent_while_nobody_speaks(deepgram, stt_settings):
    stt = DeepgramSTT("k", stt_settings.model_copy(update={"deepgram_keepalive_seconds": 0.1}))
    await stt.start(Lang.EN)
    await asyncio.sleep(0.4)
    assert deepgram.control.count("KeepAlive") >= 2
    await stt.close()


async def test_a_dropped_connection_is_reopened_on_the_next_audio(deepgram, stt):
    deepgram.drop_after_frames = 2
    await stt.start(Lang.EN)
    await stt.send_audio(FRAME)
    await stt.send_audio(FRAME)
    await asyncio.sleep(0.1)  # the server hangs up
    await speak(stt)
    transcripts = await collect_until_final(stt)
    assert deepgram.connections == 2
    assert transcripts[-1].final and transcripts[-1].text.endswith("next EMI due?")


async def test_a_drop_while_waiting_for_the_final_still_gives_one(deepgram, stt):
    deepgram.ignore_finalize = True
    deepgram.drop_after_frames = 5
    await stt.start(Lang.EN)
    for _ in range(4):
        await stt.send_audio(FRAME)
    await asyncio.sleep(0.05)
    await stt.finish()
    await stt.send_audio(FRAME)  # the fifth frame makes the server hang up
    transcripts = await collect_until_final(stt)
    assert transcripts[-1].text == "When is my"


@pytest.mark.parametrize(
    "status, error",
    [
        (http.HTTPStatus.PAYMENT_REQUIRED, QuotaExceeded),
        (http.HTTPStatus.TOO_MANY_REQUESTS, QuotaExceeded),
        (http.HTTPStatus.UNAUTHORIZED, ProviderError),
    ],
)
async def test_a_refused_connection_raises_the_right_error(deepgram, stt, status, error):
    deepgram.reject_with = status
    with pytest.raises(error) as raised:
        await stt.start(Lang.EN)
    assert type(raised.value) is error


async def test_no_key_and_no_server_are_provider_errors(stt_settings):
    with pytest.raises(ProviderError, match="DEEPGRAM_API_KEY"):
        await DeepgramSTT("", stt_settings).start(Lang.EN)
    nowhere = stt_settings.model_copy(update={"deepgram_url": "ws://127.0.0.1:9/v1/listen"})
    with pytest.raises(ProviderError, match="Could not reach"):
        await DeepgramSTT("k", nowhere).start(Lang.EN)


async def test_close_ends_events_and_tells_deepgram(deepgram, stt_settings):
    stt = DeepgramSTT("k", stt_settings)
    await stt.start(Lang.EN)
    await stt.close()
    assert [t async for t in stt.events()] == []
    await asyncio.sleep(0.05)
    assert deepgram.control == ["CloseStream"]
    assert not [t for t in stt._tasks if not t.done()]


async def test_it_drives_a_whole_turn_in_the_pipeline(deepgram, stt_settings, monkeypatch):
    """The orchestrator, with the mock VAD, on the real Deepgram client."""
    from tests.pipeline.conftest import Rig

    rig = Rig(monkeypatch)
    rig.use("stt", lambda lang: DeepgramSTT("k", stt_settings))
    await rig.start("en")
    await rig.greeted()
    await rig.say()
    await rig.reply(1)
    user = [m for m in rig.socket.of_type("transcript") if m["role"] == "user"]
    assert user[0]["final"] is False and user[0]["text"].startswith("When")
    assert (user[-1]["final"], user[-1]["text"]) == (True, "When is my next EMI due?")
    assert rig.trace(1).user_text == "When is my next EMI due?"
    await rig.session.teardown()


async def test_a_stalled_connection_fails_instead_of_blocking_the_caller(deepgram, stt_settings):
    """If Deepgram stops taking audio, send_audio must not wait: the caller's audio loop
    runs through it. The queue fills, and that is reported as a failure."""
    stt = DeepgramSTT("k", stt_settings.model_copy(update={"deepgram_max_queued_frames": 8}))
    await stt.start(Lang.EN)
    stalled = asyncio.Event()

    async def never_sends(data) -> None:
        await stalled.wait()

    stt._send = never_sends
    began = asyncio.get_running_loop().time()
    with pytest.raises(ProviderError, match="not taking audio"):
        for _ in range(20):
            await stt.send_audio(FRAME)
    assert asyncio.get_running_loop().time() - began < 0.5
    with pytest.raises(ProviderError):  # and it stays failed
        await stt.send_audio(FRAME)
    stalled.set()
    await stt.close()


async def test_strict_mode_raises_when_a_flush_goes_unanswered(deepgram, stt_settings):
    deepgram.ignore_finalize = True
    settings = stt_settings.model_copy(update={"finalize_timeout_seconds": 0.1})
    stt = DeepgramSTT("k", settings, strict=True)
    await stt.start(Lang.EN)
    await speak(stt)
    with pytest.raises(ProviderError, match="flush"):
        await collect_until_final(stt)
    await stt.close()


async def test_behind_a_fallback_an_unanswered_flush_is_redone_by_the_other_stt(
    deepgram, stt_settings
):
    from app.core.mocks import MockSTT
    from app.stt.fallback import FallbackSTT

    deepgram.ignore_finalize = True
    settings = stt_settings.model_copy(update={"finalize_timeout_seconds": 0.1})
    whisper = MockSTT(["When is my next EMI due?"])
    stt = FallbackSTT(DeepgramSTT("k", settings, strict=True), lambda: whisper)
    await stt.start(Lang.EN)
    await speak(stt)
    transcripts = await collect_until_final(stt)
    assert transcripts[-1].text == "When is my next EMI due?"  # whole, not "When is my"
    assert stt.on_fallback
    await stt.close()


async def test_close_does_not_wait_on_a_peer_that_will_not_hang_up(deepgram, stt_settings):
    stt = DeepgramSTT("k", stt_settings.model_copy(update={"close_timeout_seconds": 0.1}))
    await stt.start(Lang.EN)

    async def never(*args, **kwargs) -> None:
        await asyncio.Event().wait()

    stt._ws.close = never
    await asyncio.wait_for(stt.close(), 1.0)
