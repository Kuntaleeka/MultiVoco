import asyncio
import time
from collections.abc import AsyncIterator

import pytest

from app.core.config import Settings
from app.core.interfaces import (
    AgentEvent,
    CallContext,
    Done,
    Handoff,
    ProviderError,
    QuotaExceeded,
    TextDelta,
    ToolResult,
)
from app.core.languages import LANGUAGES, Lang, dominant_script
from app.core.mocks import (
    AGENT_REPLIES,
    GREETING,
    TTS_SAMPLE_RATES,
    USER_UTTERANCES,
    MockAgent,
    MockTTS,
    silence,
    tone,
)
from app.core.trace import Mark
from app.pipeline.config import ASK_LANGUAGE

FIRST_SENTENCE = "Sure, I can help with that."


def index_of(sent, predicate) -> int:
    return next(i for i, item in enumerate(sent) if isinstance(item, dict) and predicate(item))


def audio_after(sent, turn_id: int) -> int:
    """Bytes of audio sent for a turn."""
    start = index_of(sent, lambda m: m["type"] == "audio_start" and m["turn_id"] == turn_id)
    total = 0
    for item in sent[start + 1 :]:
        if isinstance(item, bytes):
            total += len(item)
        elif item["type"] in ("audio_start", "audio_end", "flush"):
            break
    return total


# --- A normal call ---------------------------------------------------------------


async def test_greeting_is_turn_zero(rig):
    await rig.start("en")
    await rig.greeted()
    sock = rig.socket
    types = [m["type"] for m in sock.messages]
    assert types[:2] == ["ready", "language"]
    assert sock.of_type("language")[0] == {
        "type": "language",
        "lang": "en",
        "source": "manual",
        "confidence": None,
    }
    assert sock.of_type("audio_start")[0] == {
        "type": "audio_start",
        "turn_id": 0,
        "sample_rate": TTS_SAMPLE_RATES[Lang.EN],
    }
    assert [m["value"] for m in sock.of_type("state")] == ["thinking", "speaking", "listening"]
    assert (await rig.reply(0))["text"] == GREETING
    assert audio_after(sock.sent, 0) > 0
    assert rig.spoken() == [GREETING]
    assert rig.trace(0).time_to_first_audio_ms is None  # nobody spoke before it


@pytest.mark.parametrize("lang", list(Lang))
async def test_one_turn_in_each_language(rig, lang):
    await rig.start(lang.value)
    await rig.greeted()
    await rig.say()
    reply = await rig.reply(1)

    assert reply["text"] == AGENT_REPLIES[lang]
    assert dominant_script(reply["text"]) is LANGUAGES[lang].script
    user = [m for m in rig.socket.of_type("transcript") if m["role"] == "user"]
    assert [(m["final"], m["turn_id"]) for m in user] == [(False, 1), (True, 1)]
    assert user[-1]["text"] == USER_UTTERANCES[lang]
    assert rig.socket.of_type("audio_start")[1]["sample_rate"] == TTS_SAMPLE_RATES[lang]
    assert rig.socket.of_type("audio_end")[-1] == {"type": "audio_end", "turn_id": 1}

    trace = rig.trace(1)
    marks = [
        trace.get(m)
        for m in (
            Mark.SPEECH_END,
            Mark.STT_FINAL,
            Mark.LLM_RAW_FIRST_TOKEN,
            Mark.LLM_FIRST_TOKEN,
            Mark.LLM_FIRST_SENTENCE,
            Mark.TTS_FIRST_CHUNK,
            Mark.AUDIO_SENT,
        )
    ]
    assert None not in marks and marks == sorted(marks)
    assert 0 <= trace.time_to_first_audio_ms < 500
    assert trace.language is lang and not trace.interrupted and not trace.language_switched
    assert trace.user_text == USER_UTTERANCES[lang] and trace.agent_text == AGENT_REPLIES[lang]
    assert trace.providers == {"stt": "mock", "llm": "mock", "tts": "mock"}
    assert rig.spoken() == [GREETING, AGENT_REPLIES[lang]]


async def test_token_sized_deltas_give_the_same_sentences(rig):
    class WordAgent(MockAgent):
        async def respond(self, user_text: str, ctx: CallContext) -> AsyncIterator[AgentEvent]:
            for word in AGENT_REPLIES[ctx.language].split(" "):
                await asyncio.sleep(0)
                yield TextDelta(word + " ")
            yield Done()

    rig.agent_class = WordAgent
    await rig.start("en")
    await rig.greeted()
    await rig.say()
    assert (await rig.reply(1))["text"] == AGENT_REPLIES[Lang.EN]
    partials = [
        m["text"]
        for m in rig.socket.of_type("transcript")
        if m["role"] == "agent" and m["turn_id"] == 1 and not m["final"]
    ]
    assert partials == [FIRST_SENTENCE, AGENT_REPLIES[Lang.EN]]


async def test_tool_calls_land_in_the_trace(rig):
    class ToolAgent(MockAgent):
        async def respond(self, user_text: str, ctx: CallContext) -> AsyncIterator[AgentEvent]:
            yield ToolResult("get_next_emi", {"loan_id": 1}, {"amount": "8450.00"}, 12.0)
            yield TextDelta("It is 8,450.00 rupees. ")
            yield Done()

    rig.agent_class = ToolAgent
    await rig.start("en")
    await rig.greeted()
    await rig.say()
    assert (await rig.reply(1))["text"] == "It is 8,450.00 rupees."
    assert [(c.name, c.result) for c in rig.trace(1).tool_calls] == [
        ("get_next_emi", {"amount": "8450.00"})
    ]


# --- Barge-in --------------------------------------------------------------------


async def speaking_second_sentence(rig) -> None:
    """Start a reply and wait until its second sentence is being sent."""
    rig.tts = MockTTS(ms_per_char=20, chunk_delay=0.01)
    await rig.start("en")
    await rig.greeted()
    await rig.say()
    first = len(FIRST_SENTENCE) * 20 * TTS_SAMPLE_RATES[Lang.EN] * 2 // 1000
    await rig.socket.wait_for(
        lambda s: len(s.of_type("audio_start")) > 1 and audio_after(s.sent, 1) > first + 12000
    )


async def test_barge_in_stops_audio_and_keeps_only_what_was_heard(rig):
    await speaking_second_sentence(rig)
    sock = rig.socket

    began = time.monotonic()
    await rig.audio(tone(400))
    await sock.wait_for(lambda s: s.of_type("flush"))
    assert time.monotonic() - began < 0.1 + 0.4  # 400 ms of frames, then under 100 ms
    assert sock.of_type("flush") == [{"type": "flush", "turn_id": 1}]

    flush_at = index_of(sock.sent, lambda m: m["type"] == "flush")
    await rig.send(type="playback_position", turn_id=1, ms_played=540)
    await rig.audio(silence(400))
    reply = await rig.reply(1)

    assert reply["text"] == FIRST_SENTENCE  # 540 ms is exactly the first sentence
    assert not any(m["type"] == "audio_end" and m["turn_id"] == 1 for m in sock.messages)
    trace = rig.trace(1)
    assert trace.interrupted and trace.ms_played == 540 and trace.agent_text == FIRST_SENTENCE

    # The interruption is answered as the next turn, and no audio leaked in between.
    assert (await rig.reply(2))["text"] == AGENT_REPLIES[Lang.EN]
    next_start = index_of(sock.sent, lambda m: m["type"] == "audio_start" and m["turn_id"] == 2)
    assert not any(isinstance(item, bytes) for item in sock.sent[flush_at:next_start])
    assert rig.spoken() == [GREETING, FIRST_SENTENCE, AGENT_REPLIES[Lang.EN]]
    assert not rig.trace(2).interrupted


async def test_barge_in_cuts_inside_a_sentence_on_a_word_boundary(rig):
    await speaking_second_sentence(rig)
    await rig.audio(tone(400))
    await rig.socket.wait_for(lambda s: s.of_type("flush"))
    await rig.send(type="playback_position", turn_id=1, ms_played=540 + 200)
    await rig.audio(silence(400))
    assert (await rig.reply(1))["text"] == f"{FIRST_SENTENCE} Your next"


async def test_no_playback_position_means_nothing_was_heard(rig):
    await speaking_second_sentence(rig)
    await rig.audio(tone(400) + silence(400))
    reply = await rig.reply(1)
    assert reply["text"] == ""
    assert rig.trace(1).ms_played == 0
    assert rig.spoken()[1] == ""


async def test_a_short_sound_does_not_interrupt(rig):
    await speaking_second_sentence(rig)
    await rig.audio(tone(120) + silence(400))  # under barge_in_min_ms
    reply = await rig.reply(1)
    assert reply["text"] == AGENT_REPLIES[Lang.EN]
    assert rig.socket.of_type("flush") == []
    assert not rig.trace(1).interrupted
    await asyncio.sleep(0.05)
    assert rig.session._next_turn_id == 2  # the sound was not answered either


async def test_interrupting_before_any_audio_needs_no_flush(rig):
    class SlowAgent(MockAgent):
        calls = 0

        async def respond(self, user_text: str, ctx: CallContext) -> AsyncIterator[AgentEvent]:
            SlowAgent.calls += 1
            if SlowAgent.calls == 1:
                await asyncio.sleep(5)
            async for event in super().respond(user_text, ctx):
                yield event

    rig.agent_class = SlowAgent
    await rig.start("en")
    await rig.greeted()
    await rig.say()
    await rig.socket.wait_for_state("thinking", count=2)
    await rig.say()  # they speak again while the agent is still thinking
    assert (await rig.reply(2))["text"] == AGENT_REPLIES[Lang.EN]
    assert rig.socket.of_type("flush") == []
    assert rig.trace(1).interrupted and rig.trace(1).agent_text == ""
    assert rig.spoken() == [GREETING, "", AGENT_REPLIES[Lang.EN]]


async def test_the_greeting_can_be_interrupted(rig):
    rig.tts = MockTTS(ms_per_char=20, chunk_delay=0.01)
    await rig.start("en")
    await rig.socket.wait_for(lambda s: s.audio_bytes() > 4000)
    await rig.audio(tone(400))
    await rig.socket.wait_for(lambda s: s.of_type("flush"))
    await rig.send(type="playback_position", turn_id=0, ms_played=100)
    await rig.audio(silence(400))
    assert (await rig.reply(1))["text"] == AGENT_REPLIES[Lang.EN]
    assert rig.trace(0).interrupted


# --- Language ----------------------------------------------------------------------


async def test_auto_detects_the_language_from_the_first_utterance(rig):
    rig.detect(Lang.KN, 0.95)
    await rig.start("auto")
    await rig.greeted()
    assert rig.socket.of_type("language") == []
    assert rig.spoken() == [GREETING]
    await rig.say()
    reply = await rig.reply(1)

    assert rig.socket.of_type("language") == [
        {"type": "language", "lang": "kn", "source": "audio", "confidence": 0.95}
    ]
    assert reply["text"] == AGENT_REPLIES[Lang.KN]
    assert rig.socket.of_type("audio_start")[1]["sample_rate"] == TTS_SAMPLE_RATES[Lang.KN]
    trace = rig.trace(1)
    assert trace.language is Lang.KN and trace.user_text == USER_UTTERANCES[Lang.KN]
    assert trace.get(Mark.SPEECH_END) <= trace.get(Mark.LANG_DETECTED) <= trace.get(Mark.STT_FINAL)
    # Detection buffers the first utterance, so it has no live partial transcript.
    user = [m for m in rig.socket.of_type("transcript") if m["role"] == "user"]
    assert [m["final"] for m in user] == [True]

    await rig.say()  # later turns stream straight to the STT
    await rig.reply(2)
    assert rig.trace(2).get(Mark.LANG_DETECTED) is None
    assert len(rig.socket.of_type("language")) == 1


@pytest.mark.parametrize(
    "guess, speech_ms", [((None, 0.0), 600), ((Lang.KN, 0.3), 600), ((Lang.KN, 0.95), 200)]
)
async def test_unsure_detection_asks_and_then_follows_the_answer(rig, guess, speech_ms):
    rig.detect(*guess)
    rig.stt_script[Lang.EN] = ["Kannada please"]
    await rig.start("auto")
    await rig.greeted()
    await rig.say(speech_ms)
    assert (await rig.reply(1))["text"] == ASK_LANGUAGE
    assert rig.socket.of_type("language") == []

    await rig.socket.wait_for_state("listening", count=2)
    await rig.say()
    reply = await rig.reply(2)  # greeted again, now that the language is known
    assert rig.socket.of_type("language") == [
        {"type": "language", "lang": "kn", "source": "asked", "confidence": None}
    ]
    assert reply["text"] == GREETING
    assert rig.session._ctx.extra["language_pending"] is False

    await rig.socket.wait_for_state("listening", count=3)
    await rig.say()
    assert (await rig.reply(3))["text"] == AGENT_REPLIES[Lang.KN]


async def test_script_mismatch_switches_after_two_turns(rig):
    rig.detect(Lang.EN, 0.95)
    rig.stt_script[Lang.EN] = [USER_UTTERANCES[Lang.KN]]  # English STT, Kannada script out
    await rig.start("auto")
    await rig.greeted()

    await rig.say()
    assert (await rig.reply(1))["text"] == AGENT_REPLIES[Lang.EN]
    assert [m["lang"] for m in rig.socket.of_type("language")] == ["en"]

    await rig.socket.wait_for_state("listening", count=2)
    await rig.say()
    assert (await rig.reply(2))["text"] == AGENT_REPLIES[Lang.KN]
    switch = rig.socket.of_type("language")[-1]
    assert (switch["lang"], switch["source"], switch["confidence"]) == ("kn", "script", 1.0)
    assert rig.trace(2).language_switched and rig.trace(2).language is Lang.KN
    assert not rig.trace(1).language_switched


async def test_a_manual_choice_is_never_switched(rig):
    rig.stt_script[Lang.EN] = [USER_UTTERANCES[Lang.KN]]
    await rig.start("en")
    await rig.greeted()
    for turn in (1, 2, 3):
        await rig.say()
        assert (await rig.reply(turn))["text"] == AGENT_REPLIES[Lang.EN]
        await rig.socket.wait_for_state("listening", count=turn + 1)
    assert [m["source"] for m in rig.socket.of_type("language")] == ["manual"]


async def test_hindi_and_english_do_not_trigger_a_switch(rig):
    rig.detect(Lang.EN, 0.95)
    rig.stt_script[Lang.EN] = [USER_UTTERANCES[Lang.HI]]
    await rig.start("auto")
    await rig.greeted()
    for turn in (1, 2, 3):
        await rig.say()
        await rig.reply(turn)
        await rig.socket.wait_for_state("listening", count=turn + 1)
    assert [m["lang"] for m in rig.socket.of_type("language")] == ["en"]


async def test_set_language_between_turns_applies_at_once(rig):
    await rig.start("auto")
    await rig.greeted()
    await rig.send(type="set_language", lang="bn")
    assert rig.socket.of_type("language")[-1]["source"] == "manual"
    await rig.say()
    assert (await rig.reply(1))["text"] == AGENT_REPLIES[Lang.BN]


async def test_set_language_during_a_reply_waits_for_the_turn_to_end(rig):
    rig.tts = MockTTS(ms_per_char=20, chunk_delay=0.005)
    await rig.start("en")
    await rig.socket.wait_for(lambda s: s.audio_bytes() > 4000)
    await rig.send(type="set_language", lang="kn")
    assert [m["lang"] for m in rig.socket.of_type("language")] == ["en"]
    await rig.reply(0)
    await rig.socket.wait_for(lambda s: len(s.of_type("language")) == 2)
    assert rig.socket.of_type("audio_start")[0]["sample_rate"] == TTS_SAMPLE_RATES[Lang.EN]
    await rig.say()
    assert (await rig.reply(1))["text"] == AGENT_REPLIES[Lang.KN]


# --- Errors and limits -------------------------------------------------------------


@pytest.mark.parametrize(
    "error, code", [(ProviderError, "provider_error"), (QuotaExceeded, "language_unavailable")]
)
async def test_a_tts_failure_abandons_the_turn_but_not_the_call(rig, error, code):
    class FlakyTTS(MockTTS):
        fail = False

        async def synthesize(self, text: str, language: Lang) -> AsyncIterator[bytes]:
            if FlakyTTS.fail and text != FIRST_SENTENCE:
                raise error("boom")
            async for chunk in super().synthesize(text, language):
                yield chunk

    rig.tts = FlakyTTS(ms_per_char=2)
    await rig.start("en")
    await rig.greeted()
    FlakyTTS.fail = True
    await rig.say()
    reply = await rig.reply(1)
    assert reply["text"] == FIRST_SENTENCE  # the sentence that did get spoken
    assert [m["code"] for m in rig.socket.of_type("error")] == [code]
    assert rig.socket.close_code is None
    assert rig.trace(1).interrupted

    FlakyTTS.fail = False
    await rig.socket.wait_for_state("listening", count=2)
    await rig.say()
    assert (await rig.reply(2))["text"] == AGENT_REPLIES[Lang.EN]


async def test_a_failing_trace_sink_does_not_break_the_call(rig):
    class BrokenSink:
        async def call_started(self, call):
            raise RuntimeError("database is down")

        turn_finished = call_ended = call_started

    rig.use("trace_sink", BrokenSink)
    await rig.start("en")
    await rig.greeted()
    await rig.say()
    assert (await rig.reply(1))["text"] == AGENT_REPLIES[Lang.EN]
    assert rig.socket.of_type("error") == []


async def test_handoff_ends_the_call_after_the_reply_is_spoken(rig):
    class HandoffAgent(MockAgent):
        async def respond(self, user_text: str, ctx: CallContext) -> AsyncIterator[AgentEvent]:
            yield TextDelta("I am connecting you to a colleague. ")
            yield Handoff("caller asked for a person")
            yield Done()

    rig.agent_class = HandoffAgent
    session = await rig.start("en")
    await rig.greeted()
    await rig.say()
    await asyncio.wait_for(session.closed.wait(), 3)
    assert rig.socket.close_code == 1000
    assert rig.socket.of_type("state")[-1]["value"] == "handoff"
    assert (await rig.reply(1))["text"] == "I am connecting you to a colleague."
    await session.teardown()
    call = rig.sink.calls[session.id]
    assert (call.outcome, call.handoff_reason) == ("handoff", "caller asked for a person")


@pytest.mark.parametrize(
    "raw",
    ["not json", '{"type": "nope"}', '{"type": "start", "lang": "fr"}', '{"type": "end"}'],
)
async def test_a_bad_first_message_closes_the_call(rig, raw):
    from app.pipeline.session import CallSession

    rig.session = CallSession(rig.socket, rig.config, rig.settings)
    await rig.session.handle_message(raw)
    assert [m["code"] for m in rig.socket.of_type("error")] == ["bad_message"]
    assert rig.socket.close_code == 4400


async def test_audio_before_start_and_wrong_sized_frames_are_rejected(rig):
    from app.pipeline.session import CallSession

    rig.session = CallSession(rig.socket, rig.config, rig.settings)
    await rig.session.handle_audio(bytes(1024))
    assert rig.socket.close_code == 4400

    other = type(rig.socket)()
    session = CallSession(other, rig.config, rig.settings)
    await session.handle_message('{"type": "start", "lang": "en"}')
    await session.handle_audio(bytes(1000))
    assert other.close_code == 4400
    assert other.of_type("error")[0]["code"] == "bad_message"
    await session.teardown()


async def test_end_closes_normally_and_records_the_call(rig):
    rig.detect(Lang.BN)
    session = await rig.start("auto")
    await rig.greeted()
    await rig.say()
    await rig.reply(1)
    await rig.send(type="end")
    assert rig.socket.close_code == 1000
    await session.teardown()
    call = rig.sink.calls[session.id]
    assert call.outcome == "completed" and call.ended_at is not None
    assert (call.requested_lang, call.final_lang) == ("auto", Lang.BN)
    assert [t.idx for t in rig.sink.turns] == [0, 1]


async def test_hanging_up_mid_reply_leaves_nothing_running(rig):
    rig.tts = MockTTS(ms_per_char=20, chunk_delay=0.01)
    session = await rig.start("en")
    await rig.greeted()
    await rig.say()
    await rig.socket.wait_for(lambda s: len(s.of_type("audio_start")) > 1)
    await session.teardown()  # the socket dropped: no "end"
    assert all(task.done() for task in session._tasks)
    assert rig.sink.calls[session.id].outcome == "abandoned"
    assert rig.trace(1).interrupted
    sent = len(rig.socket.sent)
    await asyncio.sleep(0.05)
    assert len(rig.socket.sent) == sent


async def test_turn_limit_ends_the_call(rig):
    rig.settings = Settings(max_turns=1)
    session = await rig.start("en")
    await rig.greeted()
    await rig.say()
    await rig.reply(1)
    await rig.socket.wait_for_state("listening", count=2)
    await rig.say()
    await asyncio.wait_for(session.closed.wait(), 3)
    assert rig.socket.close_code == 4408
    assert rig.socket.of_type("error")[-1]["code"] == "limit_reached"
    await session.teardown()
    assert rig.sink.calls[session.id].outcome == "limit"


@pytest.mark.parametrize("limits", [{"idle_timeout_seconds": 0}, {"max_call_seconds": 0}])
async def test_idle_and_length_limits_end_the_call(rig, limits):
    rig.settings = Settings(**limits)
    session = await rig.start("en")
    await asyncio.wait_for(session.closed.wait(), 3)
    assert rig.socket.close_code == 4408
    assert rig.socket.of_type("error")[-1]["code"] == "limit_reached"
