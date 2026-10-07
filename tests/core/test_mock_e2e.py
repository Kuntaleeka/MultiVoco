"""The shared definition of "not broken": one turn through every contract, on mocks.

This must pass at all times, on a fresh clone, with no API keys. The loop here is a
reference, not the product: the real one lives in app/pipeline/.
"""

import asyncio
from datetime import UTC, datetime

import pytest

from app.core import registry
from app.core.interfaces import CallContext, Done, LangGuess, TextDelta, VADEventKind
from app.core.languages import AUTO, LANGUAGES, Lang, dominant_script
from app.core.mocks import (
    AGENT_REPLIES,
    USER_UTTERANCES,
    InMemoryTraceSink,
    MockLanguageDetector,
    MockTTS,
    frames,
    silence,
    tone,
)
from app.core.trace import CallRecord, Mark, SessionClock, TurnTrace

ALL_MARKS = [
    Mark.SPEECH_END,
    Mark.STT_FINAL,
    Mark.LLM_FIRST_TOKEN,
    Mark.LLM_FIRST_SENTENCE,
    Mark.TTS_FIRST_CHUNK,
    Mark.AUDIO_SENT,
]


async def run_turn(requested: str, detected: Lang = Lang.EN):
    clock = SessionClock()
    sink = InMemoryTraceSink()
    call = CallRecord("call-1", datetime.now(UTC), requested_lang=requested)
    await sink.call_started(call)

    # 1. Caller speaks: 600 ms of tone, then silence. VAD finds the utterance.
    vad = registry.get_vad()
    utterance: list[bytes] = []
    events = []
    for frame in frames(silence(100) + tone(600) + silence(500)):
        event = await vad.process(frame)
        if event:
            events.append(event.kind)
        if events == [VADEventKind.SPEECH_START]:
            utterance.append(frame)
    assert events == [VADEventKind.SPEECH_START, VADEventKind.SPEECH_END]

    # 2. Language: given by the client, or detected from the first utterance.
    lang_marks = {}
    if requested == AUTO:
        detector = MockLanguageDetector(LangGuess(detected, 0.95))
        guess = await detector.detect_audio(b"".join(utterance))
        assert guess.lang is not None
        lang = guess.lang
        lang_marks[Mark.LANG_DETECTED] = clock.now_ms()
    else:
        lang = Lang(requested)

    ctx = CallContext(call_id=call.call_id, language=lang)
    trace = TurnTrace(call_id=call.call_id, idx=0, language=lang, region="test")
    trace.providers = {kind: registry.provider_name(kind, lang) for kind in registry.ROUTED_KINDS}
    trace.mark(Mark.SPEECH_END, 0.0)
    for mark, at in lang_marks.items():
        trace.mark(mark, at)

    # 3. STT.
    stt = registry.get_stt(lang)
    await stt.start(lang)
    for frame in utterance:
        await stt.send_audio(frame)
    await stt.finish()
    transcripts = []
    async for transcript in stt.events():
        transcripts.append(transcript)
        if transcript.final:
            break
    await stt.close()
    trace.mark(Mark.STT_FINAL, clock.now_ms())
    trace.user_text = transcripts[-1].text

    # 4. Agent, streamed. 5. TTS. 6. "Socket".
    agent = registry.get_agent(ctx)
    tts = registry.get_tts(lang)
    reply = ""
    audio = bytearray()
    finished = False
    async for event in agent.respond(trace.user_text, ctx):
        if isinstance(event, TextDelta):
            trace.mark(Mark.LLM_FIRST_TOKEN, clock.now_ms())
            reply += event.text
        elif isinstance(event, Done):
            finished = True
    assert finished
    trace.mark(Mark.LLM_FIRST_SENTENCE, clock.now_ms())
    async for chunk in tts.synthesize(reply.strip(), lang):
        trace.mark(Mark.TTS_FIRST_CHUNK, clock.now_ms())
        audio += chunk
        trace.mark(Mark.AUDIO_SENT, clock.now_ms())

    agent.commit_spoken(reply.strip(), interrupted=False)
    trace.agent_text = reply.strip()
    await sink.turn_finished(trace)
    call.final_lang = lang
    call.ended_at = datetime.now(UTC)
    await sink.call_ended(call)
    return trace, transcripts, bytes(audio), tts.sample_rate(lang), sink


def check_turn(trace, transcripts, audio, sample_rate, sink, lang):
    assert [t.final for t in transcripts] == [False, True]
    assert all(t.language is lang for t in transcripts)
    assert trace.user_text == USER_UTTERANCES[lang]

    assert trace.agent_text == AGENT_REPLIES[lang]
    assert dominant_script(trace.agent_text) is LANGUAGES[lang].script

    assert len(audio) % 2 == 0
    assert len(audio) / 2 / sample_rate > 0.5  # more than half a second of speech

    times = [trace.get(mark) for mark in ALL_MARKS]
    assert None not in times
    assert times == sorted(times)
    assert trace.time_to_first_audio_ms >= 0
    assert set(trace.stage_durations()) == {"stt", "llm", "sentence", "tts", "send"}

    assert sink.turns == [trace]
    assert sink.calls["call-1"].final_lang is lang
    assert trace.to_dict()["providers"] == {"stt": "mock", "llm": "mock", "tts": "mock"}


@pytest.mark.parametrize("lang", list(Lang))
async def test_one_turn_in_each_language(lang):
    trace, transcripts, audio, rate, sink = await run_turn(lang.value)
    check_turn(trace, transcripts, audio, rate, sink, lang)
    assert trace.get(Mark.LANG_DETECTED) is None


@pytest.mark.parametrize("detected", [Lang.KN, Lang.BN, Lang.HI])
async def test_one_turn_with_auto_detection(detected):
    trace, transcripts, audio, rate, sink = await run_turn(AUTO, detected=detected)
    check_turn(trace, transcripts, audio, rate, sink, detected)
    assert trace.get(Mark.LANG_DETECTED) is not None
    assert sink.calls["call-1"].requested_lang == AUTO


async def test_script_check_catches_a_language_switch():
    detector = registry.get_language_detector()
    assert detector.detect_text(USER_UTTERANCES[Lang.KN]).lang is Lang.KN
    assert detector.detect_text(USER_UTTERANCES[Lang.EN]) == LangGuess(None, 0.0)


async def test_tts_stops_promptly_when_cancelled():
    tts = MockTTS(chunk_delay=0.01)
    chunks = 0

    async def speak():
        nonlocal chunks
        async for _ in tts.synthesize("x" * 500, Lang.EN):  # 10 s of audio, 250 chunks
            chunks += 1

    task = asyncio.create_task(speak())
    await asyncio.sleep(0.05)
    task.cancel()
    results = await asyncio.gather(task, return_exceptions=True)
    assert isinstance(results[0], asyncio.CancelledError)
    assert 0 < chunks < 20


async def test_agent_history_keeps_only_what_was_heard():
    ctx = CallContext("call-1", Lang.EN)
    agent = registry.get_agent(ctx)
    async for _ in agent.respond("hello", ctx):
        pass
    agent.commit_spoken("Sure, I can", interrupted=True)
    assert [m.content for m in agent.history] == ["hello", "Sure, I can"]
