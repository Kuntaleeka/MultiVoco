import json

from app.core.languages import Lang
from app.core.trace import Mark, SessionClock, ToolCallTrace, TurnTrace


def make_trace() -> TurnTrace:
    trace = TurnTrace(call_id="c1", idx=0, language=Lang.KN)
    for mark, at in [
        (Mark.SPEECH_END, 1000.0),
        (Mark.STT_FINAL, 1150.0),
        (Mark.LLM_FIRST_TOKEN, 1400.0),
        (Mark.LLM_FIRST_SENTENCE, 1500.0),
        (Mark.TTS_FIRST_CHUNK, 1620.0),
        (Mark.AUDIO_SENT, 1625.0),
    ]:
        trace.mark(mark, at)
    return trace


def test_time_to_first_audio_is_audio_sent_minus_speech_end():
    assert make_trace().time_to_first_audio_ms == 625.0


def test_stages_add_up_to_time_to_first_audio():
    trace = make_trace()
    stages = trace.stage_durations()
    assert stages == {"stt": 150.0, "llm": 250.0, "sentence": 100.0, "tts": 120.0, "send": 5.0}
    assert sum(stages.values()) == trace.time_to_first_audio_ms


def test_first_mark_wins():
    trace = make_trace()
    trace.mark(Mark.AUDIO_SENT, 9999.0)
    assert trace.get(Mark.AUDIO_SENT) == 1625.0


def test_incomplete_trace_has_no_headline_metric():
    trace = TurnTrace(call_id="c1", idx=0, language=Lang.EN)
    trace.mark(Mark.SPEECH_END, 10.0)
    assert trace.time_to_first_audio_ms is None
    assert trace.stage_durations() == {}


def test_to_dict_is_json_serialisable():
    trace = make_trace()
    trace.tool_calls.append(ToolCallTrace("get_next_emi", {"loan_id": 1}, {"amount": "5000"}, 12.5))
    data = json.loads(json.dumps(trace.to_dict()))
    assert data["language"] == "kn"
    assert data["time_to_first_audio_ms"] == 625.0
    assert data["tool_calls"][0]["name"] == "get_next_emi"


def test_session_clock_is_monotonic_from_zero():
    clock = SessionClock()
    first, second = clock.now_ms(), clock.now_ms()
    assert 0 <= first <= second < 1000
