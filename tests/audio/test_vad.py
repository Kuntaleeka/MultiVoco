import asyncio
import wave
from pathlib import Path

import pytest

pytest.importorskip("onnxruntime")

from app.audio.config import VADSettings
from app.audio.vad import MODEL_PATH, SileroVAD, SpeechGate
from app.core import registry
from app.core.interfaces import FRAME_BYTES, VAD, VADEventKind
from app.core.mocks import frames, silence, tone

FIXTURES = Path(__file__).parent.parent / "fixtures"
START, END = VADEventKind.SPEECH_START, VADEventKind.SPEECH_END


def run_gate(gate: SpeechGate, probabilities: list[float]) -> list[tuple[int, VADEventKind]]:
    events = []
    for index, probability in enumerate(probabilities):
        event = gate.update(probability)
        if event:
            events.append((index, event.kind))
    return events


def gate(**overrides) -> SpeechGate:
    return SpeechGate(VADSettings(**{"end_silence_ms": 160, **overrides}))  # 5 frames


# --- The gate: probabilities in, events out ---------------------------------------


def test_speech_starts_after_two_loud_frames_and_ends_after_the_silence():
    events = run_gate(gate(), [0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert events == [(2, START), (8, END)]


def test_a_single_loud_frame_is_ignored():
    g = gate()
    assert run_gate(g, [0.9, 0.1, 0.9, 0.1, 0.9, 0.1]) == []
    assert not g.is_speech


def test_a_pause_shorter_than_the_end_silence_does_not_end_speech():
    probabilities = [0.9, 0.9] + [0.1] * 4 + [0.9] + [0.1] * 5
    assert run_gate(gate(), probabilities) == [(1, START), (11, END)]


def test_hysteresis_keeps_quiet_speech_going():
    # 0.4 is under the start threshold but over the end threshold.
    assert run_gate(gate(), [0.4] * 5) == []
    assert run_gate(gate(), [0.9, 0.9] + [0.4] * 20) == [(1, START)]


def test_is_speech_tracks_each_frame_not_the_utterance():
    g = gate()
    seen = []
    for probability in [0.9, 0.9, 0.9, 0.1, 0.1, 0.4, 0.1]:
        g.update(probability)
        seen.append(g.is_speech)
    # The trailing silence is inside the utterance but is not speech.
    assert seen == [True, True, True, False, False, True, False]


def test_it_can_detect_a_second_utterance():
    probabilities = ([0.9] * 3 + [0.0] * 5) * 2
    assert [kind for _, kind in run_gate(gate(), probabilities)] == [START, END, START, END]


def test_end_silence_is_configurable():
    quick = run_gate(gate(end_silence_ms=64), [0.9, 0.9] + [0.0] * 10)
    slow = run_gate(gate(end_silence_ms=256), [0.9, 0.9] + [0.0] * 10)
    assert quick == [(1, START), (3, END)]
    assert slow == [(1, START), (9, END)]


# --- The model ----------------------------------------------------------------------


def test_the_model_file_is_in_the_repo():
    assert MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 1_000_000


async def test_silence_and_a_pure_tone_are_not_speech():
    vad = SileroVAD()
    assert isinstance(vad, VAD)
    peaks = []
    for frame in frames(silence(500) + tone(1000) + silence(500)):
        assert await vad.process(frame) is None
        peaks.append(vad.probability)
    assert max(peaks) < 0.5
    assert not vad.is_speech


async def test_wrong_sized_frames_are_rejected():
    with pytest.raises(ValueError, match="512 samples"):
        await SileroVAD().process(bytes(FRAME_BYTES - 2))


async def test_calls_do_not_share_state():
    a, b = SileroVAD(), SileroVAD()
    for frame in frames(tone(300)):
        await a.process(frame)
    quiet = bytes(FRAME_BYTES)
    fresh = SileroVAD()
    await b.process(quiet)
    await fresh.process(quiet)
    assert b.probability == fresh.probability  # a's audio did not leak into b


async def test_inference_runs_off_the_event_loop_and_is_fast():
    vad = SileroVAD()
    await vad.process(bytes(FRAME_BYTES))  # load the model
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0)
            ticks += 1

    task = asyncio.create_task(ticker())
    loop = asyncio.get_running_loop()
    began = loop.time()
    for frame in frames(silence(32 * 100)):
        await vad.process(frame)
    per_frame_ms = (loop.time() - began) * 1000 / 100
    task.cancel()
    assert ticks >= 100  # the loop kept turning between frames
    assert per_frame_ms < 16  # at most half of real time, on any machine we run on


def test_importing_the_package_registers_it(settings_env):
    import app.audio  # noqa: F401

    settings_env(MOCK="stt,langid,llm,tts,agent,trace_sink")
    assert isinstance(registry.get_vad(), SileroVAD)


# --- Real speech, when a recording is there -----------------------------------------


def speech_fixtures() -> list[Path]:
    return sorted(FIXTURES.glob("speech_*.wav"))


@pytest.mark.skipif(not speech_fixtures(), reason="no tests/fixtures/speech_*.wav recordings")
@pytest.mark.parametrize("path", speech_fixtures(), ids=lambda p: p.name)
async def test_real_speech_is_detected_once_with_silence_around_it(path):
    with wave.open(str(path), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
        pcm = wav.readframes(wav.getnframes())
    vad = SileroVAD()
    kinds = []
    for frame in frames(silence(300) + pcm + silence(1000)):
        event = await vad.process(frame)
        if event:
            kinds.append(event.kind)
    assert kinds[0] is START and kinds[-1] is END
    assert len(kinds) % 2 == 0
