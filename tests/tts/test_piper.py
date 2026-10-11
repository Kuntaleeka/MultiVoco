"""The Piper wrapper. Most tests use a fake voice. The last ones need the voice files."""

import asyncio
import json
import threading
import time
from pathlib import Path

import pytest

import app.tts
from app.core import registry
from app.core.interfaces import TTS, ProviderError
from app.core.languages import Lang
from app.tts.config import TTSSettings, get_tts_settings
from app.tts.piper import PiperTTS

RATE = 16_000
REAL_VOICES = get_tts_settings().voices_dir
needs_voices = pytest.mark.skipif(
    not (REAL_VOICES / f"{get_tts_settings().piper_voice_en}.onnx").exists(),
    reason="Piper voice files are not downloaded",
)


class FakeChunk:
    def __init__(self, seconds: float) -> None:
        self.audio_int16_bytes = bytes(int(RATE * seconds) * 2)


class FakeVoice:
    """Two 'sentences' of audio per call. Records how many threads are inside it at once."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.texts: list[str] = []
        self.inside = 0
        self.most_inside = 0
        self._count = threading.Lock()

    def synthesize(self, text):
        self.texts.append(text)
        for seconds in (0.25, 0.13):
            with self._count:
                self.inside += 1
                self.most_inside = max(self.most_inside, self.inside)
            time.sleep(self.delay)
            with self._count:
                self.inside -= 1
            yield FakeChunk(seconds)


@pytest.fixture
def voices(tmp_path) -> TTSSettings:
    for name in ("en-test", "hi-test"):
        (tmp_path / f"{name}.onnx").write_bytes(b"")
        (tmp_path / f"{name}.onnx.json").write_text(json.dumps({"audio": {"sample_rate": RATE}}))
    return TTSSettings(
        voices_dir=tmp_path, piper_voice_en="en-test", piper_voice_hi="hi-test", _env_file=None
    )


def make_tts(settings: TTSSettings, voice: FakeVoice | None = None):
    voice = voice or FakeVoice()
    loaded: list[Path] = []

    def loader(model: Path):
        loaded.append(model)
        return voice

    return PiperTTS(settings, loader), voice, loaded


async def collect(tts, text="Hello there.", lang=Lang.EN) -> list[bytes]:
    return [piece async for piece in tts.synthesize(text, lang)]


async def test_audio_comes_out_in_short_pieces_of_pcm16(voices):
    tts, voice, _ = make_tts(voices)
    pieces = await collect(tts, "Your EMI is 8450.00 rupees.")

    assert isinstance(tts, TTS)
    assert len(b"".join(pieces)) == int(RATE * 0.25) * 2 + int(RATE * 0.13) * 2
    assert max(len(piece) for piece in pieces) == RATE // 10 * 2  # 100 ms
    assert all(len(piece) % 2 == 0 for piece in pieces)
    assert voice.texts == ["Your E M I is 8450 rupees."]  # normalised before speaking


async def test_the_sample_rate_is_known_without_loading_the_model(voices):
    tts, _, loaded = make_tts(voices)
    assert tts.sample_rate(Lang.EN) == RATE
    assert loaded == []


async def test_each_voice_is_loaded_once_and_only_when_needed(voices):
    tts, _, loaded = make_tts(voices)
    await asyncio.gather(*(collect(tts) for _ in range(4)))
    assert [model.name for model in loaded] == ["en-test.onnx"]
    await collect(tts, "नमस्ते।", Lang.HI)
    assert [model.name for model in loaded] == ["en-test.onnx", "hi-test.onnx"]


async def test_text_with_nothing_to_say_produces_no_audio(voices):
    tts, voice, loaded = make_tts(voices)
    assert await collect(tts, " *** ") == []
    assert voice.texts == [] and loaded == []


async def test_concurrent_calls_never_run_inside_piper_together(voices):
    """Piper's phonemiser is not thread-safe, and the shared TTS is called concurrently."""
    tts, voice, _ = make_tts(voices, FakeVoice(delay=0.02))
    results = await asyncio.gather(*(collect(tts, f"Sentence {i}.") for i in range(4)))
    assert voice.most_inside == 1
    assert len({len(b"".join(pieces)) for pieces in results}) == 1  # every call got its audio


async def test_cancelling_one_call_stops_it_and_leaves_the_others_alone(voices):
    tts, _, _ = make_tts(voices, FakeVoice(delay=0.05))
    heard = 0

    async def speak():
        nonlocal heard
        async for _ in tts.synthesize("Stop me.", Lang.EN):
            heard += 1

    doomed = asyncio.create_task(speak())
    other = asyncio.create_task(collect(tts, "Let me finish."))
    await asyncio.sleep(0.07)
    doomed.cancel()
    with pytest.raises(asyncio.CancelledError):
        await doomed
    assert heard < 5  # 5 pieces would be the whole reply
    assert len(b"".join(await other)) == int(RATE * 0.25) * 2 + int(RATE * 0.13) * 2
    # The lock was released: the next call goes straight through.
    assert await asyncio.wait_for(collect(tts), timeout=2)


async def test_a_language_without_a_voice_is_a_provider_error(voices):
    tts, _, _ = make_tts(voices)
    with pytest.raises(ProviderError, match="no voice for kn"):
        await collect(tts, "ನಮಸ್ಕಾರ.", Lang.KN)


async def test_a_missing_voice_file_is_a_provider_error_that_says_how_to_fix_it(voices):
    (voices.voices_dir / "hi-test.onnx").unlink()
    tts, _, _ = make_tts(voices)
    with pytest.raises(ProviderError, match="download_voices hi-test"):
        await collect(tts, "नमस्ते।", Lang.HI)
    # English still works.
    assert await collect(tts)


async def test_a_failure_inside_piper_is_a_provider_error(voices):
    class Broken:
        def synthesize(self, text):
            raise RuntimeError("onnx blew up")
            yield

    tts = PiperTTS(voices, lambda model: Broken())
    with pytest.raises(ProviderError, match="synthesis failed"):
        await collect(tts)


async def test_warm_up_loads_what_it_can_and_never_raises(voices):
    (voices.voices_dir / "hi-test.onnx").unlink()
    tts, voice, loaded = make_tts(voices)
    await tts.warm_up()
    assert [model.name for model in loaded] == ["en-test.onnx"]
    assert voice.texts == ["Hello."]


def test_the_registry_hands_out_one_shared_piper_for_english_and_hindi(settings_env):
    settings_env(MOCK="vad,stt,langid,llm,agent,trace_sink")
    assert registry.get_tts(Lang.EN) is registry.get_tts(Lang.HI) is app.tts.piper()


# --- With the real voices ----------------------------------------------------------


# Loading a real voice takes about 6 s, so this runs only with `pytest -m live`.
@pytest.mark.live
@needs_voices
@pytest.mark.parametrize(
    ("lang", "text"),
    [
        (Lang.EN, "Your next EMI of 8450.00 rupees is due on 5 November 2026."),
        (Lang.HI, "आपकी अगली EMI 8450.00 रुपये की है।"),
    ],
)
async def test_real_piper_speaks(lang, text):
    tts = PiperTTS()
    pcm = b"".join([piece async for piece in tts.synthesize(text, lang)])
    seconds = len(pcm) / 2 / tts.sample_rate(lang)
    assert 1.5 < seconds < 12
    assert any(pcm)  # not silence
