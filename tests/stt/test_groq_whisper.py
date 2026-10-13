import asyncio
import io
import wave

import httpx2 as httpx
import pytest

from app.core.interfaces import STT, ProviderError, QuotaExceeded
from app.core.languages import Lang
from app.core.mocks import tone
from app.stt.config import STTSettings
from app.stt.groq_whisper import GroqWhisperSTT, to_wav, transcribe

from .conftest import FRAME, QUIET, collect_until_final

SETTINGS = STTSettings()


def verbose(*segments: tuple[str, float, float], language: str = "english") -> dict:
    return {
        "text": " ".join(text for text, _, _ in segments),
        "language": language,
        "segments": [
            {"text": f" {text}", "avg_logprob": logprob, "no_speech_prob": no_speech}
            for text, logprob, no_speech in segments
        ],
    }


class FakeGroq:
    def __init__(self, *replies) -> None:
        self.replies = list(replies) or [verbose(("When is my next EMI due?", -0.1, 0.01))]
        self.requests: list[httpx.Request] = []
        self.delays: list[float] = []
        self.client = httpx.AsyncClient(transport=httpx.MockTransport(self._handle))

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        index = len(self.requests)
        self.requests.append(request)
        if index < len(self.delays):
            await asyncio.sleep(self.delays[index])
        reply = self.replies[min(index, len(self.replies) - 1)]
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, int):
            return httpx.Response(reply, text="nope")
        return httpx.Response(200, json=reply)

    def field(self, name: str, request: int = 0) -> str | None:
        """A form field of a multipart request, or None if it was not sent."""
        marker = f'name="{name}"\r\n\r\n'.encode()
        body = self.requests[request].content
        if marker not in body:
            return None
        return body.split(marker, 1)[1].split(b"\r\n", 1)[0].decode()


def make(groq: FakeGroq) -> GroqWhisperSTT:
    return GroqWhisperSTT("test-key", SETTINGS, groq.client)


def test_it_is_an_stt():
    assert isinstance(GroqWhisperSTT("k"), STT)


def test_wav_is_16k_mono_16_bit():
    pcm = tone(100)
    with wave.open(io.BytesIO(to_wav(pcm)), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
        assert wav.readframes(wav.getnframes()) == pcm


async def test_one_request_per_utterance_and_no_partials():
    groq = FakeGroq()
    stt = make(groq)
    await stt.start(Lang.KN)
    for _ in range(5):
        await stt.send_audio(FRAME)
    assert groq.requests == []  # nothing is sent until the caller stops
    await stt.finish()
    transcripts = await collect_until_final(stt)

    assert [(t.text, t.final, t.language) for t in transcripts] == [
        ("When is my next EMI due?", True, Lang.KN)
    ]
    assert transcripts[0].confidence == pytest.approx(0.905, abs=0.001)  # exp(-0.1)
    request = groq.requests[0]
    assert request.headers["Authorization"] == "Bearer test-key"
    assert groq.field("model") == SETTINGS.groq_model
    assert groq.field("language") == "kn"
    assert groq.field("response_format") == "verbose_json"
    assert to_wav(FRAME * 5) in request.content
    await stt.close()


async def test_segments_whisper_thinks_are_silence_are_dropped():
    groq = FakeGroq(verbose(("Pay now.", -0.2, 0.05), ("Thank you.", -0.9, 0.93)))
    stt = make(groq)
    await stt.start(Lang.EN)
    await stt.send_audio(FRAME)
    await stt.finish()
    assert (await collect_until_final(stt))[-1].text == "Pay now."

    groq.replies = [verbose(("Thank you.", -0.9, 0.93))]
    await stt.send_audio(FRAME)
    await stt.finish()
    assert (await collect_until_final(stt))[-1].text == ""
    await stt.close()


async def test_no_audio_or_silent_audio_makes_no_request():
    groq = FakeGroq()
    stt = make(groq)
    await stt.start(Lang.EN)
    await stt.finish()
    assert [(t.text, t.final) for t in await collect_until_final(stt)] == [("", True)]
    for _ in range(10):
        await stt.send_audio(QUIET)  # Whisper would answer "Thank you." to this
    await stt.finish()
    assert [(t.text, t.final) for t in await collect_until_final(stt)] == [("", True)]
    assert groq.requests == []
    await stt.close()


async def test_finals_keep_the_order_of_the_utterances():
    groq = FakeGroq(verbose(("first", -0.1, 0.0)), verbose(("second", -0.1, 0.0)))
    groq.delays = [0.15, 0.0]  # the first answer comes back last
    stt = make(groq)
    await stt.start(Lang.EN)
    for _ in range(2):
        await stt.send_audio(FRAME)
        await stt.finish()
    first = await collect_until_final(stt)
    second = await collect_until_final(stt)
    assert (first[-1].text, second[-1].text) == ("first", "second")
    await stt.close()


@pytest.mark.parametrize(
    "reply, error",
    [(429, QuotaExceeded), (500, ProviderError), (httpx.ConnectError("down"), ProviderError)],
)
async def test_failures_surface_through_events(reply, error):
    stt = make(FakeGroq(reply))
    await stt.start(Lang.EN)
    await stt.send_audio(FRAME)
    await stt.finish()
    with pytest.raises(error) as raised:
        await collect_until_final(stt)
    assert type(raised.value) is error
    await stt.close()


async def test_no_key_fails_at_start():
    with pytest.raises(ProviderError, match="GROQ_API_KEY"):
        await GroqWhisperSTT("").start(Lang.EN)


async def test_transcribe_without_a_language_lets_whisper_detect_it():
    groq = FakeGroq(verbose(("ನನ್ನ ಮುಂದಿನ EMI ಯಾವಾಗ?", -0.3, 0.02), language="kannada"))
    text, confidence, heard = await transcribe(
        tone(300), None, api_key="k", settings=SETTINGS, client=groq.client
    )
    assert (text, heard) == ("ನನ್ನ ಮುಂದಿನ EMI ಯಾವಾಗ?", "kannada")
    assert 0 < confidence < 1
    assert groq.field("language") is None


async def test_close_cancels_a_request_in_flight():
    groq = FakeGroq()
    groq.delays = [5.0]
    stt = make(groq)
    await stt.start(Lang.EN)
    await stt.send_audio(FRAME)
    await stt.finish()
    await asyncio.sleep(0.05)
    await asyncio.wait_for(stt.close(), 1.0)
    assert [t async for t in stt.events()] == []
