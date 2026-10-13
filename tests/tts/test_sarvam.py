"""The Sarvam client against a fake HTTP transport. No network, no key."""

import json
from datetime import date

import httpx2 as httpx
import pytest

import app.tts
from app.core import registry
from app.core.interfaces import TTS, ProviderError, QuotaExceeded
from app.core.languages import Lang
from app.tts.config import TTSSettings
from app.tts.sarvam import URL, SarvamTTS

KEY = "test-key-not-real"
RATE = 16_000
PCM = bytes(range(256)) * 50  # 12 800 bytes: 0.4 s at 16 kHz


class Server:
    def __init__(self, *responses) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def tts(self, today=date.today, **overrides) -> SarvamTTS:
        settings = TTSSettings(
            sarvam_api_key=KEY, sarvam_sample_rate=RATE, _env_file=None, **overrides
        )
        client = httpx.AsyncClient(transport=httpx.MockTransport(self))
        return SarvamTTS(settings, client, today)


def audio(content: bytes = PCM) -> httpx.Response:
    return httpx.Response(200, content=content)


def failure(status: int, code: str = "", message: str = "nope") -> httpx.Response:
    return httpx.Response(status, json={"error": {"code": code, "message": message}})


async def collect(tts, text="ನಿಮ್ಮ ಮುಂದಿನ EMI 8450.00 ರೂ.", lang=Lang.KN) -> list[bytes]:
    return [piece async for piece in tts.synthesize(text, lang)]


async def test_kannada_is_sent_and_pcm_comes_back_in_short_pieces():
    server = Server(audio())
    tts = server.tts()
    pieces = await collect(tts)

    assert isinstance(tts, TTS)
    assert b"".join(pieces) == PCM
    assert max(len(piece) for piece in pieces) == RATE // 10 * 2  # 100 ms
    assert tts.sample_rate(Lang.KN) == RATE

    request = server.requests[0]
    assert (request.method, str(request.url)) == ("POST", URL)
    assert request.headers["api-subscription-key"] == KEY
    assert json.loads(request.content) == {
        "text": "ನಿಮ್ಮ ಮುಂದಿನ EMI 8450 ರೂ.",
        "language_code": "kn-IN",
        "model": "bulbul:v3",
        "speaker": "priya",
        "speech_sample_rate": RATE,
        "output_audio_codec": "linear16",
        "enable_preprocessing": True,
    }


@pytest.mark.parametrize(
    ("lang", "code"),
    [(Lang.BN, "bn-IN"), (Lang.HI, "hi-IN"), (Lang.EN, "en-IN"), (Lang.KN, "kn-IN")],
)
async def test_each_language_gets_its_sarvam_code(lang, code):
    server = Server(audio())
    await collect(server.tts(), "নমস্কার।", lang)
    assert json.loads(server.requests[0].content)["language_code"] == code


async def test_a_wav_header_is_not_played_as_audio():
    header = b"RIFF" + bytes(4) + b"WAVEfmt " + bytes(20) + b"data" + bytes(4)
    server = Server(audio(header + PCM))
    assert b"".join(await collect(server.tts())) == PCM


async def test_half_a_sample_at_the_end_is_dropped():
    server = Server(audio(PCM + b"\x01"))
    pieces = await collect(server.tts())
    assert b"".join(pieces) == PCM
    assert all(len(piece) % 2 == 0 for piece in pieces)


async def test_text_with_nothing_to_say_makes_no_request():
    server = Server()
    assert await collect(server.tts(), " *** ") == []
    assert server.requests == []


# --- Failures ----------------------------------------------------------------------


@pytest.mark.parametrize("problem", [failure(503), httpx.ReadTimeout("slow")])
async def test_a_passing_failure_is_retried_once(problem):
    server = Server(problem, audio())
    assert b"".join(await collect(server.tts())) == PCM
    assert len(server.requests) == 2


async def test_two_failures_are_a_provider_error():
    server = Server(failure(503), failure(503))
    with pytest.raises(ProviderError, match="sarvam: HTTP 503") as raised:
        await collect(server.tts())
    assert not isinstance(raised.value, QuotaExceeded)


@pytest.mark.parametrize(
    "response",
    [
        failure(429, "rate_limit_exceeded_error"),
        failure(403, "insufficient_quota_error"),
        failure(429),
    ],
)
async def test_running_out_is_quota_exceeded_and_is_not_retried(response):
    server = Server(response)
    with pytest.raises(QuotaExceeded):
        await collect(server.tts())
    assert len(server.requests) == 1


async def test_a_rejected_key_is_a_provider_error_that_does_not_leak_it():
    server = Server(failure(403, "invalid_api_key_error", "bad key"))
    with pytest.raises(ProviderError) as raised:
        await collect(server.tts())
    assert not isinstance(raised.value, QuotaExceeded)
    assert KEY not in str(raised.value)
    assert len(server.requests) == 1


async def test_no_key_is_a_provider_error_and_no_request():
    server = Server()
    tts = SarvamTTS(TTSSettings(sarvam_api_key=None, _env_file=None), None)
    with pytest.raises(ProviderError, match="MULTIVOCO_SARVAM_API_KEY"):
        await collect(tts)
    assert server.requests == []


# --- The daily character cap -------------------------------------------------------


async def test_the_daily_cap_stops_requests_before_the_credit_is_spent():
    server = Server(audio(), audio())
    tts = server.tts(sarvam_daily_chars=25)
    await collect(tts, "ಹತ್ತು ಅಕ್ಷರ.")  # 12 characters
    await collect(tts, "ಹತ್ತು ಅಕ್ಷರ.")  # 24 in total
    assert tts.chars_today == 24
    with pytest.raises(QuotaExceeded, match="daily cap of 25"):
        await collect(tts, "ಹತ್ತು ಅಕ್ಷರ.")
    assert len(server.requests) == 2


async def test_the_cap_starts_again_the_next_day():
    day = [date(2026, 10, 13)]
    server = Server(audio(), audio())
    tts = server.tts(today=lambda: day[0], sarvam_daily_chars=15)
    await collect(tts, "ಹತ್ತು ಅಕ್ಷರ.")
    day[0] = date(2026, 10, 14)
    await collect(tts, "ಹತ್ತು ಅಕ್ಷರ.")
    assert tts.chars_today == 12


def test_the_registry_reaches_sarvam_through_a_route_override(settings_env):
    settings_env(
        MOCK="vad,stt,langid,llm,agent,trace_sink",
        ROUTE_OVERRIDES='{"kn": {"tts": "sarvam"}, "bn": {"tts": "sarvam"}}',
    )
    assert registry.get_tts(Lang.KN) is registry.get_tts(Lang.BN) is app.tts.sarvam()
    assert registry.get_tts(Lang.EN) is app.tts.piper()
