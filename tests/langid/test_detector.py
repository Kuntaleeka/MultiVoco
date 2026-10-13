import httpx2 as httpx
import pytest

from app.core import registry
from app.core.interfaces import LangGuess, LanguageDetector, ProviderError, QuotaExceeded
from app.core.languages import Lang
from app.core.mocks import MockSTT, tone
from app.langid.config import LangIDSettings
from app.langid.detector import SURE, BallotLanguageDetector

SPEECH = tone(1500)


class FakeDeepgram:
    """Answers each language's request with a scripted confidence and transcript."""

    def __init__(self, **votes) -> None:
        self.votes = votes  # code -> (confidence, text) | int status | Exception
        self.requests: list[httpx.Request] = []
        self.client = httpx.AsyncClient(transport=httpx.MockTransport(self._handle))

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == "GET":  # prepare() opening a connection
            return httpx.Response(405)
        vote = self.votes[request.url.params["language"]]
        if isinstance(vote, Exception):
            raise vote
        if isinstance(vote, int):
            return httpx.Response(vote, text="nope")
        confidence, text = vote
        alternative = {"confidence": confidence, "transcript": text}
        return httpx.Response(
            200, json={"results": {"channels": [{"alternatives": [alternative]}]}}
        )


def detector(fake: FakeDeepgram, **settings) -> BallotLanguageDetector:
    return BallotLanguageDetector("test-key", LangIDSettings(**settings), fake.client)


HINDI = (1.00, "नमस्ते, मेरी अगली EMI कब है?")
ENGLISH = (1.00, "Hello. I wanted to ask when my next EMI is due.")


async def test_the_most_confident_recogniser_wins():
    fake = FakeDeepgram(multi=(0.81, "नमस्कार नन्हा"), kn=(0.97, "ನಮಸ್ಕಾರ ನನ್ನ ಮುಂದಿನ"), bn=(0.79, "x"))
    d = detector(fake)
    assert isinstance(d, LanguageDetector)
    guess = await d.detect_audio(SPEECH)
    assert guess.lang is Lang.KN
    assert guess.confidence == pytest.approx(0.96)  # a lead of 0.16, where 0.10 is "sure"
    assert guess.text == "ನಮಸ್ಕಾರ ನನ್ನ ಮುಂದಿನ"  # the winner's transcript comes with it
    assert [v.code for v in d.last_votes] == ["kn", "multi", "bn"]

    assert {r.url.params["language"] for r in fake.requests} == {"multi", "kn", "bn"}
    request = fake.requests[0]
    assert request.headers["Authorization"] == "Token test-key"
    assert request.url.params["model"] == "nova-3"
    assert request.content[:4] == b"RIFF"


@pytest.mark.parametrize("vote, lang", [(HINDI, Lang.HI), (ENGLISH, Lang.EN)])
async def test_hindi_and_english_are_told_apart_by_script(vote, lang):
    fake = FakeDeepgram(multi=vote, kn=(0.6, "x"), bn=(0.7, "y"))
    assert await detector(fake).detect_audio(SPEECH) == LangGuess(lang, 1.0, vote[1])


async def test_hinglish_counts_as_hindi():
    fake = FakeDeepgram(multi=(0.92, "मेरा next EMI कब due है?"), kn=(0.64, "x"), bn=(0.72, "y"))
    assert (await detector(fake).detect_audio(SPEECH)).lang is Lang.HI


@pytest.mark.parametrize(
    "lead, sure", [(0.20, True), (0.10, True), (0.09, False), (0.01, False), (0.0, False)]
)
async def test_a_narrow_lead_is_not_sure(lead, sure):
    fake = FakeDeepgram(multi=(0.85 - lead, "नमस्कार"), bn=(0.85, "নমস্কার, আমার"), kn=(0.3, "x"))
    guess = await detector(fake).detect_audio(SPEECH)
    if lead > 0:
        assert guess.lang is Lang.BN  # on a dead heat either may be named: it is not used
    assert (guess.confidence >= SURE) is sure


async def test_the_margin_that_counts_as_sure_is_configurable():
    fake = FakeDeepgram(multi=(0.80, "a"), bn=(0.85, "নমস্কার"), kn=(0.3, "x"))
    assert (await detector(fake).detect_audio(SPEECH)).confidence < SURE
    assert (await detector(fake, sure_margin=0.05).detect_audio(SPEECH)).confidence >= SURE


async def test_audio_too_short_to_judge_is_not_sent():
    fake = FakeDeepgram()
    assert await detector(fake).detect_audio(tone(300)) == LangGuess(None, 0.0)
    assert fake.requests == []


async def test_nothing_recognised_is_no_evidence():
    fake = FakeDeepgram(multi=(0.9, ""), kn=(0.9, " "), bn=(0.0, ""))
    assert await detector(fake).detect_audio(SPEECH) == LangGuess(None, 0.0)


async def test_an_empty_transcript_cannot_win_on_confidence():
    fake = FakeDeepgram(multi=(0.99, ""), kn=(0.70, "ನಮಸ್ಕಾರ"), bn=(0.2, "x"))
    assert (await detector(fake).detect_audio(SPEECH)).lang is Lang.KN


async def test_a_missing_vote_means_ask_the_caller():
    fake = FakeDeepgram(multi=HINDI, kn=500, bn=(0.2, "x"))
    guess = await detector(fake).detect_audio(SPEECH)
    assert guess.lang is Lang.HI
    assert 0 < guess.confidence < SURE


@pytest.mark.parametrize(
    "votes, error",
    [
        (dict(multi=500, kn=500, bn=500), ProviderError),
        (dict(multi=500, kn=429, bn=500), QuotaExceeded),
        (dict(multi=402, kn=402, bn=402), QuotaExceeded),
        (dict.fromkeys(("multi", "kn", "bn"), httpx.ConnectError("down")), ProviderError),
    ],
)
async def test_when_no_recogniser_answers_it_raises(votes, error):
    with pytest.raises(error) as raised:
        await detector(FakeDeepgram(**votes)).detect_audio(SPEECH)
    assert type(raised.value) is error


async def test_no_key_is_a_provider_error():
    with pytest.raises(ProviderError, match="DEEPGRAM_API_KEY"):
        await BallotLanguageDetector("").detect_audio(SPEECH)


def test_text_detection_is_the_script_check():
    d = BallotLanguageDetector("k")
    assert d.detect_text("ನನ್ನ ಮುಂದಿನ EMI ಯಾವಾಗ?") == LangGuess(Lang.KN, 1.0)
    assert d.detect_text("mera next EMI kab due hai") == LangGuess(None, 0.0)


def test_importing_the_package_registers_it(settings_env):
    import app.langid  # noqa: F401

    settings_env(MOCK="vad,stt,llm,tts,agent,trace_sink")
    assert isinstance(registry.get_language_detector(), BallotLanguageDetector)


async def test_in_the_pipeline_a_sure_guess_sets_the_language(monkeypatch):
    from tests.pipeline.conftest import Rig

    fake = FakeDeepgram(multi=(0.76, "ना मुश्किल"), kn=(0.53, "x"), bn=(0.95, "নমস্কার, আমার পরের"))
    rig = Rig(monkeypatch)
    rig.use("langid", lambda: detector(fake, min_audio_ms=100))
    sent_to_stt: list[bytes] = []

    class RecordingSTT(MockSTT):
        async def send_audio(self, frame: bytes) -> None:
            sent_to_stt.append(frame)
            await super().send_audio(frame)

    rig.use("stt", lambda lang: RecordingSTT())
    await rig.start("auto")
    await rig.greeted()
    await rig.say()
    await rig.reply(1)
    assert rig.socket.of_type("language") == [
        {"type": "language", "lang": "bn", "source": "audio", "confidence": 1.0}
    ]
    assert rig.trace(1).language is Lang.BN
    # The detector's transcript is the user's turn. Nothing was transcribed twice.
    assert rig.trace(1).user_text == "নমস্কার, আমার পরের"
    assert sent_to_stt == []
    # While the greeting played, a connection was opened for each candidate.
    assert [r.method for r in fake.requests[:3]] == ["GET"] * 3
    await rig.session.teardown()


async def test_the_answer_to_the_language_question_is_read_from_the_ballot(monkeypatch):
    from tests.pipeline.conftest import Rig

    fake = FakeDeepgram(multi=(0.5, "hmm"), kn=(0.5, "x"), bn=(0.5, "y"))  # a dead heat
    rig = Rig(monkeypatch)
    rig.use("langid", lambda: detector(fake, min_audio_ms=100))
    await rig.start("auto")
    await rig.greeted()
    await rig.say()
    await rig.reply(1)  # the agent asks which language
    assert rig.socket.of_type("language") == []

    fake.votes = dict(multi=(0.99, "Kannada please."), kn=(0.4, "x"), bn=(0.3, "y"))
    await rig.socket.wait_for_state("listening", count=2)
    await rig.say()
    await rig.reply(2)
    chosen = rig.socket.of_type("language")[-1]
    assert (chosen["lang"], chosen["source"]) == ("kn", "asked")  # named, not the winner's
    await rig.session.teardown()


async def test_prepare_never_raises():
    await BallotLanguageDetector("").prepare()
    fake = FakeDeepgram()
    fake.client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: (_ for _ in ()).throw(httpx.ConnectError("x"))
        )
    )
    await detector(fake).prepare()
