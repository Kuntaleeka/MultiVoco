"""Which language is the caller speaking? Decided by a ballot of recognisers.

Neither Whisper's nor Deepgram's built-in language detection can tell Kannada or
Bengali from Hindi (docs/langid.md). What does separate them: transcribe the utterance
once per candidate language, at the same time, and compare how confident each
recogniser is in its own transcript. The right one is sure of itself; the others are
forcing the sounds into the wrong language, and their confidence shows it.

The winner's lead over the runner-up is the evidence. A narrow lead means "not sure",
and the orchestrator then asks the caller.
"""

import asyncio
import logging
from dataclasses import dataclass

import httpx2 as httpx

from app.core.config import get_settings
from app.core.interfaces import INPUT_SAMPLE_RATE, LangGuess, ProviderError, QuotaExceeded
from app.core.languages import Lang, Script, lang_from_script, script_counts
from app.langid.config import LangIDSettings, get_langid_settings
from app.stt.config import get_stt_settings
from app.stt.groq_whisper import to_wav

log = logging.getLogger(__name__)

# The confidence reported for a lead of exactly `sure_margin`. It matches the
# orchestrator's default threshold (PipelineConfig.min_lang_confidence).
SURE = 0.6

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        # Kept open well past the default 5 s: the connections are opened while the
        # greeting plays and used when the caller has finished their first sentence.
        _client = httpx.AsyncClient(limits=httpx.Limits(keepalive_expiry=60))
    return _client


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


@dataclass(frozen=True)
class Vote:
    """One recogniser's go at the utterance."""

    code: str  # Deepgram's language code: "multi", "kn", "bn"
    confidence: float
    text: str


class BallotLanguageDetector:
    def __init__(
        self,
        api_key: str | None = None,
        settings: LangIDSettings | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._key = api_key if api_key is not None else get_settings().deepgram_api_key
        self._s = settings or get_langid_settings()
        self._client = client
        # Deepgram code -> our languages that share it. Hindi and English share "multi".
        self._candidates: dict[str, list[Lang]] = {}
        for lang in Lang:
            code = get_stt_settings().deepgram_languages.get(lang.value, lang.value)
            self._candidates.setdefault(code, []).append(lang)
        self.last_votes: list[Vote] = []  # for scripts and debugging

    async def _vote(self, wav: bytes, code: str) -> Vote:
        try:
            response = await (self._client or _get_client()).post(
                self._s.deepgram_url,
                params={
                    "model": self._s.deepgram_model,
                    "language": code,
                    "smart_format": "true",  # the winner's text becomes the user's turn
                },
                headers={"Authorization": f"Token {self._key}", "Content-Type": "audio/wav"},
                content=wav,
                timeout=self._s.timeout_seconds,
            )
        except httpx.HTTPError as error:
            raise ProviderError(f"Could not reach Deepgram: {error!r}") from error
        if response.status_code in (402, 429):
            raise QuotaExceeded(f"Deepgram returned HTTP {response.status_code}")
        if response.status_code != 200:
            raise ProviderError(
                f"Deepgram returned HTTP {response.status_code}: {response.text[:200]}"
            )
        channels = response.json().get("results", {}).get("channels") or [{}]
        best = (channels[0].get("alternatives") or [{}])[0]
        text = (best.get("transcript") or "").strip()
        return Vote(code, float(best.get("confidence") or 0.0) if text else 0.0, text)

    async def prepare(self) -> None:
        """Open one connection per candidate ahead of time. Optional, and never raises.

        The ballot sends its requests at once, so it needs that many connections. Opening
        them costs a TLS handshake each, which is better spent while the greeting plays
        than after the caller has spoken.
        """
        if not self._key:
            return
        client = self._client or _get_client()
        headers = {"Authorization": f"Token {self._key}"}

        async def touch() -> None:
            try:
                await client.get(self._s.deepgram_url, headers=headers, timeout=5)
            except httpx.HTTPError:
                pass

        await asyncio.gather(*(touch() for _ in self._candidates))

    async def detect_audio(self, pcm: bytes) -> LangGuess:
        self.last_votes = []
        if len(pcm) / 2 / INPUT_SAMPLE_RATE * 1000 < self._s.min_audio_ms:
            return LangGuess(None, 0.0)
        if not self._key:
            raise ProviderError("MULTIVOCO_DEEPGRAM_API_KEY is not set")
        wav = to_wav(pcm)
        results = await asyncio.gather(
            *(self._vote(wav, code) for code in self._candidates), return_exceptions=True
        )
        votes = [r for r in results if isinstance(r, Vote)]
        failures = [r for r in results if isinstance(r, BaseException)]
        for failure in failures:
            if not isinstance(failure, ProviderError):
                raise failure
        if not votes:
            raise next((f for f in failures if isinstance(f, QuotaExceeded)), failures[0])
        votes.sort(key=lambda vote: vote.confidence, reverse=True)
        self.last_votes = votes
        winner = votes[0]
        if not winner.text:
            return LangGuess(None, 0.0)
        if failures:
            # A candidate did not vote, so the lead cannot be trusted. Report the
            # winner, but below the threshold, so the caller is asked.
            log.warning("language ballot incomplete: %s", failures[0])
            return LangGuess(self._language_of(winner), SURE / 2, winner.text)
        runner_up = votes[1].confidence if len(votes) > 1 else 0.0
        lead = winner.confidence - runner_up
        confidence = max(0.0, min(1.0, SURE * lead / self._s.sure_margin))
        return LangGuess(self._language_of(winner), round(confidence, 3), winner.text)

    def _language_of(self, vote: Vote) -> Lang:
        langs = self._candidates[vote.code]
        if len(langs) == 1:
            return langs[0]
        # "multi" covers Hindi and English. Devanagari in the transcript means Hindi.
        if script_counts(vote.text).get(Script.DEVANAGARI) and Lang.HI in langs:
            return Lang.HI
        return Lang.EN if Lang.EN in langs else langs[0]

    def detect_text(self, text: str) -> LangGuess:
        return LangGuess(*lang_from_script(text))
