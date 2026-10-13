"""Sarvam AI's streaming text-to-speech, for Kannada and Bengali.

It took Azure Speech's place in the plan because Azure needs a card to sign up
(section 11, note 12). The API is the HTTP stream endpoint: one POST per sentence, raw
PCM back as it is generated. Reference: docs.sarvam.ai, text-to-speech, convert-stream.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from datetime import date

import httpx2 as httpx

from app.core.interfaces import ProviderError, QuotaExceeded
from app.core.languages import Lang
from app.tts.config import TTSSettings, get_tts_settings
from app.tts.normalize import clean

log = logging.getLogger(__name__)

URL = "https://api.sarvam.ai/text-to-speech/stream"
LANGUAGE_CODES = {Lang.EN: "en-IN", Lang.HI: "hi-IN", Lang.KN: "kn-IN", Lang.BN: "bn-IN"}
MAX_CHARS = 2500  # the docs give 2500 in one place and 3500 in another

_QUOTA_CODES = ("insufficient_quota_error", "rate_limit_exceeded_error")
_RETRYABLE_STATUS = {408, 500, 502, 503, 504}


class _Retryable(Exception):
    """A failure worth one more try, as long as no audio has gone out."""


class SarvamTTS:
    """Holds no per-call state apart from the day's character count, so one is shared."""

    languages = set(LANGUAGE_CODES)

    def __init__(
        self,
        settings: TTSSettings | None = None,
        client: httpx.AsyncClient | None = None,
        today: Callable[[], date] = date.today,
    ) -> None:
        self._settings = settings or get_tts_settings()
        self._client = client
        self._today = today
        self._day = today()
        self.chars_today = 0

    def sample_rate(self, language: Lang) -> int:
        return self._settings.sarvam_sample_rate

    def _spend(self, count: int) -> None:
        """Count characters against the daily cap, before the request is made."""
        if self._today() != self._day:
            self._day, self.chars_today = self._today(), 0
        cap = self._settings.sarvam_daily_chars
        if cap and self.chars_today + count > cap:
            raise QuotaExceeded(f"sarvam: daily cap of {cap} characters reached")
        self.chars_today += count

    async def synthesize(self, text: str, language: Lang) -> AsyncIterator[bytes]:
        if language not in LANGUAGE_CODES:
            raise ProviderError(f"sarvam: no voice for {language.value}")
        if not self._settings.sarvam_api_key:
            raise ProviderError("sarvam: no API key set (MULTIVOCO_SARVAM_API_KEY)")
        spoken = clean(text)[:MAX_CHARS]
        if not spoken:
            return
        self._spend(len(spoken))
        payload = {
            "text": spoken,
            "language_code": LANGUAGE_CODES[language],
            "model": self._settings.sarvam_model,
            "speaker": self._settings.sarvam_speaker,
            "speech_sample_rate": self._settings.sarvam_sample_rate,
            "output_audio_codec": "linear16",
            # Lets Sarvam read amounts, dates, and English words inside the sentence.
            "enable_preprocessing": True,
        }
        for attempt in (1, 2):
            started = False
            try:
                async for piece in self._once(payload, language):
                    started = True
                    yield piece
                return
            except _Retryable as error:
                # Once audio has gone out, a retry would make the caller hear it twice.
                if started or attempt == 2:
                    raise ProviderError(f"sarvam: {error}") from error
                log.warning("sarvam failed (%s), retrying", error)
                await asyncio.sleep(0.2)

    async def _once(self, payload: dict, language: Lang) -> AsyncIterator[bytes]:
        client = self._client or _shared_client()
        timeout = httpx.Timeout(
            self._settings.sarvam_read_timeout_s, connect=self._settings.sarvam_connect_timeout_s
        )
        headers = {"api-subscription-key": self._settings.sarvam_api_key}
        step = self.sample_rate(language) * self._settings.tts_chunk_ms // 1000 * 2
        pending = b""
        header_checked = False
        try:
            async with client.stream(
                "POST", URL, json=payload, headers=headers, timeout=timeout
            ) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    raise _http_error(response.status_code, body)
                async for data in response.aiter_bytes():
                    pending += data
                    if not header_checked:
                        if len(pending) < 12:
                            continue
                        header_checked = True
                        if pending[:4] == b"RIFF":
                            # Should be raw PCM, but the docs do not say. Skip a WAV header.
                            start = pending.find(b"data")
                            if start < 0:
                                header_checked = False
                                continue
                            pending = pending[start + 8 :]
                    while len(pending) >= step:
                        yield pending[:step]
                        pending = pending[step:]
        except httpx.TimeoutException as error:
            raise _Retryable(f"timed out ({type(error).__name__})") from error
        except httpx.HTTPError as error:
            raise _Retryable(f"connection failed ({type(error).__name__})") from error
        # PCM16 samples are two bytes: drop a stray last byte rather than emit half a sample.
        pending = pending[: len(pending) // 2 * 2]
        if pending:
            yield pending


def _http_error(status: int, body: str) -> Exception:
    try:
        error = json.loads(body).get("error") or {}
    except (json.JSONDecodeError, AttributeError):
        error = {}
    code = str(error.get("code", "")) if isinstance(error, dict) else ""
    message = str(error.get("message", "")) if isinstance(error, dict) else ""
    detail = f"HTTP {status} {code}: {' '.join((message or body).split())[:200]}"
    if status == 429 or code in _QUOTA_CODES:
        return QuotaExceeded(f"sarvam: {detail}")
    if status in _RETRYABLE_STATUS:
        return _Retryable(detail)
    return ProviderError(f"sarvam: {detail}")


_client: httpx.AsyncClient | None = None


def _shared_client() -> httpx.AsyncClient:
    """One connection pool per process, so later sentences skip the TLS handshake."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient()
    return _client


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
    _client = None
