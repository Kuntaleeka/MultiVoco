"""Checks run on every sentence before it is released to be spoken.

Guardrail 2, tool-only figures: every number in a reply must have come from a tool
result in this call. The agent is told to write figures in digits (TTS turns them into
words), so the check reads digits in any of the four numeral systems, and rejects
magnitude words such as "thousand" or "लाख" that would let a figure slip past as words.

Known limits, to be covered by the [kn] and [bn] tasks and by the evals:
- Small numbers written as words ("five", "पाँच") are not caught.
- Magnitude words are listed for English and Hindi/Hinglish only.
- A figure the caller said is not allowed until a tool has returned it.
"""

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any

from app.core.languages import LANGUAGES, Lang, Script, script_counts, to_ascii_digits

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")

# "the last 4 digits of your phone number" is the one figure the agent's own script needs.
ALWAYS_ALLOWED = frozenset({Decimal(4)})


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


_MAGNITUDE_WORDS = frozenset(
    _nfc(word)
    for word in (
        *("hundred", "thousand", "lakh", "lakhs", "lac", "lacs", "crore", "crores"),
        *("million", "billion", "hazaar", "hazar", "sau"),
        *("सौ", "हज़ार", "हजार", "लाख", "करोड़", "करोड"),
    )
)
_TOKEN_SPLIT = re.compile(r"[\s.,;:!?।\"'()\[\]/\-–—]+")


def numbers_in(text: str) -> list[tuple[str, Decimal]]:
    """Every number written in digits, as (text as written, value)."""
    found = []
    for match in _NUMBER.finditer(to_ascii_digits(text)):
        try:
            found.append((match.group(), Decimal(match.group().replace(",", ""))))
        except InvalidOperation:
            continue
    return found


class FigureLedger:
    """The figures the agent may say: everything tools have returned so far in this call."""

    def __init__(self) -> None:
        self._allowed: set[Decimal] = set(ALWAYS_ALLOWED)

    def add(self, value: Any) -> None:
        """Record every number inside a tool result, including those inside strings and dates."""
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, int | float | Decimal):
            self._allowed.add(Decimal(str(value)))
        elif isinstance(value, str):
            self._allowed.update(number for _, number in numbers_in(value))
        elif isinstance(value, dict):
            for item in value.values():
                self.add(item)
        elif isinstance(value, list | tuple):
            for item in value:
                self.add(item)

    def unsupported(self, text: str) -> list[str]:
        """Figures in `text` that no tool result backs, as written. Empty means it passes."""
        bad = [written for written, number in numbers_in(text) if number not in self._allowed]
        tokens = _TOKEN_SPLIT.split(_nfc(text).casefold())
        bad.extend(token for token in tokens if token in _MAGNITUDE_WORDS)
        return bad


# Sessions where a reply in Latin script is normal: English, and Hindi spoken as Hinglish.
_LATIN_IS_FINE = frozenset({Lang.EN, Lang.HI})
_MAX_LATIN_ONLY_LETTERS = 12


def wrong_language(text: str, lang: Lang) -> bool:
    """True when the sentence is clearly not in the session language, judged by script."""
    counts = script_counts(text)
    latin = counts.get(Script.LATIN, 0)
    native = {script: n for script, n in counts.items() if script is not Script.LATIN}
    if lang is Lang.EN:
        return sum(native.values()) > latin
    if native:
        # A name in another script may appear, so only the dominant native script counts.
        return max(native, key=lambda script: native[script]) is not LANGUAGES[lang].script
    return lang not in _LATIN_IS_FINE and latin > _MAX_LATIN_ONLY_LETTERS


_TERMINATORS = "".join(sorted({ch for info in LANGUAGES.values() for ch in info.terminators}))
_BOUNDARY = re.compile(rf"[{re.escape(_TERMINATORS)}]+[\"')\]]*\s+")


class SentenceBuffer:
    """Cuts streamed text into sentences, so each can be checked whole before release.

    A terminator only ends a sentence when whitespace follows, which keeps decimals
    such as 8450.50 in one piece.
    """

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, text: str) -> list[str]:
        self._buffer += text
        sentences = []
        while match := _BOUNDARY.search(self._buffer):
            sentences.append(self._buffer[: match.end()])
            self._buffer = self._buffer[match.end() :]
        return sentences

    def flush(self) -> str:
        rest, self._buffer = self._buffer, ""
        return rest if rest.strip() else ""
