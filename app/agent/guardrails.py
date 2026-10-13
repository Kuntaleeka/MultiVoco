"""Checks run on every sentence before it is released to be spoken.

Guardrail 2, tool-only figures: every number in a reply must have come from a tool
result in this call. The agent is told to write figures in digits (TTS turns them into
words), so the check reads digits in any of the four numeral systems, and rejects
magnitude words such as "thousand" or "लाख" that would let a figure slip past as words.

Known limits, to be measured by the evals:
- Numbers below a hundred written as words ("five", "पाँच", "ಐದು") are not caught. Every
  amount on a loan is above that, so this leaves days of the month and small counts.
- The Kannada and Bengali word lists have not been checked by a native speaker.
- A figure the caller said is not allowed until a tool has returned it.
"""

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any

from app.core.languages import LANGUAGES, Lang, Script, script_counts, to_ascii_digits
from app.tools.base import parse_date

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

# Kannada and Bengali join and inflect these ("ಐದುಸಾವಿರ", "ಲಕ್ಷಕ್ಕೆ", "হাজারের"), so they
# are matched anywhere inside a word. The exceptions are everyday words that merely
# contain one: ಲಕ್ಷಣ (feature), ಲಕ್ಷ್ಯ (attention), লক্ষ্য (aim), লক্ষণ (sign).
# All of this needs a native speaker's check.
_MAGNITUDE_STEMS = tuple(
    _nfc(stem)
    for stem in (
        *("ನೂರ", "ಸಾವಿರ", "ಲಕ್ಷ", "ಕೋಟಿ"),  # hundred, thousand, lakh, crore
        *("হাজার", "লাখ", "লক্ষ", "কোটি"),  # thousand, lakh, lakh, crore
    )
)
_NOT_MAGNITUDES = tuple(_nfc(word) for word in ("ಲಕ್ಷಣ", "ಲಕ್ಷ್ಯ", "লক্ষ্য", "লক্ষণ"))
# Bengali hundreds: "শত" alone, or a number ending in "শো" (একশো, পাঁচশো).
_BENGALI_HUNDRED = _nfc("শত")
_BENGALI_HUNDRED_ENDING = _nfc("শো")


def _is_magnitude(token: str) -> bool:
    if token in _MAGNITUDE_WORDS or token == _BENGALI_HUNDRED:
        return True
    if token.endswith(_BENGALI_HUNDRED_ENDING) and len(token) > len(_BENGALI_HUNDRED_ENDING):
        return True
    if any(token.startswith(word) for word in _NOT_MAGNITUDES):
        return False
    return any(stem in token for stem in _MAGNITUDE_STEMS)


# Zero to nine as a caller might say them. Used only to tell "the caller read out some
# digits, written as words" from "the caller gave no digits". The Kannada and Bengali
# words need a native speaker's check.
_DIGIT_WORDS = frozenset(
    _nfc(word)
    for word in (
        *("zero", "oh", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"),
        *("shunya", "ek", "do", "teen", "char", "chaar", "paanch", "panch", "chhah", "chhe"),
        *("saat", "aath", "nau"),
        *("शून्य", "एक", "दो", "तीन", "चार", "पाँच", "पांच", "छह", "छः", "सात", "आठ", "नौ"),
        *("ಸೊನ್ನೆ", "ಒಂದು", "ಎರಡು", "ಮೂರು", "ನಾಲ್ಕು", "ಐದು", "ಆರು", "ಏಳು", "ಎಂಟು", "ಒಂಬತ್ತು"),
        *("শূন্য", "এক", "দুই", "তিন", "চার", "পাঁচ", "ছয়", "সাত", "আট", "নয়"),
    )
)
_MIN_DIGIT_WORDS = 3


_HINDI_DIGITS = "शून्य|एक|दो|तीन|चार|पाँच|पांच|छह|छः|सात|आठ|नौ"
# "साथ" (with) straight after two or more digit words is a misheard "सात" (seven). One
# digit word is not enough: "एक साथ" is the everyday phrase for "together".
_MISHEARD_SEVEN = re.compile(rf"((?:(?:{_HINDI_DIGITS})\s+){{2,}})साथ(?![ऀ-ॿ])")


def repair_misheard_digits(text: str) -> str:
    """Undo a mishearing seen in a real call, where it failed a caller's verification.

    Speech recognition wrote "दो दो एक साथ" for the digits 2 2 1 7, and the model, told
    about it in the prompt, still read three digits and the word "together".
    """
    return _MISHEARD_SEVEN.sub(r"\1सात", text)


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
        bad.extend(token for token in tokens if _is_magnitude(token))
        return bad


def identity_details_not_given(args: dict[str, Any], caller_texts: list[str]) -> list[str]:
    """Which verify_identity arguments the caller never said, so the model made them up.

    Seen with a real model: given only the phone digits, it invented a date of birth, and
    given nothing at all it invented both. Each guess cost the caller one of their three
    attempts. The check compares digits only. If the caller said no digits but did say
    several digit words (speech-to-text wrote "four eight two one"), nothing can be
    compared and the call is let through.
    """
    everything = " ".join(caller_texts)
    said = "".join(ch for ch in to_ascii_digits(everything) if ch.isdigit())
    if not said:
        tokens = _TOKEN_SPLIT.split(_nfc(everything).casefold())
        if sum(token in _DIGIT_WORDS for token in tokens) >= _MIN_DIGIT_WORDS:
            return []
        return ["the last 4 digits of the phone number", "the date of birth"]
    missing = []
    phone = "".join(ch for ch in to_ascii_digits(str(args.get("phone_last4", ""))) if ch.isdigit())
    if len(phone) == 4 and phone not in said:
        missing.append("the last 4 digits of the phone number")
    dob = parse_date(args.get("dob", ""))
    # The year as said: all four digits, or the last two ("12/04/90").
    if dob is not None and str(dob.year) not in said and f"{dob.year % 100:02d}" not in said:
        missing.append("the date of birth")
    return missing


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


# Seen live: the model separates sentences with a narrow no-break space and writes
# non-breaking hyphens. Neither is safe for the sentence splitter or for TTS.
_ODD_SPACES = re.compile("[   -​    　﻿]")
_ODD_HYPHENS = re.compile("[‐‑]")


def plain_spaces(text: str) -> str:
    return _ODD_HYPHENS.sub("-", _ODD_SPACES.sub(" ", text))


# Seen in a real call: the backup model wrote "User:" lines and answered them itself,
# and all of it was spoken. A label for the other side means the model has left its own
# turn. A label for its own side is only a prefix.
_OTHER_SPEAKER = re.compile(r"\s*(user|caller|customer|human|borrower)\s*:", re.IGNORECASE)
_OWN_LABEL = re.compile(r"^\s*(assistant|agent|ai|bot)\s*:\s*", re.IGNORECASE)


def starts_other_speaker(sentence: str) -> bool:
    return _OTHER_SPEAKER.match(sentence) is not None


def without_own_label(sentence: str) -> str:
    return _OWN_LABEL.sub("", sentence, count=1)


_TERMINATORS = "".join(sorted({ch for info in LANGUAGES.values() for ch in info.terminators}))
_ENDING = re.compile(rf"([{re.escape(_TERMINATORS)}]+)[\"')\]]*")


def _starts_sentence(terminator: str, following: str) -> bool:
    """Does `following`, coming straight after `terminator`, begin a new sentence?

    The same rule as the orchestrator's splitter (app/pipeline/splitter.py). Seen live:
    the model writes "...2026.Is there anything else?" with no space. A letter straight
    after "?", "!" or a danda starts a sentence, and so does a capital or a non-Latin
    letter straight after a full stop. A digit or a small Latin letter after a full stop
    does not, which keeps 8450.50 and example.com whole.
    """
    if not following.isalpha():
        return False
    if terminator != ".":
        return True
    return following.isupper() or not following.isascii()


class SentenceBuffer:
    """Cuts streamed text into sentences, so each can be checked whole before release.

    A sentence ends at a terminator followed by whitespace, or by the start of the next
    sentence with no space at all. The last character is never judged until the next
    one has arrived.
    """

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, text: str) -> list[str]:
        self._buffer += text
        sentences = []
        position = 0
        while match := _ENDING.search(self._buffer, position):
            end = match.end()
            if end == len(self._buffer):
                break  # what follows has not arrived yet
            if self._buffer[end].isspace():
                while end < len(self._buffer) and self._buffer[end].isspace():
                    end += 1
            elif not _starts_sentence(match.group(1)[-1], self._buffer[end]):
                position = end
                continue
            sentences.append(self._buffer[:end])
            self._buffer = self._buffer[end:]
            position = 0
        return sentences

    def flush(self) -> str:
        rest, self._buffer = self._buffer, ""
        return rest if rest.strip() else ""
