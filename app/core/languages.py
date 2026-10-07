"""Languages, scripts, numerals, and the per-language provider routing table.

Adding a language means: a Lang member, a LANGUAGES entry, a script range if the script
is new, and a DEFAULT_ROUTES entry. Nothing else in the pipeline should need to change.
"""

from dataclasses import dataclass, replace
from enum import StrEnum

from app.core.config import get_settings


class Lang(StrEnum):
    EN = "en"
    HI = "hi"
    KN = "kn"
    BN = "bn"


# Value of `lang` in the client's start message when the server should detect it.
AUTO = "auto"


class Script(StrEnum):
    LATIN = "latin"
    DEVANAGARI = "devanagari"
    KANNADA = "kannada"
    BENGALI = "bengali"


def _digits(first: int) -> str:
    return "".join(chr(first + i) for i in range(10))


@dataclass(frozen=True)
class LanguageInfo:
    lang: Lang
    name: str
    script: Script
    digits: str  # native numerals for 0-9, in order
    terminators: str  # characters that end a sentence


LANGUAGES: dict[Lang, LanguageInfo] = {
    Lang.EN: LanguageInfo(Lang.EN, "English", Script.LATIN, "0123456789", ".?!"),
    Lang.HI: LanguageInfo(Lang.HI, "Hindi", Script.DEVANAGARI, _digits(0x0966), ".?!।"),
    Lang.KN: LanguageInfo(Lang.KN, "Kannada", Script.KANNADA, _digits(0x0CE6), ".?!"),
    Lang.BN: LanguageInfo(Lang.BN, "Bengali", Script.BENGALI, _digits(0x09E6), ".?!।"),
}

_SCRIPT_RANGES: dict[Script, tuple[int, int]] = {
    Script.DEVANAGARI: (0x0900, 0x097F),
    Script.BENGALI: (0x0980, 0x09FF),
    Script.KANNADA: (0x0C80, 0x0CFF),
}

# The danda and double danda sit in the Devanagari block but Bengali uses them too,
# so they are not evidence for either script.
_SHARED_PUNCTUATION = {0x0964, 0x0965}

SCRIPT_LANG: dict[Script, Lang] = {
    Script.DEVANAGARI: Lang.HI,
    Script.KANNADA: Lang.KN,
    Script.BENGALI: Lang.BN,
}

_ASCII_DIGITS = str.maketrans(
    {
        native: str(value)
        for info in LANGUAGES.values()
        if info.script is not Script.LATIN
        for value, native in enumerate(info.digits)
    }
)


def script_of(ch: str) -> Script | None:
    """Script of a single character, or None for digits, punctuation, and spaces."""
    code = ord(ch)
    if code in _SHARED_PUNCTUATION:
        return None
    if ch.isascii():
        return Script.LATIN if ch.isalpha() else None
    for script, (lo, hi) in _SCRIPT_RANGES.items():
        if lo <= code <= hi:
            return None if ch.isdigit() else script
    return None


def script_counts(text: str) -> dict[Script, int]:
    counts: dict[Script, int] = {}
    for ch in text:
        script = script_of(ch)
        if script is not None:
            counts[script] = counts.get(script, 0) + 1
    return counts


def dominant_script(text: str) -> Script | None:
    counts = script_counts(text)
    if not counts:
        return None
    return max(counts, key=lambda script: counts[script])


def lang_from_script(text: str) -> tuple[Lang | None, float]:
    """Language implied by the script of `text`, with the share of native letters behind it.

    Latin-only text returns (None, 0.0): Hinglish and Kannada-English callers are often
    transcribed in Latin script, so Latin letters are no evidence of English.
    """
    native = {s: n for s, n in script_counts(text).items() if s is not Script.LATIN}
    if not native:
        return None, 0.0
    top = max(native, key=lambda script: native[script])
    return SCRIPT_LANG[top], native[top] / sum(native.values())


def to_ascii_digits(text: str) -> str:
    """Replace Devanagari, Kannada, and Bengali numerals with 0-9."""
    return text.translate(_ASCII_DIGITS)


@dataclass(frozen=True)
class Route:
    stt: str
    llm: str
    tts: str


# Starting point. Workstreams A, B, and C confirm or change each entry by measurement,
# first through MULTIVOCO_ROUTE_OVERRIDES, then here once the choice is settled.
DEFAULT_ROUTES: dict[Lang, Route] = {
    Lang.EN: Route(stt="deepgram", llm="groq", tts="piper"),
    Lang.HI: Route(stt="deepgram", llm="groq", tts="piper"),
    Lang.KN: Route(stt="groq_whisper", llm="groq", tts="azure"),
    Lang.BN: Route(stt="groq_whisper", llm="groq", tts="azure"),
}


def route_for(lang: Lang) -> Route:
    override = get_settings().route_overrides.get(lang.value, {})
    return replace(DEFAULT_ROUTES[lang], **override)


def shares_route(a: Lang, b: Lang) -> bool:
    """True when switching between the two languages needs no provider change."""
    return route_for(a) == route_for(b)
