"""Makes the agent's text safe to speak, in all four languages.

The agent writes figures in digits (its guardrail depends on that), and both voices
read plain numbers aloud by themselves. So this only reshapes what trips them up:
markup, currency signs, paise, ISO dates, long digit strings, and grouping commas.

`normalize` is for Piper (English, Hindi), which wants numbers with no commas.
`for_cloud` is for Sarvam, whose docs ask for commas in numbers of more than four digits.

The Kannada and Bengali words here need a native speaker's check.
"""

import re

from app.core.languages import Lang, to_ascii_digits

_RUPEES = {Lang.EN: "rupees", Lang.HI: "रुपये", Lang.KN: "ರೂಪಾಯಿ", Lang.BN: "টাকা"}
_PAISE = {Lang.EN: "paise", Lang.HI: "पैसे", Lang.KN: "ಪೈಸೆ", Lang.BN: "পয়সা"}
_PERCENT = {Lang.EN: "percent", Lang.HI: "प्रतिशत", Lang.KN: "ಪ್ರತಿಶತ", Lang.BN: "শতাংশ"}
_MONTHS = {
    Lang.EN: (
        *("January", "February", "March", "April", "May", "June", "July", "August"),
        *("September", "October", "November", "December"),
    ),
    Lang.HI: (
        *("जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त"),
        *("सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"),
    ),
    Lang.KN: (
        *("ಜನವರಿ", "ಫೆಬ್ರವರಿ", "ಮಾರ್ಚ್", "ಏಪ್ರಿಲ್", "ಮೇ", "ಜೂನ್", "ಜುಲೈ", "ಆಗಸ್ಟ್"),
        *("ಸೆಪ್ಟೆಂಬರ್", "ಅಕ್ಟೋಬರ್", "ನವೆಂಬರ್", "ಡಿಸೆಂಬರ್"),
    ),
    Lang.BN: (
        *("জানুয়ারি", "ফেব্রুয়ারি", "মার্চ", "এপ্রিল", "মে", "জুন", "জুলাই", "আগস্ট"),
        *("সেপ্টেম্বর", "অক্টোবর", "নভেম্বর", "ডিসেম্বর"),
    ),
}
# Read letter by letter. Piper otherwise says "emmy". Sarvam is sent "EMI" as it is.
_SPELL_OUT = {
    Lang.EN: {"EMI": "E M I"},
    Lang.HI: {"EMI": "ई एम आई"},
}

_MARKUP = re.compile(r"[*_#`~|<>\[\]{}]+")
_SPACES = re.compile(r"\s+")
_GROUPING = re.compile(r"(?<=\d),(?=\d)")
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_CURRENCY_BEFORE = re.compile(r"(?:₹|\bRs\.?|\bINR)\s*(\d[\d.]*\d|\d)", re.IGNORECASE)
_WHOLE = re.compile(r"(\d)\.0+(?!\d)")
_LONG_DIGITS = re.compile(r"(?<![\d.])\d{7,}(?!\.?\d)")
_FIVE_DIGITS_OR_MORE = re.compile(r"(?<![\d.])\d{5,}(?!\d)")
_PERCENT_SIGN = re.compile(r"(?<=\d)\s*%")


def _iso_date(match: re.Match, lang: Lang) -> str:
    year, month, day = (int(part) for part in match.groups())
    if not 1 <= month <= 12:
        return match.group()
    return f"{day} {_MONTHS[lang][month - 1]} {year}"


def _indian_grouping(match: re.Match) -> str:
    digits = match.group()
    head, tail = digits[:-3], digits[-3:]
    pairs = [head[max(0, end - 2) : end] for end in range(len(head), 0, -2)][::-1]
    return ",".join([*pairs, tail])


def _common(text: str, lang: Lang) -> str:
    """The steps both voices need."""
    text = _MARKUP.sub(" ", to_ascii_digits(text))
    text = _GROUPING.sub("", text)  # 1,81,250 -> 181250, regrouped later where wanted
    text = _ISO_DATE.sub(lambda match: _iso_date(match, lang), text)  # 2026-11-05
    text = _CURRENCY_BEFORE.sub(rf"\1 {_RUPEES[lang]}", text)  # ₹8450 -> 8450 rupees
    text = _WHOLE.sub(r"\1", text)  # 8450.00 -> 8450
    # 8450.50 rupees -> 8450 rupees 50 paise
    text = re.sub(
        rf"(\d+)\.(\d{{2}})\s*{re.escape(_RUPEES[lang])}",
        rf"\1 {_RUPEES[lang]} \2 {_PAISE[lang]}",
        text,
    )
    text = _PERCENT_SIGN.sub(f" {_PERCENT[lang]}", text)
    # A phone or account number is read one digit at a time, not as a quantity. A long
    # number followed by "rupees" is an amount, and is left whole.
    return re.sub(
        rf"{_LONG_DIGITS.pattern}(?!\s*{re.escape(_RUPEES[lang])})",
        lambda match: " ".join(match.group()),
        text,
    )


def normalize(text: str, lang: Lang) -> str:
    """For Piper."""
    text = _common(text, lang)
    for word, spoken in _SPELL_OUT.get(lang, {}).items():
        text = re.sub(rf"\b{word}\b", spoken, text)
    return _SPACES.sub(" ", text).strip()


def for_cloud(text: str, lang: Lang) -> str:
    """For Sarvam."""
    text = _FIVE_DIGITS_OR_MORE.sub(_indian_grouping, _common(text, lang))  # 206300 -> 2,06,300
    return _SPACES.sub(" ", text).strip()
