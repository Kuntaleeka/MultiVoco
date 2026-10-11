"""Makes the agent's text safe to speak. English and Hindi so far.

The agent writes figures in digits (its guardrail depends on that), and Piper's
phonemiser reads plain numbers aloud by itself. So this only removes what trips it up:
markup, odd spacing, currency signs, grouping commas, and trailing ".00".

Not done yet, and listed in the plan: reading loan IDs and phone digits one at a time,
and spoken dates.
"""

import re

from app.core.languages import Lang, to_ascii_digits

_RUPEES = {Lang.EN: "rupees", Lang.HI: "रुपये"}
# Read letter by letter. Piper otherwise says "emmy".
_SPELL_OUT = {
    Lang.EN: {"EMI": "E M I"},
    Lang.HI: {"EMI": "ई एम आई"},
}

_MARKUP = re.compile(r"[*_#`~|<>\[\]{}]+")
_SPACES = re.compile(r"\s+")
_GROUPING = re.compile(r"(?<=\d),(?=\d)")
_WHOLE = re.compile(r"(\d)\.0+(?!\d)")
_CURRENCY_BEFORE = re.compile(r"(?:₹|\bRs\.?|\bINR)\s*(\d[\d.]*)", re.IGNORECASE)


def normalize(text: str, lang: Lang) -> str:
    text = to_ascii_digits(text)
    text = _MARKUP.sub(" ", text)
    text = _GROUPING.sub("", text)  # 1,81,250 -> 181250, so it is read as one number
    text = _WHOLE.sub(r"\1", text)  # 8450.00 -> 8450
    rupees = _RUPEES.get(lang)
    if rupees:
        text = _CURRENCY_BEFORE.sub(rf"\1 {rupees}", text)  # ₹8450 -> 8450 rupees
    for word, spoken in _SPELL_OUT.get(lang, {}).items():
        text = re.sub(rf"\b{word}\b", spoken, text)
    return _SPACES.sub(" ", text).strip()
