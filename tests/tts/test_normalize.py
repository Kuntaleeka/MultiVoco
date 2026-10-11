import pytest

from app.core.languages import Lang
from app.tts.normalize import normalize


@pytest.mark.parametrize(
    ("text", "lang", "spoken"),
    [
        ("Your EMI is 8450.00 rupees.", Lang.EN, "Your E M I is 8450 rupees."),
        ("Outstanding is 1,81,250.50 rupees.", Lang.EN, "Outstanding is 181250.50 rupees."),
        ("You owe ₹8,450 today.", Lang.EN, "You owe 8450 rupees today."),
        ("You owe Rs. 8450.00 today.", Lang.EN, "You owe 8450 rupees today."),
        ("**Sure**, it is `due` soon.", Lang.EN, "Sure , it is due soon."),
        ("Due on 5 November 2026.", Lang.EN, "Due on 5 November 2026."),
        ("आपकी EMI ८४५०.०० रुपये है।", Lang.HI, "आपकी ई एम आई 8450 रुपये है।"),
        ("आपका बकाया ₹93,750 है।", Lang.HI, "आपका बकाया 93750 रुपये है।"),
        ("  spaced \n out  ", Lang.EN, "spaced out"),
        ("***", Lang.EN, ""),
    ],
)
def test_text_is_made_safe_to_speak(text, lang, spoken):
    assert normalize(text, lang) == spoken


def test_a_word_that_only_contains_emi_is_left_alone():
    assert normalize("A premium EMIs plan.", Lang.EN) == "A premium EMIs plan."
