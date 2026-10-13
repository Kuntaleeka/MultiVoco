import pytest

from app.core.languages import Lang
from app.tts.normalize import for_cloud, normalize


@pytest.mark.parametrize(
    ("text", "lang", "spoken"),
    [
        ("Your EMI is 8450.00 rupees.", Lang.EN, "Your E M I is 8450 rupees."),
        ("You owe ₹8,450 today.", Lang.EN, "You owe 8450 rupees today."),
        ("You owe Rs. 8450.00 today.", Lang.EN, "You owe 8450 rupees today."),
        ("**Sure**, it is `due` soon.", Lang.EN, "Sure , it is due soon."),
        ("Due on 5 November 2026.", Lang.EN, "Due on 5 November 2026."),
        ("आपकी EMI ८४५०.०० रुपये है।", Lang.HI, "आपकी ई एम आई 8450 रुपये है।"),
        ("आपका बकाया ₹93,750 है।", Lang.HI, "आपका बकाया 93750 रुपये है।"),
        ("  spaced \n out  ", Lang.EN, "spaced out"),
        ("***", Lang.EN, ""),
        # Paise are read as paise, not as "point five zero".
        (
            "Outstanding is 1,81,250.50 rupees.",
            Lang.EN,
            "Outstanding is 181250 rupees 50 paise.",
        ),
        ("बकाया 8450.50 रुपये है।", Lang.HI, "बकाया 8450 रुपये 50 पैसे है।"),
        # Seen live: the model wrote the date the way the tool returned it.
        ("It is due on 2026-10-23.", Lang.EN, "It is due on 23 October 2026."),
        ("यह 2026-11-05 को देय है।", Lang.HI, "यह 5 नवंबर 2026 को देय है।"),
        ("Not a date: 2026-13-45.", Lang.EN, "Not a date: 2026-13-45."),
        # A phone or account number is read digit by digit. An amount is not.
        ("Call 9876543210 now.", Lang.EN, "Call 9 8 7 6 5 4 3 2 1 0 now."),
        ("You owe 12500000 rupees.", Lang.EN, "You owe 12500000 rupees."),
        ("Loan 16 has 181250 left.", Lang.EN, "Loan 16 has 181250 left."),
        ("The rate is 12.5%.", Lang.EN, "The rate is 12.5 percent."),
        ("ब्याज 12% है।", Lang.HI, "ब्याज 12 प्रतिशत है।"),
    ],
)
def test_text_is_made_safe_for_piper(text, lang, spoken):
    assert normalize(text, lang) == spoken


def test_a_word_that_only_contains_emi_is_left_alone():
    assert normalize("A premium EMIs plan.", Lang.EN) == "A premium EMIs plan."


@pytest.mark.parametrize(
    ("text", "lang", "spoken"),
    [
        # Sarvam's docs ask for commas in numbers of more than four digits.
        ("ನಿಮ್ಮ ಬಾಕಿ 206300.00 ರೂಪಾಯಿ.", Lang.KN, "ನಿಮ್ಮ ಬಾಕಿ 2,06,300 ರೂಪಾಯಿ."),
        ("ನಿಮ್ಮ EMI ೮೪೫೦ ರೂಪಾಯಿ.", Lang.KN, "ನಿಮ್ಮ EMI 8450 ರೂಪಾಯಿ."),
        ("আপনার বকেয়া ₹12,50,000।", Lang.BN, "আপনার বকেয়া 12,50,000 টাকা।"),
        ("পরের EMI 10400.50 টাকা।", Lang.BN, "পরের EMI 10,400 টাকা 50 পয়সা।"),
        # Seen live: an ISO date, and an English month, inside a Kannada sentence.
        ("ದಿನಾಂಕ 2026-10-05.", Lang.KN, "ದಿನಾಂಕ 5 ಅಕ್ಟೋಬರ್ 2026."),
        ("তারিখ 2026-10-23।", Lang.BN, "তারিখ 23 অক্টোবর 2026।"),
        ("ಫೋನ್ 9876543210.", Lang.KN, "ಫೋನ್ 9 8 7 6 5 4 3 2 1 0."),
        ("**ಖಂಡಿತ**", Lang.KN, "ಖಂಡಿತ"),
        ("Your EMI is 11450.00 rupees.", Lang.EN, "Your EMI is 11,450 rupees."),
    ],
)
def test_text_is_made_safe_for_sarvam(text, lang, spoken):
    assert for_cloud(text, lang) == spoken
