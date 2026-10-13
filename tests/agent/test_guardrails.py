import unicodedata
from decimal import Decimal

import pytest

from app.agent.guardrails import (
    FigureLedger,
    SentenceBuffer,
    numbers_in,
    repair_misheard_digits,
    starts_other_speaker,
    without_own_label,
    wrong_language,
)
from app.agent.lines import GREETINGS, HANDOFF_LINE, SAFE_REPLY
from app.core.languages import Lang

EMI_RESULT = {"loan_id": 1, "emi_amount": "8450.00", "next_due_date": "2026-11-05"}


@pytest.mark.parametrize(
    ("text", "values"),
    [
        ("Your EMI is 8,450.00.", ["8450.00"]),
        ("Outstanding is 1,81,250.50 on loan 1.", ["181250.50", "1"]),  # Indian grouping
        ("आपकी EMI ८४५० है।", ["8450"]),
        ("ನಿಮ್ಮ EMI ೮೪೫೦.", ["8450"]),
        ("আপনার EMI ৮৪৫০।", ["8450"]),
        ("Due on the 5th, 2026.", ["5", "2026"]),
        ("No figures here.", []),
    ],
)
def test_numbers_are_read_in_every_numeral_system(text, values):
    assert [number for _, number in numbers_in(text)] == [Decimal(v) for v in values]


def test_figures_from_a_tool_result_may_be_said_in_any_form():
    ledger = FigureLedger()
    ledger.add(EMI_RESULT)
    for text in (
        "Your EMI of 8450.00 is due on 5 November 2026.",
        "Your EMI of 8,450 is due on 05/11/2026.",
        "ನಿಮ್ಮ EMI ೮೪೫೦, ನವೆಂಬರ್ ೫ ರಂದು.",
        "আপনার EMI ৮৪৫০, ৫ নভেম্বর।",
        "Please give the last 4 digits of your phone number.",
    ):
        assert ledger.unsupported(text) == []


@pytest.mark.parametrize(
    ("text", "bad"),
    [
        ("Your EMI is 8500.", ["8500"]),
        ("Your EMI is 8450.50.", ["8450.50"]),
        ("आपकी EMI ९००० है।", ["9000"]),
        ("ನಿಮ್ಮ EMI ೯೦೦೦.", ["9000"]),
        ("আপনার EMI ৯০০০।", ["9000"]),
        ("It is about eight thousand rupees.", ["thousand"]),
        ("करीब आठ हज़ार रुपये।", ["हज़ार"]),
        ("Lagbhag 2 lakh baaki hai.", ["2", "lakh"]),
    ],
)
def test_figures_no_tool_returned_are_caught(text, bad):
    ledger = FigureLedger()
    ledger.add(EMI_RESULT)
    assert ledger.unsupported(text) == [unicodedata.normalize("NFC", item) for item in bad]


@pytest.mark.parametrize(
    "text",
    [
        "ನಿಮ್ಮ ಬಾಕಿ ಸುಮಾರು ಎರಡು ಲಕ್ಷ ರೂಪಾಯಿ.",  # about two lakh
        "ನಿಮ್ಮ EMI ಎಂಟು ಸಾವಿರದ ಐನೂರು.",  # eight thousand five hundred
        "ಐದುಸಾವಿರ ರೂಪಾಯಿ ಕಟ್ಟಬೇಕು.",  # joined: five-thousand
        "ಒಂದು ಕೋಟಿ ರೂಪಾಯಿ.",
        "আপনার বকেয়া প্রায় দুই লাখ টাকা।",
        "আপনার EMI আট হাজার পাঁচশো টাকা।",
        "আপনার বকেয়া এক লক্ষ টাকা।",
        "এক কোটি টাকা।",
    ],
)
def test_figures_written_as_kannada_or_bengali_words_are_caught(text):
    ledger = FigureLedger()
    ledger.add(EMI_RESULT)
    assert ledger.unsupported(text) != []


@pytest.mark.parametrize(
    "text",
    [
        "ಇದರ ಲಕ್ಷಣ ಏನು ಎಂದು ನೋಡುತ್ತೇನೆ.",  # ಲಕ್ಷಣ: feature, not lakh
        "ದಯವಿಟ್ಟು ಲಕ್ಷ್ಯ ಕೊಡಿ.",  # ಲಕ್ಷ್ಯ: attention
        "ಖಂಡಿತ, ನಾನು ಈಗ ನೋಡುತ್ತೇನೆ.",
        "অনুগ্রহ করে লক্ষ্য করুন।",  # লক্ষ্য: notice
        "আপনি কবে শোধ করবেন?",  # শোধ: repay, not a hundred
        "নিশ্চয়ই, আমি এখনই দেখছি।",
    ],
)
def test_everyday_kannada_and_bengali_words_are_not_taken_for_figures(text):
    assert FigureLedger().unsupported(text) == []


@pytest.mark.parametrize(
    ("heard", "meant"),
    [
        # From a real call: the digits 2 2 1 7.
        ("आखिरी चार अंक दो दो एक साथ हैं", "आखिरी चार अंक दो दो एक सात हैं"),
        ("नौ आठ साथ छह", "नौ आठ सात छह"),
        # "एक साथ" is "together", and so is "साथ" anywhere else.
        ("मैं दोनों EMI एक साथ भर दूँगा", "मैं दोनों EMI एक साथ भर दूँगा"),
        ("मेरे साथ ऐसा क्यों हुआ", "मेरे साथ ऐसा क्यों हुआ"),
        ("दो तीन साथी आए थे", "दो तीन साथी आए थे"),  # साथी: companions
        ("My number is 4821", "My number is 4821"),
    ],
)
def test_a_misheard_hindi_seven_is_put_right_only_among_digits(heard, meant):
    assert repair_misheard_digits(heard) == meant


def test_nothing_is_allowed_before_any_tool_has_run():
    assert FigureLedger().unsupported("Your balance is 181250.50.") == ["181250.50"]


def test_true_and_false_in_a_result_are_not_figures():
    ledger = FigureLedger()
    ledger.add({"verified": True, "attempts_left": 2})
    assert ledger.unsupported("You have 2 attempts.") == []
    assert ledger.unsupported("You have 1 attempt.") == ["1"]


@pytest.mark.parametrize(
    ("text", "lang", "wrong"),
    [
        ("Your next EMI is due soon.", Lang.EN, False),
        ("आपकी अगली EMI जल्द है।", Lang.EN, True),
        ("Thank you, ರವಿ. Your loan is active.", Lang.EN, False),  # a name in another script
        ("आपकी अगली EMI जल्द है।", Lang.HI, False),
        ("Aapki agli EMI jald hi due hai.", Lang.HI, False),  # Hinglish
        ("ನಿಮ್ಮ ಮುಂದಿನ EMI ಶೀಘ್ರದಲ್ಲೇ ಇದೆ.", Lang.HI, True),
        ("ನಿಮ್ಮ ಮುಂದಿನ EMI ಶೀಘ್ರದಲ್ಲೇ ಇದೆ.", Lang.KN, False),
        ("Your next EMI is due soon.", Lang.KN, True),
        ("EMI OK.", Lang.KN, False),  # too short to judge
        ("আপনার পরের EMI শীঘ্রই।", Lang.BN, False),
        ("आपकी अगली EMI जल्द है।", Lang.BN, True),
        ("Your next EMI is due soon.", Lang.BN, True),
    ],
)
def test_wrong_language_is_judged_by_script(text, lang, wrong):
    assert wrong_language(text, lang) is wrong


@pytest.mark.parametrize("lines", [GREETINGS, SAFE_REPLY, HANDOFF_LINE])
@pytest.mark.parametrize("lang", list(Lang))
def test_fixed_lines_pass_the_guardrails_they_stand_in_for(lines, lang):
    assert FigureLedger().unsupported(lines[lang]) == []
    assert not wrong_language(lines[lang], lang)


@pytest.mark.parametrize(
    ("text", "sentences"),
    [
        # Seen in a real call: no space after the sentence.
        (
            "It is due on 23 October 2026.Is there anything else?Yes.",
            ["It is due on 23 October 2026.", "Is there anything else?", "Yes."],
        ),
        ("आपकी EMI देय है।क्या आप भुगतान करेंगे?", ["आपकी EMI देय है।", "क्या आप भुगतान करेंगे?"]),
        ("ಬಾಕಿ ಇದೆ.ಇನ್ನೇನು ಬೇಕು?", ["ಬಾಕಿ ಇದೆ.", "ಇನ್ನೇನು ಬೇಕು?"]),
        ("Pay 8450.50 at example.com today.", ["Pay 8450.50 at example.com today."]),
        ("Really?yes, really.", ["Really?", "yes, really."]),
    ],
)
def test_sentences_are_cut_even_when_the_model_leaves_out_the_space(text, sentences):
    buffer = SentenceBuffer()
    out = []
    for ch in text:  # one character at a time, the hardest way for it to arrive
        out.extend(buffer.feed(ch))
    out.append(buffer.flush())
    assert out == sentences


@pytest.mark.parametrize("line", ["User: yes", "user:Apologies", " Caller : ok", "Customer: hm"])
def test_a_line_for_the_other_speaker_is_recognised(line):
    assert starts_other_speaker(line)


@pytest.mark.parametrize("line", ["Your user ID is ready.", "Users: none.", "The caller said so."])
def test_ordinary_sentences_are_not_taken_for_speaker_labels(line):
    assert not starts_other_speaker(line)


def test_the_agents_own_label_is_only_a_prefix():
    assert without_own_label("Assistant: Sure, one moment.") == "Sure, one moment."
    assert without_own_label("Agent:Sure.") == "Sure."
    assert without_own_label("An agent: no.") == "An agent: no."


def test_sentences_are_cut_only_at_real_boundaries():
    buffer = SentenceBuffer()
    out = []
    for piece in ["Your EMI is 8450", ".50 today. Is that", " fine? आपकी EMI", " है। ठीक"]:
        out.extend(buffer.feed(piece))
    assert out == ["Your EMI is 8450.50 today. ", "Is that fine? ", "आपकी EMI है। "]
    assert buffer.flush() == "ठीक"
    assert buffer.flush() == ""
