import unicodedata
from decimal import Decimal

import pytest

from app.agent.guardrails import FigureLedger, SentenceBuffer, numbers_in, wrong_language
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


def test_sentences_are_cut_only_at_real_boundaries():
    buffer = SentenceBuffer()
    out = []
    for piece in ["Your EMI is 8450", ".50 today. Is that", " fine? आपकी EMI", " है। ठीक"]:
        out.extend(buffer.feed(piece))
    assert out == ["Your EMI is 8450.50 today. ", "Is that fine? ", "आपकी EMI है। "]
    assert buffer.flush() == "ठीक"
    assert buffer.flush() == ""
