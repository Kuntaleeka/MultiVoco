from datetime import date

import pytest

from app.agent.guardrails import FigureLedger, wrong_language
from app.agent.prompt import _STYLE, build_system_prompt
from app.core.interfaces import CallContext
from app.core.languages import LANGUAGES, Lang, Script, script_counts

TODAY = date(2026, 10, 7)


@pytest.mark.parametrize("lang", list(Lang))
def test_the_prompt_names_the_session_language_and_its_style(lang):
    prompt = build_system_prompt(CallContext("c", lang), TODAY)
    assert f"Session language: {LANGUAGES[lang].name}." in prompt
    assert _STYLE[lang] in prompt
    assert "Wednesday, 07 October 2026" in prompt


@pytest.mark.parametrize("lang", [Lang.HI, Lang.KN, Lang.BN])
def test_the_example_lines_are_in_the_right_script_and_hold_no_account_figure(lang):
    """The model copies from its examples, so they must pass the guardrails themselves."""
    examples = _STYLE[lang].split('"')[1::2]
    assert len(examples) >= 2
    for example in examples:
        native = {s: n for s, n in script_counts(example).items() if s is not Script.LATIN}
        assert max(native, key=native.get) is LANGUAGES[lang].script
        assert not wrong_language(example, lang)
        assert FigureLedger().unsupported(example) == []


def test_an_unverified_caller_gets_the_identity_rules_and_a_verified_one_does_not():
    ctx = CallContext("c", Lang.EN, failed_verifications=1)
    before = build_system_prompt(ctx, TODAY)
    assert "NOT verified yet. Attempts left: 2." in before
    assert "Never guess or fill in a phone digit or a date." in before
    assert "04/12/1990 means 12 April 1990" in before  # speech recognition is month first

    ctx.verified_customer_id = 1
    after = build_system_prompt(ctx, TODAY)
    assert "The caller is verified." in after
    assert "Attempts left" not in after


def test_a_rejected_sentence_and_what_was_already_heard_are_added_when_present():
    ctx = CallContext("c", Lang.EN)
    plain = build_system_prompt(ctx, TODAY)
    assert "stopped before this sentence" not in plain and "already heard" not in plain

    again = build_system_prompt(
        ctx, TODAY, rejected=("Your EMI is 9000.", "it states 9000"), spoken="Sure. "
    )
    assert "'Your EMI is 9000.'" in again
    assert "Reason: it states 9000" in again
    assert "already heard: 'Sure. '" in again
