import pytest

from app.core.languages import Lang
from app.core.mocks import AGENT_REPLIES
from app.pipeline.splitter import SentenceSplitter


def split(chunks: list[str]) -> list[str]:
    splitter = SentenceSplitter()
    out = [sentence for chunk in chunks for sentence in splitter.feed(chunk)]
    if (rest := splitter.flush()) is not None:
        out.append(rest)
    return out


def test_sentences_come_out_as_soon_as_they_are_complete():
    splitter = SentenceSplitter()
    assert splitter.feed("Sure, I can help") == []
    assert splitter.feed(" with that. Your next") == ["Sure, I can help with that."]
    assert splitter.feed(" EMI is due on the fifth.") == []  # could still be "fifth.5"
    assert splitter.flush() == "Your next EMI is due on the fifth."
    assert splitter.flush() is None


@pytest.mark.parametrize("lang", list(Lang))
def test_chunking_does_not_change_the_sentences(lang):
    text = AGENT_REPLIES[lang]
    whole = split([text])
    assert len(whole) == 2
    assert split(list(text)) == whole  # a character at a time
    assert split([word + " " for word in text.split(" ")]) == whole  # a word at a time
    assert split([sentence + " " for sentence in whole]) == whole  # a sentence at a time


def test_amounts_and_decimals_stay_in_one_piece():
    text = "Your EMI is Rs. 8,450.50 at 10.5% interest. It is due on 05.11.2026."
    assert split(list(text)) == [
        "Your EMI is Rs. 8,450.50 at 10.5% interest.",
        "It is due on 05.11.2026.",
    ]


def test_titles_and_initials_do_not_end_a_sentence():
    assert split(["Thank you Mr. A. Rao. Dr. Shetty will call you."]) == [
        "Thank you Mr. A. Rao.",
        "Dr. Shetty will call you.",
    ]


def test_danda_ends_a_sentence_without_needing_a_space():
    splitter = SentenceSplitter()
    assert splitter.feed("जी, मैं मदद करती हूँ।") == ["जी, मैं मदद करती हूँ।"]
    assert splitter.feed("আমি সাহায্য করছি।আপনার EMI পাঁচ তারিখে।") == [
        "আমি সাহায্য করছি।",
        "আপনার EMI পাঁচ তারিখে।",
    ]


def test_questions_exclamations_and_closing_quotes():
    assert split(['She said "pay now!" Did you? Yes...  ok']) == [
        'She said "pay now!"',
        "Did you?",
        "Yes...",
        "ok",
    ]


def test_whitespace_only_input_gives_nothing():
    assert split(["  ", "\n"]) == []
