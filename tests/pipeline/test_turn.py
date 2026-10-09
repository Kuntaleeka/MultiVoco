from app.core.languages import Lang
from app.core.trace import TurnTrace
from app.pipeline.turn import Segment, Turn


def make_turn(*segments: Segment) -> Turn:
    turn = Turn(id=1, lang=Lang.EN, trace=TurnTrace("c", 1, Lang.EN))
    turn.segments.extend(segments)
    return turn


FIRST = Segment("Sure, I can help with that.", ms=540.0, complete=True)
SECOND = Segment("Your next EMI is due on the fifth.", ms=680.0, complete=True)


def test_nothing_played_means_nothing_heard():
    assert make_turn(FIRST, SECOND).heard_text(0) == ""


def test_whole_sentences_are_exact():
    turn = make_turn(FIRST, SECOND)
    assert turn.heard_text(540) == FIRST.text
    assert turn.heard_text(1220) == f"{FIRST.text} {SECOND.text}"
    assert turn.heard_text(99_999) == f"{FIRST.text} {SECOND.text}"


def test_a_cut_inside_a_sentence_lands_on_a_word_boundary():
    turn = make_turn(FIRST, SECOND)
    heard = turn.heard_text(540 + 340)  # half of the second sentence
    assert heard.startswith(FIRST.text)
    tail = heard[len(FIRST.text) :].strip()
    assert tail and SECOND.text.startswith(tail)
    assert SECOND.text[len(tail)] == " "  # ended between words
    assert len(tail) <= len(SECOND.text) / 2


def test_a_cut_inside_the_first_word_keeps_nothing():
    assert make_turn(FIRST).heard_text(40) == ""


def test_unfinished_sentence_uses_the_rate_of_finished_ones():
    partial = Segment(SECOND.text, ms=200.0, complete=False)  # only 200 ms was sent
    turn = make_turn(FIRST, partial)
    heard = turn.heard_text(540 + 200)
    # 200 ms at 20 ms per character is 10 characters: "Your next ", so "Your next".
    assert heard == f"{FIRST.text} Your next"


def test_unfinished_first_sentence_falls_back_to_a_default_rate():
    turn = make_turn(Segment(FIRST.text, ms=300.0, complete=False))
    heard = turn.heard_text(300)
    assert heard == "" or FIRST.text.startswith(heard)
    assert len(heard) < len(FIRST.text)


def test_spoken_text_and_audio_length():
    partial = Segment("Half a", ms=100.0, complete=False)
    turn = make_turn(FIRST, partial)
    assert turn.audio_ms == 640.0
    assert turn.spoken_text() == f"{FIRST.text} Half a"
    assert turn.spoken_text(complete_only=True) == FIRST.text
