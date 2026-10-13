"""Cuts streamed agent text into sentences for TTS.

A sentence ends at a terminator followed by whitespace. Waiting for the whitespace is
what keeps "8,450.50" and "3.5%" in one piece when the text arrives token by token.
The danda needs no whitespace: it has no other use. Text may arrive a token at a time
or several sentences at once, and both give the same sentences.

Models also leave the space out ("...on 23 October 2026.Is there anything else?").
So a terminator ends a sentence without whitespace too, when what follows can only be
the start of a new one: a letter after "?" or "!", or a capital or a non-Latin letter
after a full stop. A digit or a small Latin letter after a full stop does not, which
keeps "8,450.50", "example.com", and "v2.x" whole.
"""

from app.core.languages import LANGUAGES

_TERMINATORS = frozenset("".join(info.terminators for info in LANGUAGES.values()))
_DANDA = "।"
_CLOSERS = "\"')]”’"

# Words that end in a full stop without ending the sentence. Lowercase, without the stop.
_ABBREVIATIONS = frozenset(
    {"mr", "mrs", "ms", "dr", "shri", "smt", "rs", "no", "nos", "a/c", "ac", "st", "vs", "approx"}
)


class SentenceSplitter:
    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, text: str) -> list[str]:
        """Add text. Returns the sentences that are now complete."""
        self._buffer += text
        sentences: list[str] = []
        start = 0
        i = 0
        buffer = self._buffer
        while i < len(buffer):
            if buffer[i] not in _TERMINATORS:
                i += 1
                continue
            end = i + 1
            while end < len(buffer) and (buffer[end] in _TERMINATORS or buffer[end] in _CLOSERS):
                end += 1
            if buffer[i] != _DANDA:
                if end == len(buffer):
                    break  # need the next character to know whether this ends the sentence
                if self._is_abbreviation(buffer, start, i) or not (
                    buffer[end].isspace() or self._starts_sentence(buffer[end - 1], buffer[end])
                ):
                    i = end
                    continue
            sentence = buffer[start:end].strip()
            if sentence:
                sentences.append(sentence)
            start = i = end
        self._buffer = buffer[start:]
        return sentences

    def flush(self) -> str | None:
        """The unfinished remainder, when the agent has no more text."""
        rest, self._buffer = self._buffer.strip(), ""
        return rest or None

    @staticmethod
    def _starts_sentence(terminator: str, following: str) -> bool:
        """Does `following`, coming straight after `terminator`, begin a new sentence?"""
        if not following.isalpha():
            return False
        if terminator != ".":
            return True
        return following.isupper() or not following.isascii()

    @staticmethod
    def _is_abbreviation(buffer: str, start: int, stop: int) -> bool:
        if buffer[stop] != ".":
            return False
        word = buffer[start:stop].rsplit(None, 1)[-1:] or [""]
        token = word[0].lower()
        # A single letter before a stop is an initial ("A. Rao"), not a sentence.
        return token in _ABBREVIATIONS or (len(token) == 1 and token.isalpha() and token.isascii())
