"""Turn-taking tunables. Defaults are starting points, to be tuned on real audio."""

from dataclasses import dataclass

from app.core.languages import Lang

ASK_LANGUAGE = "English, Hindi, Kannada, or Bangla?"

# What a caller might say when asked which language they want. Matched as substrings
# of a lowercased transcript.
LANGUAGE_NAMES: dict[Lang, tuple[str, ...]] = {
    Lang.EN: ("english", "inglish", "अंग्रेज़ी", "अंग्रेजी", "ಇಂಗ್ಲಿಷ್", "ইংরেজি"),
    Lang.HI: ("hindi", "हिंदी", "हिन्दी", "ಹಿಂದಿ", "হিন্দি"),
    Lang.KN: ("kannada", "ಕನ್ನಡ", "कन्नड़", "কন্নড়"),
    Lang.BN: ("bangla", "bengali", "বাংলা", "बांग्ला", "बंगाली", "ಬಂಗಾಳಿ"),
}


@dataclass(frozen=True)
class PipelineConfig:
    # Speech during a reply must last this long before it counts as an interruption.
    # Shorter sounds (a cough, "hm") are dropped.
    barge_in_min_ms: float = 250.0
    # Frames kept from before the VAD fired, so the first syllable is not clipped.
    preroll_frames: int = 8
    # How long to wait for the client's playback_position after a flush.
    position_timeout_s: float = 0.4
    # Extra time allowed for the client to finish playing before the turn is over.
    playback_margin_s: float = 0.15
    # Sentences waiting for TTS. A full queue pauses the agent's text stream.
    sentence_queue: int = 4

    # Auto-detection on the first utterance.
    min_lang_confidence: float = 0.6
    min_detect_ms: float = 400.0
    # Turns in a row whose transcript script disagrees before the language switches.
    switch_after_mismatches: int = 2


def same_group(a: Lang | None, b: Lang) -> bool:
    """Languages a caller mixes freely, served by one STT session. Hindi and English."""
    return a == b or {a, b} == {Lang.EN, Lang.HI}


def language_named_in(text: str) -> Lang | None:
    lowered = text.lower()
    for lang, names in LANGUAGE_NAMES.items():
        if any(name in lowered for name in names):
            return lang
    return None
