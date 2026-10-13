"""Piper on ONNX Runtime, for English and Hindi. No PyTorch.

One instance is shared by every call. Synthesis runs in the pool, one sentence per
job: Piper cannot be stopped inside a sentence, so cancelling takes effect at the next
piece of audio, and the piece being computed is thrown away.
"""

import json
import logging
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any

from app.core.interfaces import ProviderError
from app.core.languages import Lang, route_for
from app.core.pools import iter_blocking, run_blocking
from app.tts.config import TTSSettings, get_tts_settings
from app.tts.normalize import normalize

log = logging.getLogger(__name__)

WARM_UP_TEXT = {Lang.EN: "Hello.", Lang.HI: "नमस्ते।"}

# Piper's phonemiser (espeak-ng) keeps its language in one global, so only one thread
# may be inside Piper at a time. The lock is taken per piece, never across a yield.
_PIPER_LOCK = threading.Lock()


def _load_voice(model: Path) -> Any:
    """Build the voice with our own ONNX Runtime session, instead of PiperVoice.load.

    Measured on 2026-10-11 (docs/tts.md): with ONNX Runtime's memory arena on, the
    default, two voices take about 620 MB. With it off they take about 240 MB, for
    synthesis that is roughly a third slower. The host has 512 MB, so it is off.
    """
    # Imported late: they are heavy, and a mocked run never needs them.
    import onnxruntime
    from piper import PiperVoice
    from piper.config import PiperConfig
    from piper.voice import ESPEAK_DATA_DIR

    options = onnxruntime.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    threads = get_tts_settings().piper_threads
    if threads:
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
    config = json.loads(model.with_suffix(".onnx.json").read_text(encoding="utf-8"))
    return PiperVoice(
        config=PiperConfig.from_dict(config),
        session=onnxruntime.InferenceSession(
            str(model), sess_options=options, providers=["CPUExecutionProvider"]
        ),
        espeak_data_dir=Path(ESPEAK_DATA_DIR),
        download_dir=model.parent,
    )


class PiperTTS:
    languages = {Lang.EN, Lang.HI}

    def __init__(
        self,
        settings: TTSSettings | None = None,
        loader: Callable[[Path], Any] = _load_voice,
    ) -> None:
        self._settings = settings or get_tts_settings()
        self._loader = loader
        self._names = {
            Lang.EN: self._settings.piper_voice_en,
            Lang.HI: self._settings.piper_voice_hi,
        }
        self._voices: dict[Lang, Any] = {}
        self._rates: dict[Lang, int] = {}
        self._load_lock = threading.Lock()

    def _model(self, language: Lang) -> Path:
        if language not in self._names:
            raise ProviderError(f"piper: no voice for {language.value}")
        return self._settings.voices_dir / f"{self._names[language]}.onnx"

    def sample_rate(self, language: Lang) -> int:
        """Read from the voice's config file, so asking does not load the model."""
        if language not in self._rates:
            config = self._model(language).with_suffix(".onnx.json")
            try:
                self._rates[language] = int(
                    json.loads(config.read_text(encoding="utf-8"))["audio"]["sample_rate"]
                )
            except (OSError, KeyError, ValueError) as error:
                raise ProviderError(f"piper: cannot read {config}: {error}") from error
        return self._rates[language]

    def load(self, language: Lang) -> Any:
        """The loaded voice. Blocking: call it from the pool."""
        with self._load_lock:
            if language not in self._voices:
                model = self._model(language)
                if not model.exists():
                    raise ProviderError(
                        f"piper: voice file {model} is missing. Download it with "
                        f"`python -m piper.download_voices {model.stem} --download-dir "
                        f"{self._settings.voices_dir}`"
                    )
                self._voices[language] = self._loader(model)
            return self._voices[language]

    def _pieces(self, text: str, language: Lang) -> Iterator[bytes]:
        """Blocking generator: PCM in pieces of tts_chunk_ms. Runs in the pool."""
        voice = self.load(language)
        step = self.sample_rate(language) * self._settings.tts_chunk_ms // 1000 * 2
        sentences = iter(voice.synthesize(text))
        while True:
            with _PIPER_LOCK:
                chunk = next(sentences, None)
            if chunk is None:
                return
            pcm = chunk.audio_int16_bytes
            for start in range(0, len(pcm), step):
                yield pcm[start : start + step]

    async def synthesize(self, text: str, language: Lang) -> AsyncIterator[bytes]:
        spoken = normalize(text, language)
        if not spoken:
            return
        self.sample_rate(language)  # fail early, before any audio, if the voice is absent
        try:
            async for piece in iter_blocking(lambda: self._pieces(spoken, language)):
                yield piece
        except ProviderError:
            raise
        except Exception as error:
            raise ProviderError(f"piper: synthesis failed: {error}") from error

    async def warm_up(self) -> None:
        """Load each voice in use and synthesise once, so the first caller does not wait.

        Only languages routed to Piper are loaded. Each voice costs memory, so routing
        Hindi elsewhere (MULTIVOCO_ROUTE_OVERRIDES) keeps its voice out of the process.
        """
        for language in self._names:
            if route_for(language).tts != "piper":
                continue
            try:
                await run_blocking(self.load, language)
                async for _ in self.synthesize(WARM_UP_TEXT[language], language):
                    pass
            except ProviderError as error:
                # A missing voice must not stop the app: that language fails per call.
                log.warning("piper warm-up skipped for %s: %s", language.value, error)
