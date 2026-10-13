"""Silero VAD on ONNX Runtime, with no PyTorch.

The model gives a speech probability per 32 ms frame. SpeechGate turns that stream of
probabilities into SPEECH_START and SPEECH_END. They are separate so the gate can be
tested, and tuned, without the model.
"""

import threading
from pathlib import Path

import numpy as np

from app.audio.config import VADSettings, get_vad_settings
from app.core.interfaces import (
    FRAME_BYTES,
    FRAME_MS,
    FRAME_SAMPLES,
    INPUT_SAMPLE_RATE,
    VADEvent,
    VADEventKind,
)
from app.core.pools import run_blocking

MODEL_PATH = Path(__file__).parent / "models" / "silero_vad.onnx"

# The model wants the last 64 samples of the previous frame in front of each frame.
_CONTEXT_SAMPLES = 64
_STATE_SHAPE = (2, 1, 128)

_session = None
_session_lock = threading.Lock()


def _get_session():
    """One ONNX session per process. It holds no per-call state, so calls can share it."""
    global _session
    with _session_lock:
        if _session is None:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = 1  # one small model: more threads only add overhead
            options.inter_op_num_threads = 1
            _session = ort.InferenceSession(
                str(MODEL_PATH), sess_options=options, providers=["CPUExecutionProvider"]
            )
        return _session


class SpeechGate:
    """Probabilities in, speech start and end out."""

    def __init__(self, settings: VADSettings | None = None) -> None:
        self._s = settings or get_vad_settings()
        self._end_frames = max(1, round(self._s.end_silence_ms / FRAME_MS))
        self._speaking = False
        self._loud = 0
        self._quiet = 0
        self.is_speech = False

    def update(self, probability: float) -> VADEvent | None:
        if not self._speaking:
            self.is_speech = probability >= self._s.start_threshold
            self._loud = self._loud + 1 if self.is_speech else 0
            if self._loud >= self._s.start_frames:
                self._speaking = True
                self._loud = self._quiet = 0
                return VADEvent(VADEventKind.SPEECH_START)
            return None
        self.is_speech = probability >= self._s.end_threshold
        if self.is_speech:
            self._quiet = 0
            return None
        self._quiet += 1
        if self._quiet >= self._end_frames:
            self._speaking = False
            self._quiet = 0
            return VADEvent(VADEventKind.SPEECH_END)
        return None


class SileroVAD:
    """One instance per call: it carries the model's recurrent state for that call."""

    def __init__(self, settings: VADSettings | None = None) -> None:
        self._gate = SpeechGate(settings)
        self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros(_CONTEXT_SAMPLES, dtype=np.float32)
        self._rate = np.array(INPUT_SAMPLE_RATE, dtype=np.int64)
        self.probability = 0.0

    @property
    def is_speech(self) -> bool:
        return self._gate.is_speech

    def _infer(self, frame: bytes) -> float:
        samples = np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768.0
        window = np.concatenate((self._context, samples))[np.newaxis, :]
        output, self._state = _get_session().run(
            None, {"input": window, "state": self._state, "sr": self._rate}
        )
        self._context = samples[-_CONTEXT_SAMPLES:]
        return float(output.squeeze())

    async def process(self, frame: bytes) -> VADEvent | None:
        if len(frame) != FRAME_BYTES:
            raise ValueError(f"expected {FRAME_SAMPLES} samples, got {len(frame) // 2}")
        self.probability = await run_blocking(self._infer, frame)
        return self._gate.update(self.probability)


async def warm_up() -> None:
    """Load the model and run it once, so the first call does not pay for it."""
    await SileroVAD().process(bytes(FRAME_BYTES))
