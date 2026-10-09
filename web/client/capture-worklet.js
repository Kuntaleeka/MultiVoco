// Mic capture: mono float samples in, 16 kHz PCM16 frames of 512 samples out.
// The page asks for a 16 kHz AudioContext, in which case this only packs frames.
// If the browser gave another rate, it resamples by linear interpolation.

const TARGET_RATE = 16000;
const FRAME_SAMPLES = 512;

class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / TARGET_RATE;
    this.position = 0; // read position in the current block; may be between -1 and 0
    this.previous = 0; // last sample of the previous block
    this.frame = new Int16Array(FRAME_SAMPLES);
    this.filled = 0;
  }

  push(value) {
    const clamped = Math.max(-1, Math.min(1, value));
    this.frame[this.filled++] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
    if (this.filled === FRAME_SAMPLES) {
      const out = this.frame.buffer;
      this.port.postMessage(out, [out]);
      this.frame = new Int16Array(FRAME_SAMPLES);
      this.filled = 0;
    }
  }

  process(inputs) {
    const input = inputs[0] && inputs[0][0];
    if (!input || input.length === 0) return true;
    const n = input.length;
    let position = this.position;
    while (position < n - 1) {
      const index = Math.floor(position);
      const fraction = position - index;
      const a = index < 0 ? this.previous : input[index];
      const b = input[index + 1];
      this.push(a + (b - a) * fraction);
      position += this.ratio;
    }
    this.position = position - n;
    this.previous = input[n - 1];
    return true;
  }
}

registerProcessor("capture", CaptureProcessor);
