// Plays the agent's audio: PCM16 chunks at whatever rate each turn announces.
// Chunks are resampled to the output rate as one continuous stream, then scheduled
// back to back, so there are no clicks at chunk boundaries and no gaps between them.

const LEAD_SECONDS = 0.03; // how far ahead of "now" the first chunk of a turn starts

export class Player {
  constructor(context, onStarted) {
    this.context = context;
    this.onStarted = onStarted; // called once per turn, when its audio actually begins
    this.sources = [];
    this.turnId = null;
    this.reset(context.sampleRate);
  }

  reset(inputRate) {
    this.ratio = inputRate / this.context.sampleRate;
    this.position = 0;
    this.previous = 0;
    this.oddByte = null;
    this.nextTime = 0;
    this.startTimer = null;
  }

  // audio_start: a new turn, possibly at a new sample rate.
  begin(turnId, sampleRate) {
    this.stop();
    this.turnId = turnId;
    this.reset(sampleRate);
  }

  // One binary frame from the server.
  push(arrayBuffer) {
    if (this.turnId === null) return; // after a flush: belongs to the cancelled turn
    let bytes = new Uint8Array(arrayBuffer);
    if (this.oddByte !== null) {
      const joined = new Uint8Array(bytes.length + 1);
      joined[0] = this.oddByte;
      joined.set(bytes, 1);
      bytes = joined;
      this.oddByte = null;
    }
    if (bytes.length % 2 === 1) {
      this.oddByte = bytes[bytes.length - 1];
      bytes = bytes.subarray(0, bytes.length - 1);
    }
    if (bytes.length === 0) return;
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const input = new Float32Array(bytes.length / 2);
    for (let i = 0; i < input.length; i++) input[i] = view.getInt16(i * 2, true) / 0x8000;
    const samples = this.resample(input);
    if (samples.length > 0) this.schedule(samples);
  }

  resample(input) {
    const n = input.length;
    const out = new Float32Array(Math.ceil((n - this.position) / this.ratio) + 1);
    let count = 0;
    let position = this.position;
    while (position < n - 1) {
      const index = Math.floor(position);
      const fraction = position - index;
      const a = index < 0 ? this.previous : input[index];
      const b = input[index + 1];
      out[count++] = a + (b - a) * fraction;
      position += this.ratio;
    }
    this.position = position - n;
    this.previous = input[n - 1];
    return out.subarray(0, count);
  }

  schedule(samples) {
    const context = this.context;
    const buffer = context.createBuffer(1, samples.length, context.sampleRate);
    buffer.copyToChannel(samples, 0);
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    const first = this.sources.length === 0 && this.startTimer === null;
    const start = Math.max(this.nextTime, context.currentTime + LEAD_SECONDS);
    source.start(start);
    this.nextTime = start + buffer.duration;
    const entry = { source, start, duration: buffer.duration };
    this.sources.push(entry);
    if (first) {
      const turnId = this.turnId;
      const delay = Math.max(0, (start - context.currentTime) * 1000);
      this.startTimer = setTimeout(() => this.onStarted(turnId), delay);
    }
  }

  // Milliseconds of this turn's audio that have actually been played.
  playedMs() {
    const now = this.context.currentTime;
    let seconds = 0;
    for (const { start, duration } of this.sources) {
      seconds += Math.max(0, Math.min(duration, now - start));
    }
    return seconds * 1000;
  }

  // flush: stop now, drop everything queued, and say how much was heard.
  flush() {
    const played = this.playedMs();
    this.stop();
    this.turnId = null;
    return played;
  }

  stop() {
    if (this.startTimer !== null) clearTimeout(this.startTimer);
    this.startTimer = null;
    for (const { source } of this.sources) {
      try {
        source.stop();
      } catch {
        // already finished
      }
      source.disconnect();
    }
    this.sources = [];
  }
}
