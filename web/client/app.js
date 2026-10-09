// The call page. Protocol: docs/protocol.md.

import { Player } from "./player.js";

const INPUT_RATE = 16000;
const LANGUAGE_NAMES = { en: "English", hi: "हिन्दी", kn: "ಕನ್ನಡ", bn: "বাংলা" };
const STATE_LABELS = {
  connecting: "Connecting…",
  listening: "Listening",
  thinking: "Thinking…",
  speaking: "Speaking",
  handoff: "Transferring you to a person",
  idle: "Not connected",
};
const SOURCE_LABELS = { audio: "detected", script: "switched", asked: "chosen", manual: "" };

const el = {
  call: document.getElementById("call"),
  language: document.getElementById("language"),
  state: document.getElementById("state"),
  detected: document.getElementById("detected"),
  error: document.getElementById("error"),
  transcript: document.getElementById("transcript"),
};

let call = null; // everything belonging to the call in progress

function setState(state) {
  el.state.dataset.state = state;
  el.state.textContent = STATE_LABELS[state] ?? state;
}

function showError(message) {
  el.error.textContent = message;
  el.error.hidden = !message;
}

function showLanguage(lang, source) {
  const note = SOURCE_LABELS[source];
  el.detected.textContent = note ? `${LANGUAGE_NAMES[lang]} (${note})` : LANGUAGE_NAMES[lang];
  el.detected.hidden = false;
  if (el.language.value !== "auto" || source === "manual") el.language.value = lang;
}

// One bubble per role and turn. A newer message for the same pair replaces the text.
function showTranscript({ role, text, final, turn_id: turnId }) {
  const id = `line-${role}-${turnId}`;
  let line = document.getElementById(id);
  if (!line) {
    if (!text) return;
    line = document.createElement("li");
    line.id = id;
    line.className = role;
    el.transcript.append(line);
  }
  if (final && !text) {
    line.remove(); // the reply was cut off before any of it was heard
    return;
  }
  line.textContent = text;
  line.classList.toggle("pending", !final);
  if (final && role === "agent" && call?.cutTurn === turnId) line.classList.add("cut");
  line.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

function send(message) {
  if (call?.socket.readyState === WebSocket.OPEN) call.socket.send(JSON.stringify(message));
}

function onMessage(event) {
  if (!call) return;
  if (typeof event.data !== "string") {
    call.player.push(event.data);
    return;
  }
  const message = JSON.parse(event.data);
  switch (message.type) {
    case "ready":
      call.ready = true;
      break;
    case "state":
      setState(message.value);
      break;
    case "language":
      showLanguage(message.lang, message.source);
      break;
    case "transcript":
      showTranscript(message);
      break;
    case "audio_start":
      call.player.begin(message.turn_id, message.sample_rate);
      break;
    case "audio_end":
      break; // queued audio keeps playing to its end
    case "flush":
      call.cutTurn = message.turn_id;
      send({
        type: "playback_position",
        turn_id: message.turn_id,
        ms_played: Math.round(call.player.flush()),
      });
      break;
    case "error":
      showError(message.message);
      break;
  }
}

async function startCall() {
  showError("");
  el.transcript.replaceChildren();
  el.detected.hidden = true;
  el.call.disabled = true;
  setState("connecting");

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
  } catch (error) {
    const denied = error.name === "NotAllowedError" || error.name === "SecurityError";
    showError(
      denied
        ? "Microphone access was blocked. Allow it for this page and try again."
        : "No microphone was found.",
    );
    setState("idle");
    el.call.disabled = false;
    return;
  }

  // Capture at 16 kHz so the browser does the resampling. Play at the device rate.
  let captureContext;
  try {
    captureContext = new AudioContext({ sampleRate: INPUT_RATE });
  } catch {
    captureContext = new AudioContext(); // the worklet resamples instead
  }
  const playContext = new AudioContext();
  await captureContext.audioWorklet.addModule("capture-worklet.js");
  await Promise.all([captureContext.resume(), playContext.resume()]);

  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${location.host}/ws/call`);
  socket.binaryType = "arraybuffer";

  const player = new Player(playContext, (turnId) => {
    send({ type: "playback_started", turn_id: turnId, t_client_ms: performance.now() });
  });
  const microphone = captureContext.createMediaStreamSource(stream);
  const capture = new AudioWorkletNode(captureContext, "capture", {
    numberOfInputs: 1,
    numberOfOutputs: 0,
    channelCount: 1,
  });
  capture.port.onmessage = ({ data }) => {
    if (call?.ready && socket.readyState === WebSocket.OPEN) socket.send(data);
  };
  microphone.connect(capture);

  call = { socket, stream, captureContext, playContext, player, ready: false, cutTurn: null };

  socket.onopen = () => {
    send({ type: "start", lang: el.language.value });
    el.call.textContent = "End call";
    el.call.dataset.active = "true";
    el.call.disabled = false;
  };
  socket.onmessage = onMessage;
  socket.onerror = () => showError("The connection failed.");
  socket.onclose = (event) => {
    if (event.code === 4429 && !el.error.textContent) showError("All lines are busy.");
    cleanUp(event.code === 1000 ? 600 : 0);
  };
}

// Let the last words finish playing after a normal hang-up or handoff.
function cleanUp(lingerMs = 0) {
  if (!call) return;
  const ended = call;
  call = null;
  ended.stream.getTracks().forEach((track) => track.stop());
  ended.captureContext.close();
  setTimeout(() => {
    ended.player.stop();
    ended.playContext.close();
  }, lingerMs);
  if (el.state.dataset.state !== "handoff") setState("idle");
  el.call.textContent = "Start call";
  el.call.dataset.active = "false";
  el.call.disabled = false;
}

function endCall() {
  if (!call) return;
  send({ type: "end" });
  call.player.stop();
  call.socket.close(1000);
  cleanUp();
}

el.call.addEventListener("click", () => (call ? endCall() : startCall()));
el.language.addEventListener("change", () => {
  if (call?.ready && el.language.value !== "auto") {
    send({ type: "set_language", lang: el.language.value });
  }
});
window.addEventListener("pagehide", endCall);
