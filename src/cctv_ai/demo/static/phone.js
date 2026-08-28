// Phone browser live source.
//
// Frames are encoded to JPEG here rather than shipped raw: the uplink stays
// small (~20-40 KB/frame) and the server pays exactly one imdecode per frame.
// Sending is skipped whenever the socket's buffer is already backing up, so a
// weak uplink degrades by dropping frames instead of accumulating megabytes in
// the browser.

const KIND_VIDEO = 0x01;
const KIND_AUDIO = 0x02;
// Skip sending while this much is still queued in the socket.
const MAX_BUFFERED_BYTES = 512 * 1024;

const preview = document.getElementById("preview");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const statusEl = document.getElementById("phoneStatus");
const secureWarning = document.getElementById("secureWarning");
const statSent = document.getElementById("statSent");
const statSkipped = document.getElementById("statSkipped");
const statAudio = document.getElementById("statAudio");

let config = {
  send_fps: 5,
  frame_max_width: 480,
  jpeg_quality: 0.7,
  audio_chunk_ms: 1000,
};

let ws = null;
let stream = null;
let audioCtx = null;
let processor = null;
let sendTimer = null;
let canvas = null;
let ctx = null;
let running = false;
const stats = { sent: 0, skipped: 0, audio: 0 };

function setStatus(text, isError) {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", Boolean(isError));
}

function renderStats() {
  statSent.textContent = stats.sent;
  statSkipped.textContent = stats.skipped;
  statAudio.textContent = stats.audio;
}

// getUserMedia requires a secure context. http://localhost qualifies;
// http://<laptop-ip>:8000 from a phone does not.
if (!window.isSecureContext) {
  secureWarning.hidden = false;
  startBtn.disabled = true;
  setStatus("Insecure origin — camera and microphone are unavailable.", true);
}

async function loadConfig() {
  try {
    const res = await fetch("/api/phone/config");
    if (res.ok) config = Object.assign(config, await res.json());
  } catch (err) {
    // Defaults above are fine; the server echoes them again on "ready".
  }
}

function wsUrl() {
  const scheme = location.protocol === "https:" ? "wss://" : "ws://";
  return scheme + location.host + "/ws/ingest";
}

function prefixed(kind, buffer) {
  const out = new Uint8Array(buffer.byteLength + 1);
  out[0] = kind;
  out.set(new Uint8Array(buffer), 1);
  return out;
}

function captureFrame() {
  if (!running || !ws || ws.readyState !== WebSocket.OPEN) return;
  if (!preview.videoWidth) return;

  // Client-side backpressure: never queue on top of an already-full socket.
  if (ws.bufferedAmount > MAX_BUFFERED_BYTES) {
    stats.skipped += 1;
    renderStats();
    return;
  }

  const scale = Math.min(1, config.frame_max_width / preview.videoWidth);
  const w = Math.max(1, Math.round(preview.videoWidth * scale));
  const h = Math.max(1, Math.round(preview.videoHeight * scale));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  ctx.drawImage(preview, 0, 0, w, h);
  canvas.toBlob(
    async (blob) => {
      if (!blob || !running || !ws || ws.readyState !== WebSocket.OPEN) return;
      ws.send(prefixed(KIND_VIDEO, await blob.arrayBuffer()));
      stats.sent += 1;
      renderStats();
    },
    "image/jpeg",
    config.jpeg_quality
  );
}

// Raw PCM rather than MediaRecorder/Opus. The server then needs no audio
// decoder (no ffmpeg, no torchaudio), the sample rate is known and fixed, and
// the classifier sees the untouched signal. Int16 at 16 kHz mono is ~32 KB/s.
const TARGET_SAMPLE_RATE = 16000;

function startAudio() {
  const tracks = stream.getAudioTracks();
  if (!tracks.length) return;

  try {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  } catch (err) {
    return; // Audio is optional; video detection continues without it.
  }

  const source = audioCtx.createMediaStreamSource(stream);
  // ScriptProcessor is deprecated but is the only node available without
  // serving a separate AudioWorklet module file; the work here is trivial.
  const bufferSize = 4096;
  processor = audioCtx.createScriptProcessor(bufferSize, 1, 1);

  const chunkSamples = Math.round(
    (TARGET_SAMPLE_RATE * config.audio_chunk_ms) / 1000
  );
  let pending = [];
  let pendingLength = 0;

  processor.onaudioprocess = (ev) => {
    if (!running) return;
    const input = ev.inputBuffer.getChannelData(0);
    const ratio = ev.inputBuffer.sampleRate / TARGET_SAMPLE_RATE;

    // Decimate to the model's sample rate. Nearest-sample is adequate: the
    // mel front-end integrates over 25 ms windows regardless.
    const outLength = Math.floor(input.length / ratio);
    const out = new Int16Array(outLength);
    for (let i = 0; i < outLength; i++) {
      const s = Math.max(-1, Math.min(1, input[Math.floor(i * ratio)]));
      out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    pending.push(out);
    pendingLength += outLength;

    if (pendingLength < chunkSamples) return;

    const merged = new Int16Array(pendingLength);
    let off = 0;
    for (const part of pending) {
      merged.set(part, off);
      off += part.length;
    }
    pending = [];
    pendingLength = 0;

    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (ws.bufferedAmount > MAX_BUFFERED_BYTES) return;
    ws.send(prefixed(KIND_AUDIO, merged.buffer));
    stats.audio += 1;
    renderStats();
  };

  source.connect(processor);
  // ScriptProcessor only fires while connected to a destination. Routing it
  // through a muted gain node keeps it alive without echoing to the speaker.
  const mute = audioCtx.createGain();
  mute.gain.value = 0;
  processor.connect(mute);
  mute.connect(audioCtx.destination);
}

async function start() {
  startBtn.disabled = true;
  setStatus("Requesting camera and microphone…");
  await loadConfig();

  try {
    // Permissions are requested here, on an explicit tap — required both for
    // the prompt itself and for mobile autoplay policy.
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
      // The browser's voice-call defaults are actively harmful for acoustic
      // event detection, so all three are turned off:
      //   autoGainControl  normalises loudness — a scream's defining feature
      //   noiseSuppression is tuned to keep speech and discard everything else
      //   echoCancellation cancels audio the device is also playing, which
      //                    silently defeats testing with a played-back clip
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
  } catch (err) {
    setStatus(`Permission denied or no camera: ${err.message}`, true);
    startBtn.disabled = false;
    return;
  }

  preview.srcObject = stream;
  await preview.play().catch(() => {});

  canvas = document.createElement("canvas");
  ctx = canvas.getContext("2d");

  ws = new WebSocket(wsUrl());
  ws.binaryType = "arraybuffer";

  ws.onopen = () => setStatus("Connected. Waiting for the detector…");

  ws.onmessage = (ev) => {
    let msg;
    try {
      msg = JSON.parse(ev.data);
    } catch (err) {
      return;
    }
    if (msg.type === "ready") {
      config = Object.assign(config, msg.config || {});
      running = true;
      stopBtn.hidden = false;
      setStatus(`Streaming at ${config.send_fps} fps.`);
      sendTimer = setInterval(captureFrame, 1000 / config.send_fps);
      startAudio();
    } else if (msg.type === "error") {
      setStatus(msg.error || "Rejected by the server.", true);
      stop();
    }
  };

  ws.onerror = () => setStatus("Connection error.", true);
  ws.onclose = () => {
    if (running) setStatus("Disconnected.", true);
    stop();
  };
}

function stop() {
  running = false;
  if (sendTimer) {
    clearInterval(sendTimer);
    sendTimer = null;
  }
  if (processor) {
    try {
      processor.disconnect();
      processor.onaudioprocess = null;
    } catch (err) {
      /* already torn down */
    }
  }
  processor = null;
  if (audioCtx) {
    try {
      audioCtx.close();
    } catch (err) {
      /* already closed */
    }
  }
  audioCtx = null;
  if (ws && ws.readyState === WebSocket.OPEN) {
    try {
      ws.send(JSON.stringify({ type: "stop" }));
    } catch (err) {
      /* socket already going away */
    }
    ws.close();
  }
  ws = null;
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  preview.srcObject = null;
  stopBtn.hidden = true;
  startBtn.disabled = !window.isSecureContext;
  if (window.isSecureContext) setStatus("Stopped.");
}

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", () => {
  setStatus("Stopping…");
  stop();
});
// Releasing the camera on navigate-away matters on phones: the indicator light
// stays on otherwise.
window.addEventListener("pagehide", stop);
renderStats();
