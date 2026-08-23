const sourceLabel = document.getElementById("sourceLabel");
const filePlayer = document.getElementById("filePlayer");
const liveView = document.getElementById("liveView");
const playerHint = document.getElementById("playerHint");
const modelsEl = document.getElementById("models");
const readoutEl = document.getElementById("readout");
const eventsEl = document.getElementById("events");
const refreshBtn = document.getElementById("refreshBtn");
const uploadForm = document.getElementById("uploadForm");
const videoFile = document.getElementById("videoFile");
const uploadStatus = document.getElementById("uploadStatus");
const incidentBanner = document.getElementById("incidentBanner");
const incidentTitle = document.getElementById("incidentTitle");
const incidentMeta = document.getElementById("incidentMeta");

let events = [];
let usingFilePlayer = false;

function fmtTime(epoch) {
  if (!epoch) return "—";
  return new Date(epoch * 1000).toLocaleString();
}

function showLiveView() {
  usingFilePlayer = false;
  filePlayer.hidden = true;
  liveView.hidden = false;
  playerHint.hidden = true;
  liveView.src = "/api/stream.mjpg";
}

function showFilePlayer() {
  usingFilePlayer = true;
  liveView.hidden = true;
  playerHint.hidden = true;
  filePlayer.hidden = false;
  filePlayer.src = "/api/video?ts=" + Date.now();
}

filePlayer.addEventListener("error", () => {
  showLiveView();
});

function renderModels(models) {
  const order = ["accident", "violence", "audio"];
  modelsEl.innerHTML = "";
  for (const key of order) {
    const m = models[key];
    if (!m) continue;
    const li = document.createElement("li");
    const name = document.createElement("span");
    name.textContent = m.model_name;
    const badge = document.createElement("span");
    badge.className = "badge " + (m.loaded ? "ok" : "warn");
    badge.textContent = m.loaded ? "Loaded" : "Not loaded";
    li.append(name, badge);
    modelsEl.append(li);
  }
}

function renderReadout(preds) {
  const parts = [];
  for (const key of ["accident", "violence"]) {
    const p = preds && preds[key];
    if (!p) continue;
    parts.push(
      `<p><strong>${key}</strong>: ${p.label} <span class="conf">(${(p.confidence * 100).toFixed(1)}%)</span></p>`
    );
  }
  readoutEl.innerHTML = parts.length ? parts.join("") : '<p class="muted">No predictions yet.</p>';
}

function renderEvents() {
  if (!events.length) {
    eventsEl.innerHTML =
      '<li class="muted">None yet. Events appear only after repeated high-confidence detections.</li>';
    return;
  }
  eventsEl.innerHTML = events
    .map((e) => {
      const conf = e.confidence != null ? `${(e.confidence * 100).toFixed(1)}%` : "";
      return `<li class="event"><time>${fmtTime(e.timestamp_epoch_s)}</time><strong>${e.event_type || e.verified_label}</strong> ${conf}<div class="muted">${e.camera_id || ""}</div></li>`;
    })
    .join("");
}

function renderIncident(incident) {
  if (!incident) {
    incidentBanner.hidden = true;
    return;
  }
  incidentBanner.hidden = false;
  incidentBanner.className = "banner";
  const status = (incident.status || "").toUpperCase();
  if (status === "REPORTED") {
    incidentBanner.classList.add("reported");
    incidentTitle.textContent = "ACCIDENT REPORTED";
  } else if (status === "AWAITING_ACKNOWLEDGEMENT") {
    incidentBanner.classList.add("waiting");
    incidentTitle.textContent = "AWAITING ACKNOWLEDGEMENT";
  } else {
    incidentBanner.classList.add("detected");
    incidentTitle.textContent = status === "DETECTED" ? "INCIDENT DETECTED" : status;
  }
  const conf = incident.confidence != null ? `${(incident.confidence * 100).toFixed(1)}%` : "";
  incidentMeta.textContent = [
    incident.id,
    incident.event_type,
    incident.location,
    incident.camera_id,
    fmtTime(incident.timestamp_epoch_s),
    conf,
    incident.speech_result ? `heard: ${incident.speech_result}` : "",
  ]
    .filter(Boolean)
    .join(" · ");
}

async function refreshStatus() {
  const res = await fetch("/api/status");
  const data = await res.json();
  const cam = data.camera || {};
  sourceLabel.textContent = `${cam.source_type || ""} · ${cam.source || ""}`;

  if (cam.last_error) {
    sourceLabel.textContent += ` — ${cam.last_error}`;
  }

  renderModels(data.models || {});
  renderReadout(data.last_predictions || {});
  renderIncident(data.latest_incident);

  if (data.playback && data.playback.file_available && cam.source_type === "file" && !usingFilePlayer && !filePlayer.src) {
    showFilePlayer();
  } else if (cam.source_type !== "file") {
    showLiveView();
  } else if (cam.last_error && !usingFilePlayer) {
    playerHint.hidden = false;
    playerHint.textContent = cam.last_error;
  }
}

refreshBtn.addEventListener("click", refreshStatus);
uploadForm.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  if (!videoFile.files || !videoFile.files[0]) {
    uploadStatus.textContent = "Choose a video file first.";
    return;
  }
  uploadStatus.textContent = "Uploading…";
  const body = new FormData();
  body.append("file", videoFile.files[0]);
  const res = await fetch("/api/upload", { method: "POST", body });
  const data = await res.json();
  if (!data.ok) {
    uploadStatus.textContent = data.error || "Upload failed.";
    return;
  }
  usingFilePlayer = false;
  filePlayer.removeAttribute("src");
  filePlayer.load();
  uploadStatus.textContent = "Uploaded. Inference is running on this clip.";
  showFilePlayer();
  refreshStatus();
});
refreshStatus();
setInterval(refreshStatus, 2000);

const ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws/events");
ws.onmessage = (ev) => {
  events.unshift(JSON.parse(ev.data));
  events = events.slice(0, 40);
  renderEvents();
};
