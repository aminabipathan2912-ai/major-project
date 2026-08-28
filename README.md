# CCTV Accident & Violence Detection

Local video inference (accident + violence) with temporal verification, escalating verified
incidents to a hosted backend that places an acknowledged phone call.

> **Safety.** Official emergency numbers (100 / 108 / 112) are **never** dialed. The system calls
> only the test/operator number you configure. There is no two-way AI conversation — the call
> plays an alert and listens for the word "Done".

---

## Architecture

Three deployable pieces. The cloud host never loads a model; the phone never runs inference.

```
PHONE BROWSER  (optional live source)      LOCAL INFERENCE APP            HOSTED ALERT BACKEND
─────────────────────────────────────      ───────────────────            ────────────────────
 camera + mic  ─── HTTPS tunnel ──→  WS /ws/ingest                         PostgreSQL incidents
 JPEG frames @ 5 fps                  ↓                                    Sarvam TTS → .wav
 audio chunks @ 1 s                 FrameBuffer  ←── OpenCV (file/webcam/rtsp)
                                      ↓                                    Twilio voice call
                              accident + violence models                        ↓
                              (one clip, one preprocess, shared tensor)    caller says "Done"
                                      ↓                                         ↓
                              TemporalVerifier (min_hits / window / cooldown)  status=REPORTED
                                      ↓
                       ┌──────────────┴──────────────┐
                  WS /ws/events                 POST /api/incidents ──────────→
                  (dashboard)                   (bearer token)
```

| Piece | Runs on | Owns |
|---|---|---|
| **Local inference app** (repo root) | your laptop / edge box | video, OpenCV, PyTorch, `models/*.pt` |
| **Hosted alert backend** (`alert_backend/`) | Render (or any HTTPS host) | Postgres, Twilio, Sarvam keys |
| **Phone page** (`/phone`) | any phone browser | camera + mic capture only |

Secrets never cross: the local app holds no Twilio/Sarvam/database credentials, only a shared
bearer token.

---

# Part A — Local inference app

## A1. Install

Developed and verified on Python 3.14. On WSL/Ubuntu:

```bash
cd /mnt/d/programming/major-project
sudo apt-get install -y python3-venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## A2. Add the trained weights

```
models/accident_best.pt      # EfficientNet-B0 clip classifier, 8 frames
models/violence_best.pt
```

Produced by the notebooks in `training/` — see `training/README.md`. If a `.pt` is missing that
model reports `loaded = false` and returns **no** predictions; it never fakes one.

## A3. Configure

```bash
cp .env.example .env
```

Minimum to run locally with no phone calls:

```env
CAMERA_SOURCE_TYPE=file
CAMERA_SOURCE=tests/fixtures/file_002001.mp4
EMERGENCY_MODE=log-only
```

## A4. Run

```bash
source .venv/bin/activate
python run_demo_server.py          # PORT=8000 by default
```

Open <http://localhost:8000>. Upload a clip from the dashboard, or point `CAMERA_SOURCE` at one.

Other sources:

```bash
CAMERA_SOURCE_TYPE=webcam CAMERA_SOURCE=0                                   python run_demo_server.py
CAMERA_SOURCE_TYPE=rtsp   CAMERA_SOURCE=rtsp://user:pass@camera-ip/stream   python run_demo_server.py
```

## A5. Verify the install

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/check_preprocess_equivalence.py <a 720p or 1080p clip>
.venv/bin/python scripts/benchmark_pipeline.py <same clip>
```

The equivalence check **must print `RESULT: PASS`**. It proves the ingest downscale leaves the
tensor reaching the models bit-identical to the training transform — the one failure mode that
would silently degrade the trained weights. Run it against a 720p/1080p clip; on a small fixture
the downscale is a pass-through and the comparison proves nothing.

---

# Part B — Hosted alert backend

Only needed when you want a real phone call. Skip it entirely with `EMERGENCY_MODE=log-only`.

## B1. Database (Neon or any Postgres)

Create a database and copy its connection string. Keep `?sslmode=require` for Neon. Tables are
created automatically on first boot.

## B2. Twilio

1. Buy/verify a Twilio number → `TWILIO_FROM_NUMBER`.
2. Verify **your own** mobile as a caller ID → `TWILIO_TO_NUMBER` (E.164, e.g. `+91...`).
3. Pick a call mode:

| `TWILIO_CALL_MODE` | Behaviour | When |
|---|---|---|
| `custom` | Full flow: app TwiML, spoken "Done" acknowledgement, call-progress callbacks | Upgraded (paid) account |
| `trial-custom` | App TwiML + spoken "Done", but no progress callbacks | Trial account |
| `trial-template` | Twilio's hosted sample message only | Smoke-testing a trial account |

## B3. Sarvam TTS (optional)

Set `SARVAM_API_KEY` for generated Hindi/English alert audio. Left blank, Twilio's built-in
text-to-speech is used instead — the call still works.

## B4. Deploy to Render

Create a Web Service from this repo with **root directory `alert_backend`**:

- Build: `pip install -r requirements.txt`
- Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

(`alert_backend/Dockerfile` is there if you prefer a container deploy — it pins Python 3.12 and
serves on port 10000. Tables are created automatically on first boot either way.)

Then set every variable from `alert_backend/.env.example` in Render's Environment dashboard:

```env
DATABASE_URL=postgresql://USER:PASSWORD@HOST:5432/DB?sslmode=require
ALERT_INGEST_TOKEN=<long random secret>
PUBLIC_BASE_URL=https://your-alert-backend.onrender.com   # no trailing slash
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_FROM_NUMBER=+1...
TWILIO_TO_NUMBER=+91...
TWILIO_CALL_MODE=custom
TWILIO_VALIDATE_WEBHOOKS=true
SARVAM_API_KEY=
```

`PUBLIC_BASE_URL` is **required**: Twilio fetches TwiML and audio from it and posts webhooks back,
so it must be the public HTTPS URL of this service.

Check it is alive:

```bash
curl https://your-alert-backend.onrender.com/health
```

## B5. Point the local app at it

In the local `.env`:

```env
EMERGENCY_MODE=remote
ALERT_BACKEND_URL=https://your-alert-backend.onrender.com
ALERT_BACKEND_TOKEN=<exactly the same value as ALERT_INGEST_TOKEN>
```

Restart the local app. A verified event now creates an incident, generates audio, and calls your
number. Say **"Done"** and the incident becomes `REPORTED`.

> Render free instances sleep. The first escalation after idle may time out while the service
> wakes — the dashboard shows the delivery state.

---

# Part C — Phone as a live camera (cloudflared)

The `/phone` page streams a phone's camera and mic into the same pipeline over a WebSocket.

## C1. Why a tunnel is mandatory

`getUserMedia` only works in a **secure context**. `http://localhost` qualifies;
`http://192.168.x.x:8000` from your phone **does not** — the browser refuses camera access and no
code can work around it. A tunnel gives the local app a real HTTPS origin.

## C2. Install cloudflared

```bash
# Debian/Ubuntu/WSL
curl -L -o cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb

# macOS
brew install cloudflared
```

## C3. Run the tunnel

With the local app already running on port 8000, in a second terminal:

```bash
cloudflared tunnel --url http://localhost:8000
```

It prints a public URL:

```
https://random-words-here.trycloudflare.com
```

No account or DNS setup needed for this quick-tunnel mode. (`ngrok http 8000` works identically.)

## C4. Start monitoring

1. Open **`https://<your-tunnel>.trycloudflare.com/phone`** on the phone.
2. Tap **Start monitoring** — permissions are requested on that tap, as browsers require.
3. Allow camera and microphone.

The dashboard's Source panel switches to **Phone live**. The page's WebSocket URL is derived from
`location.host`, so it follows the tunnel automatically; `PUBLIC_BASE_URL` on the hosted backend is
unaffected because the tunnel only fronts the *local* app.

Stop, or close the tab, and the pipeline reverts to the previous source.

## C5. What is bounded

The browser cannot overwhelm the backend, regardless of what it does:

| Bound | Setting | Behaviour |
|---|---|---|
| Frame queue | `PHONE_FRAME_QUEUE_MAX=4` | drop-oldest — a fresh frame beats a complete history |
| Audio ring | `PHONE_AUDIO_BUFFER_MAX=8` | oldest chunks dropped, drops counted |
| Message size | `PHONE_MAX_MESSAGE_BYTES=1048576` | oversized messages rejected, never buffered |
| Concurrent streams | `PHONE_MAX_SESSIONS=1` | second connection refused cleanly |

Two streams into one buffer would interleave two scenes into a single clip and corrupt inference,
hence the session cap. Drop counters appear in `/api/status` and on the dashboard, so loss is
visible rather than silent.

**Audio is received and bounded, not classified.** There is no audio model. `AudioModelAdapter`
exposes `predict_audio(chunks)` returning `None`, and `AUDIO_EVENT` already exists in `EventType`,
so a real classifier drops in without touching the pipeline. Nothing fabricates an audio detection.

---

## Configuration reference

All settings live in `.env` (see `.env.example` for the annotated version).

### Source and buffering

| Key | Default | Notes |
|---|---|---|
| `CAMERA_SOURCE_TYPE` | `file` | `file` \| `webcam` \| `rtsp` (`phone` is set automatically) |
| `CAMERA_SOURCE` | — | path, device index, or RTSP URL |
| `FILE_LOOP` / `FILE_REALTIME` | `false` / `true` | play a clip once, at its native FPS |
| `FRAME_BUFFER_MAXLEN` | `32` | frames retained |
| `INFERENCE_FRAME_SIZE` | `256` | ingest downscale; matches the training transform's resize |
| `INGEST_RESIZE_EXACT` | `true` | `true` = PIL BICUBIC, bit-exact. `false` = faster cv2, **not** identical |
| `INGEST_SAMPLE_FPS` | `0` | `0` keeps every decoded frame. Raising it widens the clip's time span |

### Inference

| Key | Default | Notes |
|---|---|---|
| `INFERENCE_INTERVAL_MS` | `1000` | one cycle scores both models |
| `CLIP_FRAME_COUNT` | `8` | must match the checkpoints |
| `CLIP_PREPROCESS_FAST` | `false` | batched cv2 preprocess; verified bit-exact, still opt-in |
| `ACCIDENT_CLIP_WINDOW_SEC` | `0` | `0` = last-N frames. A value spreads frames over N seconds |
| `VIOLENCE_CLIP_WINDOW_SEC` | `0` | same; violence is the model trained on real temporal spread |
| `TORCH_NUM_THREADS` | `0` | `0` = torch default. Try `4` on a shared edge box |

> Raising a clip window past ~1 s needs more history than `FRAME_BUFFER_MAXLEN=32` holds at 30 fps
> (~1.05 s). Raise the buffer to 48–64 first.

### Verification (duplicate/cooldown protection)

| Key | Default |
|---|---|
| `ACCIDENT_CONFIDENCE_THRESHOLD` / `MIN_HITS` / `WINDOW_SEC` / `COOLDOWN_SEC` | `0.6` / `3` / `3.0` / `30` |
| `VIOLENCE_CONFIDENCE_THRESHOLD` / `MIN_HITS` / `WINDOW_SEC` / `COOLDOWN_SEC` | `0.6` / `2` / `3.0` / `20` |

An event escalates only after `MIN_HITS` high-confidence detections inside `WINDOW_SEC`, then
`COOLDOWN_SEC` blocks repeats. A recorded clip escalates each event type at most once.

### Preview, escalation, phone

| Key | Default |
|---|---|
| `PREVIEW_FPS` / `PREVIEW_MAX_WIDTH` / `PREVIEW_JPEG_QUALITY` | `10` / `640` / `80` |
| `EMERGENCY_MODE` | `log-only` \| `remote` |
| `INCIDENT_POLL_INTERVAL_SEC` | `3.0` — background refresh; `/api/status` never blocks on it |
| `PHONE_SEND_FPS` / `PHONE_FRAME_MAX_WIDTH` / `PHONE_JPEG_QUALITY` | `5` / `480` / `0.7` |
| `PHONE_AUDIO_CHUNK_MS` | `1000` |

---

## HTTP surface (local app)

| Route | Purpose |
|---|---|
| `GET /` | dashboard |
| `GET /phone` | phone capture page |
| `GET /api/status` | pipeline, models, verification, phone, sampling |
| `POST /api/upload` | upload a clip and switch to it (streamed to disk) |
| `GET /api/video` · `/api/frame.jpg` · `/api/stream.mjpg` | playback / preview |
| `GET /api/phone/config` | capture knobs the phone page reads |
| `WS /ws/events` | verified events, pushed |
| `WS /ws/ingest` | phone ingest (`0x01` video, `0x02` audio) |

---

## Repository layout

```
src/cctv_ai/
  camera/         media_source (protocol) · opencv_camera · phone_stream · frame_buffer · frame_scaler
  inference/      accident/ · violence/ · audio/ (interface only) · clip_classifier · loader
  event_engine/   temporal verification — thresholds, min-hits, cooldown
  emergency/      EmergencyProvider: log-only or remote POST
  core/           PipelineService — one loop, both models, shared tensor
  demo/           FastAPI server + dashboard + phone page
alert_backend/    hosted service: Postgres, Sarvam TTS, Twilio (no models)
training/         Kaggle notebooks that produced models/*.pt
scripts/          equivalence check, benchmark, test-clip generator
```

---

## Training the models

See `training/README.md`. Both models are **EfficientNet-B0 (pretrained) + mean-pool over 8
frames** — accurate enough, fits a free Kaggle GPU, no research complexity.

| Notebook | Dataset | Output |
|---|---|---|
| `training/accident/train_accident.ipynb` | [ckay16/accident-detection-from-cctv-footage](https://www.kaggle.com/datasets/ckay16/accident-detection-from-cctv-footage) | `accident_best.pt` |
| `training/violence/train_violence.ipynb` | [magicearth25/video-violence-detection-dataset](https://www.kaggle.com/datasets/magicearth25/video-violence-detection-dataset) | `violence_best.pt` |

Run on Kaggle with GPU + Internet on, then copy the `.pt` from `/kaggle/working` into `models/`.

Keep `training/` in the repo — it is the only record of how the weights were produced. It is safe
to exclude from *deployment*.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| Phone page: "Insecure origin" | Not HTTPS. Use the cloudflared tunnel URL, not the LAN IP (Part C) |
| Phone: permission prompt never appears | Permissions only on an explicit tap, and only over HTTPS |
| "A phone stream is already connected" | `PHONE_MAX_SESSIONS=1`. Close the other tab/device |
| Model shows `loaded = false` | `.pt` missing at the configured path — it will not fake predictions |
| No verified events | Working as designed: needs `MIN_HITS` hits above threshold inside `WINDOW_SEC` |
| `[ALERT_BACKEND] delivery failed` | Backend unreachable/asleep, or token mismatch with `ALERT_INGEST_TOKEN` |
| Call never arrives | Trial accounts only call **verified** numbers; check `TWILIO_CALL_MODE` |
| Equivalence check says FAIL | Ingest downscale is altering the model input — fix before shipping |
| Frames dropped on phone stream | Expected under weak uplink: drop-oldest by design. Lower `PHONE_SEND_FPS` |

---

## Summary

| Question | Answer |
|---|---|
| Can it call 100 / 108 / 112? | **No.** Your configured test/operator number only |
| Do I need the hosted backend? | Only for real calls. `EMERGENCY_MODE=log-only` runs everything else |
| Do I need cloudflared? | Only for the phone source. File/webcam/RTSP need no tunnel |
| Is the dashboard the product? | No — the pipeline is. The page is for testing and monitoring |
| Does anything fake a detection? | No. Missing model → no prediction. No audio model → no audio events |
