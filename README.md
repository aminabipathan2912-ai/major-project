# CCTV Accident & Violence Detection App

Verified detection can trigger **Sarvam TTS** and a **Twilio outbound call**. You say **Done**; the dashboard switches to **ACCIDENT REPORTED**.

There is no two-way AI conversation. Official police/ambulance emergency numbers are **not** dialed. Use your own test/operator numbers.

## Hosted test (required for Twilio)

You only host the **Python backend**. There is no separate frontend app.

`PUBLIC_BASE_URL` is still required because **Twilio’s cloud** (not your browser) must HTTP-call that backend:

- fetch TwiML (`/twilio/voice/...`) when the call connects
- download the Sarvam audio (`/audio/...`)
- POST what you said (`/twilio/acknowledge/...`)

Localhost is invisible to Twilio. The demo page at `/` is optional and ships with the same backend.

**Numbers:** one Twilio from-number, one to-number (your phone). Not 100/108/112.

1. Fill `.env`:
   - `PUBLIC_BASE_URL=https://your-backend.onrender.com` (no trailing slash)
   - `TWILIO_ACCOUNT_SID`
   - `TWILIO_AUTH_TOKEN`
   - `TWILIO_FROM_NUMBER`
   - `TWILIO_TO_NUMBER`
   - `SARVAM_API_KEY`
   - `EMERGENCY_MODE=voice`
   - `CAMERA_SOURCE=...`
   - `INCIDENT_LOCATION=...`
2. Host this FastAPI app with those env vars.
3. Open the hosted site. **Upload a fight/accident clip** on the page (do not rely on a video in the git repo).
4. Models run on that uploaded file. If verified, your phone rings. Say **Done**.
5. Incident status becomes **REPORTED**.

If Sarvam fails, Twilio still speaks a fallback `<Say>` of the same text.

### Twilio console
- Voice webhook URLs are generated automatically: `/twilio/voice/{incident_id}`
- Status: `/twilio/status/{call_id}`
- Speech result: `/twilio/acknowledge/{incident_id}`
- Audio: `/audio/{incident_id}.wav`

### Local run (pipeline only)

`python run_demo_server.py` still works for video/models. Voice calling needs `PUBLIC_BASE_URL` on a hosted deployment.

---

## Flow

```
Video → accident/violence models → temporal verifier
     → incident (DETECTED)
     → Sarvam TTS
     → Twilio call to your test phone
     → you say Done
     → status REPORTED → dashboard: ACCIDENT REPORTED
```

---

## Recommended mode: pre-recorded video (not live camera)

For cost and simplicity, this project is set up to run on **pre-recorded `.mp4` files** by default.

That is a good plan:
- same inference pipeline as live CCTV
- much cheaper than always-on RTSP/webcam processing
- easier to test the same clip again and again
- live RTSP/webcam remain supported later if needed

Default config:
- `CAMERA_SOURCE_TYPE=file`
- `CAMERA_SOURCE=tests/fixtures/sample.mp4` (placeholder only)

When you have a real fight/accident clip, just point `CAMERA_SOURCE` to that file.

## What this project is (in simple words)

This is a **CCTV-style video inference system**, not a big city dashboard.

The product flow is:

1. Video comes in (pre-recorded file now; webcam/RTSP later if needed)
2. Frames are buffered
3. AI models look for accidents or fighting
4. A verification layer checks if the detection is strong and repeated
5. Only then an emergency incident can be created (Sarvam + Twilio on hosted deploy)

```
Pre-recorded video (default) / Webcam / RTSP
        ↓
   Frame Buffer
        ↓
 Accident Model + Violence Model
        ↓
 Temporal Event Verification
        ↓
 Incident + Sarvam TTS + Twilio (hosted)
```

The web page at `http://localhost:8000` is only a **demo/testing screen**.  
It is not the main product.

---

## Why `tests/fixtures/sample.mp4` only shows "frame 0, frame 1..." text

That video is **not a fight video**.

It was generated automatically by `scripts/create_test_video.py` so we could test the camera pipeline inside WSL without a webcam.

What it contains:
- black background
- green text like `frame 0`, `frame 1`, `frame 2`...
- about 3 seconds of video

What it is for:
- prove the app can open a video file
- prove frames move into the buffer
- prove the demo page can show live frames
- prove the pipeline runs even when models are missing

What it is **not** for:
- detecting fights
- detecting accidents
- training AI models

So if you open the demo and only see those text frames, that is expected.

---

## Do I need a fight/accident video?

**Yes for your planned workflow — but not for basic app testing.**

Your plan (pre-recorded video, no live camera) is already supported and is now the default.

| Goal | What video do you need? |
|---|---|
| Test that the app runs | `sample.mp4` placeholder is enough |
| Run inference on realistic clips (your plan) | Your own fight/accident `.mp4` |
| Get real detections | Same clip **plus** trained models plugged in |

Using your own video **today**:
- demo will play that video
- AI will still say **model not loaded** until models are trained

### How to use your own pre-recorded video

1. Put your `.mp4` somewhere, for example:
   - `tests/fixtures/my_fight_video.mp4`
2. Run:

```bash
cd /mnt/d/Programming/major-project
source .venv/bin/activate
export CAMERA_SOURCE_TYPE=file
export CAMERA_SOURCE=tests/fixtures/my_fight_video.mp4
python run_demo_server.py
```

3. Open: `http://localhost:8000`

Live webcam/RTSP remain available later if you ever need them.

---

## What we already built (folder by folder)

```
src/cctv_ai/
  camera/         # reads webcam, file, or RTSP; buffers frames
  inference/
    accident/     # plug-in point for accident AI model
    violence/     # plug-in point for fight/violence AI model
    audio/        # reserved for future audio-event model
  event_engine/   # temporal verification (no single-frame panic)
  emergency/      # on_verified_emergency(...) provider hook
  demo/           # tiny web page for testing only
  core/           # pipeline that connects everything
```

### Camera layer
- Independent from AI models
- Supports webcam, local video file, RTSP/IP camera
- Frame buffer + reconnect behavior

### Inference layer
- Shared contract: `input → prediction → confidence → metadata`
- Accident and violence adapters exist, but are empty until you add trained weights
- If weights are missing, the app reports that clearly and returns no fake predictions

### Event verification layer
- Separate from models
- Requires confidence threshold + repeated detections over time
- Has cooldown so one short flicker does not spam emergencies

### Emergency layer
- Provider-agnostic interface: `on_verified_emergency(event)`
- Currently `log-only`
- Ready for a future voice/notification provider (not hard-coded to any vendor)

### Demo dashboard
- Shows latest frame
- Shows model status
- Shows verified events (when models exist)
- Not a full analytics product

---

## Datasets (for later model training — not used by the live app now)

We intentionally **do not** download or train models inside this application.

### Accident model training
- **Used in the notebook:** https://www.kaggle.com/datasets/ckay16/accident-detection-from-cctv-footage  
  (`Accident` vs `Non Accident` CCTV frames — free, labeled)
- Hugging Face listing (paid 10-clip preview, **not enough to train**): https://huggingface.co/datasets/ud-smart-city/car-accident-video
- GitHub reference (research only): https://github.com/iamreubengm/CCTVAccidentDetection
- Roboflow annotations (optional later): https://app.roboflow.com/satellitetraffic/accident-detection-model-6rv64-g1pdg/annotate/job/5Z3DYSw75Zm5mqfoEoqU

Target output later:
`ACCIDENT / NORMAL + confidence + timestamp + camera_id`

### Violence / fighting model training
- Kaggle violence dataset: https://www.kaggle.com/datasets/magicearth25/video-violence-detection-dataset

Target output later:
`VIOLENCE / NORMAL + confidence + timestamp + camera_id`

### Not part of the live CCTV core
These are optional analytics/research later, not required for real-time CCTV inference:
- disaster tweets
- road traffic accident CSVs
- Indian road accident tabular datasets

---

## Setup (WSL)

```bash
cd /mnt/d/Programming/major-project
sudo apt-get install -y python3.14-venv
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Run the demo (pre-recorded video)

### Default placeholder video

```bash
source .venv/bin/activate
python scripts/create_test_video.py
python run_demo_server.py
```

Then open: http://localhost:8000

### Your own fight/accident video (recommended once you have one)

```bash
export CAMERA_SOURCE_TYPE=file
export CAMERA_SOURCE=/path/to/your_video.mp4
python run_demo_server.py
```

Or put the path permanently in `.env`:

```bash
CAMERA_SOURCE_TYPE=file
CAMERA_SOURCE=tests/fixtures/my_clip.mp4
```

### Optional later: webcam / RTSP

Only needed if you want live camera input later:

```bash
export CAMERA_SOURCE_TYPE=webcam
export CAMERA_SOURCE=0
python run_demo_server.py
```

```bash
export CAMERA_SOURCE_TYPE=rtsp
export CAMERA_SOURCE=rtsp://username:password@camera-ip/stream
python run_demo_server.py
```

Copy `.env.example` to `.env` if you want permanent settings.

---

## What you should see on the demo page

After copying trained weights into `models/`:

- live/demo video
- `accident.loaded = true` and `violence.loaded = true` if the `.pt` files exist
- predicted labels + confidence in status/events
- verified events only after the temporal verifier sees repeated high-confidence hits

If a `.pt` file is missing, that model stays `loaded = false` and will not fake detections.

---

## Plug in trained weights

1. Copy from Kaggle Output into this project:
   - `models/violence_best.pt`
   - `models/accident_best.pt`
2. Copy `.env.example` to `.env` (paths are already set)
3. Install inference deps (includes PyTorch):

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

4. Run with your video:

```bash
export CAMERA_SOURCE_TYPE=file
export CAMERA_SOURCE=/path/to/your_clip.mp4
python run_demo_server.py
```

The adapters load EfficientNet-B0 clip classifiers from those checkpoints and return `label + confidence + timestamp + camera_id`. The event engine still requires repeated detections before an emergency log.

See `training/README.md` for how the notebooks produced these files.

---

## Short summary

| Question | Answer |
|---|---|
| How do I test the phone alert? | Host the app, set `PUBLIC_BASE_URL` + Twilio/Sarvam keys, use your test number |
| Can it call real 100/108/112? | **No.** Test/operator numbers only |
| Is the dashboard the main product? | **No** — the pipeline is the product |

Build first. Train models next. Plug them in after.
