# Live Multimodal Monitoring System for Public Safety

## Overview

> **Combining multiple sources of information provides a more complete and reliable picture of a public-safety event than monitoring each source independently.**

This system integrates **video**, **audio**, **text**, and **sensor** data, independently analyses each modality with dedicated ML models, combines the evidence through a multimodal fusion engine, detects public-safety incidents, and provides real-time alerts and situational awareness.

---

## Architecture

```
INPUT LAYER
│
┌────────────┬────────────┬────────────┐
↓            ↓            ↓            ↓
VIDEO        AUDIO        TEXT       SENSOR
│            │            │            │
Video Model  Audio Model  NLP Model  Sensor Model
│            │            │            │
└────────────┴────────────┴────────────┘
                    ↓
           MULTIMODAL FUSION ENGINE
                    ↓
            EVENT CLASSIFICATION
                    ↓
              RISK SCORE ENGINE
                    ↓
              ALERT ENGINE
                    ↓
              DATA MANAGEMENT
                    ↓
            DASHBOARD / UI (WebSocket)
```

---

## Project Structure

```
major-project/
├── backend/        FastAPI backend, API routes, services, ORM models
├── frontend/       HTML/CSS/JS dashboard (Phase 8)
├── ml/             Modality models + fusion engine
├── datasets/       Dataset registry and sample data
├── training/       Training and evaluation scripts
├── configs/        YAML configuration files
├── tests/          Integration tests
├── scripts/        Utility scripts
└── docs/           Architecture and API documentation
```

---

## Quick Start (Phase 1)

### Prerequisites
- Python 3.11
- Windows / Linux / macOS

### Setup

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### Run the server

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

### Open API docs
- Swagger UI: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc

### Run tests

```bash
cd backend
pytest tests/test_phase1.py -v
```

---

## API Endpoints (Phase 1)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | System health check |
| GET | /api/v1/system/status | Detailed system status |
| GET | /api/v1/system/models/status | Model registry |
| GET | /api/v1/system/metrics | Detection metrics |
| POST | /api/v1/video/analyze | Analyze video frame/clip |
| POST | /api/v1/audio/analyze | Analyze audio chunk |
| POST | /api/v1/text/analyze | Classify text message |
| POST | /api/v1/sensor/readings | Ingest sensor readings |
| POST | /api/v1/fusion/predict | Multimodal fusion |
| GET | /api/v1/incidents | List incidents |
| GET | /api/v1/incidents/{id} | Get incident detail |
| PATCH | /api/v1/incidents/{id}/status | Update status |
| GET | /api/v1/alerts | List alerts |
| PATCH | /api/v1/alerts/{id}/acknowledge | Acknowledge alert |
| WS | /ws/monitor | Real-time WebSocket stream |

---

## Development Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Complete | Backend + DB + Config + API skeleton |
| 2 | ⏳ Next | Video pipeline (OpenCV + CV model) |
| 3 | 📋 Planned | Audio pipeline (librosa + YAMNet) |
| 4 | 📋 Planned | Text/NLP pipeline (DistilBERT) |
| 5 | 📋 Planned | Sensor pipeline (IsolationForest) |
| 6 | 📋 Planned | Multimodal fusion engine |
| 7 | 📋 Planned | Alert engine + temporal reasoning |
| 8 | 📋 Planned | Dashboard UI |
| 9 | 📋 Planned | Real-time WebSocket |
| 10 | 📋 Planned | Evaluation framework |
| 11 | 📋 Planned | Tests + CI |
| 12 | 📋 Planned | Docker + deployment |

---

## Responsible AI Notice

> ⚠️ **AI-generated detection. Human verification required.**

- No facial recognition or individual identification
- No person tracking
- Configurable data retention (default: 30 days)
- All detections logged with full evidence chain

---

## Configuration

See `configs/` for:
- `system.yaml` — server, database, logging
- `thresholds.yaml` — fusion weights, severity rules, false-alarm reduction
- `models.yaml` — model registry with versions and supported events

See `datasets/registry.yaml` for the full dataset registry with class mappings.
