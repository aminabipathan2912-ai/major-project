"""
backend/app/api/routes/analytics.py
Accident analytics endpoint — serves pre-computed statistics from datasets.

Datasets:
  - Road Traffic Accidents (Kaggle)
  - Indian Road Accident 2022-2025 (MoRTH)
  - Disaster Tweets (Kaggle NLP2) — class distribution only

All figures are representative summaries derived from public datasets.
Marked as SIMULATED_DATA when real CSVs are not loaded.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter

router = APIRouter(prefix="/analytics", tags=["analytics"])

# ---------------------------------------------------------------------------
# Representative summary data (derived from public datasets)
# Replace with real CSV loading when datasets are mounted in datasets/ dir
# ---------------------------------------------------------------------------
_DISCLAIMER = (
    "SIMULATED_DATA — Representative statistics derived from public datasets "
    "(MoRTH Road Accidents in India 2022-23, Kaggle Road Traffic Accidents, "
    "Kaggle NLP2 Disaster Tweets). Not real-time. Not guaranteed accurate."
)


def _base() -> Dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "disclaimer": _DISCLAIMER,
    }


@router.get("/summary")
async def analytics_summary():
    """High-level accident summary statistics."""
    return {
        **_base(),
        "data": {
            "total_accidents_2024_projected": 480652,
            "total_fatalities_2024_projected": 177757,
            "total_injured_2024_projected": 451432,
            "yoy_change_pct": 3.2,
            "accidents_per_hour": 54.9,
            "accidents_per_km_national_highway": 0.048,
            "source": "MoRTH Road Accidents in India 2022-2023 + projected 2024",
        },
    }


@router.get("/state-wise")
async def state_wise():
    """State-wise accident and fatality data (top 10 states)."""
    return {
        **_base(),
        "data": {
            "states": [
                {"state": "Uttar Pradesh",  "accidents": 44000, "fatalities": 22000, "rank": 1},
                {"state": "Tamil Nadu",     "accidents": 64000, "fatalities": 17000, "rank": 2},
                {"state": "Maharashtra",    "accidents": 32000, "fatalities": 12000, "rank": 3},
                {"state": "Madhya Pradesh", "accidents": 51000, "fatalities": 14000, "rank": 4},
                {"state": "Rajasthan",      "accidents": 29000, "fatalities": 11000, "rank": 5},
                {"state": "Karnataka",      "accidents": 40000, "fatalities": 11000, "rank": 6},
                {"state": "Gujarat",        "accidents": 27000, "fatalities": 9000,  "rank": 7},
                {"state": "Andhra Pradesh", "accidents": 22000, "fatalities": 8000,  "rank": 8},
                {"state": "Telangana",      "accidents": 20000, "fatalities": 7000,  "rank": 9},
                {"state": "Kerala",         "accidents": 40000, "fatalities": 4500,  "rank": 10},
            ],
        },
    }


@router.get("/monthly-trends")
async def monthly_trends():
    """Monthly accident trends 2023 and 2024 (projected)."""
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    return {
        **_base(),
        "data": {
            "months": months,
            "accidents_2023": [38000,33000,40000,37000,42000,44000,47000,45000,39000,40000,37000,38000],
            "accidents_2024_projected": [39000,35000,41000,38000,43000,46000,49000,47000,41000,41000,38000,39000],
        },
    }


@router.get("/vehicle-types")
async def vehicle_types():
    """Accident distribution by vehicle type."""
    return {
        **_base(),
        "data": {
            "vehicle_types": [
                {"type": "Two-Wheelers",    "percentage": 44.2, "color": "#ff2d55"},
                {"type": "Cars/Jeeps",      "percentage": 21.5, "color": "#ff6b35"},
                {"type": "Trucks/Lorries",  "percentage": 15.8, "color": "#ffd60a"},
                {"type": "Buses",           "percentage": 4.3,  "color": "#30d158"},
                {"type": "Auto-Rickshaws",  "percentage": 6.7,  "color": "#00d4ff"},
                {"type": "Others",          "percentage": 7.5,  "color": "#bf5af2"},
            ],
            "note": "Two-wheelers account for 44% of all accidents (MoRTH 2022).",
        },
    }


@router.get("/causes")
async def accident_causes():
    """Accident cause breakdown."""
    return {
        **_base(),
        "data": {
            "causes": [
                {"cause": "Over-speeding",       "percentage": 72.0},
                {"cause": "Drunk Driving",        "percentage": 12.0},
                {"cause": "Red Light Jumping",    "percentage": 6.0},
                {"cause": "Wrong Lane",           "percentage": 4.0},
                {"cause": "Distracted Driving",   "percentage": 3.0},
                {"cause": "Poor Visibility",      "percentage": 2.0},
                {"cause": "Others",               "percentage": 1.0},
            ],
            "note": "Over-speeding is the primary cause of road accidents in India (MoRTH 2022).",
        },
    }


@router.get("/disaster-tweets")
async def disaster_tweets():
    """Kaggle NLP2 Twitter Disaster Tweets dataset class distribution."""
    return {
        **_base(),
        "data": {
            "total_tweets": 7613,
            "disaster_tweets": 3271,
            "non_disaster_tweets": 4342,
            "disaster_pct": 42.9,
            "non_disaster_pct": 57.1,
            "dataset": "Kaggle NLP2 — Natural Language Processing with Disaster Tweets",
            "source_url": "https://www.kaggle.com/competitions/nlp-getting-started",
            "note": "Training set split. Used for DisasterTextModel fine-tuning (DistilBERT).",
        },
    }
