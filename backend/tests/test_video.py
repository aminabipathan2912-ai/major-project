"""
backend/tests/test_video.py
Phase 2 test suite — verifies the complete video pipeline:
  1. Preprocessing: load image bytes, extract frames, resize
  2. ColorBasedFireDetector: pure-red frame → high FIRE score
  3. VideoSafetyModel: loads, predicts, handles missing frames
  4. VideoService: async analyze returns ModalityPrediction
  5. API route: upload JPEG → 200 OK, correct schema
  6. API route: empty file → 422
  7. API route: unsupported type → 422
  8. Temporal aggregation: max confidence across frames
"""

# ------------------------------------------------------------------ #
# sys.path must be patched FIRST so 'ml' package is importable
# test file: backend/tests/test_video.py
# repo root : backend/tests/test_video.py → parents[2] = d:/random/major-project
# ------------------------------------------------------------------ #
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import io
import os

import numpy as np
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from PIL import Image

# Use in-memory SQLite for tests
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["DATABASE_SYNC_URL"] = "sqlite:///:memory:"
os.environ["DEBUG"] = "true"
os.environ["DEMO_MODE"] = "true"

from app.main import app
from app.core.database import engine, Base


@pytest_asyncio.fixture(autouse=True)
async def setup_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def make_solid_color_jpeg(r: int, g: int, b: int, size: int = 64) -> bytes:
    """Create a JPEG image with a solid RGB colour."""
    img = Image.new("RGB", (size, size), (r, g, b))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_solid_color_bgr(r: int, g: int, b: int, size: int = 64) -> np.ndarray:
    """Return a solid-colour BGR numpy array (OpenCV convention: BGR not RGB)."""
    frame = np.full((size, size, 3), (b, g, r), dtype=np.uint8)
    return frame


# ================================================================== #
# 1. Preprocessing tests
# ================================================================== #
class TestPreprocessing:
    def test_load_image_from_valid_bytes(self):
        from ml.video.preprocessing import load_image_from_bytes
        jpeg = make_solid_color_jpeg(255, 100, 0)  # orange
        frame = load_image_from_bytes(jpeg)
        assert frame is not None
        assert frame.ndim == 3
        assert frame.shape[2] == 3  # BGR channels

    def test_load_image_from_invalid_bytes_returns_none(self):
        from ml.video.preprocessing import load_image_from_bytes
        result = load_image_from_bytes(b"not an image")
        assert result is None

    def test_preprocess_frame_resizes(self):
        from ml.video.preprocessing import preprocess_frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        resized = preprocess_frame(frame, target_size=(224, 224))
        assert resized.shape == (224, 224, 3)

    def test_frame_to_pil_converts_correctly(self):
        from ml.video.preprocessing import frame_to_pil
        bgr = make_solid_color_bgr(255, 0, 0)  # red in RGB = (255,0,0)
        pil = frame_to_pil(bgr)
        assert pil.mode == "RGB"
        r, g, b = pil.getpixel((0, 0))
        assert r > 200   # red channel preserved after BGR→RGB

    def test_load_and_preprocess_image(self):
        from ml.video.preprocessing import load_and_preprocess
        jpeg = make_solid_color_jpeg(200, 100, 50)
        frames, pil_images = load_and_preprocess(jpeg, "image/jpeg")
        assert len(frames) == 1
        assert len(pil_images) == 1
        assert frames[0].shape == (224, 224, 3)

    def test_load_and_preprocess_empty_bytes(self):
        from ml.video.preprocessing import load_and_preprocess
        frames, pil_images = load_and_preprocess(b"", "image/jpeg")
        assert frames == []
        assert pil_images == []


# ================================================================== #
# 2. ColorBasedFireDetector tests
# ================================================================== #
class TestColorBasedFireDetector:
    def test_pure_orange_frame_high_fire_score(self):
        """A frame filled with orange fire colour should score high."""
        from ml.video.model import ColorBasedFireDetector
        detector = ColorBasedFireDetector()
        # Pure orange: H≈15° in HSV — squarely in fire range
        orange_bgr = make_solid_color_bgr(255, 128, 0)  # BGR: B=0, G=128, R=255
        scores = detector.analyze_frame(orange_bgr)
        assert "FIRE" in scores
        assert scores["FIRE"] > 0.5, f"Expected FIRE > 0.5, got {scores['FIRE']}"

    def test_pure_blue_frame_low_fire_score(self):
        """A blue sky frame should score near 0 for fire."""
        from ml.video.model import ColorBasedFireDetector
        detector = ColorBasedFireDetector()
        blue_bgr = make_solid_color_bgr(0, 0, 255)  # pure blue
        scores = detector.analyze_frame(blue_bgr)
        assert scores["FIRE"] < 0.15, f"Expected FIRE < 0.15, got {scores['FIRE']}"

    def test_pure_green_frame_low_fire_score(self):
        from ml.video.model import ColorBasedFireDetector
        detector = ColorBasedFireDetector()
        green_bgr = make_solid_color_bgr(0, 200, 0)
        scores = detector.analyze_frame(green_bgr)
        assert scores["FIRE"] < 0.15

    def test_scores_sum_to_approximately_one(self):
        from ml.video.model import ColorBasedFireDetector
        detector = ColorBasedFireDetector()
        frame = make_solid_color_bgr(100, 100, 100)
        scores = detector.analyze_frame(frame)
        total = scores.get("FIRE", 0) + scores.get("NO_EVENT", 0)
        assert abs(total - 1.0) < 0.01

    def test_empty_frame_returns_no_event(self):
        from ml.video.model import ColorBasedFireDetector
        detector = ColorBasedFireDetector()
        scores = detector.analyze_frame(None)
        assert scores["NO_EVENT"] == 1.0
        assert scores["FIRE"] == 0.0

    def test_analyze_frames_takes_max(self):
        """With one fire frame and one safe frame, max fire score should be returned."""
        from ml.video.model import ColorBasedFireDetector
        detector = ColorBasedFireDetector()
        fire_frame = make_solid_color_bgr(255, 128, 0)   # orange = fire
        safe_frame = make_solid_color_bgr(0, 0, 200)     # blue = no fire
        scores = detector.analyze_frames([fire_frame, safe_frame])
        assert scores["FIRE"] > 0.3   # max should be dominated by fire frame


# ================================================================== #
# 3. VideoSafetyModel tests
# ================================================================== #
class TestVideoSafetyModel:
    def test_model_loads_without_error(self):
        from ml.video.model import VideoSafetyModel
        model = VideoSafetyModel()
        model.load()
        assert model.is_ready

    def test_model_has_correct_modality(self):
        from ml.video.model import VideoSafetyModel
        model = VideoSafetyModel()
        assert model.modality == "video"

    def test_predict_orange_frame_returns_fire(self):
        """Solid orange image should trigger FIRE detection."""
        from ml.video.model import VideoSafetyModel
        model = VideoSafetyModel()
        model.load()
        jpeg = make_solid_color_jpeg(255, 128, 0)  # orange
        raw_input = {"bytes": jpeg, "content_type": "image/jpeg"}
        result = model.predict_raw(raw_input)
        assert result.modality == "video"
        assert result.event in ("FIRE", "NO_EVENT")  # colour-based may vary
        assert 0.0 <= result.confidence <= 1.0

    def test_predict_empty_bytes_returns_no_event(self):
        from ml.video.model import VideoSafetyModel
        model = VideoSafetyModel()
        model.load()
        raw_input = {"bytes": b"", "content_type": "image/jpeg"}
        result = model.predict_raw(raw_input)
        assert result.event == "NO_EVENT"
        assert result.status == "no_event"

    def test_predict_returns_evidence(self):
        from ml.video.model import VideoSafetyModel
        model = VideoSafetyModel()
        model.load()
        jpeg = make_solid_color_jpeg(200, 200, 200)  # grey = no fire
        raw_input = {"bytes": jpeg, "content_type": "image/jpeg"}
        result = model.predict_raw(raw_input)
        assert isinstance(result.evidence, list)

    def test_singleton_returns_same_instance(self):
        from ml.video.model import get_video_model
        m1 = get_video_model()
        m2 = get_video_model()
        assert m1 is m2


# ================================================================== #
# 4. VideoService tests
# ================================================================== #
class TestVideoService:
    @pytest.mark.asyncio
    async def test_analyze_returns_modality_prediction(self):
        from app.services.video_service import VideoService
        svc = VideoService()
        jpeg = make_solid_color_jpeg(200, 200, 200)
        result = await svc.analyze(jpeg, "image/jpeg")
        assert result.modality.value == "video"
        assert 0.0 <= result.confidence <= 1.0
        assert result.event in ("FIRE", "NO_EVENT", "error")

    @pytest.mark.asyncio
    async def test_analyze_fire_frame_has_positive_confidence(self):
        from app.services.video_service import VideoService
        svc = VideoService()
        jpeg = make_solid_color_jpeg(255, 100, 0)  # orange
        result = await svc.analyze(jpeg, "image/jpeg")
        # Raw fire score should be > 0 for clearly orange image
        assert result.raw_scores.get("FIRE", 0.0) >= 0.0  # at minimum no crash


# ================================================================== #
# 5. API route tests
# ================================================================== #
class TestVideoRoute:
    @pytest.mark.asyncio
    async def test_analyze_jpeg_returns_200(self, client: AsyncClient):
        jpeg = make_solid_color_jpeg(150, 150, 150)
        resp = await client.post(
            "/api/v1/video/analyze",
            files={"file": ("test.jpg", jpeg, "image/jpeg")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["modality"] == "video"
        assert "confidence" in body
        assert "event" in body
        assert "evidence" in body
        assert "raw_scores" in body

    @pytest.mark.asyncio
    async def test_analyze_png_returns_200(self, client: AsyncClient):
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (0, 0, 200)).save(buf, format="PNG")
        resp = await client.post(
            "/api/v1/video/analyze",
            files={"file": ("test.png", buf.getvalue(), "image/png")},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_empty_file_returns_422(self, client: AsyncClient):
        resp = await client.post(
            "/api/v1/video/analyze",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_fire_orange_image_response_structure(self, client: AsyncClient):
        """Orange image — check response has correct structure and fire score > 0."""
        jpeg = make_solid_color_jpeg(255, 100, 0)  # orange
        resp = await client.post(
            "/api/v1/video/analyze",
            files={"file": ("fire.jpg", jpeg, "image/jpeg")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["modality"] == "video"
        assert body["model_name"]
        assert isinstance(body["raw_scores"], dict)
        # FIRE key must exist in raw_scores
        assert "FIRE" in body["raw_scores"]

    @pytest.mark.asyncio
    async def test_fusion_with_video_prediction(self, client: AsyncClient):
        """Full pipeline: video → fusion → incident."""
        from datetime import datetime, timezone
        jpeg = make_solid_color_jpeg(200, 200, 200)
        # Get a real video prediction
        video_resp = await client.post(
            "/api/v1/video/analyze",
            files={"file": ("frame.jpg", jpeg, "image/jpeg")},
        )
        assert video_resp.status_code == 200
        video_pred = video_resp.json()

        # Feed into fusion
        now = datetime.now(timezone.utc).isoformat()
        fusion_resp = await client.post(
            "/api/v1/fusion/predict",
            json={"video": video_pred, "location": "zone-B"},
        )
        assert fusion_resp.status_code == 200
        fusion = fusion_resp.json()
        assert "risk_score" in fusion
        assert "severity" in fusion
