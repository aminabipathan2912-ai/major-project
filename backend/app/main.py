"""
backend/app/main.py
FastAPI application factory for the Live Multimodal Monitoring System.

Architecture:
  Request → API routes → Services → DB
                       ↘ ML Inference (Phase 2–5)
                       ↘ Fusion engine (Phase 6)
                       ↘ Alert engine (Phase 7)
                       ↗ WebSocket broadcast

IMPORTANT: ML inference must NEVER be called directly inside route functions.
All inference goes through service classes.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import get_settings
from app.core.database import init_db
from app.core.logging import setup_logging

# ------------------------------------------------------------------ #
# Bootstrap logging before anything else
# ------------------------------------------------------------------ #
setup_logging()
logger = logging.getLogger(__name__)
settings = get_settings()


# ------------------------------------------------------------------ #
# Lifespan (startup / shutdown)
# ------------------------------------------------------------------ #
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("=" * 60)
    logger.info("  %s  v%s", settings.APP_NAME, settings.APP_VERSION)
    logger.info("  Demo mode: %s", settings.DEMO_MODE)
    logger.info("=" * 60)

    # Ensure data directory exists
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    # Initialise database (create tables)
    await init_db()

    logger.info("Application startup complete. Listening on %s:%d",
                settings.SERVER_HOST, settings.SERVER_PORT)
    yield

    logger.info("Application shutting down …")


# ------------------------------------------------------------------ #
# Application factory
# ------------------------------------------------------------------ #
def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Live Multimodal Monitoring System for Public Safety. "
            "Integrates video, audio, text and sensor data through "
            "AI-powered multimodal fusion for real-time incident detection."
        ),
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )

    # ---------------------------------------------------------------- #
    # CORS
    # ---------------------------------------------------------------- #
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------------------------------------------------------------- #
    # Global exception handler
    # ---------------------------------------------------------------- #
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled exception: %s — %s", type(exc).__name__, exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc) if settings.DEBUG else "Contact system administrator",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    # ---------------------------------------------------------------- #
    # Routers
    # ---------------------------------------------------------------- #
    from app.api.routes import video, audio, text, sensor, fusion, incidents, alerts, system, analytics
    from app.api.websocket import router as ws_router

    API_PREFIX = "/api/v1"

    app.include_router(video.router,     prefix=API_PREFIX)
    app.include_router(audio.router,     prefix=API_PREFIX)
    app.include_router(text.router,      prefix=API_PREFIX)
    app.include_router(sensor.router,    prefix=API_PREFIX)
    app.include_router(fusion.router,    prefix=API_PREFIX)
    app.include_router(incidents.router, prefix=API_PREFIX)
    app.include_router(alerts.router,    prefix=API_PREFIX)
    app.include_router(system.router,    prefix=API_PREFIX)
    app.include_router(analytics.router, prefix=API_PREFIX)
    app.include_router(ws_router)       # WebSocket: /ws/monitor

    # ---------------------------------------------------------------- #
    # Root health check  (must be registered BEFORE StaticFiles mount)
    # ---------------------------------------------------------------- #
    @app.get("/health", tags=["health"], include_in_schema=False)
    async def health():
        return {
            "status": "healthy",
            "version": settings.APP_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ---------------------------------------------------------------- #
    # Serve frontend static files (Phase 8)
    # Mounted at /app so API routes (/, /health, /api/*) are not shadowed.
    # In production, serve the frontend via nginx or a CDN.
    # ---------------------------------------------------------------- #
    frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
    if frontend_dir.exists():
        from fastapi.responses import RedirectResponse
        app.mount("/app", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

        @app.get("/", tags=["frontend"], include_in_schema=False)
        async def root_redirect():
            return RedirectResponse(url="/app/index.html")

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
