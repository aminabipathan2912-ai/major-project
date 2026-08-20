"""
backend/app/core/config.py
Application settings loaded from environment variables / .env file.
Uses pydantic-settings for type-safe configuration.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Application
    # ------------------------------------------------------------------ #
    APP_NAME: str = "Live Multimodal Monitoring System for Public Safety"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    DEMO_MODE: bool = True

    # ------------------------------------------------------------------ #
    # Server
    # ------------------------------------------------------------------ #
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8000

    # ------------------------------------------------------------------ #
    # Database
    # ------------------------------------------------------------------ #
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/monitoring.db"
    DATABASE_SYNC_URL: str = "sqlite:///./data/monitoring.db"

    # ------------------------------------------------------------------ #
    # Security
    # ------------------------------------------------------------------ #
    SECRET_KEY: str = "change-me-in-production"

    # ------------------------------------------------------------------ #
    # CORS
    # ------------------------------------------------------------------ #
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000"

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    LOG_LEVEL: str = "INFO"

    # ------------------------------------------------------------------ #
    # Data retention
    # ------------------------------------------------------------------ #
    DATA_RETENTION_DAYS: int = 30

    # ------------------------------------------------------------------ #
    # Paths (resolved relative to repo root)
    # ------------------------------------------------------------------ #
    @property
    def repo_root(self) -> Path:
        # backend/app/core/config.py  →  up 3 levels  →  repo root
        return Path(__file__).resolve().parents[3]

    @property
    def data_dir(self) -> Path:
        d = self.repo_root / "backend" / "data"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def logs_dir(self) -> Path:
        d = self.repo_root / "backend" / "logs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def ml_dir(self) -> Path:
        return self.repo_root / "ml"

    @property
    def configs_dir(self) -> Path:
        return self.repo_root / "configs"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
