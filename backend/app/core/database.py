"""
backend/app/core/database.py
Async SQLAlchemy engine + session factory for SQLite (Phase 1).
Drop-in replaceable with PostgreSQL by changing DATABASE_URL in .env.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# ------------------------------------------------------------------ #
# Engine
# ------------------------------------------------------------------ #
# SQLite needs check_same_thread=False for async usage
connect_args: dict = {}
if "sqlite" in settings.DATABASE_URL:
    connect_args["check_same_thread"] = False

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    connect_args=connect_args,
    pool_pre_ping=True,
)

# ------------------------------------------------------------------ #
# Session factory
# ------------------------------------------------------------------ #
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ------------------------------------------------------------------ #
# Base class for all ORM models
# ------------------------------------------------------------------ #
class Base(DeclarativeBase):
    pass


# ------------------------------------------------------------------ #
# Dependency
# ------------------------------------------------------------------ #
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields a database session per request."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ------------------------------------------------------------------ #
# Initialisation
# ------------------------------------------------------------------ #
async def init_db() -> None:
    """Create all tables on startup (dev/SQLite). Use Alembic for production."""
    from app.models import incident, prediction, evidence, alert  # noqa: F401

    logger.info("Initialising database tables …")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database ready.")
