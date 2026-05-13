"""
Async database engine and session factory for ProofyX.

Reads DATABASE_URL from environment.
Falls back to SQLite (aiosqlite) when DATABASE_URL is not set (dev mode).
"""

from __future__ import annotations

import logging
import os
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _get_database_url() -> str:
    """Resolve the async database URL from environment."""
    url = os.environ.get("DATABASE_URL", "")
    if url:
        # Support standard postgres:// URLs by converting to asyncpg
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)
        return url
    # Fallback: SQLite for local dev
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "proofyx_history.db")
    logger.info("DATABASE_URL not set — using SQLite at %s", db_path)
    return f"sqlite+aiosqlite:///{db_path}"


def get_engine() -> AsyncEngine:
    """Return the shared async engine, creating it on first call."""
    global _engine
    if _engine is None:
        url = _get_database_url()
        is_sqlite = url.startswith("sqlite")
        _engine = create_async_engine(
            url,
            echo=False,
            pool_pre_ping=not is_sqlite,
            # SQLite doesn't support pool_size
            **({"pool_size": 5, "max_overflow": 10} if not is_sqlite else {}),
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the shared session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async session."""
    factory = get_session_factory()
    async with factory() as session:
        yield session


async def init_db() -> None:
    """Create all tables (used at app startup)."""
    from db.models import Base

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialized")


async def close_db() -> None:
    """Dispose the engine (used at app shutdown)."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
