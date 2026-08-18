"""Tests for db.history — async SQLAlchemy analysis persistence."""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import db.database as database
from db.history import AnalysisHistory
from db.models import Base

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture()
async def db(monkeypatch):
    """Isolated in-memory database for each test."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(database, "_engine", engine)
    monkeypatch.setattr(database, "_session_factory", factory)

    yield AnalysisHistory()

    await engine.dispose()


class TestSaveAndRetrieve:
    async def test_save_returns_id(self, db):
        result = {
            "id": "abc123",
            "media_type": "image",
            "risk_score": 0.75,
            "risk_percent": 75.0,
            "verdict": "AI-GENERATED",
            "confidence": "HIGH",
            "model_scores": {"vit": 0.8, "efficientnet": 0.7},
        }
        saved_id = await db.save(result)
        assert saved_id == "abc123"

    async def test_get_returns_saved_result(self, db):
        result = {
            "id": "test1",
            "media_type": "image",
            "risk_score": 0.42,
            "risk_percent": 42.0,
            "verdict": "AUTHENTIC",
            "confidence": "MEDIUM",
            "model_scores": {"vit": 0.5},
            "explanation": "Some explanation",
        }
        await db.save(result)
        retrieved = await db.get("test1")

        assert retrieved is not None
        assert retrieved["id"] == "test1"
        assert retrieved["risk_score"] == 0.42
        assert retrieved["verdict"] == "AUTHENTIC"
        assert retrieved["model_scores"] == {"vit": 0.5}

    async def test_get_nonexistent_returns_none(self, db):
        assert await db.get("nonexistent") is None

    async def test_model_scores_round_trips_as_dict(self, db):
        result = {
            "id": "json1",
            "media_type": "image",
            "risk_score": 0.5,
            "risk_percent": 50.0,
            "verdict": "AUTHENTIC",
            "confidence": "LOW",
            "model_scores": {"a": 0.1, "b": 0.9},
        }
        await db.save(result)
        retrieved = await db.get("json1")
        assert isinstance(retrieved["model_scores"], dict)
        assert retrieved["model_scores"]["a"] == 0.1


class TestGetRecent:
    async def test_returns_ordered_by_timestamp(self, db):
        for i in range(5):
            await db.save({
                "id": f"item{i}",
                "timestamp": f"2024-01-0{i + 1}T00:00:00Z",
                "media_type": "image",
                "risk_score": 0.1 * i,
                "risk_percent": 10.0 * i,
                "verdict": "AUTHENTIC",
                "confidence": "LOW",
                "model_scores": {},
            })

        recent = await db.get_recent(limit=3)
        assert len(recent) == 3
        # Most recent first
        assert recent[0]["id"] == "item4"

    async def test_filter_by_media_type(self, db):
        await db.save({
            "id": "img1", "media_type": "image",
            "risk_score": 0.5, "risk_percent": 50.0,
            "verdict": "X", "confidence": "X", "model_scores": {},
        })
        await db.save({
            "id": "vid1", "media_type": "video",
            "risk_score": 0.5, "risk_percent": 50.0,
            "verdict": "X", "confidence": "X", "model_scores": {},
        })

        images = await db.get_recent(media_type="image")
        assert all(r["media_type"] == "image" for r in images)
        assert len(images) == 1


class TestCount:
    async def test_count_empty(self, db):
        assert await db.count() == 0

    async def test_count_after_saves(self, db):
        for i in range(3):
            await db.save({
                "id": f"c{i}", "media_type": "image",
                "risk_score": 0.0, "risk_percent": 0.0,
                "verdict": "", "confidence": "", "model_scores": {},
            })
        assert await db.count() == 3


class TestDelete:
    async def test_delete_existing(self, db):
        await db.save({
            "id": "del1", "media_type": "image",
            "risk_score": 0.0, "risk_percent": 0.0,
            "verdict": "", "confidence": "", "model_scores": {},
        })
        assert await db.delete("del1") is True
        assert await db.get("del1") is None

    async def test_delete_nonexistent(self, db):
        assert await db.delete("nope") is False


class TestUpdatePaths:
    async def test_update_gradcam_path(self, db):
        await db.save({
            "id": "up1", "media_type": "image",
            "risk_score": 0.0, "risk_percent": 0.0,
            "verdict": "", "confidence": "", "model_scores": {},
        })
        await db.update_paths("up1", gradcam_path="/tmp/gradcam.png")
        result = await db.get("up1")
        assert result["gradcam_path"] == "/tmp/gradcam.png"
