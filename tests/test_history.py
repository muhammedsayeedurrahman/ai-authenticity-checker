"""Tests for db.history — SQLite analysis persistence."""

from __future__ import annotations

import pytest

from db.history import AnalysisHistory


@pytest.fixture()
def db():
    """In-memory database for testing."""
    return AnalysisHistory(db_path=":memory:")


class TestSaveAndRetrieve:
    def test_save_returns_id(self, db):
        result = {
            "id": "abc123",
            "media_type": "image",
            "risk_score": 0.75,
            "risk_percent": 75.0,
            "verdict": "LIKELY MANIPULATED",
            "confidence": "HIGH",
            "model_scores": {"vit": 0.8, "efficientnet": 0.7},
        }
        saved_id = db.save(result)
        assert saved_id == "abc123"

    def test_get_returns_saved_result(self, db):
        result = {
            "id": "test1",
            "media_type": "image",
            "risk_score": 0.42,
            "risk_percent": 42.0,
            "verdict": "UNCERTAIN",
            "confidence": "MEDIUM",
            "model_scores": {"vit": 0.5},
            "explanation": "Some explanation",
        }
        db.save(result)
        retrieved = db.get("test1")

        assert retrieved is not None
        assert retrieved["id"] == "test1"
        assert retrieved["risk_score"] == 0.42
        assert retrieved["verdict"] == "UNCERTAIN"
        assert retrieved["model_scores"] == {"vit": 0.5}

    def test_get_nonexistent_returns_none(self, db):
        assert db.get("nonexistent") is None

    def test_model_scores_round_trips_as_dict(self, db):
        result = {
            "id": "json1",
            "media_type": "image",
            "risk_score": 0.5,
            "risk_percent": 50.0,
            "verdict": "UNCERTAIN",
            "confidence": "LOW",
            "model_scores": {"a": 0.1, "b": 0.9},
        }
        db.save(result)
        retrieved = db.get("json1")
        assert isinstance(retrieved["model_scores"], dict)
        assert retrieved["model_scores"]["a"] == 0.1


class TestGetRecent:
    def test_returns_ordered_by_timestamp(self, db):
        for i in range(5):
            db.save({
                "id": f"item{i}",
                "timestamp": f"2024-01-0{i + 1}T00:00:00Z",
                "media_type": "image",
                "risk_score": 0.1 * i,
                "risk_percent": 10.0 * i,
                "verdict": "UNCERTAIN",
                "confidence": "LOW",
                "model_scores": {},
            })

        recent = db.get_recent(limit=3)
        assert len(recent) == 3
        # Most recent first
        assert recent[0]["id"] == "item4"

    def test_filter_by_media_type(self, db):
        db.save({
            "id": "img1", "media_type": "image",
            "risk_score": 0.5, "risk_percent": 50.0,
            "verdict": "X", "confidence": "X", "model_scores": {},
        })
        db.save({
            "id": "vid1", "media_type": "video",
            "risk_score": 0.5, "risk_percent": 50.0,
            "verdict": "X", "confidence": "X", "model_scores": {},
        })

        images = db.get_recent(media_type="image")
        assert all(r["media_type"] == "image" for r in images)
        assert len(images) == 1


class TestCount:
    def test_count_empty(self, db):
        assert db.count() == 0

    def test_count_after_saves(self, db):
        for i in range(3):
            db.save({
                "id": f"c{i}", "media_type": "image",
                "risk_score": 0.0, "risk_percent": 0.0,
                "verdict": "", "confidence": "", "model_scores": {},
            })
        assert db.count() == 3


class TestDelete:
    def test_delete_existing(self, db):
        db.save({
            "id": "del1", "media_type": "image",
            "risk_score": 0.0, "risk_percent": 0.0,
            "verdict": "", "confidence": "", "model_scores": {},
        })
        assert db.delete("del1") is True
        assert db.get("del1") is None

    def test_delete_nonexistent(self, db):
        assert db.delete("nope") is False


class TestUpdatePaths:
    def test_update_gradcam_path(self, db):
        db.save({
            "id": "up1", "media_type": "image",
            "risk_score": 0.0, "risk_percent": 0.0,
            "verdict": "", "confidence": "", "model_scores": {},
        })
        db.update_paths("up1", gradcam_path="/tmp/gradcam.png")
        result = db.get("up1")
        assert result["gradcam_path"] == "/tmp/gradcam.png"
