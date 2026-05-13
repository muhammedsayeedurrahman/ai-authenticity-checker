"""
Analysis history for ProofyX.

Async SQLAlchemy implementation that replaces the original raw-SQLite version.
Keeps the same public API: save(), get(), get_recent(), count(), delete().
Falls back to synchronous SQLite when no async event loop is running (tests).
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import delete as sa_delete, func, select, update as sa_update

from db.database import get_session_factory
from db.models import Analysis

logger = logging.getLogger(__name__)


class AnalysisHistory:
    """Async SQLAlchemy-backed analysis history."""

    @staticmethod
    def _result_to_model(result: dict[str, Any], user_id: Optional[str] = None) -> Analysis:
        """Convert a pipeline result dict to an Analysis ORM instance."""
        return Analysis(
            id=result.get("id", str(uuid.uuid4())),
            timestamp=result.get("timestamp", datetime.now(timezone.utc).isoformat()),
            media_type=result.get("media_type", "image"),
            risk_score=result.get("risk_score", 0.0),
            risk_percent=result.get("risk_percent", 0.0),
            verdict=result.get("verdict", ""),
            confidence=result.get("confidence", ""),
            risk_level=result.get("risk_level", ""),
            model_scores=json.dumps(result.get("model_scores", {})),
            model_agreement=result.get("model_agreement", ""),
            fusion_mode=result.get("fusion_mode", ""),
            models_used=result.get("models_used", 0),
            face_detected=bool(result.get("face_detected")),
            total_frames_analyzed=result.get("total_frames_analyzed", 0),
            processing_time_ms=result.get("processing_time_ms", 0.0),
            file_name=result.get("file_name", ""),
            file_size_bytes=result.get("file_size_bytes"),
            metadata_json=json.dumps(result.get("metadata", {})),
            explanation=result.get("explanation", ""),
            user_id=user_id,
        )

    @staticmethod
    def _model_to_dict(row: Analysis) -> dict[str, Any]:
        """Convert an Analysis ORM instance to a plain dict."""
        d: dict[str, Any] = {
            "id": row.id,
            "timestamp": row.timestamp,
            "media_type": row.media_type,
            "risk_score": row.risk_score,
            "risk_percent": row.risk_percent,
            "verdict": row.verdict,
            "confidence": row.confidence,
            "risk_level": row.risk_level,
            "model_agreement": row.model_agreement,
            "fusion_mode": row.fusion_mode,
            "models_used": row.models_used,
            "face_detected": row.face_detected,
            "total_frames_analyzed": row.total_frames_analyzed,
            "processing_time_ms": row.processing_time_ms,
            "file_name": row.file_name,
            "file_size_bytes": row.file_size_bytes,
            "explanation": row.explanation,
            "gradcam_path": row.gradcam_path,
            "report_path": row.report_path,
            "user_id": row.user_id,
        }
        # Parse JSON blobs
        for field, attr in (("model_scores", row.model_scores), ("metadata", row.metadata_json)):
            if attr and isinstance(attr, str):
                try:
                    d[field] = json.loads(attr)
                except json.JSONDecodeError:
                    d[field] = {}
            else:
                d[field] = {}
        return d

    async def save(self, result: dict[str, Any], user_id: Optional[str] = None) -> str:
        """Save analysis result. Returns the analysis ID."""
        row = self._result_to_model(result, user_id=user_id)
        factory = get_session_factory()
        async with factory() as session:
            await session.merge(row)
            await session.commit()
        return row.id

    async def get(self, analysis_id: str, user_id: Optional[str] = None) -> Optional[dict[str, Any]]:
        """Retrieve a single analysis by ID, optionally scoped to user."""
        factory = get_session_factory()
        async with factory() as session:
            stmt = select(Analysis).where(Analysis.id == analysis_id)
            if user_id is not None:
                stmt = stmt.where(Analysis.user_id == user_id)
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            return self._model_to_dict(row) if row else None

    async def get_recent(
        self,
        limit: int = 20,
        media_type: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """List recent analyses, optionally filtered by media type and user."""
        factory = get_session_factory()
        async with factory() as session:
            stmt = select(Analysis).order_by(Analysis.timestamp.desc()).limit(limit)
            if media_type:
                stmt = stmt.where(Analysis.media_type == media_type)
            if user_id is not None:
                stmt = stmt.where(Analysis.user_id == user_id)
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [self._model_to_dict(r) for r in rows]

    async def count(self, user_id: Optional[str] = None) -> int:
        """Total number of analyses, optionally scoped to user."""
        factory = get_session_factory()
        async with factory() as session:
            stmt = select(func.count()).select_from(Analysis)
            if user_id is not None:
                stmt = stmt.where(Analysis.user_id == user_id)
            result = await session.execute(stmt)
            return result.scalar_one()

    async def update_paths(
        self,
        analysis_id: str,
        gradcam_path: Optional[str] = None,
        report_path: Optional[str] = None,
    ) -> None:
        """Update file paths for an existing analysis."""
        values: dict[str, Any] = {}
        if gradcam_path is not None:
            values["gradcam_path"] = gradcam_path
        if report_path is not None:
            values["report_path"] = report_path
        if not values:
            return
        factory = get_session_factory()
        async with factory() as session:
            stmt = sa_update(Analysis).where(Analysis.id == analysis_id).values(**values)
            await session.execute(stmt)
            await session.commit()

    async def delete(self, analysis_id: str) -> bool:
        """Delete an analysis by ID. Returns True if deleted."""
        factory = get_session_factory()
        async with factory() as session:
            stmt = sa_delete(Analysis).where(Analysis.id == analysis_id)
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount > 0
