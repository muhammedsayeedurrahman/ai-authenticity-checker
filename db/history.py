"""
SQLite analysis history for ProofyX.

Persists every analysis result for audit trail and UI history page.
Thread-safe via per-call connections.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from core.types import AnalysisResult

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "proofyx_history.db"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    media_type TEXT NOT NULL,
    risk_score REAL NOT NULL,
    risk_percent REAL NOT NULL,
    verdict TEXT NOT NULL,
    confidence TEXT NOT NULL,
    risk_level TEXT NOT NULL DEFAULT '',
    model_scores TEXT NOT NULL,
    model_agreement TEXT DEFAULT '',
    fusion_mode TEXT DEFAULT '',
    models_used INTEGER DEFAULT 0,
    face_detected INTEGER DEFAULT 0,
    total_frames_analyzed INTEGER DEFAULT 0,
    processing_time_ms REAL DEFAULT 0.0,
    file_name TEXT DEFAULT '',
    file_size_bytes INTEGER,
    metadata TEXT,
    explanation TEXT DEFAULT '',
    gradcam_path TEXT,
    report_path TEXT
);
"""

CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_analyses_timestamp ON analyses(timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_analyses_media_type ON analyses(media_type);",
    "CREATE INDEX IF NOT EXISTS idx_analyses_verdict ON analyses(verdict);",
]


class AnalysisHistory:
    """SQLite-backed analysis history."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or DEFAULT_DB_PATH)
        self._is_memory = self.db_path == ":memory:"
        # For :memory: DBs, keep a single persistent connection
        self._persistent_conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._is_memory:
            if self._persistent_conn is None:
                self._persistent_conn = sqlite3.connect(":memory:")
                self._persistent_conn.row_factory = sqlite3.Row
            return self._persistent_conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _close_conn(self, conn: sqlite3.Connection) -> None:
        """Close connection only if it's not the persistent in-memory one."""
        if not self._is_memory:
            conn.close()

    def _init_db(self) -> None:
        conn = self._get_conn()
        try:
            conn.execute(CREATE_TABLE_SQL)
            for idx_sql in CREATE_INDEXES_SQL:
                conn.execute(idx_sql)
            conn.commit()
        finally:
            self._close_conn(conn)

    def save(self, result: dict[str, Any]) -> str:
        """Save analysis result. Returns the analysis ID."""
        analysis_id = result.get("id", str(uuid.uuid4())[:8])
        timestamp = result.get(
            "timestamp",
            datetime.now(timezone.utc).isoformat(),
        )

        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO analyses
                   (id, timestamp, media_type, risk_score, risk_percent,
                    verdict, confidence, risk_level, model_scores,
                    model_agreement, fusion_mode, models_used,
                    face_detected, total_frames_analyzed,
                    processing_time_ms, file_name, file_size_bytes,
                    metadata, explanation)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    analysis_id,
                    timestamp,
                    result.get("media_type", "image"),
                    result.get("risk_score", 0.0),
                    result.get("risk_percent", 0.0),
                    result.get("verdict", ""),
                    result.get("confidence", ""),
                    result.get("risk_level", ""),
                    json.dumps(result.get("model_scores", {})),
                    result.get("model_agreement", ""),
                    result.get("fusion_mode", ""),
                    result.get("models_used", 0),
                    1 if result.get("face_detected") else 0,
                    result.get("total_frames_analyzed", 0),
                    result.get("processing_time_ms", 0.0),
                    result.get("file_name", ""),
                    result.get("file_size_bytes"),
                    json.dumps(result.get("metadata", {})),
                    result.get("explanation", ""),
                ),
            )
            conn.commit()
            return analysis_id
        finally:
            self._close_conn(conn)

    def get(self, analysis_id: str) -> Optional[dict[str, Any]]:
        """Retrieve a single analysis by ID."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM analyses WHERE id = ?", (analysis_id,)
            ).fetchone()
            return self._row_to_dict(row) if row else None
        finally:
            self._close_conn(conn)

    def get_recent(self, limit: int = 20, media_type: Optional[str] = None) -> list[dict[str, Any]]:
        """List recent analyses, optionally filtered by media type."""
        conn = self._get_conn()
        try:
            if media_type:
                rows = conn.execute(
                    "SELECT * FROM analyses WHERE media_type = ? ORDER BY timestamp DESC LIMIT ?",
                    (media_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM analyses ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [self._row_to_dict(r) for r in rows]
        finally:
            self._close_conn(conn)

    def count(self) -> int:
        """Total number of analyses."""
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT COUNT(*) FROM analyses").fetchone()
            return row[0] if row else 0
        finally:
            self._close_conn(conn)

    def update_paths(
        self, analysis_id: str, gradcam_path: Optional[str] = None,
        report_path: Optional[str] = None,
    ) -> None:
        """Update file paths for an existing analysis (gradcam, PDF report)."""
        conn = self._get_conn()
        try:
            updates = []
            params: list[Any] = []
            if gradcam_path is not None:
                updates.append("gradcam_path = ?")
                params.append(gradcam_path)
            if report_path is not None:
                updates.append("report_path = ?")
                params.append(report_path)
            if not updates:
                return
            params.append(analysis_id)
            conn.execute(
                f"UPDATE analyses SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            conn.commit()
        finally:
            self._close_conn(conn)

    def delete(self, analysis_id: str) -> bool:
        """Delete an analysis by ID. Returns True if deleted."""
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "DELETE FROM analyses WHERE id = ?", (analysis_id,)
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            self._close_conn(conn)

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        """Convert a sqlite3.Row to a plain dict with parsed JSON fields."""
        d = dict(row)
        # Parse JSON blobs
        for json_field in ("model_scores", "metadata"):
            if json_field in d and isinstance(d[json_field], str):
                try:
                    d[json_field] = json.loads(d[json_field])
                except json.JSONDecodeError:
                    d[json_field] = {}
        # Convert face_detected back to bool
        if "face_detected" in d:
            d["face_detected"] = bool(d["face_detected"])
        return d
