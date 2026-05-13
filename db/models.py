"""
SQLAlchemy ORM models for ProofyX.

Defines the `analyses` table matching the existing SQLite schema,
with an added `user_id` column for auth scoping.
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Float,
    Index,
    Integer,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    timestamp: Mapped[str] = mapped_column(Text, nullable=False)
    media_type: Mapped[str] = mapped_column(Text, nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_percent: Mapped[float] = mapped_column(Float, nullable=False)
    verdict: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[str] = mapped_column(Text, nullable=False)
    risk_level: Mapped[str] = mapped_column(Text, default="")
    model_scores: Mapped[str] = mapped_column(Text, nullable=False)  # JSON blob
    model_agreement: Mapped[str] = mapped_column(Text, default="")
    fusion_mode: Mapped[str] = mapped_column(Text, default="")
    models_used: Mapped[int] = mapped_column(Integer, default=0)
    face_detected: Mapped[bool] = mapped_column(Boolean, default=False)
    total_frames_analyzed: Mapped[int] = mapped_column(Integer, default=0)
    processing_time_ms: Mapped[float] = mapped_column(Float, default=0.0)
    file_name: Mapped[str] = mapped_column(Text, default="")
    file_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    explanation: Mapped[str] = mapped_column(Text, default="")
    gradcam_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    report_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)

    __table_args__ = (
        Index("idx_analyses_timestamp", "timestamp"),
        Index("idx_analyses_media_type", "media_type"),
        Index("idx_analyses_verdict", "verdict"),
        Index("idx_analyses_user_id", "user_id"),
    )
