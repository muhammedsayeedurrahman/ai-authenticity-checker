"""Initial analyses table with user_id column.

Revision ID: 001
Revises: None
Create Date: 2026-05-13
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "analyses",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("timestamp", sa.Text(), nullable=False),
        sa.Column("media_type", sa.Text(), nullable=False),
        sa.Column("risk_score", sa.Float(), nullable=False),
        sa.Column("risk_percent", sa.Float(), nullable=False),
        sa.Column("verdict", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Text(), nullable=False),
        sa.Column("risk_level", sa.Text(), server_default=""),
        sa.Column("model_scores", sa.Text(), nullable=False),
        sa.Column("model_agreement", sa.Text(), server_default=""),
        sa.Column("fusion_mode", sa.Text(), server_default=""),
        sa.Column("models_used", sa.Integer(), server_default="0"),
        sa.Column("face_detected", sa.Boolean(), server_default="0"),
        sa.Column("total_frames_analyzed", sa.Integer(), server_default="0"),
        sa.Column("processing_time_ms", sa.Float(), server_default="0.0"),
        sa.Column("file_name", sa.Text(), server_default=""),
        sa.Column("file_size_bytes", sa.Integer(), nullable=True),
        sa.Column("metadata_json", sa.Text(), nullable=True),
        sa.Column("explanation", sa.Text(), server_default=""),
        sa.Column("gradcam_path", sa.Text(), nullable=True),
        sa.Column("report_path", sa.Text(), nullable=True),
        sa.Column("user_id", sa.Text(), nullable=True),
    )
    op.create_index("idx_analyses_timestamp", "analyses", ["timestamp"])
    op.create_index("idx_analyses_media_type", "analyses", ["media_type"])
    op.create_index("idx_analyses_verdict", "analyses", ["verdict"])
    op.create_index("idx_analyses_user_id", "analyses", ["user_id"])


def downgrade() -> None:
    op.drop_table("analyses")
