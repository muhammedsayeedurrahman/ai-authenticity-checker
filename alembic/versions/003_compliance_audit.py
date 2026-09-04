"""Compliance audit log, content labels, and SLA clocks.

Revision ID: 003
Revises: 002
Create Date: 2026-09-04
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "compliance_audit_log",
        sa.Column("seq", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("id", sa.Text(), nullable=False, unique=True),
        sa.Column("org_id", sa.Text(), nullable=False),
        sa.Column("occurred_at", sa.Text(), nullable=False),
        sa.Column("actor_type", sa.Text(), nullable=False),
        sa.Column("actor_id", sa.Text(), server_default=""),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("subject_type", sa.Text(), server_default=""),
        sa.Column("subject_id", sa.Text(), server_default=""),
        sa.Column("payload_json", sa.Text(), server_default="{}"),
        sa.Column("prev_hash", sa.Text(), nullable=True),
        sa.Column("entry_hash", sa.Text(), nullable=False),
    )
    op.create_index("idx_audit_org_seq", "compliance_audit_log", ["org_id", "seq"])
    op.create_index("idx_audit_org_subject", "compliance_audit_log", ["org_id", "subject_id"])

    op.create_table(
        "content_labels",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("org_id", sa.Text(), sa.ForeignKey("organizations.id"), nullable=False),
        sa.Column("analysis_id", sa.Text(), nullable=False),
        sa.Column("external_content_ref", sa.Text(), nullable=False),
        sa.Column("uploader_ref", sa.Text(), server_default=""),
        sa.Column("media_type", sa.Text(), server_default=""),
        sa.Column("content_sha256", sa.Text(), server_default=""),
        sa.Column("label_code", sa.Text(), nullable=False),
        sa.Column("label_display", sa.Text(), server_default=""),
        sa.Column("requires_visible_label", sa.Boolean(), server_default="0"),
        sa.Column("risk_score", sa.Float(), server_default="0.0"),
        sa.Column("confidence", sa.Text(), server_default=""),
        sa.Column("verdict", sa.Text(), server_default=""),
        sa.Column("label_basis_json", sa.Text(), server_default="[]"),
        sa.Column("ruleset_version", sa.Text(), server_default=""),
        sa.Column("detector_version", sa.Text(), server_default=""),
        sa.Column("labeled_at", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("superseded_by_id", sa.Text(), nullable=True),
    )
    op.create_index("idx_content_labels_org_ref", "content_labels", ["org_id", "external_content_ref"])

    op.create_table(
        "sla_clocks",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("org_id", sa.Text(), sa.ForeignKey("organizations.id"), nullable=False),
        sa.Column("content_label_id", sa.Text(), nullable=False),
        sa.Column("analysis_id", sa.Text(), server_default=""),
        sa.Column("obligation_type", sa.Text(), server_default="takedown_3h"),
        sa.Column("started_at", sa.Text(), nullable=False),
        sa.Column("due_at", sa.Text(), nullable=False),
        sa.Column("deadline_seconds", sa.Integer(), server_default="10800"),
        sa.Column("status", sa.Text(), server_default="running"),
        sa.Column("acted_at", sa.Text(), nullable=True),
        sa.Column("action", sa.Text(), nullable=True),
        sa.Column("acted_by", sa.Text(), nullable=True),
        sa.Column("warn_notified_at", sa.Text(), nullable=True),
        sa.Column("breach_notified_at", sa.Text(), nullable=True),
        sa.Column("notes", sa.Text(), server_default=""),
    )
    op.create_index("idx_sla_org_status_due", "sla_clocks", ["org_id", "status", "due_at"])


def downgrade() -> None:
    op.drop_index("idx_sla_org_status_due", table_name="sla_clocks")
    op.drop_table("sla_clocks")
    op.drop_index("idx_content_labels_org_ref", table_name="content_labels")
    op.drop_table("content_labels")
    op.drop_index("idx_audit_org_subject", table_name="compliance_audit_log")
    op.drop_index("idx_audit_org_seq", table_name="compliance_audit_log")
    op.drop_table("compliance_audit_log")
