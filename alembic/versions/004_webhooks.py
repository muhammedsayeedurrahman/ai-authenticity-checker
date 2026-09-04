"""Webhook endpoints and delivery queue.

Revision ID: 004
Revises: 003
Create Date: 2026-09-04
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "webhook_endpoints",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("org_id", sa.Text(), sa.ForeignKey("organizations.id"), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("secret_enc", sa.Text(), nullable=False),
        sa.Column("event_types_json", sa.Text(), server_default="[]"),
        sa.Column("is_active", sa.Boolean(), server_default="1"),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("last_success_at", sa.Text(), nullable=True),
        sa.Column("last_failure_at", sa.Text(), nullable=True),
        sa.Column("consecutive_failures", sa.Integer(), server_default="0"),
    )
    op.create_index("idx_webhook_endpoints_org", "webhook_endpoints", ["org_id"])

    op.create_table(
        "webhook_deliveries",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("org_id", sa.Text(), nullable=False),
        sa.Column("endpoint_id", sa.Text(), sa.ForeignKey("webhook_endpoints.id"), nullable=False),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), server_default="pending"),
        sa.Column("attempts", sa.Integer(), server_default="0"),
        sa.Column("next_attempt_at", sa.Text(), nullable=True),
        sa.Column("response_status", sa.Integer(), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("delivered_at", sa.Text(), nullable=True),
    )
    op.create_index("idx_webhook_deliveries_org", "webhook_deliveries", ["org_id"])
    op.create_index("idx_webhook_deliveries_status_next", "webhook_deliveries", ["status", "next_attempt_at"])


def downgrade() -> None:
    op.drop_index("idx_webhook_deliveries_status_next", table_name="webhook_deliveries")
    op.drop_index("idx_webhook_deliveries_org", table_name="webhook_deliveries")
    op.drop_table("webhook_deliveries")
    op.drop_index("idx_webhook_endpoints_org", table_name="webhook_endpoints")
    op.drop_table("webhook_endpoints")
