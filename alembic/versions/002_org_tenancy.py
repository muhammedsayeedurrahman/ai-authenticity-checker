"""Organization tenancy: organizations, org_members, org_api_keys; analyses.org_id.

Revision ID: 002
Revises: 001
Create Date: 2026-09-04
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "organizations",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("slug", sa.Text(), nullable=False, unique=True),
        sa.Column("plan_tier", sa.Text(), server_default="trial"),
        sa.Column("region", sa.Text(), server_default="IN"),
        sa.Column("sla_takedown_seconds", sa.Integer(), server_default="10800"),
        sa.Column("contact_email", sa.Text(), server_default=""),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default="1"),
    )

    op.create_table(
        "org_members",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("org_id", sa.Text(), sa.ForeignKey("organizations.id"), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("role", sa.Text(), server_default="viewer"),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.UniqueConstraint("org_id", "user_id", name="uq_org_members_org_user"),
    )
    op.create_index("idx_org_members_user_id", "org_members", ["user_id"])

    op.create_table(
        "org_api_keys",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("org_id", sa.Text(), sa.ForeignKey("organizations.id"), nullable=False),
        sa.Column("key_prefix", sa.Text(), nullable=False),
        sa.Column("key_hash", sa.Text(), nullable=False, unique=True),
        sa.Column("label", sa.Text(), server_default=""),
        sa.Column("scopes", sa.Text(), server_default="[]"),
        sa.Column("created_by_user_id", sa.Text(), server_default=""),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("last_used_at", sa.Text(), nullable=True),
        sa.Column("revoked_at", sa.Text(), nullable=True),
    )
    op.create_index("idx_org_api_keys_org_id", "org_api_keys", ["org_id"])

    op.add_column("analyses", sa.Column("org_id", sa.Text(), nullable=True))
    op.create_index("idx_analyses_org_id", "analyses", ["org_id"])


def downgrade() -> None:
    op.drop_index("idx_analyses_org_id", table_name="analyses")
    op.drop_column("analyses", "org_id")
    op.drop_index("idx_org_api_keys_org_id", table_name="org_api_keys")
    op.drop_table("org_api_keys")
    op.drop_index("idx_org_members_user_id", table_name="org_members")
    op.drop_table("org_members")
    op.drop_table("organizations")
