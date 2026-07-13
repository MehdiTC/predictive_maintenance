"""Add Loop 9 monitoring and model-lifecycle persistence.

Revision ID: 20260713_0003
Revises: 20260713_0002
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260713_0003"
down_revision: str | None = "20260713_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _enum(name: str, *values: str) -> sa.Enum:
    return sa.Enum(*values, name=name, native_enum=False, create_constraint=True)


def _uuid_pk() -> sa.Column:
    return sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False)


def _created_at() -> sa.Column:
    return sa.Column(
        "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
    )


def upgrade() -> None:
    op.add_column("pipeline_runs", sa.Column("idempotency_key", sa.String(length=64)))
    op.add_column("pipeline_runs", sa.Column("phase", sa.String(length=100)))
    op.add_column(
        "pipeline_runs",
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_unique_constraint(
        "uq_pipeline_runs_idempotency_key", "pipeline_runs", ["idempotency_key"]
    )

    op.create_table(
        "data_quality_reports",
        _uuid_pk(),
        sa.Column("pipeline_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("model_version", sa.String(length=100), nullable=False),
        sa.Column("feature_version", sa.String(length=100), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "status",
            _enum("data_quality_status", "pass", "warning", "fail", "insufficient_data"),
            nullable=False,
        ),
        sa.Column("record_count", sa.BigInteger(), nullable=False),
        sa.Column("asset_count", sa.Integer(), nullable=False),
        sa.Column("failure_count", sa.Integer(), nullable=False),
        sa.Column(
            "details", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False
        ),
        _created_at(),
        sa.CheckConstraint(
            "window_start <= window_end", name="ck_data_quality_reports_ordered_window"
        ),
        sa.CheckConstraint(
            "record_count >= 0", name="ck_data_quality_reports_non_negative_record_count"
        ),
        sa.CheckConstraint(
            "asset_count >= 0", name="ck_data_quality_reports_non_negative_asset_count"
        ),
        sa.CheckConstraint(
            "failure_count >= 0", name="ck_data_quality_reports_non_negative_failure_count"
        ),
        sa.ForeignKeyConstraint(
            ["pipeline_run_id"],
            ["pipeline_runs.id"],
            name="fk_data_quality_reports_pipeline_run_id_pipeline_runs",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_data_quality_reports"),
    )
    op.create_index(
        "ix_data_quality_reports_model_window",
        "data_quality_reports",
        ["model_name", "model_version", "window_end"],
    )

    op.create_table(
        "lifecycle_asset_assignments",
        _uuid_pk(),
        sa.Column("pipeline_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("asset_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "role",
            _enum("lifecycle_asset_role", "retraining_addition", "promotion_holdout"),
            nullable=False,
        ),
        sa.Column("source_asset_id", sa.String(length=255)),
        sa.Column("row_count", sa.Integer(), nullable=False),
        _created_at(),
        sa.CheckConstraint(
            "row_count > 0", name="ck_lifecycle_asset_assignments_positive_row_count"
        ),
        sa.ForeignKeyConstraint(
            ["pipeline_run_id"],
            ["pipeline_runs.id"],
            name="fk_lifecycle_asset_assignments_pipeline_run_id_pipeline_runs",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["asset_id"],
            ["assets.id"],
            name="fk_lifecycle_asset_assignments_asset_id_assets",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_lifecycle_asset_assignments"),
        sa.UniqueConstraint("pipeline_run_id", "asset_id", name="uq_lifecycle_run_asset"),
    )
    op.create_index(
        "ix_lifecycle_assets_run_role",
        "lifecycle_asset_assignments",
        ["pipeline_run_id", "role"],
    )

    op.create_table(
        "lifecycle_events",
        _uuid_pk(),
        sa.Column("pipeline_run_id", postgresql.UUID(as_uuid=True)),
        sa.Column("event_key", sa.String(length=128), nullable=False),
        sa.Column("event_type", sa.String(length=100), nullable=False),
        sa.Column("phase", sa.String(length=100), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("from_version", sa.String(length=100)),
        sa.Column("to_version", sa.String(length=100)),
        sa.Column("actor", sa.String(length=255), nullable=False),
        sa.Column(
            "details", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False
        ),
        _created_at(),
        sa.ForeignKeyConstraint(
            ["pipeline_run_id"],
            ["pipeline_runs.id"],
            name="fk_lifecycle_events_pipeline_run_id_pipeline_runs",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_lifecycle_events"),
        sa.UniqueConstraint("event_key", name="uq_lifecycle_events_event_key"),
    )
    op.create_index(
        "ix_lifecycle_events_run_created", "lifecycle_events", ["pipeline_run_id", "created_at"]
    )
    op.create_index(
        "ix_lifecycle_events_model_created", "lifecycle_events", ["model_name", "created_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_lifecycle_events_model_created", table_name="lifecycle_events")
    op.drop_index("ix_lifecycle_events_run_created", table_name="lifecycle_events")
    op.drop_table("lifecycle_events")
    op.drop_index("ix_lifecycle_assets_run_role", table_name="lifecycle_asset_assignments")
    op.drop_table("lifecycle_asset_assignments")
    op.drop_index("ix_data_quality_reports_model_window", table_name="data_quality_reports")
    op.drop_table("data_quality_reports")
    op.drop_constraint("uq_pipeline_runs_idempotency_key", "pipeline_runs", type_="unique")
    op.drop_column("pipeline_runs", "updated_at")
    op.drop_column("pipeline_runs", "phase")
    op.drop_column("pipeline_runs", "idempotency_key")
