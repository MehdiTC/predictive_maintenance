"""Add Loop 8 replay-run state and realized prediction outcomes.

Revision ID: 20260713_0002
Revises: 20260712_0001
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260713_0002"
down_revision: str | None = "20260712_0001"
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
    op.create_table(
        "replay_runs",
        _uuid_pk(),
        sa.Column("dataset_name", sa.String(length=100), nullable=False),
        sa.Column("dataset_subset", sa.String(length=100), nullable=False),
        sa.Column("source_asset_id", sa.Integer(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("external_asset_id", sa.String(length=255), nullable=False),
        sa.Column("asset_id", postgresql.UUID(as_uuid=True)),
        sa.Column("final_cycle", sa.Integer(), nullable=False),
        sa.Column("last_confirmed_cycle", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            _enum(
                "replay_run_status",
                "created",
                "running",
                "paused",
                "completed",
                "failed",
                "cancelled",
            ),
            nullable=False,
        ),
        sa.Column(
            "mode",
            _enum("replay_mode", "step", "continuous", "accelerated"),
            nullable=False,
        ),
        sa.Column("cycle_delay_seconds", sa.Double(), nullable=False),
        sa.Column("simulated_cycle_duration_seconds", sa.Double(), nullable=False),
        sa.Column("replay_started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_advanced_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("lease_token", sa.String(length=64)),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
        sa.Column("failure_event_id", postgresql.UUID(as_uuid=True)),
        sa.Column("labels_backfilled_at", sa.DateTime(timezone=True)),
        sa.Column("evaluation_completed_at", sa.DateTime(timezone=True)),
        sa.Column("error_message", sa.Text()),
        sa.Column(
            "metadata", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False
        ),
        _created_at(),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("source_asset_id > 0", name="ck_replay_runs_positive_source_asset"),
        sa.CheckConstraint("attempt >= 1", name="ck_replay_runs_positive_attempt"),
        sa.CheckConstraint("final_cycle > 0", name="ck_replay_runs_positive_final_cycle"),
        sa.CheckConstraint(
            "last_confirmed_cycle >= 0 AND last_confirmed_cycle <= final_cycle",
            name="ck_replay_runs_confirmed_cycle_in_range",
        ),
        sa.CheckConstraint("cycle_delay_seconds >= 0", name="ck_replay_runs_non_negative_delay"),
        sa.CheckConstraint(
            "simulated_cycle_duration_seconds > 0",
            name="ck_replay_runs_positive_cycle_duration",
        ),
        sa.CheckConstraint(
            "status != 'completed' OR completed_at IS NOT NULL",
            name="ck_replay_runs_completed_run_has_finish",
        ),
        sa.CheckConstraint(
            "status != 'failed' OR error_message IS NOT NULL",
            name="ck_replay_runs_failed_run_has_error",
        ),
        sa.CheckConstraint(
            "(lease_token IS NULL) = (lease_expires_at IS NULL)",
            name="ck_replay_runs_lease_fields_paired",
        ),
        sa.ForeignKeyConstraint(
            ["asset_id"],
            ["assets.id"],
            name="fk_replay_runs_asset_id_assets",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["failure_event_id"],
            ["maintenance_events.id"],
            name="fk_replay_runs_failure_event_id_maintenance_events",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_replay_runs"),
        sa.UniqueConstraint(
            "dataset_name",
            "dataset_subset",
            "source_asset_id",
            "attempt",
            name="uq_replay_runs_source_attempt",
        ),
        sa.UniqueConstraint("external_asset_id", name="uq_replay_runs_external_asset_id"),
    )
    op.create_index("ix_replay_runs_status", "replay_runs", ["status"])
    op.create_index(
        "ix_replay_runs_source_asset", "replay_runs", ["dataset_subset", "source_asset_id"]
    )

    op.create_table(
        "prediction_outcomes",
        _uuid_pk(),
        sa.Column("prediction_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("maintenance_event_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("asset_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("cycle", sa.Integer(), nullable=False),
        sa.Column("realized_rul", sa.Integer(), nullable=False),
        sa.Column("labeled_at", sa.DateTime(timezone=True), nullable=False),
        _created_at(),
        sa.CheckConstraint("cycle > 0", name="ck_prediction_outcomes_positive_cycle"),
        sa.CheckConstraint(
            "realized_rul >= 0", name="ck_prediction_outcomes_non_negative_realized_rul"
        ),
        sa.ForeignKeyConstraint(
            ["prediction_id"],
            ["predictions.id"],
            name="fk_prediction_outcomes_prediction_id_predictions",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["maintenance_event_id"],
            ["maintenance_events.id"],
            name="fk_prediction_outcomes_maintenance_event_id_maintenance_events",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["asset_id"],
            ["assets.id"],
            name="fk_prediction_outcomes_asset_id_assets",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_prediction_outcomes"),
        sa.UniqueConstraint(
            "prediction_id",
            "maintenance_event_id",
            name="uq_prediction_outcomes_prediction_event",
        ),
    )
    op.create_index("ix_prediction_outcomes_asset", "prediction_outcomes", ["asset_id"])
    op.create_index("ix_prediction_outcomes_event", "prediction_outcomes", ["maintenance_event_id"])


def downgrade() -> None:
    op.drop_index("ix_prediction_outcomes_event", table_name="prediction_outcomes")
    op.drop_index("ix_prediction_outcomes_asset", table_name="prediction_outcomes")
    op.drop_table("prediction_outcomes")
    op.drop_index("ix_replay_runs_source_asset", table_name="replay_runs")
    op.drop_index("ix_replay_runs_status", table_name="replay_runs")
    op.drop_table("replay_runs")
