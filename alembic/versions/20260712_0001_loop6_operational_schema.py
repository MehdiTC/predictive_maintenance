"""Create the Loop 6 PostgreSQL operational schema.

Revision ID: 20260712_0001
Revises: None
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260712_0001"
down_revision: str | None = None
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
        "assets",
        _uuid_pk(),
        sa.Column("external_id", sa.String(length=255), nullable=False),
        sa.Column("dataset_name", sa.String(length=100)),
        sa.Column("dataset_subset", sa.String(length=100)),
        sa.Column("source_asset_id", sa.String(length=255)),
        sa.Column(
            "status",
            _enum("asset_status", "active", "inactive", "maintenance", "retired"),
            nullable=False,
        ),
        _created_at(),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_assets"),
        sa.UniqueConstraint("external_id", name="uq_assets_external_id"),
    )

    measurement_names = [
        *(f"operating_setting_{index}" for index in range(1, 4)),
        *(f"sensor_{index:02d}" for index in range(1, 22)),
    ]
    finite = " AND ".join(
        f"{name} NOT IN ('NaN'::double precision, 'Infinity'::double precision, "
        "'-Infinity'::double precision)"
        for name in measurement_names
    )
    op.create_table(
        "sensor_readings",
        _uuid_pk(),
        sa.Column("asset_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("cycle", sa.Integer(), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        *(sa.Column(name, sa.Double(), nullable=False) for name in measurement_names),
        sa.Column("schema_version", sa.String(length=50), nullable=False),
        sa.Column("source", sa.String(length=100), nullable=False),
        sa.Column("ingestion_id", sa.String(length=255)),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("cycle > 0", name="ck_sensor_readings_positive_cycle"),
        sa.CheckConstraint(finite, name="ck_sensor_readings_finite_measurements"),
        sa.ForeignKeyConstraint(
            ["asset_id"],
            ["assets.id"],
            name="fk_sensor_readings_asset_id_assets",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_sensor_readings"),
        sa.UniqueConstraint("asset_id", "cycle", name="uq_sensor_readings_asset_cycle"),
        sa.UniqueConstraint("ingestion_id", name="uq_sensor_readings_ingestion_id"),
    )
    op.create_index("ix_sensor_readings_asset_cycle", "sensor_readings", ["asset_id", "cycle"])
    op.create_index("ix_sensor_readings_ingested_at", "sensor_readings", ["ingested_at"])

    op.create_table(
        "predictions",
        _uuid_pk(),
        sa.Column("asset_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("sensor_reading_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("cycle", sa.Integer(), nullable=False),
        sa.Column("predicted_rul", sa.Double(), nullable=False),
        sa.Column("lower_rul", sa.Double()),
        sa.Column("upper_rul", sa.Double()),
        sa.Column(
            "risk_level", _enum("risk_level", "healthy", "warning", "critical"), nullable=False
        ),
        sa.Column("failure_within_30", sa.Boolean()),
        sa.Column("failure_within_50", sa.Boolean()),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("model_version", sa.String(length=100), nullable=False),
        sa.Column("model_alias", sa.String(length=100)),
        sa.Column("model_run_id", sa.String(length=255)),
        sa.Column("feature_version", sa.String(length=100), nullable=False),
        sa.Column("prediction_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("latency_ms", sa.Double()),
        sa.Column("request_id", sa.String(length=255)),
        _created_at(),
        sa.CheckConstraint("cycle > 0", name="ck_predictions_positive_cycle"),
        sa.CheckConstraint("predicted_rul >= 0", name="ck_predictions_non_negative_predicted_rul"),
        sa.CheckConstraint(
            "lower_rul IS NULL OR lower_rul >= 0", name="ck_predictions_non_negative_lower_rul"
        ),
        sa.CheckConstraint(
            "upper_rul IS NULL OR upper_rul >= 0", name="ck_predictions_non_negative_upper_rul"
        ),
        sa.CheckConstraint(
            "(lower_rul IS NULL AND upper_rul IS NULL) OR "
            "(lower_rul <= predicted_rul AND predicted_rul <= upper_rul)",
            name="ck_predictions_ordered_interval",
        ),
        sa.CheckConstraint(
            "latency_ms IS NULL OR latency_ms >= 0", name="ck_predictions_non_negative_latency"
        ),
        sa.CheckConstraint(
            "predicted_rul NOT IN ('NaN'::double precision, 'Infinity'::double precision, "
            "'-Infinity'::double precision) AND "
            "(lower_rul IS NULL OR lower_rul NOT IN ('NaN'::double precision, "
            "'Infinity'::double precision, '-Infinity'::double precision)) AND "
            "(upper_rul IS NULL OR upper_rul NOT IN ('NaN'::double precision, "
            "'Infinity'::double precision, '-Infinity'::double precision)) AND "
            "(latency_ms IS NULL OR latency_ms NOT IN ('NaN'::double precision, "
            "'Infinity'::double precision, '-Infinity'::double precision))",
            name="ck_predictions_finite_values",
        ),
        sa.ForeignKeyConstraint(
            ["asset_id"], ["assets.id"], name="fk_predictions_asset_id_assets", ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["sensor_reading_id"],
            ["sensor_readings.id"],
            name="fk_predictions_sensor_reading_id_sensor_readings",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_predictions"),
        sa.UniqueConstraint("request_id", name="uq_predictions_request_id"),
        sa.UniqueConstraint(
            "sensor_reading_id",
            "model_name",
            "model_version",
            name="uq_predictions_reading_model_version",
        ),
    )
    op.create_index(
        "ix_predictions_asset_timestamp", "predictions", ["asset_id", "prediction_timestamp"]
    )
    op.create_index("ix_predictions_timestamp", "predictions", ["prediction_timestamp"])

    op.create_table(
        "maintenance_events",
        _uuid_pk(),
        sa.Column("asset_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "event_type",
            _enum(
                "maintenance_event_type", "failure", "planned_maintenance", "inspection", "repair"
            ),
            nullable=False,
        ),
        sa.Column("event_cycle", sa.Integer()),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source", sa.String(length=100), nullable=False),
        sa.Column("external_event_id", sa.String(length=255)),
        sa.Column("description", sa.Text()),
        sa.Column(
            "metadata", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False
        ),
        _created_at(),
        sa.CheckConstraint(
            "event_cycle IS NULL OR event_cycle > 0",
            name="ck_maintenance_events_positive_event_cycle",
        ),
        sa.ForeignKeyConstraint(
            ["asset_id"],
            ["assets.id"],
            name="fk_maintenance_events_asset_id_assets",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_maintenance_events"),
        sa.UniqueConstraint("external_event_id", name="uq_maintenance_events_external_event_id"),
    )
    op.create_index(
        "ix_maintenance_events_asset_occurred_at", "maintenance_events", ["asset_id", "occurred_at"]
    )
    op.create_index("ix_maintenance_events_occurred_at", "maintenance_events", ["occurred_at"])

    op.create_table(
        "model_evaluations",
        _uuid_pk(),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("model_version", sa.String(length=100), nullable=False),
        sa.Column(
            "evaluation_scope",
            _enum("evaluation_scope", "replay", "online", "validation", "benchmark"),
            nullable=False,
        ),
        sa.Column("dataset_subset", sa.String(length=100)),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("sample_count", sa.BigInteger(), nullable=False),
        sa.Column("mae", sa.Double()),
        sa.Column("rmse", sa.Double()),
        sa.Column("nasa_score", sa.Double()),
        sa.Column("critical_precision", sa.Double()),
        sa.Column("critical_recall", sa.Double()),
        sa.Column("interval_coverage", sa.Double()),
        sa.Column(
            "metrics", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False
        ),
        _created_at(),
        sa.CheckConstraint(
            "window_start <= window_end", name="ck_model_evaluations_ordered_window"
        ),
        sa.CheckConstraint(
            "sample_count >= 0", name="ck_model_evaluations_non_negative_sample_count"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_model_evaluations"),
    )
    op.create_index(
        "ix_model_evaluations_model_version_created",
        "model_evaluations",
        ["model_name", "model_version", "created_at"],
    )

    op.create_table(
        "drift_reports",
        _uuid_pk(),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("model_version", sa.String(length=100), nullable=False),
        sa.Column("feature_version", sa.String(length=100), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "status",
            _enum("drift_status", "not_detected", "warning", "detected", "insufficient_data"),
            nullable=False,
        ),
        sa.Column("max_psi", sa.Double()),
        sa.Column("max_wasserstein", sa.Double()),
        sa.Column("drifted_feature_count", sa.Integer(), nullable=False),
        sa.Column(
            "details", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False
        ),
        _created_at(),
        sa.CheckConstraint("window_start <= window_end", name="ck_drift_reports_ordered_window"),
        sa.CheckConstraint(
            "max_psi IS NULL OR max_psi >= 0", name="ck_drift_reports_non_negative_max_psi"
        ),
        sa.CheckConstraint(
            "max_wasserstein IS NULL OR max_wasserstein >= 0",
            name="ck_drift_reports_non_negative_max_wasserstein",
        ),
        sa.CheckConstraint(
            "drifted_feature_count >= 0", name="ck_drift_reports_non_negative_feature_count"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_drift_reports"),
    )
    op.create_index(
        "ix_drift_reports_model_window",
        "drift_reports",
        ["model_name", "model_version", "window_end"],
    )

    op.create_table(
        "pipeline_runs",
        _uuid_pk(),
        sa.Column(
            "run_type",
            _enum(
                "pipeline_run_type",
                "ingestion",
                "monitoring",
                "retraining",
                "backfill",
                "promotion",
            ),
            nullable=False,
        ),
        sa.Column(
            "status",
            _enum("pipeline_run_status", "pending", "running", "succeeded", "failed", "cancelled"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("trigger", sa.String(length=100), nullable=False),
        sa.Column("git_sha", sa.String(length=64)),
        sa.Column("model_version", sa.String(length=100)),
        sa.Column("input_manifest_checksum", sa.String(length=64)),
        sa.Column("output_manifest_checksum", sa.String(length=64)),
        sa.Column("error_message", sa.Text()),
        sa.Column(
            "metadata", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False
        ),
        _created_at(),
        sa.CheckConstraint(
            "finished_at IS NULL OR finished_at >= started_at",
            name="ck_pipeline_runs_valid_finish_time",
        ),
        sa.CheckConstraint(
            "status NOT IN ('succeeded', 'failed', 'cancelled') OR finished_at IS NOT NULL",
            name="ck_pipeline_runs_terminal_run_has_finish",
        ),
        sa.CheckConstraint(
            "status != 'failed' OR error_message IS NOT NULL",
            name="ck_pipeline_runs_failed_run_has_error",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_pipeline_runs"),
    )
    op.create_index("ix_pipeline_runs_status_started", "pipeline_runs", ["status", "started_at"])
    op.create_index("ix_pipeline_runs_type_started", "pipeline_runs", ["run_type", "started_at"])


def downgrade() -> None:
    for table in (
        "pipeline_runs",
        "drift_reports",
        "model_evaluations",
        "maintenance_events",
        "predictions",
        "sensor_readings",
        "assets",
    ):
        op.drop_table(table)
