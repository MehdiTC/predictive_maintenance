# Operational PostgreSQL Database (Loops 6, 8, and 9)

## Purpose and boundary

PostgreSQL stores mutable operational history for future online workflows. It is separate from:

* Parquet under `data/`, which remains the immutable, checksummed training/feature snapshot layer.
* MLflow's default `data/mlflow/mlflow.db`, which stores local experiment and registry metadata.

Loop 6 established persistence; Loop 8 added replay/outcomes; Loop 9 adds monitoring and
model-lifecycle audit state. Modeling artifacts remain outside PostgreSQL.

## Relationships

```mermaid
erDiagram
    ASSETS ||--o{ SENSOR_READINGS : observes
    ASSETS ||--o{ PREDICTIONS : receives
    SENSOR_READINGS ||--o{ PREDICTIONS : produces
    ASSETS ||--o{ MAINTENANCE_EVENTS : experiences
    MODEL_EVALUATIONS }o--|| MODEL_IDENTITY : summarizes
    DRIFT_REPORTS }o--|| MODEL_IDENTITY : reports
    PIPELINE_RUNS ||--o{ DATA_QUALITY_REPORTS : produces
    PIPELINE_RUNS ||--o{ LIFECYCLE_ASSET_ASSIGNMENTS : assigns
    PIPELINE_RUNS ||--o{ LIFECYCLE_EVENTS : audits
    ASSETS ||--o{ LIFECYCLE_ASSET_ASSIGNMENTS : participates

    ASSETS {
        uuid id PK
        string external_id UK
        string status
    }
    SENSOR_READINGS {
        uuid id PK
        uuid asset_id FK
        int cycle UK
        float settings_1_to_3
        float sensors_01_to_21
    }
    PREDICTIONS {
        uuid id PK
        uuid asset_id FK
        uuid sensor_reading_id FK
        string model_name
        string model_version
        float predicted_rul
    }
    MAINTENANCE_EVENTS {
        uuid id PK
        uuid asset_id FK
        string event_type
        datetime occurred_at
    }
```

`MODEL_IDENTITY` is conceptual: evaluations and drift reports store `model_name` and
`model_version` directly so operational records remain meaningful if an external MLflow registry
is unavailable. There is deliberately no duplicated local model-registry table.

## Tables, constraints, and indexes

* `assets`: unique external ID; generic dataset/source identity; constrained lifecycle status.
* `sensor_readings`: positive cycle; finite three settings and 21 anonymous sensors; unique and
  indexed `(asset_id, cycle)`; ingestion timestamp/index and optional unique ingestion ID.
* `predictions`: exact model/run/feature identity; non-negative finite RUL and latency; complete,
  ordered intervals; unique `(sensor_reading_id, model_name, model_version)`; asset/timestamp and
  fleet-recency indexes; optional unique request ID.
* `maintenance_events`: failure, planned maintenance, inspection, or repair; positive optional
  event cycle; delayed `occurred_at`; asset/time indexes; optional unique external event ID.
* `model_evaluations`: replay/online/validation/benchmark scopes; ordered windows; explicit common
  metrics plus JSONB secondary metrics; model-version/creation index.
* `drift_reports`: calculated Loop 9 summaries; ordered windows, constrained status, non-negative
  distances/count; per-feature JSONB detail, reference identity, and model/window index.
* `pipeline_runs`: ingestion/monitoring/retraining/backfill/promotion types; constrained lifecycle;
  unique optional idempotency key, durable phase/metadata, terminal finish and failed-error checks;
  status/time and type/time indexes.
* `replay_runs` (Loop 8, revision `20260713_0002`): durable replay progress and phase state —
  source dataset/subset/asset and attempt (unique together), unique operational external asset ID,
  replay-internal final cycle, last confirmed cycle bounded by it, constrained status/mode,
  paired advance-lease fields, failure-event/backfill/evaluation stamps, error text, and JSONB
  configuration metadata; status and source-asset indexes. No prediction endpoint reads it.
* `prediction_outcomes` (Loop 8): realized delayed labels — unique
  `(prediction_id, maintenance_event_id)`, non-negative realized RUL, restrictive foreign keys to
  predictions, maintenance events, and assets; asset and event indexes. Predictions stay immutable.
* `data_quality_reports` (Loop 9, revision `20260713_0003`): champion/feature/window identity,
  constrained pass/warning/fail/insufficient status, record/asset/failure counts, and detailed
  checks/availability in JSONB; linked restrictively to its pipeline run.
* `lifecycle_asset_assignments` (Loop 9): unique `(pipeline_run_id, asset_id)` role assignment to
  retraining addition or promotion holdout, with source identity and row count.
* `lifecycle_events` (Loop 9): append-only idempotent event key, phase, actor, from/to model
  versions, and JSONB gate/approval/rejection/alias/refresh/rollback evidence.

All operational foreign keys are restrictive. No asset or reading deletion cascades into history.
All timestamps are timezone-aware. Command objects reject naive timestamps and non-finite values
before a SQL statement; database checks backstop core numeric and lifecycle invariants.

## Repository and transaction contract

Repositories accept a `Session`, return typed ORM records, use SQLAlchemy 2 `select()`, and never
commit. `session_scope(factory)` owns the transaction:

```python
with session_scope(factory) as session:
    asset = AssetRepository(session).create(NewAsset(external_id="asset-001"))
    SensorReadingRepository(session).insert(reading_command_for(asset.id))
```

An exception rolls the whole block back. Repository methods can therefore compose inside a future
application-service transaction. Expected immutable-key conflicts use domain errors; unexpected
database failures remain infrastructure failures and cause rollback.

## Idempotency

Sensor retries with identical `(asset_id, cycle)` data return the existing record. Any changed
timestamp, setting, sensor value, schema/source, or ingestion identity raises
`SensorReadingConflictError`; data is never overwritten. Batch insertion uses a savepoint and is
all-or-nothing.

Prediction retries with identical `(sensor_reading_id, model_name, model_version)` output return
the existing record. Changed output or identity fields raise `PredictionConflictError`.
Maintenance events are idempotent only when `external_event_id` is provided.

## Configuration and local setup

Only `postgresql+psycopg://` URLs are accepted. Copy `.env.example`, replace placeholders, and keep
the operational and test databases separate:

```bash
createdb turbine_guard
createdb turbine_guard_test
export TURBINE_GUARD_DATABASE_URL='postgresql+psycopg://USER:PASSWORD@localhost:5432/turbine_guard'
export TURBINE_GUARD_DATABASE_TEST_URL='postgresql+psycopg://USER:PASSWORD@localhost:5432/turbine_guard_test'
```

The integration guard refuses a driver other than psycopg or a database name without `test`.

## Migrations and lifecycle commands

```bash
uv run alembic upgrade head
uv run alembic current
uv run alembic history
uv run alembic downgrade -1  # development only; drops all Loop 6 tables at the initial revision
uv run pytest -m postgres tests/integration/test_postgres_operational.py
```

Migration `20260712_0001` creates the Loop 6 schema, `20260713_0002` adds Loop 8 replay/outcomes,
and `20260713_0003` adds Loop 9 lifecycle persistence. Alembic reads the typed operational URL;
credentials are not stored in `alembic.ini`. Production setup never uses
`Base.metadata.create_all()`.

## Readiness

`check_database_connection(engine)` executes `SELECT 1`, returning false for driver, timeout, or
connection errors. Loop 7 online mode installs it as a required readiness dependency together with
the champion and feature contract. Offline/test app construction can disable online inference or
inject checks, so unrelated workflows still open no connection.

## Production and backup considerations

Use managed PostgreSQL, TLS, least-privilege application and migration roles, encrypted backups,
point-in-time recovery, migration rehearsal, connection-pool sizing against provider limits, and
monitoring for failed connections/slow queries. Never place credentials in Git. Retention,
archival, restore drills, partitioning, and read replicas depend on observed volume and are not
implemented here.

## Limitations

Loop 9 provides local phase orchestration but no scheduler or distributed worker. Bulk-copy
optimization, partitioning, Docker, and deployment do not exist. UUID creation and full sensor
payload comparison add small overhead appropriate here.
