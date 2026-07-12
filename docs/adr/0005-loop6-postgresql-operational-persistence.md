# ADR 0005 — Loop 6 PostgreSQL operational persistence

**Status:** Accepted (2026-07-12)

## Context

Future ingestion, inference, delayed feedback, monitoring, and workflow loops need concurrent,
transactional operational state. Parquet remains the immutable training-snapshot format and
MLflow's local SQLite database remains experiment/registry metadata; neither is the operational
database. Loop 6 must create this boundary without adding API routes, online features, replay,
monitoring calculations, orchestration, or deployment.

## Decisions

### Synchronous SQLAlchemy and psycopg

Use SQLAlchemy 2.x typed declarative mappings and synchronous psycopg 3. Current application and
model operations are synchronous, database work is short, and no established async architecture
justifies a second execution model. Engines and session factories are explicit and lazy; importing
the package opens no connection. `pool_pre_ping`, bounded pool/overflow, connect timeout, statement
timeout, and recycling are configurable.

### PostgreSQL-specific schema and JSONB

PostgreSQL supplies UUID, timezone-aware timestamps, JSONB, constraints, and concurrency-safe
`ON CONFLICT`. Core searchable fields remain relational. JSONB is limited to secondary evaluation
metrics, detailed drift results, event metadata, and pipeline metadata. String enums use named
CHECK constraints rather than native PostgreSQL enum types, making value changes and downgrades
less operationally awkward while preserving database validation.

### Repository and transaction ownership

Application services call focused repositories over a caller-provided `Session`. Repository
methods flush but never commit. `session_scope()` owns one transaction and guarantees commit on
success or rollback on error; repositories can also join a larger application transaction.
Savepoints translate expected uniqueness failures without invalidating the outer transaction.
There is no generic repository framework or implicit global session.

### Idempotency

`(asset_id, cycle)` uniquely identifies a sensor reading. PostgreSQL conflict handling followed by
an exact immutable-payload comparison returns an identical row and raises
`SensorReadingConflictError` for a mismatch. Batches preflight duplicate keys and run inside a
savepoint so a conflict publishes no partial batch.

`(sensor_reading_id, model_name, model_version)` uniquely identifies a prediction. Exact repeated
output returns the existing prediction; changed output raises `PredictionConflictError`. Optional
ingestion/request IDs add retry identities but do not replace the core keys.

Maintenance events deduplicate only when the producer supplies `external_event_id`. Content-based
deduplication is rejected because two legitimate inspections or repairs may otherwise collapse.

### Identity, timestamps, and deletion

Application-generated UUIDs are stable across ingestion sources and avoid sequence coordination.
Commands require timezone-aware timestamps; PostgreSQL stores `TIMESTAMP WITH TIME ZONE`. Asset,
reading, prediction, and event foreign keys use `ON DELETE RESTRICT`. Operational history is never
automatically cascade-deleted; retention/anonymization requires a future explicit policy.

## Dependency decision

Direct additions are SQLAlchemy (`>=2,<3`) for typed ORM/session behavior, Alembic (`>=1.14,<2`)
for versioned production schema management, and psycopg with its binary distribution (`>=3.2,<4`)
for the modern synchronous PostgreSQL driver without local compiler-header requirements. No async
driver, repository framework, SQLite test substitute, Docker, testcontainers, or server package is
added.

## Consequences

PostgreSQL integration behavior requires a developer-provided, dedicated database whose name
contains `test`; otherwise marked tests skip. Alembic, not `Base.metadata.create_all()`, owns
production setup. JSONB keeps secondary results extensible but is not a substitute for relational
model identity, windows, status, counts, or primary metrics. Loop 7 must build API schemas and
application services above these repository contracts rather than exposing ORM objects directly.
