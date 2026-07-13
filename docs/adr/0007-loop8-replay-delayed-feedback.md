# ADR 0007 — Loop 8 replay state, delayed labels, concurrency, and recovery

**Status:** Accepted (2026-07-13)

## Context

Loop 8 must replay held-out Loop 3 trajectories one cycle at a time through
the real Loop 7 HTTP contract, keep every future observation and the failure
cycle away from the inference path, emit the failure outcome only at the end
of the trajectory, backfill realized RUL labels, and evaluate historical
predictions — all durably, idempotently, and recoverable from interruption,
without a queue, scheduler, or workflow engine.

## Decisions

### A dedicated `replay_runs` table instead of overloading `pipeline_runs`

`pipeline_runs` is a generic workflow audit row. Replay needs typed cycle
progress, source/operational asset linkage, per-phase completion stamps,
lease fields, and constraints tying them together (`last_confirmed_cycle <=
final_cycle`, completed-requires-timestamp, failed-requires-error). Modelling
that in JSONB metadata would remove database validation, indexing, and
`FOR UPDATE` granularity. The focused table is the clearly cleaner design;
`pipeline_runs` remains for future orchestration loops.

The run row is also the isolation boundary: the final source cycle is stored
only here, and no prediction endpoint reads this table, so the online path
structurally cannot observe an asset's future.

### A normalized `prediction_outcomes` table for realized labels

Realized labels are not added to `predictions`: predictions stay immutable,
and a separate table keyed `(prediction_id, maintenance_event_id)` links each
label to the exact outcome event that produced it. This supports multiple
outcomes or reevaluations per prediction, makes backfill idempotent through
`ON CONFLICT DO NOTHING` plus exact content comparison (retry timestamps are
bookkeeping, not content), and lets a changed label surface as a typed
conflict instead of a silent overwrite.

### Failure events through the application service, not a new endpoint

Loop 7 deliberately shipped no maintenance-event ingestion route. The replay
worker is a trusted local component that already holds operational-database
access for its own state, so the failure event is written through the
existing `MaintenanceEventRepository` inside the same transaction that stamps
the run row. Deduplication rests on the deterministic external event ID
(`replay-run:<run_id>:failure`). `POST /v1/maintenance-events` is
intentionally deferred until an external feedback producer exists; adding it
now would create an unauthenticated public write surface no client needs.

### Claim-lease concurrency (optimistic, no lock across HTTP)

Two workers advancing one run must never send the same next cycle twice, yet
row locks must not be held during HTTP waits. Each advance therefore claims a
lease (`lease_token`, `lease_expires_at`) on the run row in one short
`FOR UPDATE` transaction, performs the HTTP send lockless, and confirms with
its token in a second short transaction. Rivals cannot claim an active lease;
a crashed worker's lease expires and the successor's byte-identical resend
reconciles via the API's exact-retry idempotency; a confirmation with a lost
token fails explicitly. A PostgreSQL advisory lock was rejected because it
pins a connection across the wait, and plain optimistic versioning was
rejected because it detects duplicates only after both workers have already
sent.

### Phase-stamped recovery state machine

Statuses are `created → running → (paused | failed) → running → completed`,
with `cancelled` reserved for runs superseded by an explicit force restart.
Delayed feedback is three ordered phases — failure event, label backfill,
evaluation — each committed atomically together with its stamp on the run row
(`failure_event_id`, `labels_backfilled_at`, `evaluation_completed_at`), so a
crash between phases leaves an unambiguous durable position and resume
re-enters at the earliest incomplete phase. Every phase is idempotent in its
own right (external event ID, outcome uniqueness, stamped evaluation), so the
stamps are an optimization and the data model is the safety net.

### Force restart creates a new attempt; history is never deleted

Operational history is append-only (Loop 6 `RESTRICT` foreign keys). A forced
restart cancels an incomplete run and begins attempt `n+1` against a fresh
operational asset (`replay-FD001-009-r2`), leaving all earlier readings,
predictions, events, and labels intact and queryable.

### Deterministic simulated timestamps

Each run fixes a UTC epoch; cycle `t` observes at
`epoch + (t−1) × simulated_cycle_duration` (configurable, default one
second). This keeps retry payloads byte-identical (required for exact-retry
idempotency), keeps source cycle, simulated observation time, and real
ingestion time distinct, and avoids implying any real cycle duration.

### Evaluation reuses Loop 4 code and is grouped by stored model identity

Delayed evaluation calls the existing `modeling` metric implementations and
groups rows by the `(model_name, model_version)` stored on each prediction,
persisting per-asset (`replay_asset`) and cross-run (`replay_aggregate`) rows
in the existing `model_evaluations` table. Replay results never feed back
into Loop 4 champion selection, and no automated promotion exists.

## Dependency decision

`httpx` moved from the dev group to runtime dependencies (same locked
version). The replay client needs a runtime HTTP client; relying on MLflow's
transitive `requests` would make the dependency accidental (the ADR 0006
precedent), and `httpx` lets tests inject FastAPI's `TestClient` — an
`httpx.Client` subclass — as the in-process client. No retry library, queue,
scheduler, worker framework, or orchestration dependency was added.

## Consequences

Replay progress, outcomes, and evaluations are durable and reconstructible
from PostgreSQL alone; any phase can be re-run safely. The cost is one extra
table pair and a lease protocol that adds two short transactions per cycle —
negligible at replay throughput. The deferred maintenance-event endpoint
means external systems cannot yet deliver outcomes; that surface belongs to
the loop that introduces real feedback producers. Aggregate evaluations are
snapshots per completed-run set; Loop 9 monitoring will decide their cadence.
