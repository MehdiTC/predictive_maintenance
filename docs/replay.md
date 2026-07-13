# Continuous Sensor Replay and Delayed Feedback (Loop 8)

Loop 8 simulates a continuously operating predictive-maintenance environment.
The ten FD001 engines held out in the Loop 3 `replay` split — which no model
ever trained, validated, or calibrated on — are streamed one cycle at a time
through the real Loop 7 ingestion API. When a replayed trajectory reaches its
final observed cycle, a failure event is emitted, realized RUL labels are
backfilled for every historical prediction, and the delayed predictions are
evaluated with the Loop 4 metric implementations.

## Why held-out assets

Replay engines were isolated from all fitting-related partitions in Loop 3
(seed-42 asset-level split; see `docs/features.md`). Replaying them therefore
produces genuinely out-of-sample online behavior: the champion has never seen
these trajectories, and the delayed evaluation measures what operators would
actually have experienced.

## Lifecycle

```text
held-out replay asset (verified source)
        │  one POST /v1/sensor-readings per cycle, in order
        ▼
readings + champion predictions persisted by the Loop 7 API
        │  final observed cycle confirmed
        ▼
failure maintenance event (idempotent external event ID)
        ▼
realized labels: realized_rul = final_cycle − cycle  (prediction_outcomes)
        ▼
delayed evaluation per stored model version (model_evaluations, scope=replay)
        ▼
replay run marked completed
```

Each stage after ingestion is a separate, individually idempotent phase whose
completion is stamped on the durable `replay_runs` row
(`failure_event_id`, `labels_backfilled_at`, `evaluation_completed_at`,
`completed_at`). Resume always continues from the earliest incomplete phase.

## Ground-truth isolation

The replay subsystem knows each complete trajectory and its failure cycle;
the online inference path never does:

* The API receives only the current cycle's 26 sensor/setting values through
  the standard `SensorReadingRequest` contract — nothing else.
* The final cycle and full trajectory live only in `replay_runs` and the
  verified Parquet source; no prediction endpoint reads either.
* Feature construction inside the API uses only cycles already ingested
  (Loop 7 guarantees contiguous history per asset).
* Realized labels are written to a separate `prediction_outcomes` table and
  only after the failure event exists. Predictions are never mutated.
* Tests prove that mutating future source rows cannot change earlier payloads
  or earlier stored predictions, and that no label or failure event exists
  before the final cycle is ingested.

## Replay data source and integrity

The source of raw cycles is the validated Loop 2 trajectory Parquet
(`data/processed/cmapss/FD001/train_FD001.parquet`) restricted to the replay
partition of the verified split manifest. Before any cycle is sent, the full
provenance chain is re-verified by SHA-256:

```text
feature_manifest.split_manifest_sha256  → split_manifest.json
split_manifest.source_report_sha256     → processing_report.json
processing_report outputs               → train_FD001.parquet
```

A tampered, missing, non-contiguous, or non-finite input fails before a
partial replay can start. Replay RUL labels are never read during ingestion —
the processed trajectory Parquet does not even carry them. Loop 3 feature
files are never modified.

## Identity mapping

Each replay run row records the mapping between:

* the **source asset ID** (FD001 unit number from the split manifest),
* the **operational external asset ID** (`replay-FD001-009`, or
  `replay-FD001-009-r2` for a forced second attempt), and
* the **replay run ID** (UUID primary key of `replay_runs`).

## Modes and control

```bash
uv run python scripts/replay_sensor_data.py start --asset-id 9 --mode step
uv run python scripts/replay_sensor_data.py step   --run-id <UUID>
uv run python scripts/replay_sensor_data.py start --asset-id 9 --mode continuous --delay 1.0
uv run python scripts/replay_sensor_data.py start --asset-id 9 --mode accelerated
uv run python scripts/replay_sensor_data.py start --all --mode accelerated
uv run python scripts/replay_sensor_data.py stop   --run-id <UUID>
uv run python scripts/replay_sensor_data.py resume --run-id <UUID>
uv run python scripts/replay_sensor_data.py status --run-id <UUID> [--json]
uv run python scripts/replay_sensor_data.py status --all
uv run python scripts/replay_sensor_data.py evaluate-aggregate
```

* **step** advances exactly one cycle per invocation.
* **continuous** advances automatically, waiting `--delay` seconds between
  cycles.
* **accelerated** streams an entire lifecycle without waiting.
* `--max-cycles N` bounds how far a single invocation advances.
* `stop` requests a stop after the cycle currently in flight; the run pauses
  and can be resumed later.
* Repeating `start` for an asset is idempotent: an incomplete run is resumed
  (a paused or failed one is reactivated) and a completed run is returned
  unchanged.
* `--force-restart` explicitly begins a new attempt: an incomplete run is
  marked `cancelled` and a fresh operational asset
  (`…-r<attempt>`) receives the new stream. **Nothing is deleted** — earlier
  attempts, their readings, predictions, events, and labels all remain.
* `--api-base-url` and `--data-dir` override the corresponding settings;
  everything else comes from `TURBINE_GUARD_REPLAY_*` environment variables.

Exit codes: `0` success, `1` replay failure (with a durable error recorded on
the run), `2` usage error.

## Replay state persistence

`replay_runs` (Alembic revision `20260713_0002`) durably stores the source
identity and attempt, the operational asset, the final source cycle
(replay-internal ground truth), the last confirmed cycle, status
(`created/running/paused/completed/failed/cancelled`), mode, delay, simulated
cycle duration, phase stamps, lease fields, error details, and configuration
metadata (including the verified source checksums). Progress never exists
only in memory: a cycle is recorded only after the API confirms it.

## Concurrency protection

Advancement uses a **claim lease** on the run row (an optimistic-concurrency
variant): a worker claims in one short `SELECT … FOR UPDATE` transaction,
sends the cycle over HTTP with **no database lock held**, then confirms
progress with its claim token in a second short transaction. A competing
worker cannot claim an active lease, so the same next cycle is never sent
twice. If a worker crashes mid-send, its lease simply expires
(`TURBINE_GUARD_REPLAY_LEASE_SECONDS`), and the next worker's identical
resend reconciles through the API's exact-retry idempotency. A confirmation
attempted with a lost token fails explicitly.

## Recovery behavior

The recovery principle is *reconcile, don't assume*: payloads are
deterministic per run and cycle (fixed simulated `observed_at`, deterministic
`ingestion_id`), so resending is always safe.

| Failure | Behavior |
| --- | --- |
| Crash after the API accepted a cycle but before progress update | Resume resends the same cycle; the API answers `200` idempotent; progress catches up. |
| Timeout / uncertain outcome | Bounded retries resend the identical payload (`TURBINE_GUARD_REPLAY_MAX_SEND_ATTEMPTS`, exponential backoff). |
| API `503` (model or database outage) | Retried with backoff; on exhaustion the run is marked `failed` with the error recorded; `resume` retries later. |
| Conflicting existing reading (`409`) | Permanent: the run is marked `failed`; no endless retry. |
| Missing or tampered source | Rejected by checksum verification before any cycle is sent; a run created from different checksums refuses to continue. |
| Failure event emitted, labels missing | Resume re-enters at the backfill phase; the event is reused via its external ID, never duplicated. |
| Labels backfilled, evaluation missing | Resume re-enters at the evaluation phase; backfill is not repeated. |

## Failure-event semantics

When (and only when) the final cycle is confirmed ingested — re-verified
against the stored readings — a `maintenance_events` row of the existing
`failure` type is written **through the application service layer** with:

* the operational asset, `event_cycle = final_cycle`, and the simulated
  occurrence time of the final cycle,
* `source = "replay"` and metadata linking the dataset, subset, source asset,
  attempt, and replay run ID,
* the deterministic external event ID `replay-run:<run_id>:failure`, which
  makes repeated completion attempts exactly idempotent.

`POST /v1/maintenance-events` is deliberately deferred: the replay subsystem
is a trusted local component that already owns operational-database access
for its own state, and a public maintenance-ingestion API belongs with the
future feedback surface (see ADR 0007).

## Delayed labels

For an asset failing at cycle `T`, every stored prediction at cycle `t`
receives `realized_rul = T − t` in the `prediction_outcomes` table
(`(prediction_id, maintenance_event_id)` unique). Labels are validated to be
non-negative, zero at the final cycle, decreasing by exactly one per cycle,
and internally consistent; conflicting or physically impossible outcomes
raise instead of being stored. Backfill is idempotent — an exact repeat
returns existing rows, a changed label raises a typed conflict. Predicted and
realized values are never merged: predictions stay immutable, and one
prediction can be linked to multiple outcome events if reevaluation ever
requires it.

## Delayed evaluation

Evaluation reuses the Loop 4 implementations unmodified:
`modeling.metrics.regression_metrics` (MAE, RMSE, R², NASA asymmetric score),
`modeling.alerts.alert_metrics` at the critical (30) and warning (50)
horizons (precision, recall, F1, false alarms and false alarms per 1,000
cycles, first-alert lead times, missed failures, timely status), and
`modeling.conformal.interval_metrics` (empirical coverage, average/median
width).

Rows are grouped by the **model identity actually stored with each
prediction** (`model_name`, `model_version`); if the champion changes during
an asset's lifecycle, each version receives its own evaluation row, and no
metric ever blends versions without an explicit label. Results persist in the
existing `model_evaluations` table with `evaluation_scope = replay`:

* per-asset rows carry `metrics.aggregation = "replay_asset"` plus the replay
  run, source asset, attempt, and failure-event linkage;
* `evaluate-aggregate` persists `metrics.aggregation = "replay_aggregate"`
  rows spanning all completed runs (idempotent over the same run set).

Replay evaluation is evidence about online behavior only. It is never used to
retroactively change the Loop 4 champion, and no automated promotion exists.

Note that the champion predicts a capped-125 target while realized labels are
uncapped, so early-life rows inflate raw regression error by construction;
the NASA score and alert metrics are the operationally meaningful views. This
caveat is recorded with the evaluation rather than "corrected" by touching
the labels.

## Timestamp simulation

C-MAPSS provides no real timestamps. Each run records a UTC
`replay_started_at` epoch, and cycle `t` is assigned the simulated
observation time `replay_started_at + (t − 1) × simulated_cycle_duration`
(default 1 second, configurable). Three timestamps stay distinct: the source
cycle index, the simulated `observed_at`, and the actual `ingested_at`
written by PostgreSQL. One C-MAPSS cycle does **not** equal any real
duration; the cycle length is purely a simulation assumption.

## Observability

The replay worker emits structured JSON logs carrying run IDs and cycles, and
`ReplayMetrics` counters/gauges/histograms (runs started, cycles sent and
accepted, retries, failures, failure events, backfills, evaluations, active
and completed runs, cycle latency). Run and asset identifiers appear only in
logs, never as Prometheus labels.

## Testing

Unit tests cover source verification and tampering, payload generation and
leakage-safety, client retry/conflict/timeout behavior against a mock
transport, label math and invariants, hand-calculated evaluation metrics,
per-version grouping, and the full orchestrator lifecycle (start/step/
continuous/accelerated/pause/resume/stop, force restart, recovery from every
partial phase, lease concurrency) against an in-memory store implementing the
same contract as PostgreSQL.

Guarded integration tests (`TURBINE_GUARD_DATABASE_TEST_URL`, marker
`postgres`) replay a small fixture trajectory end-to-end over HTTP and real
PostgreSQL — persistence, failure event, backfill, evaluation, idempotent
completion, forced restart, uncertain-outcome recovery, and future-data
isolation. An optional `real_data` test replays one real held-out FD001
engine through the registered MLflow champion.

## Limitations

* One worker process per run is the intended deployment; the lease exists to
  make accidental concurrency safe, not to parallelize a single run.
* The replay CLI is a short-lived process, so its Prometheus metrics are
  demonstrative; the API's `/metrics` endpoint remains the primary scrape
  target.
* If a champion version changes after a run's labels were backfilled and new
  predictions are later added for old readings, they are not re-evaluated
  automatically; rerunning evaluation is a Loop 9 concern.
* Simulated timestamps make staleness checks on replayed assets approximate.
