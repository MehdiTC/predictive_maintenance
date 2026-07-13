# STATUS.md

## Project Status

**Project:** TurbineGuard
**Current phase:** Loop 9 complete and validated
**Active loop:** None — awaiting review before Loop 10
**Overall status:** Production lifecycle monitoring, structured retraining decisions, leakage-safe
candidate fitting, same-holdout champion comparison, blocking promotion gates, explicit approval,
safe MLflow alias promotion/rollback, serving-cache refresh, and phase-based recovery are
implemented and documented (ADR 0008, `docs/monitoring.md`). All requested quality gates and 398
tests pass with real PostgreSQL and local MLflow enabled; migration `20260713_0003` is at head.
The live champion remained v1 because the available one newly labeled asset is correctly below the
five-asset/two-holdout safety thresholds. No Loop 10 functionality exists.
**Last updated:** 2026-07-13

---

## Project Objective

Build a public, production-style predictive-maintenance ML platform using NASA turbine degradation data.

The completed system will demonstrate:

* Reproducible data acquisition
* Data validation and processing
* Exploratory data analysis
* Leakage-safe time-series feature engineering
* Remaining Useful Life modeling
* Maintenance-oriented evaluation
* Experiment tracking
* Model versioning
* FastAPI deployment
* PostgreSQL persistence
* Continuous sensor replay
* Delayed outcome collection
* Drift and performance monitoring
* Conditional retraining
* Model promotion gates
* Dockerized local deployment
* CI/CD
* A public dashboard and demo

See `PROJECT_SPEC.md` for the complete design.

---

## Current Repository State

Loops 0–9 are complete and validated:

```text
├── src/turbine_guard/
│   ├── api/                    # create_app() factory, health routes/schemas
│   ├── config/settings.py      # typed BaseSettings (+ data_dir, cmapss_source_url)
│   ├── data/
│   │   ├── acquisition.py      # idempotent C-MAPSS FD001 acquisition (+ public verify_raw_layer)
│   │   ├── manifest.py         # provenance manifest models + persistence
│   │   ├── schema.py           # canonical column names/dtypes (SCHEMA_VERSION 1)
│   │   ├── parsing.py          # typed parser with line-numbered ParseError
│   │   ├── validation.py       # structured checks + stats; optional FD001 canonical profile
│   │   ├── processing.py       # verify → parse → validate → Parquet + JSON report
│   │   ├── cli.py              # acquisition CLI
│   │   └── process_cli.py      # processing CLI
│   ├── features/               # Loop 3: labels, splits, features (model-free)
│   │   ├── config.py           # frozen-dataclass FeatureConfig/SplitConfig/RulConfig/BuildConfig
│   │   ├── labels.py           # RUL = T_i − t, optional cap, validation, test benchmark
│   │   ├── splits.py           # deterministic asset-level 70/15/5/10 partitioning
│   │   ├── builder.py          # stateless FeatureBuilder + IncrementalFeatureState
│   │   ├── manifest.py         # SplitManifest + FeatureManifest models/persistence
│   │   ├── pipeline.py         # verify inputs → labels → split → features → outputs
│   │   └── build_cli.py        # feature-build CLI
│   ├── modeling/               # Loop 4 models, metrics, alerts, conformal, simulation, artifacts
│   ├── tracking/               # Loop 5 MLflow adapter, pyfunc, registry, aliases, CLI
│   ├── replay/                 # Loop 8: verified source, HTTP client, lease-protected state,
│   │                           #   orchestrator, realized labels, delayed evaluation, CLI
│   ├── monitoring/             # Loop 9: reports, decisions, safe data split, candidate,
│   │                           #   gates, durable lifecycle service, CLI
│   ├── services/health.py
│   └── logging_config.py       # structured JSON logging
├── scripts/download_data.py     # thin wrappers over turbine_guard CLIs
├── scripts/process_data.py
├── scripts/build_features.py
├── scripts/train_models.py
├── scripts/mlflow_models.py
├── scripts/model_lifecycle.py
├── notebooks/01_eda.ipynb       # the single primary EDA notebook, executed
├── docs/data_contract.md        # raw structure, canonical schema, validation rules, outputs
├── docs/features.md             # Loop 3 contract: labels, splits, features, manifests
├── docs/modeling.md             # Loop 4 roles, formulas, policies, artifacts, limitations
├── docs/adr/0001-…, 0003-…      # Loop 2–4 material decision records
├── tests/                       # unit fixtures + optional local FD001 integration
├── data/                        # gitignored: raw, manifests, processed, features
└── pyproject.toml / uv.lock / Makefile / .pre-commit-config.yaml / .env.example / README.md
```

Data layers on disk after `make acquire && make process && make features`:

```text
data/
├── raw/cmapss/FD001/            # immutable, read-only (0444), checksum-verified
├── manifests/cmapss_fd001.json
├── processed/cmapss/FD001/
│   ├── train_FD001.parquet      # 20,631 rows, 100 engines, canonical schema
│   ├── test_FD001.parquet       # 13,096 rows, 100 engines
│   ├── rul_FD001.parquet        # 100 official test RUL values
│   └── processing_report.json   # machine-readable checks/stats/checksums/provenance
└── features/cmapss/FD001/
    ├── train.parquet            # 14,407 rows, 70 engines: ids, split, rul, 552 features
    ├── validation.parquet       # 3,160 rows, 15 engines
    ├── calibration.parquet      # 909 rows, 5 engines
    ├── replay.parquet           # 2,155 rows, 10 engines (isolated for Loop 8)
    ├── test_features.parquet    # 13,096 rows, 100 engines, no targets
    ├── test_labels.parquet      # 100-row official RUL benchmark (evaluation only)
    ├── split_manifest.json      # seed, strategy, asset IDs/counts/rows per partition
    └── feature_manifest.json    # feature definition, columns, checksums, provenance
```

Loop 4 model artifacts remain under `data/models/cmapss/FD001/`. Optional Loop 5 runtime state uses
`data/mlflow/` by default and remains gitignored. PostgreSQL operational persistence is implemented
under `database/` and Alembic (through revision `20260713_0003`). Loop 7 serves the online API,
Loop 8 adds replay/delayed feedback, and Loop 9 adds lifecycle monitoring and promotion. Docker,
deployment, distributed orchestration, and other Loop 10 functionality remain absent deliberately.

---

## Current Loop

Loop 9 is complete and validated. Do not begin Loop 10 without explicit approval.

---

## Completed Work

* [x] Defined the overall project mission, dataset, target, and architecture (planning).
* [x] Implemented Loop 0 — repository foundation (2026-07-12).
* [x] Implemented Loop 1 — dataset acquisition and manifesting (2026-07-12).
* [x] Implemented Loop 2 — validation, processing, and EDA (2026-07-12).
* [x] Implemented Loop 3 — labels, asset-level splits, and leakage-safe features (2026-07-12).
* [x] Implemented Loop 4 — offline modeling, evaluation, uncertainty, and simulated maintenance
  policy (2026-07-12).
* [x] Implemented and validated Loop 5 — MLflow tracking and model registry (2026-07-12).
* [x] Implemented and validated Loop 6 — PostgreSQL operational persistence (2026-07-12).
* [x] Implemented and validated Loop 7 — FastAPI online inference service (2026-07-13).
* [x] Implemented and validated Loop 8 — continuous sensor replay and delayed feedback
  (2026-07-13).
* [x] Implemented and validated Loop 9 — monitoring, retraining, candidate evaluation, and model
  promotion (2026-07-13).

---

## Loop 9 Implementation Notes

1. **Monitoring reference.** The exact champion's checksummed Loop 3 training role is the only
   reference. A stable, versioned artifact stores moments, missingness, decile proportions, and
   101 quantiles for all 552 features and is tagged/logged against the exact MLflow model version.
   Validation, calibration, replay, and official test data are excluded.
2. **Reports.** Data quality checks accepted-reading structure and availability; drift calculates
   PSI, quantile-integral Wasserstein distance, missingness shift, standardized mean shift, and
   standardized standard-deviation shift. Delayed performance reuses Loop 4 regression,
   alert/episode/lead-time, and conformal metrics over Loop 8 outcomes. Reports persist in the
   existing `drift_reports`/`model_evaluations` tables and new `data_quality_reports` table.
3. **Decisions.** Typed thresholds produce exactly `no_action`, `monitor`, `retrain`, or `blocked`.
   Signals include minimum labeled assets/rows, interval, MAE/RMSE loss, critical recall, false
   alarms, coverage, feature drift, data-quality failure, and manual force. Force never bypasses
   data or holdout safety.
4. **Retraining isolation.** The point-model base is original training data plus successfully used
   operational additions. Newly completed labeled assets are deterministically split by asset into
   fit additions and a promotion holdout. Protected validation/calibration and official NASA test
   data are never opened; an asset cannot appear in both new roles.
5. **Candidate evaluation.** The champion's existing Loop 4 family, parameters, target, features,
   and horizons are recovered; no new family/search was added. Candidate fitting reuses Loop 4.
   Candidate, the frozen champion, and the Loop 4 median baseline share one fingerprinted holdout.
   Comparison includes accuracy, NASA, alert/lead-time, uncertainty, latency, and artifact size.
6. **Gates and registry.** Twelve explicit blocking gates cover quality/data/artifact validity,
   naive-baseline improvement, bounded RMSE/NASA loss, recall, false alarms, coverage, latency,
   size, and MLflow reload equivalence. Registration orders `candidate` then verified `challenger`;
   approval is required by default before old champion → `archived` and candidate → `champion`.
   Rejection preserves champion; rollback validates a numbered version and archives the displaced
   version.
7. **Recovery and audit.** Alembic `20260713_0003` adds report, assignment, and event tables plus
   pipeline idempotency/phase timestamps. Artifacts and comparisons are checksum-verified, MLflow
   runs/versions are lifecycle-keyed, alias operations are re-entrant, and events are append-only
   and idempotent. Interrupted phases resume without duplicating reports, versions, or history.
8. **Serving refresh.** Loop 7 now loads/validates the replacement before swapping the cached model;
   failure preserves the loaded object. Promotion/rollback persist refresh success or error.
9. **CLI/dependencies.** `scripts/model_lifecycle.py` supports monitor, status, force/evaluate,
   promotion dry-run/approval/rejection, rollback, and refresh. No dependency was added.
10. **Live outcome.** The 2026-07-01–07-15 window had 221 accepted readings across two assets and
    passed data quality. Against training-only reference v1, 309/552 features crossed detected
    drift; delayed champion-v1 metrics over 201 labeled rows were MAE 26.12, RMSE 32.49, NASA
    7144.70, critical recall 0.645, false alarms 0/1000, coverage 0.478, width 37.88. The trigger
    was correctly `blocked`: only one new labeled asset/201 rows and no safe two-asset holdout.
    Repeating the exact window returned the same run UUID. Manual force remained blocked; no
    candidate/version/alias was created, and production `champion`, `candidate`, and `challenger`
    remained v1.

---

## Loop 8 Implementation Notes

1. **Replay source.** `replay/source.py` reads raw cycles from the validated Loop 2 trajectory
   Parquet restricted to the Loop 3 replay partition, re-verifying the SHA-256 chain feature
   manifest → split manifest → processing report → Parquet before exposing any row. Wrong-split,
   missing, tampered, non-contiguous, and non-finite inputs fail before a partial replay. Replay
   RUL labels are never read during ingestion; Loop 3 outputs are never modified.
2. **Replay client.** `replay/client.py` builds exact `SensorReadingRequest` payloads (one cycle
   only, deterministic simulated `observed_at` and `ingestion_id`) and POSTs them to
   `/v1/sensor-readings` over httpx. Timeouts and 5xx are retried with bounded exponential backoff
   by resending the identical payload (reconciliation = the API's exact-retry idempotency);
   409/422 are permanent; the confirmed asset/cycle identity is verified on every response.
3. **Durable state.** New `replay_runs` table (Alembic `20260713_0002`) records source
   dataset/subset/asset/attempt, unique operational external asset ID, replay-internal final
   cycle, last confirmed cycle, status/mode/delay, simulated cycle duration, phase stamps, lease
   fields, error text, and source checksums. No prediction endpoint reads it.
4. **Concurrency.** Advancement uses a claim lease: claim in one short `FOR UPDATE` transaction,
   send over HTTP with no lock held, confirm with the token. Rivals cannot claim an active lease
   (so the same next cycle is never sent twice); crashed leases expire and idempotent resends
   reconcile; stale confirmations fail explicitly.
5. **Failure events.** After the final cycle is verified persisted, a `failure` maintenance event
   is written through the existing repository with external event ID
   `replay-run:<run_id>:failure` (exactly-once across retries), `event_cycle = final_cycle`, the
   simulated occurrence time, and replay lineage metadata. `POST /v1/maintenance-events` is
   deliberately deferred (ADR 0007).
6. **Delayed labels.** New `prediction_outcomes` table stores `realized_rul = T − t` per stored
   prediction, unique per `(prediction_id, maintenance_event_id)`, backfilled only after the
   event exists. Labels are validated (non-negative, zero at the final cycle, −1 per cycle,
   conflict/impossible detection); predictions remain immutable; backfill is idempotent.
7. **Delayed evaluation.** Reuses Loop 4 `regression_metrics`, `alert_metrics` (critical 30 /
   warning 50), and `interval_metrics`, grouped by the model identity stored with each
   prediction; per-asset (`replay_asset`) and cross-run (`replay_aggregate`) rows persist in
   `model_evaluations` with scope `replay`. Replay results never change the Loop 4 champion.
8. **Lifecycle.** `replay/engine.py` supports step/continuous/accelerated modes, `--max-cycles`,
   stop-after-current-cycle, resume-from-earliest-incomplete-phase, idempotent repeated starts,
   and documented force restart (cancels an incomplete run, new attempt + fresh operational asset
   `…-rN`; nothing is deleted). Phase transitions are single transactions, so partial-phase crash
   states are recoverable and were tested.
9. **CLI.** `scripts/replay_sensor_data.py` (`replay/cli.py`): `start` (`--asset-id`/`--all`,
   `--mode`, `--delay`, `--max-cycles`, `--force-restart`), `step`, `resume`, `status`
   (`--run-id`/`--all`, `--json`), `stop`, `evaluate-aggregate`; configurable
   `--api-base-url`/`--data-dir`; exit codes 0/1/2; concise summaries on stdout, structured JSON
   logs for diagnostics.
10. **Observability.** `ReplayMetrics` (own registry): runs started/active/completed, cycles
    sent/accepted, retries, failures, failure events, backfills, evaluations, cycle latency; run
    and asset IDs appear only in structured logs, never as metric labels.
11. **Settings and dependencies.** Seven typed `TURBINE_GUARD_REPLAY_*` settings (`.env.example`
    updated). The only dependency change is promoting `httpx>=0.27` from the dev group to runtime
    (already in `uv.lock`; needed by the replay client at runtime; lets tests inject FastAPI's
    `TestClient`, an `httpx.Client` subclass). No queue, scheduler, retry library, or
    orchestration dependency.
12. **Timestamps.** Per-run UTC epoch; cycle `t` observes at `epoch + (t−1) × simulated cycle
    duration` (default 1 s, configurable). Source cycle, simulated observation time, and real
    ingestion time stay distinct; no real-hour claim is implied.
13. **Tests.** 70 new unit tests (343 total pass locally): source integrity/split enforcement,
    payload determinism and future-mutation immunity, client retry/conflict/timeout semantics,
    label math/invariants/conflicts, hand-calculated evaluation metrics and per-version grouping,
    full orchestrator lifecycle/recovery/concurrency against an in-memory store implementing the
    same contract. Five new guarded PostgreSQL integration tests cover the complete HTTP
    lifecycle, uncertain-outcome recovery, partial-phase resume, force restart, and outcome
    idempotency/conflicts, plus one optional real-FD001 replay with the registered champion.
14. **Validation (all user-run on 2026-07-13):** `uv sync` (168 resolved / 163 checked), Ruff
    format (132 files) and lint, strict Mypy (75 files), full pytest (364 collected; 349 passed
    with PostgreSQL tests skipped, then all 21 integration tests passed with
    `TURBINE_GUARD_DATABASE_TEST_URL` set — including the six replay tests and the real-FD001
    champion replay), and all pre-commit hooks. One pre-existing Loop 6 test pinned the old head
    revision literally; it now asserts the database matches Alembic's actual head and also
    requires the two Loop 8 tables (a strengthening, verified passing).
15. **Live validation (real FD001 engine 9, registered champion, 2026-07-13):** migration
    `20260712_0001 → 20260713_0002` applied and `alembic current` at head. Step mode advanced
    exactly one cycle per invocation; resume streamed all 201 cycles through HTTP; the failure
    event was emitted exactly once at cycle 201 (`replay-run:<run_id>:failure`); 201 realized
    labels were backfilled (0/1/2 at cycles 201/200/199); per-asset and aggregate evaluations
    persisted with scope `replay`. Completed-run `start` was a no-op and repeated
    `evaluate-aggregate` inserted no duplicate (exactly two evaluation rows total). Loop 3/4
    reruns returned `already_built`/`already_trained`.
16. **Real delayed-evaluation result (engine 9, 201 cycles, champion v1):** MAE 26.12, RMSE
    32.49, NASA score 7144.70, critical recall 0.645, interval coverage 0.478. These are
    computed against realized *uncapped* RUL, so early-life rows (true RUL 125–200 versus the
    capped-125 champion) inflate regression error and break interval coverage by construction;
    this is the honest operator-facing view, documented in `docs/replay.md`, and is not
    comparable to Loop 4's capped-domain replay metrics.

---

## Loop 7 Implementation Notes

1. `/v1` routes cover atomic sensor ingestion, asset list/detail/health, recent predictions,
   current champion metadata, and currently observable monitoring summary. A direct prediction
   route is deliberately deferred because one configured champion adds no distinct contract.
2. Sensor requests strictly validate positive cycles, timezone-aware UTC-normalized timestamps,
   finite three settings and 21 anonymous sensors, source/schema identifiers, and forbidden extras.
3. First cycle auto-creates a generic active asset. New cycles must be contiguous from 1; gaps and
   new older records return 409. Exact historical retries remain valid; changed duplicates return
   typed conflicts and are never overwritten.
4. One caller-owned transaction covers asset resolution/locking, reading, same-asset bounded
   history, shared `FeatureBuilder`, champion prediction, and version-pinned prediction. Feature or
   model failure rolls the whole request back.
5. The feature configuration is reconstructed from the verified Loop 3 manifest. Exact feature
   version and ordered 552-column equality are checked against the registered pyfunc signature.
6. The MLflow champion is loaded lazily/preloaded from the configurable alias into a thread-safe
   per-process cache. Registry/run evidence, feature contract, checksum, lineage, and load time are
   retained; explicit refresh exists without automated watching.
7. Structured safe errors and `X-Request-ID` correlation cover validation, conflicts, not-found,
   database/model/feature availability, and unexpected failures. Full sensor payloads and secrets
   are not logged.
8. Per-app Prometheus registries expose bounded-label HTTP, ingestion, prediction, failure,
   latency, risk, readiness, and model identity metrics without collector collisions.
9. Online lifespan creates/disposes engine resources, optionally preloads the champion, and makes
   PostgreSQL, model, and feature compatibility required readiness checks. Offline tests can disable
   or inject resources.
10. The only direct dependency added is `prometheus-client>=0.21,<1`; no Loop 8 replay/feedback,
    monitoring calculations, retraining, orchestration, Docker, or deployment functionality exists.
11. Automated checks passed: `uv sync` resolved 168/checked 163 packages; 113 files formatted;
    Ruff lint passed; strict Mypy passed over 66 source files; 273 tests passed, including three new
    PostgreSQL tests with real champion equality and forced rollback; all pre-commit hooks passed.
    Live validation ingested 20 contiguous real FD001 cycles, persisted exactly 20 readings and 20
    predictions, returned exact retries with HTTP 200/all idempotency flags, rejected a changed
    duplicate with HTTP 409, and exposed asset detail/health/recent predictions in UTC.
12. Live readiness passed with database/model/feature checks true. PostgreSQL-unavailable and
    missing-champion demonstrations returned 503 with accurate individual checks. Loop 3/4 reruns
    returned `already_built`/`already_trained`; the registry champion remained exactly equivalent
    with maximum absolute difference 0.0. No Loop 8 implementation exists.

---

## Loop 6 Implementation Notes

1. The operational store is PostgreSQL through synchronous SQLAlchemy 2.x and psycopg 3. Engine
   creation is lazy; pool, overflow, recycle, pool/connect/statement timeouts, echo, development
   URL, and guarded test URL are typed settings separate from MLflow's SQLite backend.
2. Seven typed ORM tables cover assets, immutable sensor cycles, model-version-pinned predictions,
   maintenance/failure events, model evaluations, stored drift reports, and pipeline runs. UUID
   primary keys, timezone-aware timestamps, JSONB extension fields, named checks, restrictive
   foreign keys, explicit nullability, and common-query indexes are defined.
3. Alembic revision `20260712_0001` creates the complete schema from an empty database and safely
   downgrades to base in development. Credentials are absent from `alembic.ini`; autogeneration
   metadata comes from the typed ORM. Programmatic Alembic runs preserve existing application
   loggers.
4. Focused repositories use SQLAlchemy 2 `select()` and caller-provided sessions. They flush but
   never commit. `session_scope()` owns commit/rollback; sensor batches use a savepoint so conflicts
   cannot publish partial data.
5. Sensor identity is `(asset_id, cycle)` and prediction identity is
   `(sensor_reading_id, model_name, model_version)`. PostgreSQL `ON CONFLICT` plus exact immutable
   payload comparison returns identical retries and raises typed conflicts for changed data.
   Maintenance events deduplicate only through an explicit external event ID.
6. Small frozen commands validate finite numerics, positive cycles, aware timestamps, enums,
   prediction intervals, ordered windows, and pipeline lifecycle before ORM insertion. Database
   checks independently backstop core invariants.
7. Database readiness executes `SELECT 1`, converts connection/timeout failures to unavailable,
   and is injectable. The API keeps an empty readiness map by default, preserving Loop 0 behavior.
8. Direct dependencies added: `SQLAlchemy>=2,<3`, `alembic>=1.14,<2`, and
   `psycopg[binary]>=3.2,<4`. No async stack, SQLite substitute, Docker, testcontainers,
   orchestration, or repository framework was added.
9. A dedicated local PostgreSQL 17 development database migrated from empty to head. A separate
   guarded `turbine_guard_test` database passed all six integration tests. Development downgrade
   to base and re-upgrade to head both passed; all eight expected relations were restored.
10. Full validation passed: 253 tests, Ruff formatting/lint, strict Mypy, and all pre-commit hooks.
    Loop 3 returned `already_built`, Loop 4 returned `already_trained`, and the MLflow champion
    matched exactly with maximum absolute difference 0.0. No Loop 7 routes exist.

---

## Loop 5 Implementation Notes

1. MLflow is an optional post-completion adapter. Ordinary Loop 4 training remains tracking-free;
   `--track-with-mlflow` consumes only a complete checksum-verified local execution.
2. The default local backend is SQLite (`data/mlflow/mlflow.db`) with filesystem artifacts under
   `data/mlflow/artifacts`; tracking URI, experiment, model name, artifact location, aliases,
   registration, promotion, run prefix, project tag, and environment are typed settings.
3. One parent run represents the complete execution and one nested child represents every candidate.
   Tags/parameters/metrics/artifacts cover raw-validation-split-feature-config-code lineage, all
   validation metrics, ranks/eligibility, and champion calibration/replay/official/policy evidence.
4. The registered flavor is a small custom pyfunc over the existing checksummed `ModelBundle`; it
   returns point/lower/upper RUL and risk class without duplicating preprocessing or interval logic.
5. The explicit signature contains the ordered 552 feature columns only. Missing columns fail;
   MLflow reorders named inputs and ignores extras as the documented explicit policy.
6. Registry versions are deduplicated by champion-bundle SHA. `candidate` and `challenger` follow a
   verified version; `champion` requires Loop 4 eligibility plus enabled promotion; a displaced
   champion receives `archived` and all historical versions remain.
7. Exact training-manifest SHA plus registry-behavior SHA identifies an already-logged execution.
   Explicit flags create new run history and, separately, a new version. Tampered local artifacts
   fail before MLflow mutation.
8. `scripts/mlflow_models.py` inspects parent/child runs, metrics, versions, and aliases, and verifies
   load-by-alias/version prediction equivalence. A registry-aware model card is logged as an artifact.
9. Direct dependency added: `mlflow>=3.5,<4`; no other direct dependency and no Loop 6 technology.
10. Full user validation passed: 232 tests, Ruff format/lint, strict Mypy, and all pre-commit hooks.
    Persistent FD001 tracking logged all 14 candidates under parent run
    `ba656718a5654a0c9c536411906562b2`; source child
    `f9288609d202432a99d45e3025ac80c5` registered version 1 with `candidate`, `challenger`, and
    `champion`. Alias/version loads matched the local bundle exactly (maximum difference 0.0).
11. The identical rerun returned `already_trained` and `already_logged`, retained the same parent,
    14 child runs, version 1, and aliases. Before/after Loop 3 and Loop 4 checksum diff was empty.
12. The MLflow UI launched successfully on the configured SQLite backend and returned HTTP 200.
    The observed CloudPickle/pip-resolution/Starlette/joblib messages are upstream warnings; model
    load, UI, and all tests passed.

---

## Loop 4 Implementation Notes

1. Exact Loop 3 manifests/checksums, column order, role labels/counts, and disjoint asset IDs are
   verified before fitting; model features are never inferred from numeric columns.
2. Constant median, Ridge, histogram-gradient-boosting, and XGBoost approaches are evaluated over
   uncapped and capped-125 targets with a small recorded manual grid and fixed seeds.
3. Ridge preprocessing is train-only median imputation plus missing indicators and scaling; tree
   candidates use native structural-null support. Pipelines are serialized with the model.
4. Validation alone determines operational eligibility and champion ranking. Calibration only
   fits the finite-sample conformal residual quantile. Replay and official final-row metrics cannot
   affect selection.
5. Reports include row/asset regression metrics, lifecycle/length slices, NASA asymmetric score,
   alert rows and collapsed episodes, lead time, coverage/width, latency/size, coefficients/
   importance, and explicitly simulated normalized-cost maintenance policies.
6. Candidate pipelines, champion bundle, metadata, JSON/CSV/Markdown reports, and a completion
   manifest are local, checksummed, tamper-detected, idempotent, and deliberately not registered.
7. ADR 0003 records imputation, target comparison, alerts, conformal approximation, champion
   policy, simulation, and joblib safety decisions. Full behavior is in `docs/modeling.md`.
8. **Real FD001 result:** all 14 configurations trained. Capped-125 Ridge (`alpha=1`) won on
   validation common-domain RMSE 14.448 with critical recall 0.763 and 0.633 false alarms per
   1,000 cycles. Replay RMSE was 13.955; official final-row RMSE was 14.407.
9. **Intervals and simulation:** replay empirical 90% interval coverage was 0.898 with average
   width 39.148. In the base normalized simulation, predictive cost was 42.83 versus reactive
   120.0 (−64.3%), with 10 planned interventions, no simulated failures/misses, and 261 cycles of
   useful life forfeited. These are simulated normalized units, not currency or claimed savings.

---

## Loop 3 Implementation Notes

1. **RUL labels** (`features/labels.py`): uncapped `rul = T_i − t` (int64) always generated; optional `rul_capped = min(rul, cap)` only when a cap is configured (CLI `--rul-cap`), uncapped always preserved. Validation enforces non-negativity, RUL 0 at each final cycle, and exact −1 steps. The official test set gets **no fabricated per-row labels**: `test_labels.parquet` is a 100-row evaluation benchmark (`asset_id`, `final_cycle`, `rul`) encoding the positional row *i* ↔ asset *i+1* correspondence explicitly (asserted, documented as unverifiable from content).
2. **Splits** (`features/splits.py`): by `asset_id`, never by row. Default seed 42, `numpy.random.default_rng` permutation of sorted unique IDs, largest-remainder rounding. FD001: exactly **70 train / 15 validation / 5 calibration / 10 replay** engines (14,407 / 3,160 / 909 / 2,155 rows). `AssetSplit` construction proves disjointness; replay and calibration are isolated; the official test set is untouched.
3. **FeatureBuilder** (`features/builder.py`) is **stateless**: features at cycle `t` are a pure function of that asset's observations ≤ `t` (trailing windows, grouped per asset, sorted internally). This makes leakage-safety, cross-asset isolation, fit isolation, and training-serving consistency structural. `IncrementalFeatureState` feeds one cycle at a time through the *same* batch code; offline == incremental at every cycle (tested, incl. restart via `from_history`).
4. **Features:** 24 source columns × 23 = **552 features**: current, delta_1, rolling mean/std/min/max/range/slope over windows 5/10/20, EWM mean spans 5/10/20 (`adjust=True`). Families/windows/spans/min_periods configurable; deterministic names and stable order; `feature_columns()` is the authoritative contract. Rolling slope = OLS of value vs cycle over the trailing window, vectorized via grouped rolling sums; degenerate fits → 0.0; guarded division, no infinities.
5. **Early-cycle/missing-value policy** (ADR 0002): early rows preserved; structurally undefined values (first-cycle delta, single-observation std) left **null**; **no imputation or scaling in Loop 3** — deferred to the Loop 4 model pipeline (fit on train assets only). Manifest records `imputation: null` and per-file null counts.
6. **Outputs** (`features/pipeline.py`): six Parquet files + `split_manifest.json` + `feature_manifest.json` under `data/features/cmapss/FD001/`. Column order fixed: identifiers, `split`, targets (absent from test features), then features. Loop 2 inputs are checksum-verified before building; outputs are atomic, checksummed, idempotent (`already_built`), tamper-detected, `--force` rebuilds — mirroring Loops 1–2. Config changes (seed, cap, feature config) correctly trigger rebuilds.
7. **Manifests:** split manifest records seed/strategy/version, asset IDs+counts+row counts per partition, source-report checksum, git commit, timestamp. Feature manifest records feature/split/schema versions, full feature configuration, ordered feature columns, identifier/target/metadata column groups, RUL cap, imputation policy, inputs and outputs by SHA-256, row/asset counts by split, null counts, git commit, timestamp, seed — sufficient for a future training run to identify exactly what it used.
8. **Configuration** follows the established frozen-dataclass convention (`FeatureConfig`, `SplitConfig`, `RulConfig`, `BuildConfig`) — no YAML layer introduced (ADR 0002 §6); no scattered constants.
9. **Dependencies:** none added. pandas/NumPy/PyArrow/pydantic/stdlib only; scikit-learn, tsfresh-style libraries, feature stores all rejected as unnecessary.
10. **Tests:** 77 new (183 total). Labels, splits (determinism, coverage, isolation), feature families (each verified against manual/pandas reference), unsorted input, asset-boundary isolation, config effects, leakage (future-row mutation, future-row append incl. exhaustive per-cycle perturbation, cross-asset, fit isolation, replay exclusion), offline-vs-incremental equality at every cycle + restart reconstruction, persistence (round-trip, idempotency, tamper, force, input immutability), CLI (success, rerun, missing input, invalid config → exit 1). Three real-FD001 integration tests (auto-skip without local data).

---

## Loop 2 Implementation Notes

1. **Canonical schema** (`schema.py`, version 1): `asset_id`/`cycle` as int64; `operating_setting_1..3` and `sensor_01..21` as float64. Sensors stay anonymous — no physical interpretations assigned (dataset documentation provides none).
2. **Parser** (`parsing.py`): line-level whitespace splitting with exact field-count enforcement and line-numbered `ParseError`s; no silent row drops; trailing whitespace and blank lines handled; deterministic; preserves raw bytes. `NaN`/`inf` strings parse as numeric by design — rejecting non-finite values is the validation layer's job.
3. **Validation** (`validation.py`): structured pydantic `ValidationCheck`s and per-dataset stats instead of assertions. Required checks (block publication) are separated from warnings (constant/near-constant columns — reported, never deleted). General structural validation is separate from the optional FD001 canonical-count profile (`validate_canonical` flag; the CLI always applies it).
4. **Processing** (`processing.py`): raw layer re-verified against the manifest before parsing (refactored `verify_raw_layer` now public). Atomic writes (temp+rename), SHA-256 output checksums, idempotent re-runs (`already_processed`, nothing rewritten), tamper/missing-output detection with `--force` rebuild — mirroring the Loop 1 model. Trajectory outputs sorted by `(asset_id, cycle)`; source order never relied upon.
5. **Validated/processed layers collapsed into one step** for FD001: validation gates publication, so the Parquet under `data/processed/` *is* the validated output. Documented in `docs/data_contract.md`.
6. **Real FD001 results:** all 36 dataset checks + 1 cross-check pass. Train 20,631 rows / 100 engines (lives 128–362, median 199); test 13,096 rows / 100 engines (histories 31–303, median 133.5); RUL 100 values (7–145, median 86). Zero missing/duplicate/non-finite values. Constant: `operating_setting_3`, sensors 01/10/18/19; near-constant (relative std < 1e-4): sensors 05/06/08/13/16.
7. **Key EDA nuance:** sensors 08 and 13 are flagged near-constant only because their means are large — they show clear monotonic lifecycle trends (mean within-engine Spearman ≈ 0.65+). Loop 3 must select features by trend/information content, not raw variance. Strongest trends: sensors 11, 12, 04, 07; no-trend sensors match the constant/near-constant set.
8. **EDA notebook** (`notebooks/01_eda.ipynb`, committed executed): consumes only the Parquet outputs; covers counts, lengths, quality, constant columns, settings, lifecycle trends (Spearman-via-ranks, no SciPy), representative traces, distributions, correlation matrix, train-vs-test comparison (mechanical censoring, not drift), and a "Findings and implications for Loop 3" section. No RUL labels, RUL correlations, rolling features, splits, or models.
9. **Dependencies** (ADR 0001): runtime pandas + pyarrow; dev-only pandas-stubs, matplotlib, nbconvert, ipykernel. No validation framework (Great Expectations excluded by spec; typed Python checks instead). SciPy deliberately avoided. Note: uv resolved **pandas 3.0.3** — code targets the pandas 3 API.
10. **Tests:** 61 new (106 total). Parsing/validation/processing/CLI units are fully offline with 26-column fixtures; one integration test runs against the locally acquired FD001 data by copying it to a temp dir (repo `data/` never mutated) and auto-skips when the dataset is absent (`real_data` marker).

## Loop 1 Implementation Notes (retained)

1. **Source URL.** The historical `ti.arc.nasa.gov` download is gone and `data.nasa.gov` returns 404. The working source is NASA's S3 mirror `https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip` (12.4 MB), the configurable default (`TURBINE_GUARD_CMAPSS_SOURCE_URL` / `--url`). `file://` URLs are supported and used by tests.
2. **Nested archives.** Extraction searches nested zips (depth-limited); flat and nested layouts accepted.
3. **Immutability model.** Atomic writes, read-only (0444) raw files, checksum verification on re-run; mismatch → clear error, `--force` to replace.
4. **No pinned upstream checksums** — NASA publishes none; verify-what-you-acquired model with the manifest as source of truth.
5. **Manifest fields:** dataset/subset, source name+URL, UTC timestamp, versions, git commit, archive checksum, per-file SHA-256/size/record/asset counts.
6. **Acquired FD001 matches canonical characteristics** (verified again in Loop 2).
7. **Stdlib only in Loop 1**; `acquire()` is a plain callable (Prefect wrapper deferred).
8. **`data/` is gitignored** (root-anchored `/data/`).

## Loop 0 Implementation Notes (retained)

1. Repo initialized (`git init -b main`); Loop 0 committed as `d95b528`, Loop 1 as `94ae615`.
2. `uv` via Homebrew; Python 3.12.13 managed by uv.
3. Structured logging uses the standard library only (JSON formatter).
4. Pre-commit hooks are local `uv run` commands, versions synced with `uv.lock`.
5. `/health/ready` returns 200 with an empty check map; 503 path implemented and tested.
6. Package name `turbine-guard`; repo directory name differs (`predictive_maintenance`).

---

## Current Architecture Decisions

### Accepted

* Use NASA C-MAPSS FD001 for the first complete version.
* Split data by asset rather than individual rows.
* Keep replay assets out of initial model training.
* Use one shared feature pipeline for training and serving.
* Use tabular models before considering deep sequence models.
* Use MLflow for experiment tracking and model registration.
* Use Prefect for workflows.
* Use FastAPI for inference and operational endpoints.
* Use PostgreSQL for operational state.
* Use Docker Compose for local orchestration.
* Use a lightweight server-rendered dashboard.
* Use Render as the initial planned public deployment target.
* Avoid unnecessary distributed-system components.
* Loop 0: stdlib JSON logging; local `uv run` pre-commit hooks; application factory pattern.
* Loop 1: configurable source URL with `file://` support; verify-on-rerun checksum model; immutable read-only raw layer; manifests and datasets excluded from git.
* Loop 2 (ADR 0001): pandas + pyarrow runtime; typed hand-rolled validation (no Great Expectations/pandera); matplotlib/nbconvert/ipykernel dev-only; SciPy deferred; validated and processed layers collapsed into one gated step; required-vs-warning check separation; canonical FD001 profile separate from general validation.
* Loop 3 (ADR 0002): 70/15/5/10 asset-level split via seeded permutation + largest-remainder counts; preserve early-cycle rows with structural nulls, defer imputation/scaling to Loop 4; trailing asset-grouped windows with min_periods=1 and OLS rolling slope (degenerate → 0.0); optional off-by-default RUL cap alongside the always-present uncapped target; stateless FeatureBuilder with a history-replaying incremental state; frozen-dataclass configuration (no YAML layer).
* Loop 4 (ADR 0003): train-only Ridge imputation/indicators/scaling; native-null histogram gradient boosting and XGBoost; uncapped/capped-125 targets with common-domain ranking; collapsed first-alert episodes; row-level split conformal approximation; validation-only eligibility/ranking with simplicity tolerance; normalized simulated costs; checksummed joblib artifacts.
* Loop 8 (ADR 0007): focused `replay_runs` state table instead of overloading `pipeline_runs`; normalized `prediction_outcomes` keyed `(prediction_id, maintenance_event_id)` preserving immutable predictions; failure events written through the application service with deterministic external event IDs (public maintenance-event endpoint deferred); claim-lease concurrency with no lock across HTTP; phase-stamped recovery from the earliest incomplete phase; force restart = new attempt + fresh operational asset, never deletion; deterministic simulated timestamps; delayed evaluation reuses Loop 4 metrics grouped by stored model identity; httpx promoted from dev to runtime.

### Implemented so far

Loops 0–9 are complete and validated. Containerization, CI/CD, deployment, and dashboard work
remain design-only.

---

## Known Risks

1. NASA C-MAPSS is simulated turbofan data rather than actual power-plant sensor data.
2. Sensor columns are anonymous; no physical names are assigned.
3. Online collection will be simulated through historical trajectory replay.
4. Public free-tier deployment limitations may affect persistent MLflow artifacts or background workers.
5. Maintenance-cost results will be simulated and must not be presented as real industrial savings.
6. The system could become overengineered if future tools are introduced without a clear need.
7. Time-series leakage is a major modeling risk; Loop 3 added explicit structural protections and tests (future-row mutation/append, cross-asset isolation, fit isolation, replay exclusion), which future loops must keep passing.
8. Retraining on very few newly labeled replay assets may not provide meaningful improvements;
   Loop 9 therefore blocks fitting/promotion below configurable asset, row, and holdout minima.
9. NASA hosting has moved before and may move again; the source URL is configuration, and `file://` acquisition provides a manual fallback.
10. Relative-variance heuristics mislead on FD001 (sensors 08/13). Loop 3 deliberately generates
    features for **all** columns; Loop 4 reports importance/coefficient concentration but does not
    redesign the stable feature contract without a separate ablation study.
11. The shortest test history is 31 cycles. Loop 4 preserves those rows and handles their
    structural nulls through train-only Ridge imputation or native-null tree models.
12. pandas 3.x is newer than most tutorials/snippets assume; future code must target the pandas 3 API.
13. Split conformal calibration uses temporally dependent rows within five calibration assets;
    replay coverage is empirical evidence, not a strict trajectory-level exchangeability guarantee.
14. Loop 9 freezes the champion conformal calibrator for retrained candidates because no new
    calibration role is available; poor promotion-holdout coverage blocks promotion.
15. The default local SQLite MLflow registry and serving-model cache are process-local operational
    constraints; multi-process refresh coordination is deferred beyond Loop 9.
16. Loop 7 stores accepted readings, not a durable rejected-request stream, so the data-quality
    report can only count rejected inputs when its producer supplies them explicitly.

---

## Immediate Next Action

Review and commit Loop 9. Do not begin Loop 10 without explicit approval.

---

## Loop 3 Exit Criteria — all satisfied

* [x] RUL labels correct: uncapped `rul = T_i − t`, final-cycle RUL 0, exact −1 steps, validated; optional capped target preserved alongside uncapped.
* [x] Official test set protected: no fabricated per-row labels; positional benchmark table documented as evaluation-only.
* [x] Splits deterministic (seed 42), asset-level, non-overlapping, complete coverage: 70/15/5/10 engines.
* [x] Replay and calibration assets isolated from all fitting-related partitions.
* [x] Feature generation uses only current and historical per-asset data (trailing windows, grouped by asset, sorted internally).
* [x] Offline and incremental feature outputs match at every cycle (tested on fixtures and a real engine; restart reconstruction covered).
* [x] Leakage tests pass: future-row mutation, future-row append, exhaustive per-cycle perturbation, cross-asset isolation, fit isolation, replay exclusion.
* [x] Model-ready Parquet outputs generated for train/validation/calibration/replay/test features/test labels.
* [x] Split and feature manifests generated with checksums, versions, configuration, and provenance.
* [x] Outputs reproducible and idempotent; tamper detection and `--force` rebuild work; raw and Loop 2 layers unchanged (checksum-verified).
* [x] Documentation matches implementation (`docs/features.md`, `docs/data_contract.md` boundary update, ADR 0002, README, Makefile).
* [x] No Loop 4 functionality implemented (no models, training, evaluation metrics, scaling, imputation, MLflow, conformal prediction).

---

## Loop 2 Exit Criteria — all satisfied

* [x] Data contract defined and documented (`docs/data_contract.md`, canonical schema v1).
* [x] Parser implemented in `src/`, reusable and independently tested (never notebook-only).
* [x] Structural + semantic validation with structured results; required failures block publication.
* [x] Invalid fixtures fail clearly (wrong column counts with line numbers, non-numeric values, duplicates, gaps, non-finite values, empty files, tampered outputs — all covered by tests).
* [x] Real FD001 data passes all checks; asset and cycle integrity verified (unique pairs, contiguous 1..n).
* [x] Validated Parquet outputs generated reproducibly (atomic, checksummed, idempotent, `--force` rebuild).
* [x] Machine-readable validation report generated (`processing_report.json`).
* [x] Single primary EDA notebook executes top to bottom on the processed Parquet; findings documented in the notebook and here.
* [x] Raw files unchanged (checksums verified before/after processing).
* [x] `STATUS.md`, `TASKS.md`, `README.md` updated; ADR 0001 added.
* [x] No Loop 3 functionality implemented (no labels, capped RUL, splits, rolling features, feature manifests, scaling, or models).

---

## Validation Status

All Loop 9 commands run on 2026-07-13 (macOS, PostgreSQL 17, Python 3.12.13):

| Check | Status | Detail |
| --- | --- | --- |
| `uv sync` | Pass | 168 packages resolved / 163 checked; no dependency added |
| Ruff format/lint | Pass | 152 files already formatted; all checks passed |
| Mypy (strict) | Pass | No issues in 88 source files |
| Pytest (full) | Pass | 398/398 passed in 4:13 with the real PostgreSQL test URL enabled |
| PostgreSQL integration | Pass | Migration/head, report/assignment/event persistence, phase recovery, idempotency, approval/rejection audit, plus all preexisting integrations |
| Local MLflow integration | Pass | Candidate tracking/registration idempotency, reload equality, candidate/challenger aliases, approved promotion, rejection preservation, and rollback |
| Migration | Pass | `20260713_0003 (head)` on the development database; guarded upgrade/current test passed |
| Live monitoring | Pass | Training reference v1 (14,407 training rows/70 assets); quality/drift/performance persisted; exact rerun reused run `2199a607-30e8-4c7b-b8a6-e3cb4a3945df` |
| Trigger safety | Pass | Live drift/performance trigger blocked on one asset/201 rows/no safe holdout; manual force also blocked |
| Registry integrity | Pass | Live v1 champion reload prediction difference 0.0; no live candidate version or alias movement on the blocked run |
| Serving refresh | Pass | Explicit refresh loaded champion v1; failed-replacement unit test preserved the old cached object |
| Loop 0–8 regression | Pass | Full suite, real FD001 integrations, replay, API, database, and MLflow behavior remain green |
| Loop 10 boundary | Pass | No Docker, CI service, dashboard, frontend, Kubernetes, auth, or distributed orchestration code |

All Loop 8 commands run on 2026-07-13 (macOS, PostgreSQL 17, Python 3.12.13):

| Check | Status | Detail |
| --- | --- | --- |
| `uv sync` | Pass | 168 resolved / 163 checked; httpx promoted to runtime |
| Ruff format/lint | Pass | 132 files formatted; all checks passed |
| Mypy (strict) | Pass | No issues in 75 source files (9 new replay/database modules) |
| Pytest (full) | Pass | 364 collected; 349 + 15 PostgreSQL-guarded, all passing with the test DB |
| Replay integration | Pass | Six real-PostgreSQL tests: HTTP lifecycle, uncertain-outcome recovery, partial-phase resume, force restart, outcome idempotency/conflict, real-FD001 champion replay |
| Pre-commit | Pass | Ruff format, Ruff lint, and Mypy hooks passed |
| Migration | Pass | `20260712_0001 → 20260713_0002` applied; `alembic current` at head; head test made dynamic and strengthened with the new tables |
| Live replay | Pass | Engine 9, 201 cycles via step + resume; one failure event at cycle 201; 201 labels (final = 0, −1 per cycle); per-asset + aggregate evaluations persisted |
| Idempotency | Pass | Completed-run `start` no-op; repeated `evaluate-aggregate` inserted nothing; exact retries returned HTTP 200 idempotent |
| Ground-truth isolation | Pass | No event/labels before the final cycle; final cycle stored only in `replay_runs`; earlier predictions unchanged after backfill |
| Loop 3–7 integrity | Pass | `already_built` / `already_trained`; champion `capped_125--ridge_alpha_1` unchanged |
| Loop 9 boundary | Pass | No drift calculations, retraining, promotion, Prefect, or Docker code |

All Loop 7 commands run on 2026-07-12/13 (macOS, PostgreSQL 17, Python 3.12.13):

| Check | Status | Detail |
| --- | --- | --- |
| `uv sync` | Pass | 168 packages resolved; 163 checked |
| Ruff format/lint | Pass | 113 files formatted; all checks passed |
| Mypy (strict) | Pass | No issues in 66 source files |
| Pytest | Pass | 273 passed incl. PostgreSQL and real registered champion |
| Pre-commit | Pass | Ruff format, Ruff lint, and Mypy hooks passed |
| Live readiness | Pass | Database, model, and feature contract all true |
| Live inference | Pass | 20 contiguous real FD001 cycles returned stored champion predictions |
| Persistence | Pass | Exactly 20 readings and 20 predictions, cycles 1–20 |
| Idempotency/conflict | Pass | Exact retry HTTP 200/all flags true; changed cycle HTTP 409 |
| Asset/query API | Pass | Detail, health/trend, recent-first predictions, model metadata |
| UTC contract | Pass | All external timestamps serialize with `Z` |
| Metrics | Pass | Prometheus exposition with bounded labels; monitoring summary available |
| Rollback | Pass | Forced model failure left no asset, reading, or prediction |
| Failure readiness | Pass | Bad PostgreSQL and missing champion each returned accurate 503 checks |
| Offline/online equality | Pass | Real HTTP prediction matched shared offline features/champion |
| Loop 3–6 integrity | Pass | `already_built`, `already_trained`, registry max difference 0.0 |
| Loop 8 boundary | Pass | No replay, delayed feedback, backfill, or maintenance ingestion code |

All Loop 6 commands run on 2026-07-12 (macOS, PostgreSQL 17, Python 3.12.13 via uv):

| Check | Status | Detail |
| --- | --- | --- |
| `uv sync` | Pass | 167 packages resolved; 162 checked |
| Ruff format check | Pass | 100 files already formatted |
| Ruff lint check | Pass | All checks passed |
| Mypy (strict) | Pass | No issues in 57 source files |
| Pytest | Pass | 253 passed, including six real PostgreSQL and real FD001/MLflow tests |
| Pre-commit (all files) | Pass | Ruff format, Ruff lint, and Mypy hooks passed |
| Empty migration | Pass | `20260712_0001 (head)` created all seven operational tables |
| Migration reversal | Pass | Downgrade to base and re-upgrade to head restored eight relations incl. Alembic |
| Repository integration | Pass | Assets, sensors, predictions, events, supporting tables, transactions |
| Idempotency | Pass | Identical sensor/prediction retries returned existing; conflicts rejected |
| Transaction safety | Pass | Batch savepoint and session commit/rollback behavior verified |
| Readiness | Pass | Real PostgreSQL success plus unavailable/timeout paths verified |
| Loop 3–4 integrity | Pass | `already_built` / `already_trained`; recorded checksums verified |
| MLflow integrity | Pass | Champion alias output matched local bundle; max difference 0.0 |
| Loop 7 boundary | Pass | No `/v1`, ingestion, prediction, maintenance, or recent-prediction routes |

All Loop 5 commands run on 2026-07-12 (macOS, Python 3.12.13 via uv):

| Check | Status | Detail |
| --- | --- | --- |
| `uv sync` | Pass | 165 packages resolved; 160 checked |
| Ruff format check | Pass | 86 files already formatted |
| Ruff lint check | Pass | All checks passed |
| Mypy (strict) | Pass | No issues in 48 source files |
| Pytest | Pass | 232 passed, including real FD001 MLflow integration |
| Pre-commit (all files) | Pass | Ruff format, Ruff lint, and Mypy hooks passed |
| Persistent tracked FD001 run | Pass | 1 parent; 14 candidate children; selected capped-125 Ridge |
| Registry | Pass | `TurbineGuard-FD001-RUL` version 1, status `READY` |
| Aliases | Pass | `candidate`, `challenger`, and `champion` all point to version 1 |
| Alias/version load | Pass | 552 features; rich four-column output; max difference 0.0 |
| Tracking idempotency | Pass | `already_logged`; same parent, 14 children, and version 1 |
| Loop 3/4 integrity | Pass | Before/after SHA-256 diff empty |
| MLflow UI | Pass | SQLite-backed UI launched; root and experiment search returned HTTP 200 |
| Loop 6 boundary | Pass | No database package, Alembic tree, PostgreSQL imports, compose, or Dockerfile |

All Loop 4 commands run on 2026-07-12 (macOS, Python 3.12.13 via uv):

| Check | Status | Detail |
| --- | --- | --- |
| `uv sync` | Pass | 114 packages resolved; 109 checked |
| Ruff format check | Pass | 76 files formatted (one indicated file formatted, focused check passed) |
| Ruff lint check | Pass | All checks passed |
| Mypy (strict) | Pass | No issues in 42 source files |
| Pytest | Pass | 219 passed, including real FD001 input-contract integration |
| Pre-commit (all files) | Pass | Ruff format, Ruff lint/fix, and mypy hooks passed |
| Real FD001 forced training | Pass | 14 candidates; champion `capped_125--ridge_alpha_1` |
| Idempotent rerun | Pass | `already_trained`; current inputs/configuration/artifact checksums verified |
| Replay evaluation | Pass | MAE 10.541; RMSE 13.955; critical recall 0.768; no missed failures |
| Official NASA benchmark | Pass | Final-row MAE 11.689; RMSE 14.407; NASA score 309.229 |
| Conformal evaluation | Pass | Replay coverage 0.898; average width 39.148; official coverage 0.890 |
| Simulated policy | Pass | Base predictive 42.83 vs reactive 120.0 normalized units |
| Loop 3 feature integrity | Pass | Before/after SHA-256 diff empty |

Prior Loop 3 validation (retained for history):

| Check                          | Status | Detail                                                        |
| ------------------------------ | ------ | ------------------------------------------------------------- |
| `uv sync`                      | Pass   | 106 packages resolved; no new dependencies in Loop 3           |
| Ruff format check              | Pass   | 54 files already formatted                                     |
| Ruff lint check                | Pass   | All checks passed                                              |
| Mypy (strict)                  | Pass   | No issues in 29 source files                                   |
| Pytest                         | Pass   | 183 passed (77 new; incl. 3 real-data feature integration tests) |
| Pre-commit (all files)         | Pass   | ruff format, ruff check, mypy                                  |
| Real FD001 feature build       | Pass   | `make features`: 6 Parquet outputs + 2 manifests, 552 features |
| Idempotent re-run demo         | Pass   | `already_built`; output mtimes unchanged; exit code 0          |
| Missing-input failure demo     | Pass   | Nonexistent data dir → exit code 1, no outputs                 |
| Parquet load verification      | Pass   | All 6 outputs load; shapes (14407/3160/909/2155, 556), (13096, 555), (100, 3) |
| Future-row leakage demo (real) | Pass   | Mutating sensor_04 after cycle 100 leaves features ≤ 100 identical |
| Offline vs incremental (real)  | Pass   | Engine 1, 192 cycles: equal at every cycle                     |
| Raw + processed integrity      | Pass   | All raw and Loop 2 Parquet checksums unchanged after the build |

Known non-blocking upstream warnings: FastAPI's compatibility import emits the existing
`StarletteDeprecationWarning`; joblib 1.5.3 emits a NumPy 2.5 shape-assignment deprecation while
reloading compressed artifacts. MLflow additionally warns about trusted CloudPickle loading, an
unresolved installed pip version in its generated environment, and its UI's deprecated Starlette
WSGI bridge. Tests, registered-model loading, prediction equality, and the UI all pass.

---

## Last Completed Loop

**Loop 9 — Monitoring, Retraining, Candidate Evaluation, and Model Promotion** (2026-07-13).

---

## Next Planned Loop

After Loop 9 is reviewed and separately approved: **Loop 10 — Containers and CI/CD**.
Do not begin it automatically.
