# TASKS.md

## Task Management Rules

* Work on only one implementation loop at a time.
* The active loop is the only loop authorized for implementation.
* Do not begin the next loop automatically.
* Mark tasks complete only after implementation and validation.
* Update this file at the end of every loop.
* Add newly discovered work under the appropriate loop.
* Record unresolved issues instead of hiding or bypassing them.

---

# Active Loop

Loop 11 — Dashboard and Public Deployment. Implementation and local validation are complete,
including the zero-cost architecture (ADR 0011: free Render web service + Neon free PostgreSQL +
immutable checksum-pinned deployment bundle, replacing the ~$20.80/month paid Blueprint). The
remaining steps require only the repository owner's free accounts: Neon database, published bundle,
free Render service, and public HTTPS verification. Do not begin Loop 12.

---

# Completed Loops

## Loop 4 — Model Training, Evaluation, Uncertainty, and Policy Simulation

**Status:** Complete (2026-07-12) — validated locally; awaiting review before Loop 5

* [x] Verify Loop 3 checksums, schemas, ordered features, split roles, and asset isolation before fit.
* [x] Fit all learned preprocessing on training rows only; preserve structural early-cycle rows.
* [x] Implement the constant training-median baseline.
* [x] Implement Ridge with median imputation, missing indicators, and scaling.
* [x] Implement histogram gradient boosting with native missing-value support.
* [x] Implement deterministic CPU XGBoost candidates with native missing-value support.
* [x] Use the identical asset splits for every target/model candidate.
* [x] Compare uncapped RUL with an explicit 125-cycle capped target on compatible truth.
* [x] Implement MAE, RMSE, R-squared, NASA asymmetric score, asset aggregation, and slices.
* [x] Implement configurable row alerts, collapsed episodes, false alarms, missed failures, and lead time.
* [x] Implement finite-sample split conformal intervals using calibration rows only.
* [x] Implement configurable reactive/predictive normalized-cost simulation and sensitivity cases.
* [x] Implement validation-only operational eligibility, metric ranking, and simpler-model tie preference.
* [x] Serialize every candidate and a reloadable champion bundle with checksummed metadata.
* [x] Generate JSON, CSV, and Markdown evaluation/selection/interpretation reports.
* [x] Add the deterministic training CLI, idempotency, tamper detection, and `--force` rebuild.
* [x] Add deterministic unit tests and a lightweight real-FD001 contract integration test.
* [x] Add ADR 0003 and `docs/modeling.md`; update README/status/task documentation.
* [x] Run all requested quality gates and the real FD001 training/inspection commands. (219 tests;
  all quality gates pass; 14 real candidates trained; idempotent rerun and input integrity verified.)
* [x] Validate Loop 4 acceptance criteria. (Capped-125 Ridge alpha 1 selected on validation;
  replay/official/conformal/simulation reports verified; no Loop 5 functionality.)

---

## Loop 0 — Repository Foundation

**Status:** Complete (2026-07-12) — reviewed and committed as `d95b528`
**Priority:** Active
**Objective:** Establish a clean, typed, tested Python and FastAPI foundation without implementing data or ML functionality.

### Preparation

* [x] Read `CLAUDE.md`.
* [x] Read `PROJECT_SPEC.md`.
* [x] Read `STATUS.md`.
* [x] Inspect the current repository.
* [x] Confirm that no existing implementation will be overwritten. (Repository contained only planning documents.)
* [x] Produce a concrete Loop 0 implementation checklist.

### Python project

* [x] Configure Python 3.12. (`.python-version` pins 3.12; uv provisioned 3.12.13.)
* [x] Initialize dependency management with `uv`. (uv 0.11.24 installed via Homebrew.)
* [x] Create `pyproject.toml`.
* [x] Create and commit `uv.lock`. (Created and staged; initial git commit left for review — see unresolved issues.)
* [x] Create the `src/turbine_guard/` package.
* [x] Create a minimal `tests/` structure. (`tests/conftest.py`, `tests/unit/`.)
* [x] Add package metadata.
* [x] Add appropriate development dependency groups. (PEP 735 `dev` group: ruff, mypy, pytest, httpx, pre-commit.)

### Quality tooling

* [x] Configure Ruff formatting. (line-length 100, target py312.)
* [x] Configure Ruff linting. (A, ARG, B, C4, E, F, I, N, PT, PTH, RUF, SIM, T20, UP, W.)
* [x] Configure Mypy. (`strict = true` with the pydantic plugin.)
* [x] Configure Pytest. (`--strict-markers --strict-config`.)
* [x] Add pre-commit configuration. (Local `uv run` hooks so versions match `uv.lock`; installed and passing on all files.)
* [x] Add useful test markers only when needed. (None needed yet.)
* [x] Avoid suppressing type or lint errors without justification. (Zero suppressions in the codebase.)

### Configuration

* [x] Implement typed application settings. (`Settings(BaseSettings)` in `src/turbine_guard/config/settings.py`.)
* [x] Load settings from environment variables. (`TURBINE_GUARD_` prefix; optional `.env` file.)
* [x] Define an application environment setting. (`development` / `testing` / `production` enum.)
* [x] Define a log-level setting. (Validated literal, case-insensitive.)
* [x] Add `.env.example`.
* [x] Ensure real `.env` files are ignored. (`.env` and `.env.*` ignored; `.env.example` kept.)
* [x] Add configuration tests. (Defaults, env overrides, casing, invalid values, `.env` loading, caching.)

### Logging

* [x] Implement a structured logging foundation. (Stdlib JSON formatter; UTC timestamps; `extra=` fields merged.)
* [x] Make log level configurable. (From settings.)
* [x] Avoid configuring logging repeatedly during imports. (Configured only inside `create_app()`; idempotent handler replacement.)
* [x] Add tests for important logging configuration behavior where practical. (Formatter fields, extras, exceptions, idempotency, stdout JSON output.)

### FastAPI foundation

* [x] Create an application factory. (`create_app()` in `src/turbine_guard/api/app.py`; no module-level app.)
* [x] Add `GET /health/live`.
* [x] Add `GET /health/ready`. (Returns 200 with empty checks; 503 `not_ready` path implemented and tested.)
* [x] Define typed response models. (`LivenessResponse`, `ReadinessResponse`.)
* [x] Keep health logic separate from route definitions where reasonable. (`services/health.py` vs `api/routes/health.py`.)
* [x] Add API tests. (Liveness, readiness, 503 path, OpenAPI schema, docs page, settings injection.)
* [x] Confirm OpenAPI documentation is generated. (`/openapi.json` lists both health paths; `/docs` serves HTML.)
* [x] Avoid adding database or model dependencies. (Runtime deps: fastapi, pydantic, pydantic-settings, uvicorn only.)

### Repository support files

* [x] Create `.gitignore`.
* [x] Create `Makefile`.
* [x] Create initial `README.md`.
* [x] Document installation.
* [x] Document development commands.
* [x] Document how to run the API.
* [x] Document how to run tests and quality checks.
* [x] Add any minimal editor or tooling configuration that is clearly justified. (None justified yet; nothing added.)

### Validation

* [x] Run `uv sync`. (Pass — 45 packages resolved, editable install of `turbine-guard 0.1.0`.)
* [x] Run `uv run ruff format --check .`. (Pass — 17 files already formatted.)
* [x] Run `uv run ruff check .`. (Pass — all checks passed.)
* [x] Run `uv run mypy src`. (Pass — no issues in 12 source files, strict mode.)
* [x] Run `uv run pytest`. (Pass — 24 passed; one upstream deprecation warning, see unresolved issues.)
* [x] Start the API locally. (uvicorn with `--factory` on port 8123.)
* [x] Verify `/health/live`. (`HTTP 200 {"status":"alive"}` via curl.)
* [x] Verify `/health/ready`. (`HTTP 200 {"status":"ready","checks":{}}` via curl.)
* [x] Verify `/docs`. (`HTTP 200`, `text/html`.)
* [x] Fix all failures caused by Loop 0. (No failures remained.)

### Documentation and completion

* [x] Update `STATUS.md` with the actual implementation state.
* [x] Mark completed Loop 0 tasks in this file.
* [x] Record unresolved issues. (See "Unresolved Issues" at the end of this file.)
* [x] Record validation command results. (In `STATUS.md` and above.)
* [x] Confirm no Loop 1 functionality was implemented. (No data, ML, DB, MLflow, Prefect, Docker, or CI code exists.)
* [x] Stop and request review before beginning Loop 1.

### Out of scope for Loop 0

Do not implement:

* Dataset downloading
* NASA C-MAPSS parsing
* Data manifests
* Parquet processing
* EDA
* Feature engineering
* Model training
* MLflow
* Prefect
* PostgreSQL
* SQLAlchemy models
* Alembic migrations
* Sensor replay
* Drift monitoring
* Retraining
* Docker Compose
* Public deployment
* Dashboard functionality

---

## Loop 1 — Dataset Acquisition and Manifesting

**Status:** Complete (2026-07-12) — awaiting review before Loop 2

* [x] Implement NASA C-MAPSS acquisition. (`src/turbine_guard/data/acquisition.py`, CLI in `data/cli.py`, thin `scripts/download_data.py`; stdlib only, zero new dependencies.)
* [x] Start with FD001. (Subset fixed to FD001; member names derived from the subset string.)
* [x] Store raw source files immutably. (Atomic temp+rename writes; files marked read-only 0444 under `data/raw/cmapss/FD001/`; mismatches error instead of silently repairing; `--force` for deliberate replacement.)
* [x] Calculate file checksums. (SHA-256 for the archive and every extracted file; verified on each re-run.)
* [x] Generate acquisition manifests. (`data/manifests/cmapss_fd001.json`, written atomically; pydantic-typed model in `data/manifest.py`.)
* [x] Record source and retrieval metadata. (Dataset name, subset, source name + URL, UTC retrieval timestamp, acquisition version, tool version, git commit, per-file sizes and record/asset counts.)
* [x] Make acquisition idempotent. (Re-run verifies checksums → `already_acquired`; cached archive reused; demonstrated live and covered by tests.)
* [x] Create small test fixtures. (In-memory fixture zips — flat and nested — served over `file://`; tests are fully offline.)
* [x] Add acquisition flow. (Plain callable `acquire(config)`; the Prefect `acquire_dataset_flow` wrapper is deferred to the orchestration loop per the loop plan.)
* [x] Document dataset provenance. (README "Dataset acquisition" section + manifest `notes`: simulated turbofan data, anonymous sensor channels, no physical interpretations assigned.)
* [x] Validate Loop 1 acceptance criteria. (All validation commands pass; real acquisition + idempotent re-run demonstrated; see `STATUS.md`.)

---

## Loop 2 — Validation and EDA

**Status:** Complete (2026-07-12) — awaiting review before Loop 3

* [x] Define the raw data contract. (`docs/data_contract.md`; canonical schema v1 in `data/schema.py`.)
* [x] Parse C-MAPSS files. (`data/parsing.py`: line-level field-count enforcement, line-numbered `ParseError`s, no silent row drops, deterministic, raw bytes untouched.)
* [x] Assign explicit column names. (`asset_id`, `cycle`, `operating_setting_1..3`, `sensor_01..21` — anonymous sensors, no invented physical meanings.)
* [x] Validate types and schema. (int64 IDs/cycles, float64 settings/sensors; column set, order, and dtypes checked post-parse.)
* [x] Validate asset and cycle integrity. (Positive IDs/cycles, unique `(asset_id, cycle)`, cycles contiguous 1..n per asset, order-independent.)
* [x] Detect duplicates and invalid records. (Duplicates, missing, non-finite, malformed rows; constant/near-constant columns reported as warnings, kept.)
* [x] Produce validated data. (Validation gates publication; failed required check → no output written. FD001 canonical-count profile separate from general validation.)
* [x] Produce processed Parquet data. (`data/processed/cmapss/FD001/{train,test,rul}_FD001.parquet` + `processing_report.json`; atomic, checksummed, idempotent, tamper-detected, `--force` rebuild; `make process`.)
* [x] Create the single primary EDA notebook. (`notebooks/01_eda.ipynb`, executes top to bottom on the Parquet outputs via `make eda`; no labels/features/splits/models.)
* [x] Document EDA findings. (Notebook "Findings and implications for Loop 3" section + `STATUS.md`; key nuance: sensors 08/13 near-constant by relative std yet strongly trending.)
* [x] Validate Loop 2 acceptance criteria. (All validation commands pass; 106 tests; real FD001 processing, idempotent re-run, Parquet load, and notebook execution demonstrated; raw checksums unchanged.)

---

## Loop 3 — Labels, Splits, and Features

**Status:** Complete (2026-07-12) — awaiting review before Loop 4

* [x] Generate RUL labels. (`features/labels.py`: uncapped `rul = T_i − t` always; validation enforces non-negativity, final-cycle 0, exact −1 steps. Official test set: no fabricated per-row labels; 100-row `test_labels.parquet` benchmark with the positional correspondence asserted explicitly.)
* [x] Compare raw and capped RUL targets. (Both *generatable*: optional `rul_capped = min(rul, cap)` via config/`--rul-cap`, uncapped always preserved; the actual model comparison is a Loop 4 experiment per the spec.)
* [x] Implement asset-level train, validation, calibration, and replay splits. (`features/splits.py`: seeded `default_rng` permutation + largest-remainder counts → exactly 70/15/5/10 engines on FD001; deterministic, disjoint, complete; manifest with asset IDs, counts, row counts, seed, strategy, checksums.)
* [x] Protect official test data. (Untouched benchmark; never used for training/selection; test features carry no target columns; Loop 2 inputs checksum-verified before every build.)
* [x] Implement the shared `FeatureBuilder`. (`features/builder.py`: single stateless implementation for offline batch and single-asset incremental generation; no notebook/serving forks; `IncrementalFeatureState` replays history through the same batch code.)
* [x] Add rolling-window features. (552 features: current, delta_1, rolling mean/std/min/max/range/slope over trailing windows 5/10/20, EWM mean spans 5/10/20; grouped per asset; configurable families/windows/min_periods; deterministic names and stable order; early-cycle structural nulls preserved, imputation deferred to Loop 4 — ADR 0002.)
* [x] Add feature manifests. (`split_manifest.json` + `feature_manifest.json`: versions, configuration, ordered feature columns, column groups, input/output SHA-256s, row/asset/null counts, git commit, timestamp, seed; idempotent rebuilds with tamper detection and `--force`.)
* [x] Test offline and online feature consistency. (Equality at every cycle on fixtures and a real engine; restart reconstruction via `from_history`; out-of-order cycles rejected.)
* [x] Add explicit future-data leakage tests. (Future-row mutation, future-row append, exhaustive per-cycle perturbation, cross-asset isolation, fit isolation, replay exclusion — unit + real-data variants.)
* [x] Validate Loop 3 acceptance criteria. (All validation commands pass: 183 tests; real `make features` build + idempotent rerun demonstrated; raw and processed layers unchanged; see `STATUS.md`.)

---

# Planned Loops

## Loop 4 — Baseline Modeling and Evaluation

**Status:** Complete above

* [x] Implement constant baseline.
* [x] Implement Ridge regression.
* [x] Implement a tree-based baseline.
* [x] Implement XGBoost.
* [x] Use identical splits for all models.
* [x] Add MAE and RMSE.
* [x] Add NASA asymmetric score.
* [x] Add failure-horizon metrics.
* [x] Add false-alarm and lead-time metrics.
* [x] Add lifecycle slice evaluation.
* [x] Add conformal prediction intervals.
* [x] Add configurable maintenance-policy simulation.
* [x] Generate an evaluation report.
* [x] Select the champion using explicit criteria.
* [x] Validate Loop 4 acceptance criteria.

---

## Loop 5 — MLflow Integration

**Status:** Complete and validated (2026-07-12)

* [x] Configure typed local/remote MLflow tracking and artifact settings.
* [x] Add a SQLite-backed local registry default and gitignore runtime state.
* [x] Keep core Loop 4 modeling usable with tracking disabled.
* [x] Create one parent run and one nested child per candidate.
* [x] Log candidate parameters, validation metrics, eligibility, and rank.
* [x] Log raw acquisition, validation, split, feature, training-config, Git, and code lineage.
* [x] Log candidate pipelines/configs and complete champion reports/contracts/checksums.
* [x] Log champion calibration, replay, official-test, policy, and performance metrics.
* [x] Package the existing champion bundle as a rich MLflow pyfunc.
* [x] Add an explicit ordered-feature signature and valid input example.
* [x] Test reload equality, ordering, missing columns, extra columns, and preprocessing preservation.
* [x] Register the selected champion with source-run and evaluation metadata.
* [x] Implement `candidate`, `challenger`, `champion`, and `archived` aliases.
* [x] Preserve previous champion versions and test alias reassignment.
* [x] Implement exact-execution and bundle-checksum idempotency plus explicit force modes.
* [x] Reject tampered local artifacts before logging/registration.
* [x] Generate and log a registry-aware champion model card.
* [x] Add tracked-training, UI, inspection, state, and equivalence commands.
* [x] Add temporary-SQLite unit/integration coverage with no external server or UI.
* [x] Add ADR 0004 and MLflow documentation; update README/configuration.
* [x] Run the complete validation suite and real tracked FD001 workflow from the user terminal.
  (232 tests; all quality gates pass; 14 candidates; registry version 1; aliases verified.)
* [x] Validate Loop 5 acceptance criteria and close the loop based on actual command output.

---

## Loop 6 — PostgreSQL Operational Layer

**Status:** Complete and validated (2026-07-12) — awaiting review before Loop 7

* [x] Configure typed PostgreSQL/psycopg settings, pool, connect, and statement timeouts.
* [x] Implement SQLAlchemy 2 typed models for all seven operational tables.
* [x] Configure Alembic from typed settings without stored credentials.
* [x] Implement deterministic initial revision `20260712_0001` and downgrade.
* [x] Add asset, sensor-reading, prediction, and maintenance-event repositories.
* [x] Add evaluation, drift-report storage-only, and pipeline-run repositories.
* [x] Enforce `(asset_id, cycle)` and model/reading prediction uniqueness.
* [x] Add restrictive foreign keys, checks, JSONB fields, and common-query indexes.
* [x] Implement caller-owned transactions and rollback-safe sensor batches.
* [x] Implement exact-retry idempotency and typed conflict behavior.
* [x] Add injectable database readiness while preserving dependency-free defaults.
* [x] Add guarded PostgreSQL integration tests and fast unit tests.
* [x] Add ADR 0005, database documentation/schema diagram, settings, and lifecycle commands.
* [x] Validate Loop 6 acceptance criteria. (PostgreSQL 17 empty upgrade/current/downgrade/re-upgrade;
  six guarded integration tests; 253 total tests; Ruff/Mypy/pre-commit pass; Loop 3–5 integrity and
  MLflow equivalence verified; no Loop 7 routes.)

---

## Loop 7 — FastAPI Inference Service

**Status:** Complete and validated (2026-07-13) — awaiting review before Loop 8

* [x] Add thin versioned `/v1` routes.
* [x] Implement atomic sensor ingestion and champion prediction.
* [x] Defer direct prediction endpoint and document why it adds no distinct capability yet.
* [x] Implement asset list, detail, and health endpoints.
* [x] Implement current-model and operational monitoring-summary endpoints.
* [x] Load/cache the MLflow champion by alias with explicit refresh.
* [x] Use the shared Loop 3 feature configuration and builder.
* [x] Store readings and version-pinned predictions atomically.
* [x] Return RUL, interval, risk, horizons, model/run/feature identity, and latency.
* [x] Add structured safe errors, request IDs, and Prometheus metrics.
* [x] Require database, model, and feature compatibility in readiness.
* [x] Add strict contiguous-cycle, idempotency, conflict, gap, and rollback policies.
* [x] Add unit/API and guarded PostgreSQL/real-champion integration tests.
* [x] Add ADR 0006 and online inference documentation.
* [x] Validate Loop 7 acceptance criteria. (273 tests; Ruff/Mypy/pre-commit pass; real PostgreSQL
  and champion equality; 20-cycle live ingestion; persistence/idempotency/conflict/UTC responses;
  readiness success and individual dependency failures; Loop 3–6 integrity; no Loop 8 code.)

---

## Loop 8 — Replay and Delayed Feedback

**Status:** Complete and validated (2026-07-13) — awaiting review before Loop 9

* [x] Build held-out trajectory replay. (Checksum-verified Loop 3 replay split only; wrong-split,
  missing, tampered, non-contiguous, and non-finite sources rejected before any send.)
* [x] Add configurable replay speed. (Step, continuous with configurable delay, accelerated;
  `--max-cycles`; pause/stop/resume; typed `TURBINE_GUARD_REPLAY_*` settings.)
* [x] Reveal only current and historical cycles. (One `SensorReadingRequest` per cycle through the
  real `POST /v1/sensor-readings` contract; final cycle stored only in `replay_runs`, which no
  prediction endpoint reads.)
* [x] Emit maintenance or failure outcomes. (Idempotent `failure` event via the existing
  repository with external event ID `replay-run:<run_id>:failure`, only after the final reading is
  verified persisted; API endpoint deliberately deferred — ADR 0007.)
* [x] Backfill labels after outcomes become available. (`prediction_outcomes` table, Alembic
  `20260713_0002`; `realized_rul = T − t`; validated invariants; idempotent; conflict detection;
  predictions immutable.)
* [x] Join historical predictions to realized outcomes. (Delayed evaluation reusing Loop 4
  metrics/alerts/interval code, grouped by stored model version; per-asset and aggregate rows in
  `model_evaluations` with scope `replay`.)
* [x] Add a complete lifecycle integration test. (Guarded PostgreSQL + HTTP tests: full lifecycle,
  uncertain-outcome recovery, partial-phase resume, force restart, outcome idempotency/conflict,
  optional real-FD001 replay with the registered champion.)
* [x] Verify that inference cannot access future data. (Unit and integration tests: payload key
  allowlist, in-order sends, future-source-mutation immunity, no event/labels before the final
  cycle, early predictions unchanged after completion.)
* [x] Add durable replay state, lease-based concurrency, and crash recovery. (Claim lease with no
  lock across HTTP; resume from the earliest incomplete phase; repeated commands idempotent;
  documented force restart.)
* [x] Add replay CLI, structured logs, and low-cardinality Prometheus metrics.
* [x] Add ADR 0007 and `docs/replay.md`; update README, `.env.example`, `docs/database.md`, and
  `docs/online_inference.md`.
* [x] Validate Loop 8 acceptance criteria. (All quality gates and pre-commit pass; 364 tests
  including six real-PostgreSQL replay integration tests and a real-FD001 champion replay;
  migration `20260713_0002` applied and current; live 201-cycle replay of held-out engine 9 with
  exactly-once failure event, correct label backfill, per-asset and aggregate evaluations,
  idempotent restart and aggregate re-run; `already_built`/`already_trained` integrity; no Loop 9
  code.)

---

## Loop 9 — Monitoring and Retraining

**Status:** Complete and validated (2026-07-13) — awaiting review before Loop 10

* [x] Implement persisted data-quality monitoring. (Counts, duplicates, missing/non-finite and
  out-of-range values, cycle gaps/order, feature-history sufficiency, sensor availability.)
* [x] Implement champion/feature-version-bound training-only reference distributions.
* [x] Implement feature drift reports. (PSI, quantile-integral Wasserstein distance, missingness,
  normalized mean, and normalized standard-deviation shifts over all 552 features.)
* [x] Calculate delayed online model metrics by reusing Loop 4 regression, alert/lead-time, and
  conformal metrics over Loop 8 immutable outcomes.
* [x] Define configurable `no_action`, `monitor`, `retrain`, and `blocked` decisions, including
  manual force that cannot bypass data-quality, minimum-data, or holdout safety.
* [x] Implement leakage-safe retraining data assembly. (Original training role plus eligible
  completed operational assets; asset-level disjoint additions/holdout; protected roles closed.)
* [x] Retrain the champion family/target with the existing Loop 4 fit path; add no model family.
* [x] Compare candidate, non-refitted champion, and naive baseline on one fingerprinted holdout.
* [x] Implement every blocking promotion gate, MLflow reload equivalence, candidate/challenger
  aliases, approval-by-default promotion, rejection, and numbered-version rollback.
* [x] Make lifecycle runs auditable and resumable. (`pipeline_runs` phase/idempotency metadata,
  asset assignments, append-only events, checksum-verified artifacts, lifecycle-keyed MLflow.)
* [x] Reuse and harden Loop 7 model-cache refresh so failed replacement loads preserve the cache.
* [x] Add the monitoring/lifecycle management CLI and operator documentation/ADR 0008.
* [x] Test healthy/unhealthy quality, no/induced drift, degraded performance, all decisions,
  data isolation, too-few-assets blocking, identical holdout, every gate, registration, rejection,
  approved promotion, aliases, rollback, refresh failure, recovery, and idempotent reruns.
* [x] Validate Loop 9 acceptance criteria. (398 tests with real PostgreSQL and local MLflow;
  migration `20260713_0003`; live monitor/idempotency, blocked-safe force, champion equivalence,
  and safe refresh demonstrations; no dependency or Loop 10 functionality added.)

---

## Loop 10 — Containers and CI/CD

**Status:** Complete and validated (2026-07-13) — awaiting review before Loop 11

* [x] Add a production Dockerfile. (Python 3.12 slim, pinned uv, locked non-editable wheel,
  multi-stage build, numeric non-root user, runtime-only dependency group, no mutable state.)
* [x] Add Docker Compose. (Production-style network, explicit dependencies/profiles, no source
  bind mounts, four persistent named volumes.)
* [x] Containerize the API. (Typed Uvicorn entry point, migration/health dependencies, port 8000,
  real readiness and graceful shutdown.)
* [x] Containerize the workflow worker. (Profile-gated one-shot Loop 9 `monitor`; no scheduler,
  automatic retraining, or automatic promotion.)
* [x] Containerize the replay service. (Profile-gated safe `status --all` default; explicit
  start/step/resume/accelerated commands documented.)
* [x] Configure PostgreSQL. (PostgreSQL 17, example local credentials, `pg_isready`, operational
  database only, persistent volume.)
* [x] Configure MLflow. (HTTP tracking/registry/artifact proxy; persistent SQLite metadata and
  artifact volumes; narrow Compose/localhost host allowlist.)
* [x] Add service health checks. (PostgreSQL, MLflow, and API; API requires database, champion,
  and exact feature compatibility, not merely a live process.)
* [x] Add GitHub Actions. (Pull requests and pushes to main; pinned action majors and uv cache.)
* [x] Run formatting, linting, typing, and tests in CI. (Dedicated Python quality job.)
* [x] Test migrations in CI. (Empty PostgreSQL upgrade/current/check, complete downgrade/re-upgrade,
  and all guarded PostgreSQL integration tests.)
* [x] Build the Docker image in CI. (BuildKit/GHA cache plus non-root, imports, CLI, settings, and
  signal-shutdown contract checks.)
* [x] Run an API smoke test. (Offline deterministic champion fixture, migrations, real readiness,
  docs/metrics/representative ingestion, worker/replay CLIs, dependency restarts, persistence.)
* [x] Validate Loop 10 acceptance criteria. (402 tests across generic and real-PostgreSQL runs;
  all quality gates, final image build, image contract, Compose rendering, and isolated stack smoke
  pass; no Loop 11 code.)

---

## Loop 11 — Dashboard and Public Deployment

**Status:** Active — implementation/local validation complete; public Render verification pending

* [x] Build fleet overview.
* [x] Build asset-health view.
* [x] Display RUL and prediction intervals.
* [x] Display warnings and critical alerts.
* [x] Display bounded/filterable prediction history.
* [x] Display champion version, lineage, aliases, and lifecycle status.
* [x] Display bounded drift status and top-feature detail.
* [x] Display recent online performance with explicit capped/uncapped labeling.
* [x] Add constrained replay controls using the Loop 8 service behavior.
* [x] Add read-only dashboard API projections, pagination/limits, and deterministic ordering.
* [x] Add responsive templates, minimal CSS/JavaScript, Plotly charts, and degraded/empty states.
* [x] Prepare and validate the Render Blueprint. (Replaced 2026-07-14 by the zero-cost ADR 0011
  Blueprint: one free web service, no disks, no Render database, no MLflow service.)
* [x] Configure secrets outside Git and document rotation. (Neon URL and bundle URL/SHA-256 are
  dashboard-entered `sync: false` values; Render generates the application secret.)
* [x] Configure and locally verify PostgreSQL, MLflow, artifact, generated-data, replay, and
  monitoring persistence across restart. (Local Compose reference topology, 2026-07-13.)
* [x] Implement the deployment-bundle export/restore tooling and manifest (ADR 0011): export from
  the verified live champion, registry-identity snapshot, per-file SHA-256, pinned archive.
* [x] Implement bundle-mode serving (`TURBINE_GUARD_MODEL_SOURCE=deployment_bundle`) behind the
  shared `ChampionLoader` protocol, with no MLflow import in the demo process.
* [x] Implement the cold-start demo entry point: pinned restore → `alembic upgrade head` → serve;
  never trains; idempotent and self-healing.
* [x] Verify bundle serving equals the MLflow pyfunc exactly (max prediction difference 0.0) and
  the app boots ready in bundle mode against real PostgreSQL.
* [x] Document the zero-cost architecture, bundle workflow, cold start, persistence matrix,
  free-tier tradeoffs, deployment, and troubleshooting; add ADR 0011 and amend ADR 0010.
* [x] Add dashboard, data-correctness, replay-safety, security, Blueprint, bundle, and
  real-PostgreSQL tests. (438 total tests pass.)
* [x] Validate the recruiter-facing demo locally in the production image.
* [ ] Create the free Neon database and enter its connection string in Render (owner).
* [ ] Publish the exported bundle at a revision-pinned URL and enter URL + SHA-256 (owner).
* [ ] Create the free Render Blueprint and verify the cold-start restore/migration logs (owner).
* [ ] Verify the assigned public dashboard, OpenAPI, liveness, and readiness HTTPS URLs.
* [ ] Validate the bounded public replay demo on Render.
* [ ] Validate Loop 11 acceptance criteria.

---

## Loop 12 — Portfolio Finishing

**Status:** Not started

* [ ] Finalize the README.
* [ ] Add an architecture diagram.
* [ ] Add a system sequence diagram.
* [ ] Add final model metrics.
* [ ] Add maintenance-policy results.
* [ ] Add demo GIF or video.
* [ ] Finalize the model card.
* [ ] Document limitations.
* [ ] Document scaling paths.
* [ ] Document local reproduction.
* [ ] Verify that all reported results are reproducible.
* [ ] Review the repository for confidential information.
* [ ] Prepare resume bullets.
* [ ] Prepare interview explanations.
* [ ] Validate the final definition of done.

---

# Backlog

The following are optional and must not be implemented until the complete core project works:

* [ ] Compare FD001 against another C-MAPSS subset.
* [ ] Experiment with a sequence model.
* [ ] Add sensor-ablation studies.
* [ ] Add model explainability reports.
* [ ] Add load testing.
* [ ] Add shadow deployment behavior.
* [ ] Add canary-style model evaluation.
* [ ] Add cloud object storage.
* [ ] Add a more realistic synthetic drift generator.
* [ ] Evaluate whether TimescaleDB provides meaningful value.
* [ ] Document a hypothetical Kafka and Kubernetes scaling architecture without implementing it.

---

# Unresolved Issues

Updated at Loop 4 completion (2026-07-12). None block Loop 4 acceptance.

1. ~~No initial git commit exists yet.~~ **Resolved:** Loop 0 committed as `d95b528`, Loop 1 as `94ae615`, Loop 2 as `aa27dfc`. Loop 3 changes are intentionally uncommitted pending review (same policy).
2. **Upstream deprecation warning in the test suite.** Importing `fastapi.testclient` with starlette 1.3.1 emits `StarletteDeprecationWarning: Using httpx with starlette.testclient is deprecated; install httpx2 instead.` The warning originates in FastAPI's own compatibility import, not project code; FastAPI's TestClient currently requires `httpx`. Revisit when FastAPI completes its migration.
3. **NASA hosting stability.** The legacy `ti.arc.nasa.gov` C-MAPSS URL is dead and `data.nasa.gov` returns 404; acquisition defaults to NASA's S3 mirror (`phm-datasets.s3.amazonaws.com`). If that moves too, set `TURBINE_GUARD_CMAPSS_SOURCE_URL` (or `--url`, including `file://` for a manually downloaded archive). No code change should be needed.
4. **pandas 3.x.** `uv` resolved pandas 3.0.3 (not 2.x). Loops 2–3 code targets the pandas 3 API (e.g., `method="spearman"` now requires SciPy, avoided via rank-then-Pearson). Future loops must not assume pandas 2 behavior.
5. ~~Positional RUL correspondence must be encoded explicitly in Loop 3.~~ **Resolved:** `build_test_benchmark_labels` asserts the row *i* ↔ test unit *i + 1* correspondence (count equality + contiguous 1..N asset IDs) and `docs/features.md` documents that it cannot be verified from file contents alone.
6. ~~**Early-cycle structural nulls.**~~ **Resolved in Loop 4:** Ridge fits median imputation,
   missing indicators, and scaling on training rows only; histogram gradient boosting and XGBoost
   use native missing-value support. Tests prove held-out values cannot alter fitted preprocessing.
7. **Incremental state holds full history.** `IncrementalFeatureState` retains each asset's complete observation history (cheap at C-MAPSS trajectory lengths ≤ 362, and keeps EWM exactly equal to batch). If a future dataset has much longer histories, a bounded-buffer variant with a running EWM accumulator would be needed (documented in ADR 0002).
8. **Upstream joblib/NumPy deprecation warning.** Joblib 1.5.3 assigns NumPy array shapes while
   reloading compressed artifacts, which NumPy 2.5 warns is deprecated. Serialization and reload
   equality tests pass; revisit when joblib updates the implementation.
