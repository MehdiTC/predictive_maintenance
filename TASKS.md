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

None. Loops 0–4 are complete. Loop 5 must not begin without explicit approval.

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

**Status:** Not started

* [ ] Configure MLflow tracking.
* [ ] Log training parameters and metrics.
* [ ] Log dataset checksums.
* [ ] Log feature versions.
* [ ] Log split asset IDs.
* [ ] Log Git commit SHA.
* [ ] Log model artifacts and evaluation plots.
* [ ] Register candidate models.
* [ ] Implement model aliases.
* [ ] Generate model cards.
* [ ] Load a model by registry alias.
* [ ] Validate Loop 5 acceptance criteria.

---

## Loop 6 — PostgreSQL Operational Layer

**Status:** Not started

* [ ] Configure PostgreSQL integration.
* [ ] Implement SQLAlchemy models.
* [ ] Configure Alembic.
* [ ] Implement migrations.
* [ ] Add asset storage.
* [ ] Add sensor-reading storage.
* [ ] Add prediction storage.
* [ ] Add maintenance-event storage.
* [ ] Add monitoring and pipeline-run storage as needed.
* [ ] Enforce `(asset_id, cycle)` uniqueness.
* [ ] Add indexes and foreign keys.
* [ ] Add repository abstractions.
* [ ] Add integration tests.
* [ ] Validate Loop 6 acceptance criteria.

---

## Loop 7 — FastAPI Inference Service

**Status:** Not started

* [ ] Add versioned API routing.
* [ ] Implement sensor-ingestion endpoint.
* [ ] Implement direct prediction endpoint.
* [ ] Implement asset endpoints.
* [ ] Implement current-model endpoint.
* [ ] Implement monitoring-summary endpoint.
* [ ] Load the MLflow champion.
* [ ] Use the shared feature pipeline.
* [ ] Store incoming observations.
* [ ] Store model predictions.
* [ ] Return model-version metadata.
* [ ] Add structured errors.
* [ ] Add request IDs.
* [ ] Add service metrics.
* [ ] Expand readiness checks.
* [ ] Add integration tests.
* [ ] Validate Loop 7 acceptance criteria.

---

## Loop 8 — Replay and Delayed Feedback

**Status:** Not started

* [ ] Build held-out trajectory replay.
* [ ] Add configurable replay speed.
* [ ] Reveal only current and historical cycles.
* [ ] Emit maintenance or failure outcomes.
* [ ] Backfill labels after outcomes become available.
* [ ] Join historical predictions to realized outcomes.
* [ ] Add a complete lifecycle integration test.
* [ ] Verify that inference cannot access future data.
* [ ] Validate Loop 8 acceptance criteria.

---

## Loop 9 — Monitoring and Retraining

**Status:** Not started

* [ ] Implement data-quality monitoring.
* [ ] Implement feature drift reports.
* [ ] Add PSI.
* [ ] Add Wasserstein distance.
* [ ] Monitor missingness and distribution changes.
* [ ] Calculate delayed online model metrics.
* [ ] Define retraining triggers.
* [ ] Implement the retraining workflow.
* [ ] Implement candidate evaluation gates.
* [ ] Implement champion promotion.
* [ ] Make promotion auditable.
* [ ] Test induced drift.
* [ ] Test candidate rejection.
* [ ] Test candidate promotion.
* [ ] Validate Loop 9 acceptance criteria.

---

## Loop 10 — Containers and CI/CD

**Status:** Not started

* [ ] Add a production Dockerfile.
* [ ] Add Docker Compose.
* [ ] Containerize the API.
* [ ] Containerize the workflow worker.
* [ ] Containerize the replay service.
* [ ] Configure PostgreSQL.
* [ ] Configure MLflow.
* [ ] Add service health checks.
* [ ] Add GitHub Actions.
* [ ] Run formatting, linting, typing, and tests in CI.
* [ ] Test migrations in CI.
* [ ] Build the Docker image in CI.
* [ ] Run an API smoke test.
* [ ] Validate Loop 10 acceptance criteria.

---

## Loop 11 — Dashboard and Public Deployment

**Status:** Not started

* [ ] Build fleet overview.
* [ ] Build asset-health view.
* [ ] Display RUL and prediction intervals.
* [ ] Display warnings and critical alerts.
* [ ] Display prediction history.
* [ ] Display model version.
* [ ] Display drift status.
* [ ] Display recent online performance.
* [ ] Add replay controls where safe.
* [ ] Prepare Render deployment.
* [ ] Configure public HTTPS access.
* [ ] Configure secrets safely.
* [ ] Verify persistent storage requirements.
* [ ] Validate the public demo.
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
