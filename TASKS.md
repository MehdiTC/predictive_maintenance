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

None. Loops 0 and 1 are complete and awaiting review; Loop 2 must not begin without explicit approval.

---

# Completed Loops

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

# Planned Loops

## Loop 2 — Validation and EDA

**Status:** Blocked until Loop 1 approval

* [ ] Define the raw data contract.
* [ ] Parse C-MAPSS files.
* [ ] Assign explicit column names.
* [ ] Validate types and schema.
* [ ] Validate asset and cycle integrity.
* [ ] Detect duplicates and invalid records.
* [ ] Produce validated data.
* [ ] Produce processed Parquet data.
* [ ] Create the single primary EDA notebook.
* [ ] Document EDA findings.
* [ ] Validate Loop 2 acceptance criteria.

---

## Loop 3 — Labels, Splits, and Features

**Status:** Not started

* [ ] Generate RUL labels.
* [ ] Compare raw and capped RUL targets.
* [ ] Implement asset-level train, validation, calibration, and replay splits.
* [ ] Protect official test data.
* [ ] Implement the shared `FeatureBuilder`.
* [ ] Add rolling-window features.
* [ ] Add feature manifests.
* [ ] Test offline and online feature consistency.
* [ ] Add explicit future-data leakage tests.
* [ ] Validate Loop 3 acceptance criteria.

---

## Loop 4 — Baseline Modeling and Evaluation

**Status:** Not started

* [ ] Implement constant baseline.
* [ ] Implement Ridge regression.
* [ ] Implement a tree-based baseline.
* [ ] Implement XGBoost.
* [ ] Use identical splits for all models.
* [ ] Add MAE and RMSE.
* [ ] Add NASA asymmetric score.
* [ ] Add failure-horizon metrics.
* [ ] Add false-alarm and lead-time metrics.
* [ ] Add lifecycle slice evaluation.
* [ ] Add conformal prediction intervals.
* [ ] Add configurable maintenance-policy simulation.
* [ ] Generate an evaluation report.
* [ ] Select the champion using explicit criteria.
* [ ] Validate Loop 4 acceptance criteria.

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

Updated at Loop 1 completion (2026-07-12). None block Loop 1 acceptance.

1. ~~No initial git commit exists yet.~~ **Resolved:** Loop 0 was committed as `d95b528`. Loop 1 changes are staged but intentionally uncommitted pending review (same policy).
2. **Upstream deprecation warning in the test suite.** Importing `fastapi.testclient` with starlette 1.3.1 emits `StarletteDeprecationWarning: Using httpx with starlette.testclient is deprecated; install httpx2 instead.` The warning originates in FastAPI's own compatibility import, not project code; FastAPI's TestClient currently requires `httpx`. Revisit when FastAPI completes its migration.
3. **NASA hosting stability.** The legacy `ti.arc.nasa.gov` C-MAPSS URL is dead and `data.nasa.gov` returns 404; acquisition defaults to NASA's S3 mirror (`phm-datasets.s3.amazonaws.com`). If that moves too, set `TURBINE_GUARD_CMAPSS_SOURCE_URL` (or `--url`, including `file://` for a manually downloaded archive). No code change should be needed.
