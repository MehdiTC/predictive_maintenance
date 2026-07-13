# TurbineGuard

A production-style predictive-maintenance ML platform for turbine and rotating-equipment sensor data.

> An independently developed predictive-maintenance platform inspired by industrial power-generation use cases. It uses public NASA turbine degradation data and contains no proprietary client data or implementation details.

## Project status

The project is built in bounded implementation loops (see [PROJECT_SPEC.md](PROJECT_SPEC.md) for the full design, [STATUS.md](STATUS.md) for current state, and [TASKS.md](TASKS.md) for the loop plan).

**Loops 0–10 are implemented**. The API, PostgreSQL, MLflow, explicit bootstrap, lifecycle worker,
and held-out replay now share one non-root production image and a health-ordered Compose topology.
GitHub Actions covers Python quality, real PostgreSQL migrations/integration, local MLflow, image
contracts, and a deterministic end-to-end API smoke test. Public deployment and the dashboard are
Loop 11 and are not implemented.

## What the finished system will do

* Acquire and version NASA C-MAPSS run-to-failure sensor data.
* Build leakage-safe time-series features shared between training and serving.
* Train and evaluate Remaining Useful Life models with calibrated uncertainty.
* Serve predictions and maintenance-risk alerts through a documented FastAPI service.
* Replay held-out engine trajectories as a live sensor stream with delayed failure labels.
* Monitor data quality, drift, and online model performance, and retrain behind promotion gates.

## Requirements

* [uv](https://docs.astral.sh/uv/) — manages Python 3.12 and all dependencies automatically.
* Docker with Compose v2 — required only for the multi-service local stack and container checks.

Install uv on macOS:

```bash
brew install uv
# or
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

```bash
git clone <repository-url>
cd <repository-directory>
uv sync                # creates .venv with Python 3.12 and installs everything
cp .env.example .env   # optional: local settings overrides
make hooks             # optional: install pre-commit hooks
```

## Containerized stack

A clean Docker volume needs one explicit, idempotent champion bootstrap before API readiness can
pass:

```bash
cp .env.example .env
docker compose build
docker compose up -d --wait postgres mlflow
docker compose --profile bootstrap run --rm bootstrap
docker compose up -d --wait api
```

Normal restarts then use `make docker-up`; `make docker-down` preserves all named volumes. MLflow is
at <http://127.0.0.1:5000>, API docs at <http://127.0.0.1:8000/docs>, and readiness proves the
database, champion, and feature contract rather than merely checking the process. Worker and replay
are explicit profiles, so startup never retrains/promotes or streams a full trajectory.

See [docs/containers.md](docs/containers.md) for services, network flow, bootstrap, profiles,
volumes, migrations, replay/worker commands, reset safety, CI-equivalent checks, and troubleshooting.

## Running the API

```bash
make run
# equivalent to: uv run uvicorn --factory turbine_guard.api.app:create_app --reload
```

Then verify:

```bash
curl http://127.0.0.1:8000/health/live    # {"status":"alive"}
curl http://127.0.0.1:8000/health/ready   # {"status":"ready","checks":{}}
```

Interactive OpenAPI documentation: <http://127.0.0.1:8000/docs>

## Dataset acquisition

The project uses the **NASA C-MAPSS Turbofan Engine Degradation Simulation** dataset (subset FD001) from the NASA Prognostics Center of Excellence. It is *simulated* run-to-failure data; sensor channels are anonymous, and this project deliberately does not assign them physical interpretations (such as vibration or temperature).

```bash
make acquire
# equivalent to: uv run python scripts/download_data.py
# options: --url <https:// or file:// archive>  --data-dir <dir>  --force
```

This downloads the source archive (cached under `data/raw/cmapss/`), extracts the FD001 files unchanged into an immutable raw layer, and writes a provenance manifest:

```text
data/
├── raw/cmapss/
│   ├── <source archive>.zip          # cached download
│   └── FD001/                        # immutable raw layer (read-only files)
│       ├── train_FD001.txt
│       ├── test_FD001.txt
│       └── RUL_FD001.txt
└── manifests/
    └── cmapss_fd001.json             # provenance manifest
```

The manifest records the dataset name and subset, source name and URL, retrieval timestamp (UTC), acquisition version, git commit, and — per file — SHA-256 checksum, size in bytes, record count, and asset (engine unit) count.

Acquisition is **idempotent**: re-running verifies every raw file against the manifest checksums and downloads nothing when they match. If a raw file was modified or deleted, acquisition fails with a clear error instead of silently repairing; use `--force` to deliberately re-download and replace the raw layer.

Offline or if NASA hosting moves: download the archive manually, then point acquisition at it with `--url file:///path/to/archive.zip` (or set `TURBINE_GUARD_CMAPSS_SOURCE_URL`). Both flat archives and archives with a nested `CMAPSSData.zip` are supported.

The `data/` directory is gitignored — datasets are never committed.

## Data processing and validation

```bash
make process
# equivalent to: uv run python scripts/process_data.py
# options: --data-dir <dir>  --force
```

This verifies the raw layer against the acquisition manifest, parses the whitespace-delimited
files into the canonical typed schema (`asset_id`, `cycle`, `operating_setting_1..3`,
`sensor_01..21` — sensors are anonymous and are deliberately not given physical names), runs
structural and semantic validation (schema/dtypes, unique and contiguous cycles per asset,
finite values, canonical FD001 counts), and writes validated Parquet outputs plus a
machine-readable report:

```text
data/processed/cmapss/FD001/
├── train_FD001.parquet       # 20,631 rows, 100 engines
├── test_FD001.parquet        # 13,096 rows, 100 engines
├── rul_FD001.parquet         # 100 official test RUL values
└── processing_report.json    # checks, stats, checksums, provenance
```

A failed required check blocks publication — no output is written. Constant/near-constant
columns are reported as warnings and kept. Re-running is idempotent (nothing is rewritten when
inputs and outputs are unchanged); tampered outputs fail loudly, and `--force` rebuilds. The full
contract is documented in [docs/data_contract.md](docs/data_contract.md).

## Exploratory data analysis

```bash
make eda
# equivalent to: uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

[notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) consumes the validated Parquet outputs (never
the raw text), executes top to bottom, and covers trajectory lengths, data quality,
constant/near-constant columns, operating settings, sensor lifecycle trends, distributions,
correlation structure, train-vs-test differences, and the implications for Loop 3 feature
engineering.

## Feature generation (labels, splits, features)

```bash
make features
# equivalent to: uv run python scripts/build_features.py
# options: --data-dir <dir>  --seed <int>  --rul-cap <int>  --force
```

This turns the validated Parquet into a reproducible, model-ready feature layer:

* **RUL labels** — uncapped `rul = T_i − t` per training row, plus an optional capped `rul_capped`.
* **Asset-level splits** — deterministic train (70 %), validation (15 %), calibration (5 %), and
  replay (10 %) partitions, split by engine (never by row); the official test set is untouched.
* **Leakage-safe features** — one shared `FeatureBuilder` produces trailing-window features
  (current, previous-cycle delta, rolling mean/std/min/max/range/slope, EWM mean) grouped per
  asset, using only observations up to the current cycle. The same builder drives future online
  inference via a single-asset incremental interface; offline and incremental outputs are proven
  equal.

```text
data/features/cmapss/FD001/
├── train.parquet            # model-ready train partition (labels + features)
├── validation.parquet
├── calibration.parquet
├── replay.parquet
├── test_features.parquet    # official test features (no per-row labels)
├── test_labels.parquet      # official test RUL benchmark (evaluation only)
├── split_manifest.json      # asset IDs, counts, seed, strategy
└── feature_manifest.json    # feature definition, versions, checksums, provenance
```

Structurally-undefined early-cycle values (e.g. the first-cycle delta) are left null; imputation
and scaling are deferred to the model pipeline (Loop 4). Re-running is idempotent, tampered outputs
fail loudly, and `--force` rebuilds. The full contract — feature definitions, rolling semantics,
leakage protections, and manifest structure — is in [docs/features.md](docs/features.md).

## Offline model training and evaluation

```bash
make train
# equivalent to: uv run python scripts/train_models.py
# options: --data-dir <dir> --output-dir <dir> --seed <int> --rul-cap <int>
#          --critical-horizon <int> --warning-horizon <int>
#          --conformal-coverage <float> --force
```

The command verifies the Loop 3 manifests, checksums, exact ordered feature contract, and disjoint
asset roles before fitting. Preprocessing is learned from training rows only; validation selects
the champion, calibration fits only the conformal residual quantile, and replay plus the official
NASA final-row benchmark are evaluated only after selection.

Candidates include a training-median constant, Ridge with median imputation/missing indicators/
scaling, histogram gradient boosting with native null handling, and XGBoost with native null
handling. Uncapped RUL and a 125-cycle capped target are compared using a compatible late-life
domain. Reports combine regression, asset/lifecycle slices, alert episodes and lead time,
uncertainty coverage/width, latency/size, interpretation, and explicitly simulated normalized-cost
maintenance policies.

Artifacts are written under `data/models/cmapss/FD001/` and are checksummed/idempotent. Joblib
files use pickle semantics and must only be loaded from trusted, checksum-verified sources. See
[docs/modeling.md](docs/modeling.md) for formulas, alert definitions, policy assumptions, artifact
layout, and limitations.

## MLflow tracking and model registry

```bash
make train-tracked
# equivalent to: uv run python scripts/train_models.py --track-with-mlflow
make mlflow-inspect
make mlflow-verify
make mlflow-ui             # UI at http://127.0.0.1:5000
```

The local default uses SQLite metadata at `data/mlflow/mlflow.db` and filesystem artifacts under
`data/mlflow/artifacts`. One parent represents a complete Loop 4 execution; each candidate is a
nested child. The selected child stores held-out metrics, reports, model card, and an MLflow pyfunc
that returns point/interval RUL and risk level using the existing champion bundle. Registered model
aliases are `candidate`, `challenger`, `champion`, and (after replacement) `archived`.

Tracking is opt-in, so ordinary `make train` and all local checksummed artifacts remain usable
without an MLflow store. See [docs/mlflow.md](docs/mlflow.md) for logged fields, idempotency,
registry semantics, loading commands, remote-store configuration, and limitations.

## Development commands

| Command             | Purpose                                    |
| ------------------- | ------------------------------------------ |
| `make install`      | Install the package and dev dependencies   |
| `make format`       | Format code with Ruff                      |
| `make format-check` | Check formatting without modifying files   |
| `make lint`         | Lint with Ruff                             |
| `make lint-fix`     | Lint and apply safe fixes                  |
| `make typecheck`    | Type-check `src` with mypy (strict)        |
| `make test`         | Run the pytest suite                       |
| `make check`        | Run all quality gates                      |
| `make run`          | Run the API locally with auto-reload       |
| `make acquire`      | Download the C-MAPSS FD001 dataset         |
| `make process`      | Validate raw data, write Parquet + report  |
| `make features`     | Build RUL labels, splits, and features     |
| `make train`        | Train/evaluate Loop 4 offline RUL models   |
| `make train-tracked` | Track Loop 4 and register the champion    |
| `make mlflow-ui`    | Launch the local MLflow UI                 |
| `make mlflow-inspect` | Inspect runs, versions, and aliases      |
| `make mlflow-verify` | Compare registry champion to local bundle |
| `make db-check`    | Check PostgreSQL connectivity/revision      |
| `make db-upgrade`  | Apply operational schema migrations         |
| `make db-current`  | Show the current Alembic revision           |
| `make db-history`  | Show migration history                      |
| `make db-downgrade` | Downgrade one revision (development only) |
| `make db-test`     | Run guarded PostgreSQL integration tests    |
| `make eda`          | Execute the EDA notebook top to bottom     |
| `make hooks`        | Install pre-commit hooks                   |
| `make docker-build` | Build the reusable production image        |
| `make docker-up`    | Start infrastructure, migrations, and API  |
| `make docker-down`  | Stop the stack and preserve named volumes  |
| `make docker-migrate` | Apply migrations through the owner service |
| `make docker-bootstrap` | Explicitly create/register the champion |
| `make docker-worker` | Run one safe monitoring window            |
| `make docker-replay-status` | Inspect replay state without starting it |
| `make docker-test`  | Verify image contracts and shutdown        |
| `make docker-smoke` | Run the isolated deterministic stack smoke |

## Configuration

Settings are typed (`pydantic-settings`) and loaded from environment variables with the `TURBINE_GUARD_` prefix, or from a local `.env` file (gitignored). See [.env.example](.env.example).

| Variable                     | Default       | Description                                        |
| ---------------------------- | ------------- | -------------------------------------------------- |
| `TURBINE_GUARD_APP_NAME`     | `turbine-guard` | Human-readable application name                  |
| `TURBINE_GUARD_ENVIRONMENT`  | `development` | `development`, `testing`, or `production`          |
| `TURBINE_GUARD_LOG_LEVEL`    | `INFO`        | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL` |
| `TURBINE_GUARD_DATA_DIR`     | `data`        | Base directory for the data layers                 |
| `TURBINE_GUARD_CMAPSS_SOURCE_URL` | NASA S3 mirror | C-MAPSS archive URL (`https://` or `file://`) |
| `TURBINE_GUARD_MLFLOW_TRACKING_URI` | `sqlite:///data/mlflow/mlflow.db` | Tracking/registry URI |
| `TURBINE_GUARD_MLFLOW_EXPERIMENT_NAME` | `TurbineGuard-FD001-Offline-Modeling` | Experiment |
| `TURBINE_GUARD_MLFLOW_REGISTERED_MODEL_NAME` | `TurbineGuard-FD001-RUL` | Registry name |
| `TURBINE_GUARD_DATABASE_URL` | local PostgreSQL URL | Operational state; psycopg only |
| `TURBINE_GUARD_DATABASE_TEST_URL` | unset | Dedicated integration DB; name must contain `test` |
| `TURBINE_GUARD_ONLINE_INFERENCE_ENABLED` | `true` | Enable `/v1` resources and routes |
| `TURBINE_GUARD_MODEL_PRELOAD_ENABLED` | `true` | Warm and verify champion during lifespan |
| `TURBINE_GUARD_API_HOST` | `127.0.0.1` | API bind host (`0.0.0.0` in Compose) |
| `TURBINE_GUARD_API_PORT` | `8000` | API bind and published Compose port |
| `TURBINE_GUARD_ASSET_STALE_AFTER_SECONDS` | `300` | Asset-health staleness threshold |
| `TURBINE_GUARD_CORS_ALLOWED_ORIGINS` | `[]` | Explicit origins; disabled by default |
| `TURBINE_GUARD_TRUSTED_HOSTS` | local/test hosts | Explicit host allowlist |
| `TURBINE_GUARD_REPLAY_API_BASE_URL` | `http://127.0.0.1:8000` | Inference API the replay client targets |
| `TURBINE_GUARD_REPLAY_CYCLE_DELAY_SECONDS` | `1.0` | Wait between cycles in continuous mode |
| `TURBINE_GUARD_REPLAY_SIMULATED_CYCLE_DURATION_SECONDS` | `1.0` | Simulated cycle length (simulation assumption) |
| `TURBINE_GUARD_REPLAY_LEASE_SECONDS` | `120` | Exclusive advance-claim duration per worker |
| `TURBINE_GUARD_MONITORING_WINDOW_DAYS` | `30` | Default UTC monitoring lookback |
| `TURBINE_GUARD_RETRAINING_MIN_NEW_ASSETS` | `5` | Minimum new completed labeled assets |
| `TURBINE_GUARD_RETRAINING_MIN_HOLDOUT_ASSETS` | `2` | Minimum disjoint promotion assets |
| `TURBINE_GUARD_PROMOTION_APPROVAL_REQUIRED` | `true` | Prevent automatic champion replacement |

## Operational PostgreSQL persistence

Loop 6 stores assets, sensor readings, model-version-pinned predictions, maintenance events,
model evaluations, drift report records, and pipeline runs in PostgreSQL. Repositories never
commit implicitly; callers own atomic transaction boundaries. Sensor cycles and predictions are
idempotent on explicit database uniqueness keys, with conflicting retries rejected rather than
overwritten. PostgreSQL is independent from MLflow's local SQLite metadata.

See [docs/database.md](docs/database.md) for the schema diagram, constraints/indexes, setup,
migration commands, test safety guard, transaction semantics, readiness injection, and limitations.

## Online inference and asset health

With PostgreSQL running, the Loop 3 feature layer present, and an MLflow `champion` registered:

```bash
export TURBINE_GUARD_DATABASE_URL='postgresql+psycopg://localhost:5432/turbine_guard'
export TURBINE_GUARD_MLFLOW_TRACKING_URI='sqlite:///data/mlflow/mlflow.db'
uv run alembic upgrade head
make run
```

The API exposes versioned sensor ingestion, asset summaries/details/health, recent predictions,
current model metadata, an operational monitoring summary, readiness, and Prometheus metrics.
Ingestion auto-creates a generic asset at cycle 1, requires contiguous new cycles, and commits the
reading and prediction atomically. Exact retries are idempotent; conflicting cycles return 409.

See [docs/online_inference.md](docs/online_inference.md) for contracts, examples, consistency,
transactions, errors, metrics, readiness, security limitations, and the boundary before replay.

Logs are emitted as single-line JSON objects; fields passed via `extra=` on logging calls are merged into the payload.

## Sensor replay and delayed feedback

Loop 8 replays the ten held-out FD001 engines one cycle at a time through the real ingestion API,
never exposing future observations or the failure cycle to the inference path. When a trajectory
ends, a failure event is emitted, realized RUL labels (`final_cycle − cycle`) are backfilled for
every historical prediction, and delayed evaluations (Loop 4 metrics, grouped by the model version
stored with each prediction) are persisted. Progress is durable in PostgreSQL, interruptions
resume from the earliest incomplete phase, and repeated commands are idempotent.

```bash
# with the API running (make run) and the migration applied (uv run alembic upgrade head)
uv run python scripts/replay_sensor_data.py start --asset-id 9 --mode accelerated
uv run python scripts/replay_sensor_data.py status --run-id <UUID>
uv run python scripts/replay_sensor_data.py evaluate-aggregate
```

See [docs/replay.md](docs/replay.md) for modes, ground-truth isolation, recovery, concurrency,
timestamp simulation, and limitations.

## Monitoring, retraining, and promotion

Loop 9 persists data-quality, all-feature drift, and delayed production-performance reports against
the exact champion's training-only reference. It returns an explicit `no_action`, `monitor`,
`retrain`, or `blocked` decision. Eligible completed assets are split at asset level into cumulative
fit additions and a disjoint promotion holdout; protected validation/calibration and official NASA
test roles are excluded.

Candidate, frozen champion, and the existing median baseline use the same holdout. Blocking gates
cover quality, data sufficiency, regression/NASA regression, critical alerts, false alarms,
coverage, latency, artifact size, and MLflow reload equality. Approval is required by default;
rejection preserves `champion`; promotion/rollback archive the displaced version and use the Loop 7
load-before-swap cache refresh.

```bash
uv run alembic upgrade head
make monitor
make lifecycle-status
uv run python scripts/model_lifecycle.py force-retraining
uv run python scripts/model_lifecycle.py approve-promotion --run-id <UUID>
uv run python scripts/model_lifecycle.py rollback --version <N>
```

See [docs/monitoring.md](docs/monitoring.md) for windows, formulas, thresholds, leakage policy,
phase recovery, every CLI operation, and limitations.

## Project layout

```text
├── src/turbine_guard/
│   ├── api/            # FastAPI app factory, routes, response schemas
│   ├── config/         # typed environment-based settings
│   ├── data/           # acquisition, manifests, schema, parsing, validation, processing
│   ├── features/       # RUL labels, asset-level splits, FeatureBuilder, manifests, pipeline
│   ├── modeling/       # offline models, metrics, conformal, simulation, selection, artifacts
│   ├── tracking/       # optional MLflow runs, pyfunc, registry, aliases, inspection
│   ├── replay/         # held-out replay, delayed labels, delayed evaluation, CLI
│   ├── monitoring/     # quality/drift/performance, retraining, gates, lifecycle CLI
│   ├── services/       # business logic used by the API layer
│   └── logging_config.py
├── scripts/
│   ├── download_data.py
│   ├── process_data.py
│   ├── build_features.py
│   ├── train_models.py
│   ├── mlflow_models.py
│   ├── replay_sensor_data.py
│   └── model_lifecycle.py
├── notebooks/
│   └── 01_eda.ipynb    # the single primary EDA notebook (make eda)
├── docs/
│   ├── data_contract.md
│   ├── features.md
│   ├── modeling.md
│   ├── mlflow.md
│   ├── database.md
│   ├── online_inference.md
│   ├── replay.md
│   ├── monitoring.md
│   ├── containers.md
│   └── adr/
├── Dockerfile / compose.yaml / .dockerignore
├── .github/workflows/ci.yml
├── tests/
│   ├── conftest.py
│   ├── unit/
│   └── integration/    # local tests against the acquired FD001 data (auto-skipped)
├── data/               # gitignored: raw, manifests, processed (make acquire/process)
├── pyproject.toml      # project metadata + ruff/mypy/pytest configuration
├── Makefile
└── .env.example
```
