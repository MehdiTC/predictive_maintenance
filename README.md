# TurbineGuard

A production-style predictive-maintenance ML platform for turbine and rotating-equipment sensor data.

> An independently developed predictive-maintenance platform inspired by industrial power-generation use cases. It uses public NASA turbine degradation data and contains no proprietary client data or implementation details.

## Project status

The project is built in bounded implementation loops (see [PROJECT_SPEC.md](PROJECT_SPEC.md) for the full design, [STATUS.md](STATUS.md) for current state, and [TASKS.md](TASKS.md) for the loop plan).

**Loops 0–3 are complete**: a typed, tested Python 3.12 package with environment-based settings, structured JSON logging, a minimal FastAPI service exposing liveness and readiness endpoints, reproducible checksummed acquisition of the NASA C-MAPSS FD001 dataset, a validated Parquet processing pipeline with a machine-readable data-quality report, an executable EDA notebook, and a leakage-safe feature layer: RUL labels, deterministic asset-level train/validation/calibration/replay splits, one shared `FeatureBuilder` (offline batch and single-asset incremental), and checksummed split/feature manifests. Modeling and the online system arrive in later loops.

## What the finished system will do

* Acquire and version NASA C-MAPSS run-to-failure sensor data.
* Build leakage-safe time-series features shared between training and serving.
* Train and evaluate Remaining Useful Life models with calibrated uncertainty.
* Serve predictions and maintenance-risk alerts through a documented FastAPI service.
* Replay held-out engine trajectories as a live sensor stream with delayed failure labels.
* Monitor data quality, drift, and online model performance, and retrain behind promotion gates.

## Requirements

* [uv](https://docs.astral.sh/uv/) — manages Python 3.12 and all dependencies automatically.

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
| `make eda`          | Execute the EDA notebook top to bottom     |
| `make hooks`        | Install pre-commit hooks                   |

## Configuration

Settings are typed (`pydantic-settings`) and loaded from environment variables with the `TURBINE_GUARD_` prefix, or from a local `.env` file (gitignored). See [.env.example](.env.example).

| Variable                     | Default       | Description                                        |
| ---------------------------- | ------------- | -------------------------------------------------- |
| `TURBINE_GUARD_APP_NAME`     | `turbine-guard` | Human-readable application name                  |
| `TURBINE_GUARD_ENVIRONMENT`  | `development` | `development`, `testing`, or `production`          |
| `TURBINE_GUARD_LOG_LEVEL`    | `INFO`        | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL` |
| `TURBINE_GUARD_DATA_DIR`     | `data`        | Base directory for the data layers                 |
| `TURBINE_GUARD_CMAPSS_SOURCE_URL` | NASA S3 mirror | C-MAPSS archive URL (`https://` or `file://`) |

Logs are emitted as single-line JSON objects; fields passed via `extra=` on logging calls are merged into the payload.

## Project layout

```text
├── src/turbine_guard/
│   ├── api/            # FastAPI app factory, routes, response schemas
│   ├── config/         # typed environment-based settings
│   ├── data/           # acquisition, manifests, schema, parsing, validation, processing
│   ├── features/       # RUL labels, asset-level splits, FeatureBuilder, manifests, pipeline
│   ├── services/       # business logic used by the API layer
│   └── logging_config.py
├── scripts/
│   ├── download_data.py
│   ├── process_data.py
│   └── build_features.py
├── notebooks/
│   └── 01_eda.ipynb    # the single primary EDA notebook (make eda)
├── docs/
│   ├── data_contract.md
│   ├── features.md
│   └── adr/
├── tests/
│   ├── conftest.py
│   ├── unit/
│   └── integration/    # local tests against the acquired FD001 data (auto-skipped)
├── data/               # gitignored: raw, manifests, processed (make acquire/process)
├── pyproject.toml      # project metadata + ruff/mypy/pytest configuration
├── Makefile
└── .env.example
```
