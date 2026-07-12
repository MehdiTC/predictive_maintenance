# TurbineGuard

A production-style predictive-maintenance ML platform for turbine and rotating-equipment sensor data.

> An independently developed predictive-maintenance platform inspired by industrial power-generation use cases. It uses public NASA turbine degradation data and contains no proprietary client data or implementation details.

## Project status

The project is built in bounded implementation loops (see [PROJECT_SPEC.md](PROJECT_SPEC.md) for the full design, [STATUS.md](STATUS.md) for current state, and [TASKS.md](TASKS.md) for the loop plan).

**Loops 0–1 are complete**: a typed, tested Python 3.12 package with environment-based settings, structured JSON logging, a minimal FastAPI service exposing liveness and readiness endpoints, and reproducible, checksummed acquisition of the NASA C-MAPSS FD001 dataset. Parsing/validation, feature engineering, modeling, and the online system arrive in later loops.

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
│   ├── data/           # dataset acquisition and provenance manifests
│   ├── services/       # business logic used by the API layer
│   └── logging_config.py
├── scripts/
│   └── download_data.py
├── tests/
│   ├── conftest.py
│   └── unit/
├── data/               # gitignored: raw layer + manifests (make acquire)
├── pyproject.toml      # project metadata + ruff/mypy/pytest configuration
├── Makefile
└── .env.example
```
