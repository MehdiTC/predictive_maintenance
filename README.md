# TurbineGuard

A production-style predictive-maintenance ML platform for turbine and rotating-equipment sensor data.

> An independently developed predictive-maintenance platform inspired by industrial power-generation use cases. It uses public NASA turbine degradation data and contains no proprietary client data or implementation details.

## Project status

The project is built in bounded implementation loops (see [PROJECT_SPEC.md](PROJECT_SPEC.md) for the full design, [STATUS.md](STATUS.md) for current state, and [TASKS.md](TASKS.md) for the loop plan).

**Loop 0 — repository foundation — is complete**: a typed, tested Python 3.12 package with environment-based settings, structured JSON logging, and a minimal FastAPI service exposing liveness and readiness endpoints. Dataset acquisition, feature engineering, modeling, and the online system arrive in later loops.

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
| `make hooks`        | Install pre-commit hooks                   |

## Configuration

Settings are typed (`pydantic-settings`) and loaded from environment variables with the `TURBINE_GUARD_` prefix, or from a local `.env` file (gitignored). See [.env.example](.env.example).

| Variable                     | Default       | Description                                        |
| ---------------------------- | ------------- | -------------------------------------------------- |
| `TURBINE_GUARD_APP_NAME`     | `turbine-guard` | Human-readable application name                  |
| `TURBINE_GUARD_ENVIRONMENT`  | `development` | `development`, `testing`, or `production`          |
| `TURBINE_GUARD_LOG_LEVEL`    | `INFO`        | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL` |

Logs are emitted as single-line JSON objects; fields passed via `extra=` on logging calls are merged into the payload.

## Project layout

```text
├── src/turbine_guard/
│   ├── api/            # FastAPI app factory, routes, response schemas
│   ├── config/         # typed environment-based settings
│   ├── services/       # business logic used by the API layer
│   └── logging_config.py
├── tests/
│   ├── conftest.py
│   └── unit/
├── pyproject.toml      # project metadata + ruff/mypy/pytest configuration
├── Makefile
└── .env.example
```
