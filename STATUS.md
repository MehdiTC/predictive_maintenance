# STATUS.md

## Project Status

**Project:** TurbineGuard
**Current phase:** Loop 0 implemented and validated
**Active loop:** None — Loop 0 complete, awaiting review before Loop 1
**Overall status:** Foundation in place; no data or ML functionality yet
**Last updated:** 2026-07-12

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

Loop 0 (Repository Foundation) is implemented:

```text
├── src/turbine_guard/
│   ├── api/
│   │   ├── app.py              # create_app() factory
│   │   ├── routes/health.py    # GET /health/live, GET /health/ready
│   │   └── schemas/health.py   # LivenessResponse, ReadinessResponse
│   ├── config/settings.py      # typed BaseSettings, TURBINE_GUARD_ env prefix
│   ├── services/health.py      # ReadinessResult, check_readiness()
│   └── logging_config.py       # JSON formatter + configure_logging()
├── tests/
│   ├── conftest.py             # settings/app/client fixtures
│   └── unit/                   # 24 tests: settings, logging, health service, API
├── pyproject.toml              # hatchling, PEP 735 dev group, ruff/mypy/pytest config
├── uv.lock                     # committed lockfile (Python 3.12.13 via uv)
├── Makefile                    # install/format/lint/typecheck/test/check/run/hooks
├── .pre-commit-config.yaml     # local hooks running ruff + mypy through uv
├── .env.example                # documented defaults; real .env gitignored
├── .gitignore
└── README.md                   # setup, commands, configuration, layout
```

No dataset, feature, model, database, MLflow, Prefect, replay, monitoring, Docker, or CI functionality exists yet — deliberately.

---

## Current Loop

None active. Loop 0 is complete and awaiting review. The next loop (Loop 1 — Dataset Acquisition and Manifesting) must not begin without explicit approval.

---

## Completed Work

* [x] Defined the overall project mission.
* [x] Selected NASA C-MAPSS as the primary dataset.
* [x] Defined Remaining Useful Life as the primary modeling target.
* [x] Defined the planned continuous replay and delayed-feedback architecture.
* [x] Selected the core technology stack.
* [x] Removed unnecessary core infrastructure such as Kafka, Kubernetes, Spark, and Feast.
* [x] Defined implementation loops and acceptance criteria.
* [x] Created the persistent project specification.
* [x] Implemented Loop 0.

---

## Loop 0 Implementation Notes

Decisions and environment facts recorded during implementation:

1. **Git repository initialized.** The project directory was not its own git repository (it sat nested inside an unrelated, commitless repo rooted at the home directory). `git init -b main` was run in the project directory so pre-commit hooks and future CI have a proper repo root. All Loop 0 files are staged; **no commit has been created** — the initial commit is left for review.
2. **`uv` installed via Homebrew** (uv 0.11.24). `uv sync` provisioned Python 3.12.13 and the virtualenv.
3. **Structured logging uses the standard library only** (a small JSON formatter in `logging_config.py`). `structlog` was deliberately not added, per the dependency policy: stdlib handles the requirement.
4. **Pre-commit hooks are local `uv run` commands** (ruff format, ruff check --fix, mypy) rather than mirrored hook repos, so hook tool versions always match `uv.lock`.
5. **Readiness semantics:** `/health/ready` returns 200 with an empty `checks` map because Loop 0 has no external dependencies. The 503 `not_ready` path is implemented and covered by a test that injects a failing check; later loops add real database/model checks.
6. **Runtime dependencies are only** `fastapi`, `pydantic`, `pydantic-settings`, `uvicorn`; dev group: `ruff`, `mypy`, `pytest`, `httpx` (TestClient transport), `pre-commit`.
7. **Package name is `turbine-guard`** (import name `turbine_guard`) per the spec, although the repository directory is named `predictive_maintenance`.
8. **Settings scope:** `app_name`, `environment` (development/testing/production), `log_level` — the settings Loop 0 needs, nothing speculative. Values load from prefixed environment variables or a local `.env` file; validation is case-insensitive.

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
* Loop 0 additions: stdlib JSON logging (no structlog); local `uv run` pre-commit hooks; application factory pattern (`create_app`) with no module-level app instance.

### Implemented so far

Only the Loop 0 foundation. All data/ML/persistence/orchestration decisions remain design-only.

---

## Known Risks

1. NASA C-MAPSS is simulated turbofan data rather than actual power-plant sensor data.
2. Sensor columns may not have physical names such as vibration or temperature.
3. Online collection will be simulated through historical trajectory replay.
4. Public free-tier deployment limitations may affect persistent MLflow artifacts or background workers.
5. Maintenance-cost results will be simulated and must not be presented as real industrial savings.
6. The system could become overengineered if future tools are introduced without a clear need.
7. Time-series leakage is a major modeling risk and must be covered by explicit tests.
8. Retraining on very few newly labeled replay assets may not provide meaningful improvements.

---

## Immediate Next Action

Review Loop 0. After explicit approval, begin **Loop 1 — Dataset Acquisition and Manifesting**.

Also decide whether to create the initial git commit from the currently staged Loop 0 files.

---

## Loop 0 Exit Criteria — all satisfied

* [x] The package installs successfully with `uv` (`uv sync`, editable install of `turbine-guard 0.1.0`).
* [x] The repository uses a valid `src/` layout.
* [x] The FastAPI application starts (verified with uvicorn on port 8123).
* [x] `/health/live` returns `200 {"status": "alive"}` (unit test + live curl).
* [x] `/health/ready` returns `200 {"status": "ready", "checks": {}}` — appropriate for the dependency-free application (unit test + live curl); 503 path tested via injected failing check.
* [x] Typed settings load from environment variables (and optional `.env`), with validation tests.
* [x] Structured logging is configured (single-line JSON, UTC timestamps, `extra=` fields merged, idempotent setup).
* [x] Tests pass (24 passed).
* [x] Ruff formatting passes.
* [x] Ruff linting passes.
* [x] Mypy (strict) passes for `src`.
* [x] Setup instructions are documented in `README.md`.
* [x] `STATUS.md` and `TASKS.md` are updated.
* [x] No future-loop functionality has been implemented.

---

## Validation Status

All commands run on 2026-07-12 (macOS, Python 3.12.13 via uv):

| Check                  | Status | Detail                                             |
| ---------------------- | ------ | -------------------------------------------------- |
| `uv sync`              | Pass   | 45 packages resolved; lockfile committed            |
| Ruff format check      | Pass   | 17 files already formatted                          |
| Ruff lint check        | Pass   | All checks passed                                   |
| Mypy (strict)          | Pass   | No issues in 12 source files                        |
| Pytest                 | Pass   | 24 passed, 1 upstream deprecation warning           |
| FastAPI liveness test  | Pass   | Test suite + live `curl` → `HTTP 200`               |
| FastAPI readiness test | Pass   | Test suite + live `curl` → `HTTP 200`               |
| `/docs` + OpenAPI      | Pass   | `HTTP 200`, both health paths documented            |
| Pre-commit hooks       | Pass   | `pre-commit run --all-files`: ruff format, ruff check, mypy |

Known warning (non-blocking, upstream): importing `fastapi.testclient` with starlette 1.3.1 emits `StarletteDeprecationWarning: Using httpx with starlette.testclient is deprecated; install httpx2 instead.` The warning comes from FastAPI's own compatibility import, not project code. Revisit when FastAPI migrates.

---

## Last Completed Loop

**Loop 0 — Repository Foundation** (2026-07-12).

---

## Next Planned Loop

After Loop 0 is reviewed and explicitly approved:

**Loop 1 — Dataset Acquisition and Manifesting**

Do not begin Loop 1 automatically.
