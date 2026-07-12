# CLAUDE.md

## Project

This repository contains **TurbineGuard**, a portfolio-grade predictive-maintenance ML platform built using public NASA turbine degradation data.

The system will eventually:

* Acquire and validate historical sensor data.
* Build leakage-safe time-series features.
* Train and evaluate Remaining Useful Life models.
* Track experiments and models using MLflow.
* Serve predictions using FastAPI.
* Store operational data in PostgreSQL.
* Replay held-out asset trajectories as streaming sensor data.
* Receive delayed failure outcomes.
* Monitor data quality, drift, and model performance.
* Retrain and evaluate candidate models.
* Promote approved model versions.
* Run locally through Docker Compose.
* Support a public dashboard and deployment.

The complete design and implementation requirements are defined in `PROJECT_SPEC.md`.

---

## Required Reading

At the beginning of every implementation session, read:

1. `CLAUDE.md`
2. `PROJECT_SPEC.md`
3. `STATUS.md`
4. `TASKS.md`
5. Any relevant files under `docs/adr/`
6. Existing source code and tests related to the current task

Do not begin implementation before understanding the current repository state and active loop.

---

## Source of Truth

Use the following priority when instructions conflict:

1. The current user instruction
2. `CLAUDE.md`
3. `PROJECT_SPEC.md`
4. Accepted Architecture Decision Records
5. `TASKS.md`
6. Existing implementation

Do not silently deviate from the project specification.

When a deviation is necessary:

1. Explain the reason.
2. Describe the tradeoff.
3. Add or update an Architecture Decision Record.
4. Update the relevant project documentation.

---

## Working Method

The project is implemented through bounded development loops.

For each loop:

1. Read the required project files.
2. Inspect the existing implementation.
3. Restate the loop objective.
4. Identify relevant requirements and constraints.
5. Create a concrete implementation checklist.
6. Implement only the current loop.
7. Add or update tests.
8. Run all required validation commands.
9. Fix failures caused by the implementation.
10. Update `STATUS.md`.
11. Update `TASKS.md`.
12. Stop at the current loop’s acceptance boundary.

Do not automatically continue into the next loop.

Prefer a complete vertical slice over several incomplete abstractions.

---

## Core Engineering Rules

### Scope discipline

* Complete only the active task or loop.
* Do not add speculative infrastructure.
* Do not create placeholders for distant features unless required by the current design.
* Do not add a major dependency without a clear current need.
* Do not perform unrelated refactors.
* Do not implement future loops early.

### Code quality

* Use Python 3.12.
* Use a `src/` package layout.
* Use type annotations for application code.
* Keep functions and classes focused.
* Prefer explicit interfaces over hidden behavior.
* Avoid global mutable state.
* Avoid duplicated business logic.
* Use dependency injection where it improves testing and separation of concerns.
* Add docstrings to public modules, classes, and functions where their purpose is not obvious.
* Keep configuration outside business logic.
* Use structured logging instead of scattered `print` statements.

### Testing

* Every implemented behavior must have appropriate tests.
* Prefer small deterministic fixtures.
* Do not require downloading the full dataset during unit tests.
* Unit tests must not depend on public internet access.
* Integration tests must isolate external state when practical.
* Fix flaky or nondeterministic tests rather than repeatedly rerunning them.
* Do not weaken tests simply to make them pass.

### Documentation

When behavior, architecture, setup, or contracts change, update the relevant documentation in the same loop.

Keep documentation consistent with the implementation. Do not claim that unimplemented functionality exists.

---

## Machine Learning Rules

### Data leakage

Never use future sensor readings to create features for an earlier prediction.

For a prediction at cycle `t`, feature generation may only use observations from cycles less than or equal to `t`.

Required leakage protections include:

* Splitting data by asset, not by individual row.
* Keeping online replay assets out of initial training.
* Hiding replay failure times from the inference path.
* Testing that changing future observations cannot change earlier features.
* Fitting preprocessing components only on training data.

### Training-serving consistency

Use one shared feature-generation implementation for:

* Offline training
* Batch evaluation
* Online inference
* Monitoring
* Retraining

Do not create separate notebook and production implementations of the same feature logic.

### Evaluation

Do not evaluate the system using only one aggregate regression metric.

The completed project must eventually include:

* MAE
* RMSE
* NASA asymmetric score
* Failure-horizon precision and recall
* False-alarm rate
* Alert lead time
* Slice evaluation
* Prediction-interval evaluation
* Configurable maintenance-policy simulation

Do not invent real industrial cost savings. Business results must be labeled as simulations using explicit assumptions.

### Model selection

Do not hard-code XGBoost or any model as the winner.

Train and evaluate multiple baselines using identical data splits. Select models using explicit evaluation criteria.

### Reproducibility

Training runs must eventually record:

* Dataset version or checksum
* Feature version
* Split asset IDs
* Hyperparameters
* Random seed
* Git commit
* Metrics
* Model artifacts
* Relevant plots and reports

---

## Confidentiality and Public Framing

This repository is an independent public project inspired by industrial predictive-maintenance use cases.

Do not include:

* Accenture source code
* Genelba data
* Confidential system designs
* Internal documents
* Client-specific thresholds
* Client-specific costs
* Proprietary implementation details
* Claims that this public repository was deployed at Genelba

Use public NASA data and independently designed architecture.

Acceptable framing:

> An independently developed predictive-maintenance platform inspired by industrial power-generation use cases. It uses public NASA turbine degradation data and contains no proprietary client data or implementation details.

---

## Approved Core Stack

Use the following technologies when their corresponding loop is reached:

* Python 3.12
* `uv`
* Pandas
* NumPy
* SciPy
* Scikit-learn
* XGBoost
* FastAPI
* Pydantic
* Uvicorn
* PostgreSQL
* SQLAlchemy
* Alembic
* MLflow
* Prefect
* Docker
* Docker Compose
* Pytest
* Ruff
* Mypy
* GitHub Actions
* Prometheus Python client
* Lightweight server-rendered dashboard with Plotly

A listed technology does not need to be installed before its implementation loop.

---

## Technologies Excluded from the Core Project

Do not add the following unless the user explicitly changes the architecture:

* Kafka
* Kubernetes
* Spark
* Feast
* Kubeflow
* Airflow
* Terraform
* A separate React frontend
* A distributed feature store
* A large monitoring platform added only for screenshots

These technologies may be discussed in scaling documentation without being implemented.

---

## Dependency Policy

Before adding a dependency:

1. Verify that the standard library or an existing dependency cannot reasonably handle the requirement.
2. Confirm that the dependency is required by the current loop.
3. Prefer established, actively maintained libraries.
4. Avoid overlapping libraries that solve the same problem.
5. Document major architectural dependencies in an ADR when appropriate.

Do not install every future project dependency during repository initialization.

---

## Repository Conventions

Expected high-level structure:

```text
turbine-guard/
├── CLAUDE.md
├── PROJECT_SPEC.md
├── STATUS.md
├── TASKS.md
├── README.md
├── pyproject.toml
├── uv.lock
├── compose.yaml
├── Dockerfile
├── Makefile
├── .env.example
├── .github/
├── alembic/
├── configs/
├── data/
├── docs/
│   └── adr/
├── notebooks/
├── scripts/
├── src/
│   └── turbine_guard/
└── tests/
```

Only create directories needed by the active loop. Do not generate empty architecture for every future component.

---

## Configuration

* Use typed, environment-based application settings.
* Commit `.env.example`, not real secrets.
* Never commit credentials, tokens, passwords, or private connection strings.
* Keep configurable thresholds out of core business logic.
* Use UTC for persisted timestamps.
* Make random seeds explicit where applicable.

---

## API Requirements

When API functionality is implemented:

* Use Pydantic request and response models.
* Version public endpoints under `/v1`.
* Use consistent structured error responses.
* Provide liveness and readiness endpoints.
* Expose model version information with predictions.
* Use idempotency where duplicate sensor submissions are possible.
* Generate OpenAPI documentation.
* Add unit and integration tests.

Liveness indicates that the process is running.

Readiness indicates that the service has access to the dependencies required to handle requests.

---

## Database Requirements

When the database loop is reached:

* Use PostgreSQL.
* Use SQLAlchemy.
* Manage schema changes with Alembic.
* Add appropriate foreign keys, uniqueness constraints, and indexes.
* Enforce uniqueness for `(asset_id, cycle)` sensor observations.
* Store the model version with each prediction.
* Test migrations from an empty database.

Do not use an in-memory database as proof that PostgreSQL-specific production behavior works.

---

## Completion Requirements

Before declaring a loop complete:

1. All loop acceptance criteria are satisfied.
2. Relevant tests exist.
3. Required validation commands pass.
4. Documentation matches the implementation.
5. `STATUS.md` is updated.
6. `TASKS.md` is updated.
7. Remaining limitations are stated clearly.
8. No future loop has been implemented accidentally.

If an acceptance criterion cannot be completed, document:

* What failed
* Why it failed
* What was attempted
* The exact remaining work
* Whether the repository remains in a usable state

---

## Default Validation Commands

Use the commands appropriate to the current repository state.

The standard Python validation suite is:

```bash
uv sync
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
```

When applicable, also run:

```bash
docker compose config
docker compose build
```

Report command failures accurately. Do not state that a command passed unless it was executed successfully.

---

## Stop Condition

After completing the current loop:

* Update `STATUS.md`.
* Mark completed work in `TASKS.md`.
* Report the files changed.
* Report validation results.
* List any unresolved issues.
* Stop.

Do not begin the next loop without a new instruction.
