# TurbineGuard: End-to-End Predictive Maintenance ML Platform

## 1. Project Mission

Build a production-style predictive-maintenance platform for turbine and rotating-equipment sensor data.

The system will:

1. Acquire and version historical run-to-failure data.
2. Validate, process, explore, and transform multivariate sensor time series.
3. Train models that estimate Remaining Useful Life.
4. Convert RUL predictions into actionable maintenance-risk alerts.
5. Continuously ingest newly arriving sensor readings.
6. Store predictions and operational outcomes.
7. Detect data drift and model-performance degradation.
8. Retrain and evaluate candidate models.
9. Register and promote approved model versions.
10. Serve predictions through a documented FastAPI service.
11. Provide a public dashboard and reproducible Docker deployment.

This is an independent public implementation inspired by previous industrial predictive-maintenance experience. It must not contain Accenture code, Genelba data, internal architecture, confidential thresholds, client-specific business rules, or claims that the public system was deployed at Genelba.

Recommended public wording:

> An independently developed predictive-maintenance platform inspired by industrial power-generation use cases. It uses public NASA turbine degradation data and contains no proprietary client data or implementation details.

---

# 2. Primary Prediction Problem

## Core ML target: Remaining Useful Life

For asset (i) at operating cycle (t):

[
RUL_{i,t} = T_i - t
]

where (T_i) is the failure cycle for that asset.

The model predicts how many operating cycles remain before failure.

## Operational decision target

The system also derives:

[
P(\text{failure within } H \text{ cycles})
]

Initially:

* Warning horizon: 50 cycles
* Critical horizon: 30 cycles
* Healthy: predicted RUL above 50 cycles
* Warning: predicted RUL between 30 and 50 cycles
* Critical: predicted RUL below 30 cycles

All thresholds must be configurable rather than embedded throughout the code.

## API output

A prediction should contain:

* Predicted RUL
* Lower uncertainty bound
* Upper uncertainty bound
* Probability or indicator of failure within 30 cycles
* Risk level
* Asset ID
* Current cycle
* Model name and version
* Prediction timestamp
* Data-quality warnings

This is more useful than simply returning a binary failure prediction.

---

# 3. Data Strategy

## Primary dataset

Use NASA C-MAPSS, beginning with the FD001 subset.

Why FD001 first:

* Multiple independent engine units
* Complete run-to-failure trajectories
* Multivariate operating settings and sensor measurements
* Well-defined RUL problem
* Small enough to process locally
* Complex enough to demonstrate time-series ML engineering
* Suitable for replaying historical data as a live sensor stream

The project must clearly identify the source as simulated turbofan degradation data. It must not rename anonymous sensors as “vibration” or “temperature” unless the dataset documentation explicitly provides that interpretation.

## Data partitions

Do not randomly split individual rows.

Split by asset or engine ID:

* Initial training engines: approximately 70%
* Validation and calibration engines: approximately 15%
* Online replay engines: approximately 15%
* Official test data: untouched final benchmark

The online replay engines must never be used during initial training.

## Delayed-label simulation

The replay service has access to complete trajectories, but the prediction service must see only the current and previous cycles.

The true failure cycle must remain hidden until the replayed engine reaches failure. At that point, a maintenance or failure event is emitted, allowing the monitoring pipeline to calculate the true historical RUL for previous predictions.

This creates a realistic delayed-feedback loop and prevents future-data leakage.

## Data layers

### Raw layer

Immutable downloaded source files.

Each acquisition creates a manifest containing:

* Source name
* Retrieval timestamp
* Source filename
* File checksum
* Dataset subset
* Number of records
* Number of assets
* Schema version
* Git commit, when applicable

### Validated layer

Schema-checked, typed data with:

* Column names
* Asset IDs
* Operating cycles
* Operational settings
* Sensor values
* Duplicate checks
* Missing-value checks
* Numeric range checks
* Monotonic cycle checks within each asset

### Processed layer

Clean Parquet files partitioned by dataset subset or asset group.

### Feature layer

Versioned model-ready feature snapshots with:

* Feature definition version
* Source manifest checksum
* Training asset IDs
* Feature-generation parameters
* Output checksum

A feature store is intentionally excluded. One model and one feature pipeline do not justify adding Feast. Training and serving will share the same Python feature-building package.

---

# 4. EDA and Data Understanding

EDA must answer engineering and modeling questions rather than only produce visualizations.

Required analysis:

1. Number and length of asset trajectories.
2. Sensor missingness and invalid values.
3. Near-constant sensors.
4. Sensor behavior throughout the asset lifecycle.
5. Correlation between sensors and RUL.
6. Distribution differences across engines.
7. Effects of operational settings.
8. Sensors that show degradation trends.
9. Sensors whose behavior is mostly noise.
10. Failure-horizon class balance.
11. Potential leakage variables.
12. Differences between training, validation, replay, and final test units.

Required visualizations:

* Asset trajectory lengths
* Sensor traces for representative assets
* Sensor distributions
* Correlation matrix
* Sensor-versus-RUL plots
* Rolling sensor statistics
* Failure-horizon class balance
* Train-versus-replay feature distributions

Only one EDA notebook should exist. Reusable cleaning and feature logic must live under `src/`, not inside notebooks.

---

# 5. Feature Engineering

The primary model will use tabular features generated from recent sensor history.

For each current cycle, generate features from windows such as:

* Current sensor value
* Previous-cycle difference
* Rolling mean
* Rolling standard deviation
* Rolling minimum
* Rolling maximum
* Rolling range
* Rolling slope
* Exponentially weighted mean
* Operational settings
* Current operating cycle

Initial rolling windows:

* 5 cycles
* 10 cycles
* 20 cycles

Features at cycle (t) may only use observations from cycles less than or equal to (t).

The code must include an explicit automated leakage test that verifies that changing future readings does not change an earlier feature row.

## Training-serving consistency

Implement one shared component:

```text
FeatureBuilder
```

It must be used by:

* Offline dataset processing
* Model training
* Batch evaluation
* Online inference
* Monitoring
* Retraining

There must not be separate notebook and API implementations of the same feature logic.

---

# 6. Model Strategy

## Baselines

The first models should be deliberately simple:

1. Constant median-RUL baseline
2. Ridge regression
3. Random forest or histogram gradient boosting
4. XGBoost regressor

The expected champion is XGBoost, but it must win through evaluation rather than being hard-coded as the winner.

## Why not begin with an LSTM

A deep sequential model would add:

* More complicated training
* Longer experiments
* More preprocessing
* More difficult serving
* More difficult interpretation
* Greater risk of focusing on architecture rather than system quality

An LSTM, temporal convolution model, or Transformer may be added later only as an experiment against the established production baseline.

## RUL target experiments

Compare:

* Raw RUL
* Capped RUL during the early healthy stage

The cap value must be treated as a hyperparameter and logged in MLflow.

## Uncertainty estimation

Use split conformal prediction:

1. Train the point prediction model.
2. Calculate absolute residuals on a separate calibration set.
3. Choose the appropriate residual quantile.
4. Return a prediction interval around each RUL estimate.

This adds useful maintenance-risk information without requiring a complex Bayesian model.

---

# 7. Evaluation Framework

## Regression metrics

Track:

* MAE
* RMSE
* (R^2), for context rather than primary selection
* NASA asymmetric scoring function
* Error by lifecycle stage
* Prediction-interval coverage
* Prediction-interval width

## Maintenance-alert metrics

Convert RUL into operational horizons and track:

* Precision for failure within 30 cycles
* Recall for failure within 30 cycles
* F1
* PR-AUC
* False alarms per 1,000 operating cycles
* Missed failures
* Mean alert lead time
* Median alert lead time
* Alerts that occurred too early
* Alerts that occurred too late

Accuracy must not be used as the primary classification metric because healthy observations are much more common.

## Business-policy simulation

Create a configurable maintenance-policy evaluator comparing:

### Reactive policy

Maintenance occurs only after failure.

### Predictive policy

Maintenance occurs when the system emits a critical alert.

Use normalized cost units rather than pretending to know real Genelba costs.

Example configurable components:

* Unplanned failure cost
* Planned inspection cost
* Planned repair cost
* Cost of replacing equipment too early
* Downtime cost per cycle
* Missed-failure cost

Report:

* Relative maintenance cost
* Number of unplanned failures
* Number of planned interventions
* Average useful life retained
* Average warning lead time

The README must clearly label this as a simulation based on assumptions.

## Slice evaluation

Evaluate performance by:

* Short versus long asset histories
* Early, middle, and late lifecycle
* Operating-condition ranges
* Individual sensor-availability conditions
* Asset groups
* Drifted versus non-drifted data

---

# 8. MLflow Lifecycle

Every training run must log:

* Git commit SHA
* Dataset-manifest checksum
* Feature version
* Asset IDs in each split
* Model type
* Hyperparameters
* Random seed
* Training duration
* Evaluation metrics
* Model artifact
* Feature list
* Calibration residual information
* Diagnostic plots
* Model card
* Dependency information

## Registry aliases

Use:

* `candidate`
* `challenger`
* `champion`
* `archived`

The FastAPI service loads the `champion` model.

## Candidate promotion gate

A candidate may be promoted only when:

1. Data-validation tests pass.
2. Reproducibility tests pass.
3. Candidate beats the naive baseline.
4. Candidate is not materially worse than the champion on RMSE.
5. Candidate is not materially worse on the NASA score.
6. Critical-horizon recall remains acceptable.
7. False alarms do not increase beyond the configured tolerance.
8. Prediction-interval coverage remains acceptable.
9. Inference latency remains below the service threshold.
10. All unit and integration tests pass.

The default production-style behavior should require approval before replacing the champion.

For automated demonstrations, an environment variable may enable automatic promotion after all gates pass.

---

# 9. Continuous Online System

## Sensor replay service

Create a replay process that:

1. Selects a held-out asset trajectory.
2. Reads one cycle at a time.
3. Sends the observation to the ingestion endpoint.
4. Waits for a configurable interval.
5. Continues until failure.
6. Emits a failure or maintenance outcome.
7. Starts another held-out asset.

Support accelerated replay so an entire lifecycle can be demonstrated in minutes.

## Ingestion flow

For every incoming observation:

1. Validate the API schema.
2. Enforce uniqueness on asset ID and cycle.
3. Store the raw observation.
4. Check that cycles arrive in a valid order.
5. Determine whether sufficient history exists.
6. Generate current rolling features.
7. Load the champion model.
8. Generate a prediction.
9. Calculate an uncertainty interval.
10. Apply the alert policy.
11. Store the prediction.
12. Return the result to the sender.
13. Update service metrics.

## Feedback flow

When a failure or maintenance event arrives:

1. Record the event.
2. Calculate labels for eligible historical observations.
3. Join predictions with realized outcomes.
4. Calculate online performance.
5. Update monitoring tables.
6. Determine whether retraining conditions are satisfied.

## Retraining conditions

Retraining should not run after every observation.

Run it when:

* Enough newly labeled assets are available, and
* A scheduled retraining interval has passed, or
* Drift exceeds a configured threshold, or
* Model performance falls below a configured threshold

Example triggers:

* At least five newly labeled assets
* Rolling MAE worsens by more than 15%
* Critical-alert recall falls below its accepted level
* Several important features exceed the drift threshold

All values must remain configurable.

---

# 10. Monitoring

## Data-quality monitoring

Track:

* Incoming record count
* Rejected records
* Duplicate records
* Missing values
* Out-of-range values
* Out-of-order cycles
* Assets without sufficient history
* Sensor availability

## Data-drift monitoring

For important model features, calculate:

* Population Stability Index
* Wasserstein distance
* Changes in missingness
* Changes in mean and standard deviation

Do not add a large monitoring platform solely to create a screenshot. Implement these metrics clearly in Python and store the reports.

## Model monitoring

Once labels become available, track:

* Rolling MAE
* Rolling RMSE
* Alert precision
* Alert recall
* False-alarm rate
* Average lead time
* Prediction-interval coverage
* Prediction distribution
* Risk-level distribution

## Service monitoring

Expose Prometheus-compatible metrics for:

* Request count
* Request failures
* Prediction count
* Prediction latency
* Model-load failures
* Validation failures
* Active model version
* Database failures
* Retraining runs
* Promotion success or failure

A full Grafana deployment is optional. The public dashboard should show the most important metrics directly.

---

# 11. FastAPI Design

## Main endpoints

```text
POST /v1/sensor-readings
POST /v1/predictions
POST /v1/maintenance-events

GET /v1/assets
GET /v1/assets/{asset_id}
GET /v1/assets/{asset_id}/health
GET /v1/predictions/recent
GET /v1/models/current
GET /v1/monitoring/summary

GET /health/live
GET /health/ready
GET /metrics
GET /docs
```

`POST /v1/sensor-readings` should normally generate and return the latest prediction when sufficient history exists.

## API engineering requirements

* Pydantic request and response models
* Explicit API versioning
* Consistent error responses
* Idempotent sensor insertion
* Dependency injection for database and model access
* Structured JSON logging
* Request IDs
* Readiness and liveness checks
* Model-version information in every prediction
* Environment-based configuration
* No secrets committed to Git
* OpenAPI documentation
* Integration tests using a temporary test database

---

# 12. Data Storage

Use PostgreSQL for the operational system.

Core tables:

```text
assets
sensor_readings
predictions
maintenance_events
model_evaluations
drift_reports
pipeline_runs
```

Important constraints and indexes:

* Unique `(asset_id, cycle)` sensor readings
* Index sensor readings by `(asset_id, cycle)`
* Index predictions by asset and timestamp
* Foreign keys from predictions to assets
* Model version stored with every prediction
* Timestamps stored in UTC
* Database migrations managed with Alembic

Parquet files remain the primary storage format for immutable training snapshots.

PostgreSQL is sufficient for this project’s volume. Kafka, TimescaleDB, and a distributed feature store are unnecessary at the initial scale.

---

# 13. Workflow Orchestration

Use Prefect for:

* Dataset acquisition
* Dataset validation
* Feature generation
* Training
* Candidate evaluation
* Model registration
* Monitoring
* Retraining
* Model promotion
* Batch backfills

Required flows:

```text
acquire_dataset_flow
build_training_dataset_flow
train_candidate_flow
evaluate_candidate_flow
monitor_production_flow
retrain_if_needed_flow
backfill_labels_flow
```

Each task should be:

* Independently testable
* Retryable where appropriate
* Idempotent where possible
* Configurable
* Observable through structured logs
* Callable directly as a Python function in tests

---

# 14. Deployment and Demo

## Local deployment

A single command should start the system:

```bash
docker compose up --build
```

Core containers:

```text
api
worker
replay
postgres
mlflow
```

The dashboard may be served by the FastAPI application to avoid creating a separate frontend service.

## Public deployment

Recommended deployment target: Render.

Public architecture:

* Dockerized FastAPI web service
* Managed PostgreSQL
* Private MLflow service or persistent model artifact storage
* Scheduled monitoring and retraining job
* GitHub-triggered deployment
* HTTPS public endpoint
* Persistent disk or object storage for artifacts

The public demo does not need industrial scale. It needs to be reliable, understandable, and reproducible.

## Dashboard

The dashboard should show:

1. Fleet overview
2. Current asset health
3. Latest sensor measurements
4. Predicted RUL
5. Prediction interval
6. Healthy, warning, or critical status
7. Recent alerts
8. Prediction history
9. Model version
10. Drift status
11. Recent model performance
12. Replay controls
13. Last retraining result

Use a simple server-rendered dashboard with lightweight JavaScript and Plotly. Do not build a full React application.

## Recruiter demo flow

A two-minute demo should show:

1. Start the replay.
2. Sensor readings arrive continuously.
3. RUL decreases as degradation progresses.
4. The system enters warning and critical states.
5. A failure event arrives.
6. Historical predictions receive delayed labels.
7. Monitoring metrics update.
8. Retraining conditions are evaluated.
9. A candidate model appears in MLflow.
10. The model passes or fails the promotion gate.
11. The API reports its current champion version.

---

# 15. CI/CD and Software Engineering

## GitHub Actions checks

Every pull request should run:

1. Dependency installation
2. Ruff formatting and linting
3. Mypy type checking
4. Unit tests
5. Integration tests
6. Model smoke test
7. Docker image build
8. API health test
9. Migration test
10. Leakage test

Deployment from the main branch occurs only after the checks pass.

## Engineering standards

Use:

* Python 3.12
* `uv` for dependency management
* `ruff` for formatting and linting
* `mypy` for type checking
* `pytest`
* Pre-commit hooks
* SQLAlchemy
* Alembic
* Structured logging
* Type annotations
* Docstrings for public interfaces
* Configuration through environment variables and typed settings

Avoid large functions, hidden notebook state, global mutable model objects, and duplicated feature logic.

---

# 16. Selected Technology Stack

## Core

| Concern                | Choice                       |
| ---------------------- | ---------------------------- |
| Language               | Python                       |
| Tabular processing     | Pandas, NumPy                |
| Statistical utilities  | SciPy                        |
| ML                     | Scikit-learn, XGBoost        |
| API                    | FastAPI, Pydantic, Uvicorn   |
| Database               | PostgreSQL, SQLAlchemy       |
| Migrations             | Alembic                      |
| Experiment tracking    | MLflow                       |
| Model registry         | MLflow Model Registry        |
| Orchestration          | Prefect                      |
| Containers             | Docker, Docker Compose       |
| Service metrics        | Prometheus Python client     |
| Dashboard              | FastAPI templates and Plotly |
| Testing                | Pytest                       |
| Linting and formatting | Ruff                         |
| Type checking          | Mypy                         |
| CI/CD                  | GitHub Actions               |
| Deployment             | Render                       |

## Explicitly excluded from the core project

* Kafka
* Kubernetes
* Spark
* Feast
* Kubeflow
* Airflow
* Terraform
* Grafana
* Great Expectations
* Deep-learning serving platforms
* Multiple cloud providers
* A React frontend

These tools are not inherently bad. They are excluded because the project’s workload does not justify them.

The README should include a “Scaling this system” section explaining where Kafka, Kubernetes, online feature stores, and distributed processing would become appropriate.

---

# 17. Repository Structure

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
│   └── workflows/
├── alembic/
├── configs/
│   ├── data.yaml
│   ├── features.yaml
│   ├── model.yaml
│   ├── monitoring.yaml
│   └── business_policy.yaml
├── data/
│   ├── raw/
│   ├── validated/
│   ├── processed/
│   └── manifests/
├── docs/
│   ├── architecture.md
│   ├── data_contract.md
│   ├── model_card.md
│   ├── evaluation.md
│   ├── deployment.md
│   ├── scaling.md
│   └── adr/
├── notebooks/
│   └── 01_eda.ipynb
├── scripts/
│   ├── download_data.py
│   ├── replay_sensor_data.py
│   ├── benchmark_api.py
│   └── seed_database.py
├── src/
│   └── turbine_guard/
│       ├── api/
│       ├── config/
│       ├── contracts/
│       ├── data/
│       ├── database/
│       ├── evaluation/
│       ├── features/
│       ├── models/
│       ├── monitoring/
│       ├── replay/
│       ├── services/
│       └── workflows/
└── tests/
    ├── unit/
    ├── integration/
    ├── contract/
    ├── model/
    └── end_to_end/
```

---

# 18. Claude Code Operating Contract

Claude Code must begin every loop by reading:

```text
CLAUDE.md
PROJECT_SPEC.md
STATUS.md
TASKS.md
Relevant ADRs
Relevant existing code and tests
```

## Rules

1. Complete only one scoped loop at a time.
2. Inspect the existing implementation before editing.
3. Do not introduce a new major dependency without an Architecture Decision Record.
4. Do not replace an established component without documenting the tradeoff.
5. Keep reusable logic outside notebooks.
6. Write or update tests with each implementation.
7. Run the required validation commands before finishing.
8. Update `STATUS.md` and `TASKS.md`.
9. Document unresolved assumptions.
10. Do not claim completion when acceptance tests fail.
11. Preserve backward compatibility unless the loop explicitly changes a contract.
12. Prefer a small complete vertical slice over several incomplete abstractions.
13. Never use future sensor observations during feature generation.
14. Never expose replay-only ground truth to the prediction service.
15. Do not add Kubernetes, Kafka, Spark, or a feature store during core development.

## Standard loop structure

Every Claude Code instruction should contain:

```text
LOOP NAME
OBJECTIVE
WHY THIS LOOP EXISTS
FILES TO READ
FUNCTIONAL REQUIREMENTS
NON-FUNCTIONAL REQUIREMENTS
OUT OF SCOPE
TESTS TO ADD
COMMANDS TO RUN
ACCEPTANCE CRITERIA
DOCUMENTATION TO UPDATE
STOP CONDITION
```

Claude must stop at the acceptance boundary instead of continuing into the next loop.

---

# 19. Implementation Loops

## Loop 0: Repository foundation

Build:

* Python package
* Dependency configuration
* Linting
* Type checking
* Testing
* Basic configuration
* Logging
* Project control documents
* Empty application health endpoint

Exit criteria:

* Package installs cleanly
* Tests run
* Ruff and mypy pass
* FastAPI health endpoint responds
* No ML logic yet

## Loop 1: Dataset acquisition and manifesting

Build:

* NASA dataset downloader
* Checksum verification
* Immutable raw storage
* Dataset manifest
* Idempotent acquisition flow

Exit criteria:

* Dataset can be acquired from scratch
* Re-running does not corrupt or duplicate data
* Manifest contains required provenance fields
* Tests use small fixtures rather than downloading the full dataset

## Loop 2: Validation and EDA

Build:

* Data contract
* Parser
* Validation rules
* Processed Parquet output
* EDA notebook
* Dataset report

Exit criteria:

* Invalid fixtures fail clearly
* Real data passes
* Asset and cycle integrity checks work
* EDA findings are documented

## Loop 3: Labels, splits, and features

Build:

* RUL label generation
* Engine-level splitting
* Replay holdout isolation
* FeatureBuilder
* Feature manifests
* Leakage tests

Exit criteria:

* No engine appears in multiple splits
* Future-data leakage tests pass
* Offline and single-asset feature generation match

## Loop 4: Baseline modeling and evaluation

Build:

* Naive baseline
* Ridge model
* Tree-based model
* XGBoost model
* Evaluation framework
* Maintenance-policy simulation
* Conformal interval calibration

Exit criteria:

* All models use identical splits
* Champion selection is metric-based
* XGBoost is not assumed to win
* Evaluation report contains regression and alert metrics
* Business assumptions are configurable and clearly labeled

## Loop 5: MLflow integration

Build:

* Experiment tracking
* Dataset lineage
* Artifact logging
* Registry
* Candidate and champion aliases
* Model card generation

Exit criteria:

* A complete training run is reproducible
* Model can be loaded by registry alias
* Git SHA, dataset checksum, and feature version are logged

## Loop 6: PostgreSQL operational layer

Build:

* Database models
* Alembic migrations
* Repositories
* Assets, sensor readings, predictions, and events
* Idempotency constraints

Exit criteria:

* Migration works on an empty database
* Duplicate sensor cycles are handled safely
* Repository integration tests pass

## Loop 7: FastAPI inference service

Build:

* Sensor-ingestion endpoint
* Prediction endpoint
* Asset-health endpoints
* Model loader
* Shared online feature generation
* API documentation
* Metrics instrumentation

Exit criteria:

* A sensor reading can travel from request to database to model to stored prediction
* Responses contain model version
* Invalid data receives a clear response
* Readiness fails when the model or database is unavailable

## Loop 8: Replay and delayed feedback

Build:

* Held-out trajectory replay
* Configurable replay speed
* Failure-event emission
* Delayed label backfill
* End-to-end lifecycle test

Exit criteria:

* Prediction service never sees future rows
* Failure labels become available only after the event
* Historical online predictions can be evaluated correctly

## Loop 9: Monitoring and retraining

Build:

* Data-quality reports
* Drift calculations
* Online performance reports
* Retraining trigger
* Candidate evaluation gate
* Promotion workflow

Exit criteria:

* Drift can be deliberately induced in a test
* Poor candidate is rejected
* Acceptable candidate can receive the champion alias
* Promotion event is auditable

## Loop 10: Containers and CI/CD

Build:

* Production Dockerfile
* Docker Compose stack
* GitHub Actions
* Service health checks
* Container smoke tests

Exit criteria:

* Fresh clone can start with documented commands
* CI builds and tests the image
* API, database, MLflow, worker, and replay communicate correctly

## Loop 11: Dashboard and public deployment

Build:

* Fleet dashboard
* Asset-health view
* Alert history
* Model and drift information
* Render deployment configuration
* Deployment documentation

Exit criteria:

* Public API is reachable
* Public dashboard demonstrates continuous replay
* Secrets are not committed
* Deployment occurs only after CI passes

## Loop 12: Portfolio finishing

Build:

* Final README
* Architecture diagram
* Metrics table
* Demo GIF or video
* Model card
* Limitations
* Scaling design
* Resume-ready project summary

Exit criteria:

* A recruiter can understand the problem, architecture, model, and outcome within one minute
* One-command local reproduction works
* All reported metrics can be regenerated
* No confidential industrial information is present

---

# 20. Definition of Done

The project is complete when:

* Raw data acquisition is reproducible.
* Data lineage is recorded.
* Splits occur by engine, not row.
* Leakage tests pass.
* At least three meaningful model baselines are compared.
* Predictions include calibrated uncertainty.
* Maintenance metrics complement regression metrics.
* A continuous replay produces predictions.
* Failure labels arrive with a delay.
* Monitoring evaluates realized outcomes.
* Retraining uses newly labeled data.
* Candidate models pass an explicit gate.
* MLflow stores runs and model versions.
* FastAPI serves a champion model.
* PostgreSQL stores online state.
* Docker Compose starts the complete system.
* GitHub Actions validates every change.
* A public dashboard demonstrates the system.
* Documentation explains deliberate architecture tradeoffs.
* The README distinguishes public simulation from the Accenture engagement.

---

# 21. Design Review and Rework

## Initial design considered

The initial architecture included:

* Kafka
* Airflow
* Feast
* Kubernetes
* Grafana
* MLflow
* FastAPI
* PostgreSQL
* Spark
* Separate React frontend

### Initial score: 91.2/100

Main weaknesses:

* Too many infrastructure tools
* More time spent configuring services than understanding ML systems
* Kafka unnecessary for one replay producer
* Kubernetes unnecessary for portfolio traffic
* Feature store unnecessary for one model
* Spark unnecessary for dataset size
* Separate frontend distracts from ML engineering
* Harder for recruiters to run locally
* Greater chance of unfinished or fragile components

## Revisions

Removed:

* Kafka
* Kubernetes
* Spark
* Feast
* Grafana as a core dependency
* Separate frontend framework
* Airflow in favor of lighter Prefect workflows

Retained:

* Data lineage
* Delayed labels
* Training-serving consistency
* Experiment tracking
* Model registry
* Workflow orchestration
* Continuous ingestion
* Monitoring
* Automated evaluation gates
* API deployment
* CI/CD
* Containers
* Operational database
* Public demo

The result demonstrates architecture judgment rather than infrastructure accumulation.

---

# 22. Final Score

| Category                                 |  Weight |    Score |
| ---------------------------------------- | ------: | -------: |
| Problem framing and industrial realism   |      10 |      9.8 |
| Data engineering and continuous feedback |      20 |     19.6 |
| Modeling and evaluation rigor            |      15 |     14.8 |
| MLOps and model lifecycle                |      20 |     19.6 |
| Deployment and system design             |      15 |     14.5 |
| Software engineering quality             |      10 |      9.8 |
| Demo and recruiter clarity               |       5 |      4.9 |
| Complexity discipline                    |       5 |      5.0 |
| **Total**                                | **100** | **98.0** |

## Final assessment

This plan scores above 97 because it:

* Represents a genuine end-to-end ML system.
* Contains a meaningful delayed-feedback loop.
* Demonstrates model and data lifecycle ownership.
* Includes rigorous evaluation beyond accuracy.
* Shows training-serving consistency.
* Includes deployment and observability.
* Uses highly recognizable MLE technologies.
* Can be explained clearly in interviews.
* Avoids unnecessary distributed infrastructure.
* Is complex enough to demonstrate high technical ability while remaining realistically finishable.
