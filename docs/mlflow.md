# MLflow Experiment Tracking and Registry (Loop 5)

Loop 5 records the existing Loop 4 pipeline in MLflow. It does not change candidate definitions,
training, validation-only selection, conformal calibration, replay evaluation, official-test
evaluation, or normalized-cost policy simulation.

## Architecture and experiment structure

```text
verified Loop 4 local execution
  -> parent MLflow run (lineage + complete-execution summary)
     -> one nested child per target/model candidate
     -> selected child gets held-out metrics, reports, model card, and pyfunc
  -> verified champion model version
  -> candidate / challenger / champion aliases
```

The default experiment is `TurbineGuard-FD001-Offline-Modeling`; the default registered model is
`TurbineGuard-FD001-RUL`. Parent names are
`fd001-offline-<training-manifest-sha-prefix>`; child names append the candidate ID.

Host-only development defaults store experiment/registry metadata in SQLite at
`data/mlflow/mlflow.db`, with artifacts under `data/mlflow/artifacts`. Compose instead runs an MLflow
HTTP service with a persistent SQLite metadata volume and a separate proxied filesystem artifact
volume. The locked MLflow 3.14 registry currently emits a PostgreSQL type-mismatched model-version
lookup, so the supported single-host Compose backend remains SQLite rather than sharing the
operational PostgreSQL database. Both modes use the same tracking/artifact configuration; modeling
code does not change. Neither local topology is high availability.

## What is logged

Every run carries project, dataset/subset, environment, Git SHA, code version, random seed,
feature/split/evaluation versions, timestamp, and SHA-256 lineage for raw acquisition, validation,
split, feature, training configuration, and completed training artifacts.

Candidate child parameters include model family and hyperparameters, target/cap, preprocessing,
missing indicators, scaling, feature count, rolling windows, alert horizons, seed, threshold rule,
and conformal target coverage. The full candidate and training configurations are JSON artifacts;
the 552 feature names are one ordered JSON artifact rather than 552 parameters.

Candidate metrics use these prefixes:

* `validation/` and `validation/common_domain/`: MAE, RMSE, R², NASA score.
* `validation/critical/` and `validation/warning/`: precision, recall, F1, PR-AUC, false alarms,
  lead time, timely-warning percentage, and missed failures.
* `performance/`: training time, inference latency, and artifact size.
* `selection/`: numeric eligibility and deterministic common-domain rank.

The selected child additionally logs `calibration/`, `replay/`, `official_test/`, and `policy/`
metrics. Artifacts include candidate pipelines/configs; all champion reports; the local champion
bundle and metadata; ordered features; alert/conformal/policy contracts; lineage/checksum files;
dependency snapshots; a model card; and the packaged MLflow model. Raw or processed NASA datasets
are never copied into MLflow.

## Model contract

The MLflow flavor is a custom `pyfunc` that loads the checksummed Loop 4 `ModelBundle`. It reuses the
saved training-only Ridge preprocessing, cap, conformal state, and thresholds. Output columns are:

* `predicted_rul`
* `lower_rul`
* `upper_rul`
* `risk_level` (`healthy`, `warning`, or `critical`)

The signature requires exactly the ordered Loop 3 model features and excludes `asset_id`, `cycle`,
`split`, and `rul`. Missing required inputs fail. MLflow named-schema enforcement reorders expected
columns and ignores extra columns before calling the wrapper; this is intentional and tested. A
small complete feature input example and serving input example are stored with the model.

## Registry and aliases

Registration is enabled by default for tracked training. The source is the selected candidate child
run. Version tags include source run, Git SHA, manifest checksums, target, selection rationale,
validation/replay/official RMSE, registration time, and champion checksum.

Alias meanings:

* `candidate`: newest verified version presented to the registry.
* `challenger`: verified and ready for comparison.
* `champion`: passed the existing Loop 4 eligibility/selection gates and promotion is enabled.
* `archived`: most recently displaced champion; older versions remain preserved by number.

Loop 9 extends this contract: a retrained version receives `candidate`, then must reload with exact
prediction equivalence before receiving `challenger`. Explicit gates run before approval; approval
archives the displaced champion and assigns the new `champion`; rejection does not move it. Manual
rollback validates a numbered version before the same archive/champion transition. Lifecycle UUID
and bundle tags prevent duplicate runs/versions during recovery.

## Commands

Tracked training (an unchanged local run is verified and may be tracked without retraining):

```bash
uv run python scripts/train_models.py --track-with-mlflow
```

Explicit new run history or model version:

```bash
uv run python scripts/train_models.py --track-with-mlflow --force-mlflow-run
uv run python scripts/train_models.py --track-with-mlflow --force-mlflow-run \
  --force-new-model-version
```

Launch the UI for the local store:

```bash
uv run mlflow ui --backend-store-uri sqlite:///data/mlflow/mlflow.db --port 5000
```

Inspect parent/child runs, candidate metrics, versions, source runs, and aliases:

```bash
uv run python scripts/mlflow_models.py inspect
```

Load by champion alias or exact version and compare against the local bundle:

```bash
uv run python scripts/mlflow_models.py verify --alias champion
uv run python scripts/mlflow_models.py verify --version 1
```

List configured local runtime state:

```bash
uv run python scripts/mlflow_models.py state
```

All commands use typed environment settings and exit nonzero on failure. Set
`TURBINE_GUARD_MLFLOW_REGISTRATION_ENABLED=false` together with
`TURBINE_GUARD_MLFLOW_PROMOTE_CHAMPION=false` to track without registry mutation.

## Idempotency and safety

A completed training-manifest checksum identifies the parent execution. An identical finished run
returns `already_logged`. The champion-bundle checksum prevents an unnecessary duplicate registered
version even when a new parent/child run is explicitly requested. Force controls are opt-in and do
not overwrite runs or delete versions. All Loop 4 artifacts are checksum-verified first; tampering
fails. Joblib, cloudpickle, and MLflow pyfunc loading can execute code, so only trusted stores and
artifacts should be loaded.
