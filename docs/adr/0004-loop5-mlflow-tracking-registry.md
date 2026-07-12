# ADR 0004 — Loop 5 MLflow tracking, pyfunc packaging, registry aliases, and idempotency

**Status:** Accepted (2026-07-12)

## Context

Loop 4 already owns trustworthy modeling: it verifies lineage, trains every bounded candidate,
selects on validation only, calibrates uncertainty, evaluates held-out roles, and writes a
checksummed local completion manifest. Loop 5 must add experiment comparison and a model registry
without making that logic depend on MLflow or weakening local reproducibility.

MLflow's registry requires a database-backed backend. A legacy file-only backend is therefore not
enough for the required local aliases and version loading.

## Decisions

### Post-completion adapter and parent/child runs

MLflow integration is an explicit adapter over a completed Loop 4 artifact set. The ordinary
`train_models()` function remains unaware of MLflow. `--track-with-mlflow` first trains or verifies
the local outputs, then creates one parent run for that execution and one nested child per candidate.
The selected child additionally receives calibration, replay, official-test, policy, reports, and
the packaged model. This avoids scattering global `mlflow.log_*` calls through modeling code and
also allows a previously completed local run to be tracked without retraining.

### SQLite metadata and local artifacts

The development default is `sqlite:///data/mlflow/mlflow.db`, with artifacts under
`data/mlflow/artifacts`. SQLite is embedded, supports the registry, and needs no service. Both the
tracking URI and artifact location are typed settings; the adapter uses ordinary MLflow clients and
can point at a remote server without a modeling change. SQLite is for one-machine development, not
concurrent production workloads or shared durable storage.

### Custom pyfunc over the existing bundle

The registered flavor is a small MLflow `pyfunc`, not bare `mlflow.sklearn`. The Ridge pipeline is
sklearn-compatible, but the required prediction contract also includes target clipping, conformal
intervals, alert thresholds, and risk classification already owned by `ModelBundle`. The wrapper
checksum-verifies and loads that bundle, calls its `predict`/`predict_interval` methods, and adds no
duplicate preprocessing. It returns point/lower/upper RUL and risk level.

The explicit signature contains exactly the 552 ordered feature-manifest columns and excludes
identifiers, split metadata, and targets. Missing columns fail MLflow schema enforcement. MLflow's
documented named-schema behavior reorders expected columns and ignores extra columns; that is the
explicit extra-column policy. The wrapper still requires the sanitized order to equal the manifest.

### Registry alias semantics

After registration and load/prediction verification, the new version receives `candidate` and
`challenger`. It receives `champion` only when the recorded Loop 4 selected candidate is eligible and
promotion is enabled. A displaced champion remains immutable and receives `archived`; no historical
version is deleted. `archived` identifies the most recently displaced champion, while all older
versions remain queryable by number and tags.

### Two-level idempotency

The SHA-256 of the completed `training_manifest.json` identifies an exact tracked execution. A
finished parent run with this tag returns `already_logged`. Registration separately deduplicates on
the verified champion-bundle SHA-256, so a forced tracking run need not create another model version.
`--force-mlflow-run` creates new run history; `--force-new-model-version` explicitly creates a new
version. Neither deletes or overwrites history. Local tampering fails before logging or registration.

## Dependency decision

`mlflow>=3.5,<4` is the only new direct dependency. It supplies tracking, SQLite-backed registry
access, artifact/model packaging, signatures, aliases, and loading. Its transitive packages are
MLflow implementation requirements; no PostgreSQL, SQLAlchemy, Alembic, cloud SDK, object-store,
serving, monitoring, orchestration, or Docker dependency is added directly in this loop.

## Consequences

* Every Loop 4 candidate becomes independently searchable and comparable without changing fitting.
* Local artifacts remain the authoritative reproducible source and can be used with tracking off.
* The registered model exposes the richer operational prediction contract and can be loaded by alias
  or version.
* Local SQLite and filesystem artifacts are not suitable for multi-host concurrency, high
  availability, or ephemeral deployments; a remote MLflow server/artifact store is a configuration
  change reserved for later deployment work.
