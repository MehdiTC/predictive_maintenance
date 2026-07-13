# Monitoring and model lifecycle (Loop 9)

## Monitoring window and reference

`model_lifecycle monitor` uses a configurable 30-day half-open UTC window by default; explicit
`--window-start` and `--window-end` make demonstrations and reruns exact. Accepted raw readings are
checked for counts, duplicates, missing/non-finite/out-of-range values, sequence gaps/order,
sufficient history, and per-sensor availability. PostgreSQL rejects duplicate/missing/non-finite
accepted rows structurally; the report still calculates and records every check.

Production features are reconstructed from each selected asset's complete history with the shared
Loop 3 `FeatureBuilder`, then restricted to window rows. The reference is built only from the
checksummed original `train.parquet` and stored under
`data/monitoring/references/<model>/v<version>/training_reference.json`. It is associated with the
exact champion/feature version in MLflow. Every feature receives PSI over training-quantile bins,
one-dimensional Wasserstein distance from empirical quantiles, absolute missingness shift, and
mean/standard-deviation shifts normalized by training standard deviation.

Delayed labels that became available in the window are joined to immutable predictions for the
exact champion version. Loop 4 regression, alert/episode/lead-time, and conformal functions produce
MAE, RMSE, NASA score, critical/warning precision and recall, false alarms, lead time, coverage,
interval width, prediction distribution, and risk distribution. Reports persist in
`data_quality_reports`, `drift_reports`, and `model_evaluations`.

## Trigger policy

The decision is one of `no_action`, `monitor`, `retrain`, or `blocked`. Configurable signals cover
minimum new labeled assets/rows, elapsed interval, relative MAE/RMSE loss, critical recall, false
alarms, interval coverage, drifted-feature count, data-quality failure, and manual force. A trigger
with too little data or no safe holdout is `blocked`; force does not override safety.

Defaults include five new labeled assets, 500 rows, two promotion-holdout assets, 30 days, 15%
performance degradation, 0.60 critical recall, 0.85 coverage, and three drifted features. All
thresholds are typed `TURBINE_GUARD_*` settings in `.env.example`.

## Retraining and comparison

The point-model fit base is the verified original Loop 3 training role plus prior successfully
promoted operational additions plus this run's new retraining additions. New completed labeled
assets are deterministically assigned at asset level to retraining additions or promotion holdout.
No asset can occupy both roles in one run. The official NASA test and existing
validation/calibration roles are never opened by the retraining loader.

The initial candidate is only the champion's established Loop 4 family/configuration and target.
Its frozen interval policy is carried forward without rereading calibration data; holdout coverage
remains blocking. Candidate, non-refitted champion, and Loop 4 median baseline receive the exact
same holdout rows and ordered features. The comparison records a holdout fingerprint, MAE, RMSE,
NASA score, critical precision/recall, false alarms, lead time, coverage, width, latency, and size.

## Gates, aliases, approval, and rollback

Blocking gates require data quality, enough data, a valid bundle, beating the naive baseline,
bounded RMSE/NASA regression, minimum recall/coverage, bounded false alarms/latency/size, and exact
MLflow reload equivalence. Registry order is:

```text
register -> candidate -> reload/equivalence -> challenger -> gates -> approval
approval: old champion -> archived; candidate -> champion
rejection: champion unchanged
rollback: validate numbered version; current -> archived; target -> champion
```

Approval is required by default. `lifecycle_events` is append-only and records gates, rejection,
approval, aliases, refresh results, actors, reasons, and rollback. The Loop 7 cache loads a
replacement before swapping; a load failure keeps the current object.

## Recovery and commands

`pipeline_runs.idempotency_key` identifies an exact monitoring window or retraining input set.
`phase` and JSON metadata checkpoint assignment, fit artifact/checksum, comparison, registry
version/equivalence, gates, approval, aliases, and refresh. Database report/event writes share the
checkpoint transaction. MLflow runs and versions carry the lifecycle UUID, so a restart reuses
them instead of creating duplicates.

```bash
uv run alembic upgrade head
uv run python scripts/model_lifecycle.py monitor
uv run python scripts/model_lifecycle.py status
uv run python scripts/model_lifecycle.py force-retraining
uv run python scripts/model_lifecycle.py evaluate-candidate
uv run python scripts/model_lifecycle.py dry-run-promotion --run-id <UUID>
uv run python scripts/model_lifecycle.py approve-promotion --run-id <UUID>
uv run python scripts/model_lifecycle.py reject-candidate --run-id <UUID> --reason "reason"
uv run python scripts/model_lifecycle.py rollback --version <N>
uv run python scripts/model_lifecycle.py refresh-serving-model
```

## Limitations

The default MLflow backend is local SQLite and model-cache refresh is per process. No schedule
runner, distributed lock service, dashboard, container, deployment, or Loop 10 functionality is
included. Rejected HTTP-request counts are not durably available from the preexisting Loop 7 API;
the quality calculator accepts an explicit count and labels its source. C-MAPSS remains simulated
public data, and thresholds are demonstration defaults rather than industrial limits.
