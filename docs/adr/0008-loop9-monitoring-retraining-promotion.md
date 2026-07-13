# ADR 0008 — Loop 9 monitoring, retraining, promotion, and recovery

**Status:** Accepted (2026-07-13)

## Context

Loop 9 must turn delayed Loop 8 outcomes into production lifecycle decisions without duplicating
Loop 4 modeling/metrics, Loop 5 MLflow packaging, Loop 7 loading, or Loop 8 evaluation. It must keep
protected offline roles out of retraining, compare candidate and champion on one new asset-level
holdout, require approval by default, and recover safely across PostgreSQL and MLflow mutations.

## Decisions

### Training-only, champion-bound reference

The verified Loop 3 `train.parquet` is the only drift reference. A compact artifact records all 552
feature distributions: missingness, moments, decile histogram proportions, and 101 quantiles. Its
stable content identity is bound to the exact MLflow model and feature version and tagged/logged on
that version. Validation, calibration, replay, and official NASA test roles never enter the
reference. PSI, one-dimensional Wasserstein distance (quantile-integral form), missingness shift,
standardized mean shift, and standardized standard-deviation shift are explicit NumPy/Pandas code.

### Existing report tables plus focused lifecycle additions

`drift_reports`, `model_evaluations`, and `pipeline_runs` remain authoritative. Revision
`20260713_0003` adds a first-class `data_quality_reports` table, a unique pipeline idempotency key
and phase, asset-role assignments, and append-only lifecycle events. Report insertion and phase
checkpoint share one database transaction. Re-entering a phase reads the checkpoint or verifies
the existing artifact/version; no report, model version, event, or alias change is duplicated.

### Leakage-safe retraining and frozen uncertainty policy

The point model fits the original training role plus eligible completed labeled operational assets.
New assets are deterministically split by SHA-256 identity into retraining additions and a disjoint
promotion holdout. Previously promoted additions remain in the cumulative fit base. Protected
validation/calibration data and official test data are not opened by the retraining loader.

The initial Loop 9 candidate is the champion's existing Loop 4 family, hyperparameters, target,
features, and horizons. No new family or search is introduced. The candidate carries forward the
champion's frozen conformal calibrator without rereading protected calibration data; empirical
coverage and width on the new promotion holdout are blocking gates. Recalibrating from future new
labeled assets would require a separately isolated calibration role and is intentionally deferred.

### Explicit triggers and blocking gates

Trigger output is exactly `no_action`, `monitor`, `retrain`, or `blocked`. Manual force affects the
trigger signal but never bypasses data quality, minimum rows/assets, or safe-holdout requirements.
Candidate, frozen champion, and Loop 4 constant-median baseline use the same holdout frame. Every
gate is independently recorded; there is no weighted score that can mask a safety failure.

### Alias order, approval, rollback, and serving refresh

Candidate registration follows: register, assign `candidate`, reload/equivalence check, assign
`challenger`, evaluate gates, then await explicit approval by default. Approval moves the prior
champion to `archived` before assigning the candidate to `champion`. Rejection never changes
`champion`. Rollback validates a numbered version, archives the displaced champion, and restores
the target. Alias operations are ordered and resumable rather than transactionally atomic because
MLflow exposes no multi-alias transaction.

Loop 7 refresh now loads and validates a replacement before swapping the process cache. Failed
refresh therefore leaves the working model object intact. Refresh is explicit per process; Loop 10
deployment/process coordination remains out of scope.

## Dependency decision

No dependency is added. NumPy, Pandas, scikit-learn, MLflow, SQLAlchemy, Alembic, and PostgreSQL
already provide the required behavior. No monitoring platform, scheduler, queue, cache, container,
or distributed orchestration system is introduced.

## Consequences

The lifecycle is auditable and deterministic, but a 30-day default window is an application
default rather than a claim about physical turbine time. PostgreSQL contains accepted readings;
rejected-record counts must be supplied by the window producer because Loop 7 predates a durable
request-audit table. Local MLflow SQLite and per-process model caches remain single-machine
constraints until deployment work in later loops.
