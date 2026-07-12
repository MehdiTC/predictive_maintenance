# ADR 0006 — Loop 7 online inference, ordering, cache, and transaction policy

**Status:** Accepted (2026-07-12)

## Context

Loop 7 connects the established FastAPI factory, PostgreSQL repositories, shared feature builder,
and registered MLflow champion. The service must prevent training-serving skew and partial writes
without implementing replay, delayed feedback, monitoring calculations, retraining, orchestration,
or deployment.

## Decisions

### Automatic generic asset creation

The first accepted cycle automatically creates an active asset using the supplied external ID.
It does not label the asset as an engine or infer dataset metadata. The reading preserves its
source. New assets must begin at cycle 1. PostgreSQL uniqueness and a row lock serialize ingestion
for the same asset, including concurrent first requests.

### Strict contiguous history

New readings must be exactly the next cycle. Gaps, previously missing older cycles, and new
out-of-order records return `409 history_conflict`. An exact retry of any existing cycle remains
valid; changed data for that cycle returns `409 sensor_reading_conflict`. This conservative policy
avoids silently changing historical rolling features and predictions. A future explicit backfill
workflow may recompute affected predictions, but that belongs after Loop 8.

### One atomic ingestion transaction

Asset resolution/creation, reading persistence, history query, feature construction, model
prediction, and prediction persistence share one caller-owned SQLAlchemy transaction. Feature,
model, or database failure rolls back the asset (if new), reading, and prediction. This prioritizes
a simple invariant: every reading accepted through the prediction endpoint has a corresponding
champion result. A future raw-ingestion endpoint may deliberately use a separate transaction.

### Shared feature reconstruction

The service reads only the same asset and only cycles through the submitted cycle, explicitly
ordered. `FeatureBuilder` is constructed from the verified Loop 3 feature manifest. Its version and
ordered columns must match the MLflow signature. Early-cycle structural nulls are preserved and
handled by the already-fitted model pipeline.

### Lazy, lock-protected champion cache

One champion is loaded per process from `models:/<registered-model>@champion`. Loading is lazy and
optionally preloaded during application lifespan; no registry access occurs on import. A reentrant
lock prevents duplicate concurrent loads. Registry version, source run, lineage, model evidence,
feature version, signature, and load time are captured. `refresh()` is explicit; automated model
watching or promotion is not implemented.

### Idempotent prediction behavior

The existing Loop 6 identities remain authoritative. An identical reading retry returns the same
reading. If that reading already has a prediction from the current model version, the prediction is
returned without inference (`200`). If the champion version changes, the existing reading may gain
a new version-pinned prediction (`201`). The response distinguishes reading, prediction, and whole
request idempotency.

### Required readiness and per-app metrics

When online inference is enabled, readiness requires PostgreSQL, champion loadability, and
feature-contract compatibility. Startup records preload failure but still serves liveness and a
failing readiness response. Tests may disable online resources or inject checks. Prometheus uses a
collector registry per app instance, avoiding duplicate collectors and global mutable test state.
Labels never include asset IDs, request IDs, payloads, or raw errors.

### No direct prediction endpoint

`POST /v1/predictions` is deferred. With one configured champion, it would expose the same operation
as idempotently resubmitting a stored cycle while adding another input contract. Explicit historical
re-scoring becomes meaningful when model comparison/promotion workflows exist.

## Dependency decision

`prometheus-client>=0.21,<1` is the only direct addition. Loop 7 imports its registry, counter,
histogram, gauge, info, and exposition APIs; relying on MLflow to install it transitively would make
the application dependency accidental. No authentication framework, async database driver, model
server, cache, worker, orchestration, replay, or container dependency is added.

## Consequences

Inference is consistent and auditable but holds an asset row lock during feature construction and
prediction. That is acceptable for current trajectory lengths and one-process demo traffic.
Metrics are process-local; horizontally scaled aggregation requires Prometheus scraping. Automatic
asset creation is convenient for replay and clients but means production deployments need
authentication, authorization, rate limits, and an asset enrollment policy before accepting
untrusted traffic.
