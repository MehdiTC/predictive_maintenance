# TurbineGuard Architecture

TurbineGuard is an independently developed predictive-maintenance ML platform inspired by industrial
power-generation use cases. It uses the public NASA C-MAPSS FD001 turbofan degradation dataset and
contains no proprietary client data or implementation details.

This document describes the major components, their data flow, and the lifecycle of an online
prediction. Practical commands live in the [operations runbook](operations.md).

## Problem, data, and target

* **Problem.** Predict Remaining Useful Life (RUL) for each engine so that maintenance can be
  planned before failure, and turn continuous RUL predictions into timely warning/critical alerts.
* **Data.** NASA C-MAPSS FD001: 100 training engines run to failure and 100 test engines truncated
  before failure, each a multivariate trajectory of 3 operating settings and 21 anonymous sensors
  sampled per operating cycle. The data is *simulated*; sensor channels are never assigned physical
  meanings.
* **Target.** `RUL = final_cycle − current_cycle`, optionally capped at 125 cycles. The deployed
  champion predicts the capped target.

![Offline data foundation and engine-level splits](assets/data-splits.svg)

## Component overview

![TurbineGuard system overview](assets/system-architecture.svg)

Key structural properties encoded in this diagram:

* **One shared feature implementation.** The same `FeatureBuilder` produces features for offline
  training and for each online request; there is no separate notebook/serving fork. This is the
  primary defense against training-serving skew.
* **Two interchangeable champion sources.** Locally, the API loads the champion from the MLflow
  registry by alias. In the zero-cost public demo, it loads an immutable, checksum-pinned
  deployment bundle exported from that same registered version — no MLflow process runs publicly,
  so the public registry cannot be mutated. Both implement one `ChampionLoader` protocol and were
  verified to produce identical predictions (max difference 0.0).
* **Leakage boundaries are structural.** Data is split by engine, not by row; replay and
  calibration engines are held out of training; features at cycle `t` depend only on observations
  `≤ t`; and the replay path never exposes the failure cycle to inference.

## Data and control flow (offline → online → feedback)

1. **Acquire.** Download the C-MAPSS archive to an immutable, read-only, checksum-verified raw
   layer and write a provenance manifest.
2. **Process & validate.** Parse the whitespace-delimited files into a typed canonical schema, run
   structural/semantic validation, and publish validated Parquet plus a machine-readable report. A
   failed required check blocks publication.
3. **Feature build.** Generate RUL labels, deterministic asset-level splits (70 train / 15
   validation / 5 calibration / 10 replay), and 552 leakage-safe trailing-window features, with
   split and feature manifests.
4. **Train & select.** Fit four model families (constant baseline, Ridge, histogram gradient
   boosting, XGBoost) over uncapped and capped-125 targets on identical splits. Select the champion
   on validation only, using explicit operational gates and a complexity-tolerance rule. Calibrate
   conformal intervals on the calibration split only.
5. **Track & register.** Log every candidate to MLflow and register the champion with `candidate`,
   `challenger`, `champion` (and later `archived`) aliases, full lineage, and checksums.
6. **Serve.** FastAPI ingests one sensor cycle at a time, rebuilds features through the shared
   builder, predicts point/interval RUL and risk class with the loaded champion, and stores the
   version-pinned prediction in PostgreSQL atomically with the reading.
7. **Replay & delayed feedback.** Held-out engine trajectories are streamed one cycle at a time
   through the real ingestion API. When a trajectory ends, a failure event is emitted, realized RUL
   labels are backfilled for every historical prediction, and delayed evaluations are persisted.
8. **Monitor & retrain.** Data-quality, all-feature drift, and delayed-performance reports run
   against the champion's training-only reference and yield an explicit `no_action` / `monitor` /
   `retrain` / `blocked` decision. Retraining is leakage-safe and reuses the modeling pipeline fit path;
   candidates must clear every blocking promotion gate (including MLflow reload equivalence) and,
   by default, human approval before the champion is replaced.

### Training, calibration, and registration

![Training, calibration, champion selection, and registration](assets/training-registration.svg)

### Monitoring, retraining, and promotion

![Delayed evaluation and guarded model promotion](assets/monitoring-promotion.svg)

## Serving one online prediction

This is the request path exercised on every `POST /v1/sensor-readings` — including every cycle the
replay client sends.

![Online serving request and persistence path](assets/serving-request.svg)

The serving process supports two interchangeable sources for the same approved champion:

![Champion model sources and prediction contract](assets/model-loading.svg)

The failure cycle and future observations are never available on this path: the replay source
reveals only cycles `≤ t`, and the final cycle is stored only in a replay-state table that no
prediction endpoint reads. Identical repeated cycles are idempotent, conflicting duplicates return
an error, and a feature or model failure rolls back the sensor-reading transaction.

## Deployment topologies

| Concern | Local reference (Docker Compose) | Public demo (pinned bundle) |
| --- | --- | --- |
| API | Container in shared image | One free Render web service |
| Operational DB | PostgreSQL 17 container | External Neon free-tier PostgreSQL |
| Champion source | Live MLflow registry (alias) | Immutable checksum-pinned bundle |
| MLflow service | Runs (SQLite metadata volume) | None (registry mutation impossible) |
| Retraining/promotion | Demonstrable via lifecycle CLI | Read-only snapshot; not available |
| Migrations | One-shot `migrate` service | Alembic at container start |

Both topologies run the same application image and code; only the champion source and the presence
of the MLflow/retraining plane differ. Setup and verification commands are in the
[operations runbook](operations.md).

## Deliberate non-goals

Kafka, Kubernetes, Spark, a feature store, and a separate frontend framework are intentionally out
of scope for this single-model reference system. The current boundaries—idempotent ingestion,
stateless serving, a shared feature builder, and registry aliases—leave clear seams for those
components if scale or organizational requirements justify them.
