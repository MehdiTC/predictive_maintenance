# Scaling Paths (Design Only)

This document describes how TurbineGuard *would* scale under real production load. Nothing here is
implemented — it is a deliberate record of the architecture judgment behind the current, smaller
system, and the conditions under which each heavier component would earn its place.

The core project intentionally excludes Kafka, Kubernetes, Spark, a feature store, and a separate
frontend framework. The initial design included all of them and scored poorly for spending more
effort configuring infrastructure than demonstrating ML systems engineering. Each was removed
because a single-model, portfolio-scale system does not need it. The value is in knowing *when* the
tradeoff flips.

## Trigger conditions and the component each unlocks

| Current design | Scale trigger that would justify a change | Heavier component to add |
| --- | --- | --- |
| One replay producer over HTTP | Many real asset feeds, backpressure, replay/audit of the raw stream | Kafka (or a managed equivalent) as the ingestion log |
| One FastAPI process on one host | Sustained concurrent traffic beyond one instance; need rolling deploys, autoscaling | Kubernetes / a managed autoscaler + multiple stateless API replicas |
| Pandas/NumPy batch on one machine | Feature build no longer fits comfortably in memory / wall-clock | Spark or Dask for distributed feature computation |
| Shared `FeatureBuilder`, no store | Many models/teams need consistent, low-latency shared features online | A feature store (e.g. Feast) for online/offline parity at scale |
| Server-rendered dashboard | Rich, interactive, multi-user operator UI is a first-class product | A separate frontend framework + API gateway |
| Single-host SQLite MLflow | Many concurrent writers, HA registry, org-wide model governance | Managed MLflow / registry on a replicated database |
| Per-process champion cache | Coordinated hot model reload across many replicas | Shared cache / model-server sidecar + broadcast invalidation |

## Ingestion at scale (Kafka)

Today a single replay client POSTs one cycle at a time to `/v1/sensor-readings`, and idempotency is
enforced by database uniqueness on `(asset_id, cycle)`. With many real producers you would put a log
(Kafka) between producers and the API: producers append raw readings to a partitioned topic keyed by
`asset_id` (preserving per-asset ordering), a consumer group performs feature build + prediction,
and the same `(asset_id, cycle)` uniqueness key provides exactly-once persistence. The important
point is that the current idempotency contract already survives this move — the log replaces the
direct HTTP call without changing the correctness model.

## Compute at scale (Kubernetes, Spark)

The API is already a stateless process with real readiness checks (database, champion, feature
contract) and clean SIGTERM shutdown, so horizontal replication behind a load balancer is a
packaging change, not a redesign. Migrations are already owned by a single one-shot step, which is
exactly the pattern a Kubernetes init-container/Job would use. Distributed feature computation
(Spark/Dask) becomes worthwhile only when the offline build stops fitting on one machine; because
the `FeatureBuilder` is a pure per-asset function, it partitions by `asset_id` without changing
feature semantics.

## Features at scale (feature store)

A feature store earns its complexity when multiple models and teams need consistent features served
online with low latency and guaranteed offline/online parity. TurbineGuard achieves that parity today
with one shared builder used by both training and serving, which is sufficient for one model. The
migration path is to register the existing feature definitions as the store's transformations,
keeping the builder as the single source of truth for the computation.

## Registry and governance at scale

The single-host SQLite MLflow store is fine for one maintainer but is not highly available. At
organizational scale it would move to managed MLflow backed by a replicated database, with the
existing alias model (`candidate` / `challenger` / `champion` / `archived`), blocking promotion
gates, and approval-by-default carried over unchanged — those are already the governance primitives,
independent of where the metadata lives.

## Guiding principle

Add a heavy component only when a concrete load or organizational trigger makes the standard library
or an existing dependency insufficient — the same dependency policy the project applies loop to loop.
The current system is designed so that each of these moves is an additive change at a clean seam
(the ingestion contract, the stateless API, the pure feature function, the alias-based registry),
not a rewrite.
