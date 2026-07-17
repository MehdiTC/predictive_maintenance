# ADR 0010 — Loop 11 dashboard, public replay policy, and Render persistence

**Status:** Accepted (2026-07-13). The paid Render topology and persistence decisions below were
superseded by ADR 0011 (2026-07-14): the public demo now uses one free Render web service, external
Neon PostgreSQL, and an immutable checksum-pinned deployment bundle instead of a live MLflow
service and persistent disks. The dashboard, read-API, and public replay-policy decisions remain
accepted and implemented.

## Context

TurbineGuard already has a typed FastAPI service, PostgreSQL operational history, a durable replay
state machine, persisted monitoring reports, and an MLflow champion. Loop 11 must make those
capabilities understandable in a public demo without creating a second frontend application,
copying business logic, exposing protected split assets, or placing state on ephemeral cloud disks.

Render persistent disks attach to only one paid service. They are not available to build or
pre-deploy commands and prevent horizontal scaling. The locked MLflow version also has a verified
PostgreSQL model-version incompatibility recorded in ADR 0009.

## Decisions

### Server-rendered pages over shared read services

FastAPI serves Jinja pages under `/dashboard` and typed JSON projections under `/v1`. A single
read-side `DashboardService` owns bounded, deterministic PostgreSQL and registry projections used
by both surfaces. Templates contain no database queries or prediction logic. CSS is local; small
JavaScript modules progressively enhance RUL, interval, sensor, and drift charts with Plotly. Tables,
labels, empty states, and core navigation remain usable when chart JavaScript is unavailable.

No React application, frontend build tool, CSS framework, or Python Plotly dependency is added.
Jinja2 is made a direct dependency instead of relying on MLflow's transitive installation.

### Public replay is a policy wrapper, not a second state machine

All mutations call the Loop 8 `ReplayOrchestrator`. Public-demo mode permits one configured replay
split asset, a bounded accelerated batch, a per-process client cooldown, at most three append-only
attempts, and an explicitly confirmed reset. The existing PostgreSQL lease rejects concurrent
advancement. The API never accepts arbitrary commands or asset paths, and the response suppresses
the stored final cycle until the run is complete.

Controls are read-only by default. Writable non-demo mode requires an environment-provided admin
token. Writable modes also require an application secret for same-origin form protection. This is
deliberately smaller than an authentication system; a production asset-control surface would need
real identity, authorization, and distributed rate limiting.

### Two services and one managed database on Render

The Blueprint defines one public Docker web service (FastAPI, dashboard, bounded replay), one
private Docker MLflow service, and one managed PostgreSQL database. There is no Redis, worker,
separate replay service, or public MLflow UI.

The web service owns a disk mounted at `/var/lib/turbine-guard` for acquired/processed data,
features, local bundles, monitoring references, and lifecycle artifacts. PostgreSQL owns operational
and replay/report state. The MLflow private service owns one disk containing its SQLite registry
backend and proxied artifacts. Keeping metadata and artifacts on the same service/disk makes alias
and artifact state survive restart while respecting the established MLflow/PostgreSQL exception.

Alembic runs in the paid web service's pre-deploy phase. Render health routing uses liveness so a
new but not-yet-bootstrapped deployment can start and present a degraded page; `/health/ready`
remains strict and fails until PostgreSQL, the champion, and feature contract are available.

### Explicit bootstrap and degraded operation

Normal startup never downloads data or trains. After the first infrastructure deploy, an operator
runs the existing idempotent bootstrap from the web service runtime, where the persistent disk is
mounted and private MLflow is reachable. The process is safe to repeat and does not create a new
registered version for identical verified artifacts.

If Starter memory is insufficient, the operator upgrades temporarily or restores trusted,
checksum-verified application and MLflow disk snapshots prepared with the same artifact-proxy URI.
The dashboard handles an empty database, absent champion, absent reports, and temporary MLflow
failure as explicit unavailable/insufficient-data states rather than raw errors.

## Consequences

The minimum topology is understandable and inexpensive, but attached disks bind both services to
one instance and introduce brief deploy downtime. Monitoring remains an explicit web-service shell
command because a Render cron job cannot mount the web service's disk. The public rate limiter is
per process; this is adequate for the deliberately single-instance disk-backed demo, not a general
distributed-control design. A free-only Render deployment cannot satisfy restart persistence.

