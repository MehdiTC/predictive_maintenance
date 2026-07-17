# ADR 0011 — Zero-cost public demo: Render Free, Neon PostgreSQL, and an immutable deployment bundle

**Status:** Accepted (2026-07-14). Supersedes the deployment topology of ADR 0010; the dashboard,
read-API, and public-replay-policy decisions of ADR 0010 are unchanged.

## Context

The ADR 0010 topology (two paid Render Starter services, two persistent disks, and a paid managed
PostgreSQL database) is correct under strict restart-persistence requirements but costs about
$20.80/month. For a portfolio recruiter demo the owner requires zero hosting cost. Recruiters
interact with the demo for minutes; production-grade compute durability is not what the demo is
demonstrating — the local Docker Compose stack remains the reproducible reference implementation
of the full topology, including live MLflow.

Free-tier constraints that shaped this decision:

* Render free web services sleep when idle, cannot attach persistent disks, cannot use paid
  pre-deploy commands, and lose local filesystem changes on restart.
* Render's free managed PostgreSQL expires after 30 days, which is unacceptable mid-job-search.
* Neon's free PostgreSQL plan has no time limit (0.5 GB storage, capped monthly compute) and
  serves ordinary `postgresql://` connections, so the existing SQLAlchemy/psycopg/Alembic layer
  works unchanged.
* An always-on MLflow service was the largest cost item and exists only to serve one read-only
  champion in production; nothing in the public demo may mutate a registry anyway.

Alternatives considered: Hugging Face Spaces + Neon was rejected because Spaces' documented
outbound networking is limited to ports 80/443/8080 while ordinary PostgreSQL is on 5432; using
Neon's HTTPS data API instead would have replaced much of the existing repository layer solely for
hosting. Running the full bootstrap (acquisition, 552-feature build, training, registration) on
every cold start was rejected: long cold starts, pointless retraining, more failure modes, and a
registry state that could drift between restarts.

## Decisions

### One free web service plus external Neon PostgreSQL

`render.yaml` now defines exactly one free Docker web service. Operational state (assets,
readings, predictions, replay runs, outcomes, monitoring reports, lifecycle events) lives in a
Neon free-tier PostgreSQL database whose connection string is entered in the Render dashboard
(`sync: false`), never committed. The existing URL normalization accepts Neon's standard
`postgresql://` string. Alembic migrations run at container start (free plans have no pre-deploy
phase); the service is single-instance, so start-time migrations do not race.

### The champion is an immutable exported deployment bundle, not a live registry

A new `turbine_guard.deployment` package exports a checksum-pinned `tar.gz` containing exactly
what serving needs: the champion `ModelBundle`, the Loop 3 feature and split manifests, and the
Loop 2 processed trajectory source plus report (so the Loop 8 replay integrity chain still
verifies end to end). A typed `deployment_manifest.json` captures the registry identity at export
time: registered model name, version, champion alias and all alias assignments, source run ID,
metrics, target definition, feature version and count, git commit, lineage, and per-file SHA-256
checksums. Export loads the champion through the normal MLflow serving path first and refuses to
package an artifact whose checksum differs from the registered one.

The bundle is published outside the application repository as a revision-pinned artifact (a
Hugging Face dataset/model repository fits the ML-portfolio framing; a versioned GitHub Release
asset also works). The application repository continues to contain no generated data or model
artifacts, preserving the original repository rule.

### Bundle-mode serving without importing MLflow

`TURBINE_GUARD_MODEL_SOURCE=deployment_bundle` selects a `DeploymentBundleLoader` that verifies
the restored files against the deployment manifest (champion checksum verified before
deserialization) and adapts the Loop 4 `ModelBundle` to the same serving interface as the MLflow
pyfunc. The shared rich-output logic moved into `ModelBundle.predict_rich`, used by both paths, so
there is one implementation of the serving contract. Serving types now live in the MLflow-free
`turbine_guard.serving.champion` module and the loaders are imported lazily, so the demo process
never imports MLflow — relevant on a 512 MB free instance. Alias information comes from the
loader (`registry_aliases()`): live registry values in MLflow mode, the exported snapshot in
bundle mode, and the dashboard labels the snapshot explicitly. Model-registry mutation is
structurally impossible in the public demo.

### Cold start restores; it never trains

`scripts/start_demo.py` (1) downloads the pinned bundle, verifies the archive SHA-256 against
`TURBINE_GUARD_DEPLOYMENT_BUNDLE_SHA256`, safely extracts (stdlib `data` tar filter), verifies
every file checksum, and restores atomically with a completion marker written last; (2) runs
`alembic upgrade head` against Neon; (3) starts the normal API entry point. Restore is idempotent
and self-healing: a verified prior restore is a no-op, a corrupted file is re-restored. No demo
seeding is required — dashboard pages handle the empty state, the bounded public replay control
creates the first data, and everything written afterwards persists in Neon across compute
restarts.

## Consequences

* Hosting cost is $0. The tradeoffs are stated openly in the deployment documentation: idle
  sleep with a cold-start delay, limited monthly free compute/database allowances, ephemeral
  compute filesystem (bundle re-restored on restart), and a read-only public model registry.
* Promotion, retraining, and live MLflow remain demonstrable only in the local Compose stack;
  the dashboard says so rather than implying otherwise.
* A new champion reaches the public demo by exporting a new bundle, publishing it at a new
  revision, and updating the pinned URL/SHA-256 — an explicit, auditable release step.
* ADR 0010's Render Starter/disk/managed-PostgreSQL Blueprint is superseded; its dashboard and
  replay-safety decisions stand.
