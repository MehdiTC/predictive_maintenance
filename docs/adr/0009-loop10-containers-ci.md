# ADR 0009 — Loop 10 container topology, bootstrap, migrations, and CI fixtures

**Status:** Accepted (2026-07-13)

## Context

The API, replay engine, lifecycle service, Alembic migrations, and MLflow clients already share one
Python package and typed settings. Loop 10 must make those components reproducible without adding a
queue, scheduler, second model server, or automatic retraining. API readiness cannot pass until a
real registry champion and its checksummed feature manifest both exist, so container startup must
make that prerequisite explicit.

## Decisions

### One immutable application image

API, migration, bootstrap, worker, replay, MLflow, and management commands use one multi-stage
Python 3.12 image. `uv sync --locked --no-dev --no-editable` installs the locked wheel into an
isolated runtime environment. The final image contains the installed package, Alembic tree, and
thin operational scripts, but excludes tests, notebooks, development tools, `.env`, and mutable
data. It runs as numeric user/group `10001:10001`; imports never depend on an editable source tree.

### Separate operational and MLflow metadata stores

The Compose PostgreSQL service owns only `turbine_guard` operational state. MLflow metadata stays in
a service-owned SQLite database on a dedicated named volume, and artifacts remain on another named
volume served through MLflow's artifact proxy. The locked MLflow 3.14 server creates its PostgreSQL
`model_versions.version` column as an integer but binds its post-registration lookup as text;
PostgreSQL correctly rejects that comparison and champion registration fails. Retaining SQLite is
therefore the clean, explicit single-host exception allowed by the Loop 10 specification. It also
preserves the existing operational/registry boundary without patching dependency internals. Real
PostgreSQL remains mandatory for TurbineGuard runtime state, migrations, and CI integration tests.

### One migration owner

The one-shot `migrate` service is the sole schema owner. API waits for its successful completion;
worker and replay depend on the same completed service. No long-running process applies migrations,
so concurrent services cannot race and API startup failure cannot leave an ambiguous migration
owner.

### Explicit profiles and bootstrap

Normal startup never trains or promotes. `bootstrap` is an explicit profile that idempotently runs
the established acquisition, validation, feature, training, tracking, registration, and initial
champion path. It writes generated data to a named volume and the model to MLflow. `worker` is a
one-shot `monitor` command in the `ops` profile; it may write monitoring reports but neither
repeatedly retrains nor promotes. `replay` is a manual profile whose default command is read-only
`status --all`; operators must explicitly select start/step/resume behavior.

### Deterministic CI champion

CI uses an explicit `bootstrap-ci` profile. It creates a deterministic, schema-complete miniature
FD001 archive without network access, then runs the real acquisition, processing, 552-feature,
four-family modeling, MLflow pyfunc, registration, alias, and readiness contracts. Candidate grids
and eligibility thresholds are bounded for smoke-test runtime and the registered model name is CI
specific. Production contracts and loaders are unchanged; the fixture is never selected by normal
bootstrap.

### CI job boundaries

GitHub Actions separates Python quality, real-PostgreSQL migration/integration, focused temporary
MLflow integration, and production-image/Compose smoke checks. The container job verifies the
configured non-root user, imports, CLI availability, service-name settings, clean signal shutdown,
migrations, real readiness, representative ingestion, profile CLIs, and named-volume persistence
across restarts. It does not download NASA data or depend on developer artifacts.

## Consequences

Fresh-clone startup has an intentional bootstrap step before readiness can pass; this is more
honest than silently training on every API restart. Local credentials, SQLite metadata, and
filesystem artifacts are development-only. PostgreSQL, MLflow metadata, MLflow artifacts, and
generated project state have distinct named volumes and distinct reset consequences. Worker
scheduling, public deployment, high availability, TLS, secrets management, and a dashboard remain
outside Loop 10.
