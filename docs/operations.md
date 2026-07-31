# Operations

This runbook covers local reproduction, the Docker Compose stack, replay, monitoring, and routine
quality checks.

## Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Docker with Compose v2 for the multi-service stack

Install the locked development environment:

```bash
uv sync
cp .env.example .env
```

Every application setting is optional and documented in `.env.example`. Keep credentials and local
overrides in `.env`, which is ignored by Git.

## Reproduce the offline pipeline

```bash
make reproduce
```

The target runs acquisition, validation and Parquet conversion, feature generation, and model
training in dependency order. Individual stages are also available:

```bash
make acquire
make process
make features
make train
```

The pipeline is checksum-aware and idempotent. If a generated artifact is unexpectedly modified,
the command fails instead of silently accepting it. Generated files live under `data/` and are not
committed.

Run the exploratory notebook after processing:

```bash
make eda
```

## MLflow on the host

Track a completed training execution and register its selected model:

```bash
make train-tracked
make mlflow-ui
```

MLflow opens at <http://127.0.0.1:5000>. Inspect the registry or verify that the registered champion
matches the local bundle with:

```bash
make mlflow-inspect
make mlflow-verify
```

## Bootstrap the Compose stack

The API readiness check requires PostgreSQL, a feature manifest, and a registered champion. On a
clean Docker volume, bootstrap those dependencies once:

```bash
docker compose build
docker compose up -d --wait postgres mlflow
docker compose --profile bootstrap run --rm bootstrap
docker compose up -d --wait api
```

Endpoints:

- Dashboard: <http://127.0.0.1:8000/>
- OpenAPI: <http://127.0.0.1:8000/docs>
- Liveness: <http://127.0.0.1:8000/health/live>
- Readiness: <http://127.0.0.1:8000/health/ready>
- MLflow: <http://127.0.0.1:5000>

After the initial bootstrap:

```bash
make docker-up
make docker-logs
make docker-down
```

`make docker-down` preserves named volumes. It does not delete database or registry state.

## Replay and monitoring

The replay profile is read-only by default and reports current state:

```bash
make docker-replay-status
```

Start an accelerated held-out trajectory from the host:

```bash
uv run python scripts/replay_sensor_data.py start --asset-id 9 --mode accelerated
uv run python scripts/replay_sensor_data.py status --all
```

Run one monitoring window and inspect lifecycle state:

```bash
make monitor
make lifecycle-status
```

Model promotion requires all configured gates and, by default, explicit approval. The public demo
uses a pinned bundle and does not expose retraining or registry mutation.

## Database migrations

```bash
make db-current
make db-history
make db-upgrade
```

Compose owns migrations through its one-shot `migrate` service. Avoid running concurrent migration
owners against the same database.

## Verification

Run the standard local gate:

```bash
make check
```

Container-specific checks:

```bash
make docker-test
make docker-smoke
```

PostgreSQL integration tests require a separate database whose name contains `test`. Configure
`TURBINE_GUARD_DATABASE_TEST_URL` before running tests marked `postgres`.

## Shutdown and troubleshooting

```bash
make docker-down
docker compose ps
docker compose logs api mlflow postgres
```

If readiness fails, check PostgreSQL connectivity, the current Alembic revision, the feature
manifest, and the `champion` alias. On a completely clean volume, repeat the bootstrap sequence.
