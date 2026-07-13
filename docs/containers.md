# Containers and CI/CD (Loop 10)

## Architecture and responsibilities

All Python processes use the same `turbine-guard` application image.

| Service | Responsibility | Default behavior |
| --- | --- | --- |
| `postgres` | TurbineGuard operational state | Long-running, healthy after `pg_isready` |
| `mlflow` | Tracking UI, registry, and artifact proxy | Long-running on port 5000 |
| `migrate` | Sole Alembic migration owner | One-shot `upgrade head` |
| `api` | FastAPI inference, health, docs, and Prometheus metrics | Long-running on port 8000 |
| `worker` | Loop 9 operational worker | Profile `ops`; one-shot monitoring window |
| `replay` | Loop 8 HTTP replay client and delayed feedback | Profile `replay`; safe default is `status --all` |
| `bootstrap` | Initial data/features/model/champion creation | Profile `bootstrap`; explicit and idempotent |
| `bootstrap-ci` | Offline deterministic smoke champion | Profile `ci`; CI only |

Network flow:

```text
replay --HTTP--> api --> turbine_guard database
                    \--> mlflow
worker -------------> turbine_guard database + mlflow
bootstrap ----------> generated-data volume + mlflow
mlflow -------------> metadata volume + artifact volume
```

PostgreSQL contains only the `turbine_guard` application database managed by Alembic. MLflow owns a
separate persistent SQLite metadata volume and never uses the operational database. The locked
MLflow 3.14 release binds registered-model version lookups as text against its PostgreSQL integer
column, so its registry cannot be configured cleanly with PostgreSQL in this release. SQLite keeps
the state domains separate and is the documented single-host Compose exception; the operational
application and migration tests still use the real PostgreSQL dialect.

## Image design

The multi-stage `Dockerfile` uses `python:3.12-slim-bookworm`. A pinned `uv` binary installs
`uv.lock` with `--locked --no-dev --no-editable`; the runtime stage copies only the completed
environment, Alembic files, and operational scripts. The application is therefore installed as a
wheel and does not rely on an editable checkout or `PYTHONPATH`. The final image runs as
`10001:10001`, contains no `.env` or generated state, and uses `/app` as its working directory.

## Environment

Copy the safe local examples before Compose use:

```bash
cp .env.example .env
```

The important Compose inputs are `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`,
`MLFLOW_BACKEND_STORE_URI`, `POSTGRES_PORT`, `MLFLOW_PORT`, `TURBINE_GUARD_API_PORT`,
environment/log level, experiment/registered-model names, and replay timing. The committed
credentials are non-production examples only. Compose deliberately constructs internal URLs using
service names:

```text
postgresql+psycopg://...@postgres:5432/turbine_guard
http://mlflow:5000
http://api:8000
```

It overrides host-oriented `localhost` URLs from `.env`. Never commit a real `.env`; never put
production credentials in `compose.yaml` or the image.

## First bootstrap and normal startup

API readiness requires PostgreSQL, the registered `champion`, and exact feature/model schema
compatibility. A clean volume therefore needs an explicit bootstrap. It is intentionally separate
from ordinary startup because full FD001 acquisition and training must not run whenever API starts.

```bash
cp .env.example .env
docker compose build
docker compose up -d --wait postgres mlflow
docker compose --profile bootstrap run --rm bootstrap
docker compose up -d --wait api
```

The bootstrap runs acquire → process → features → train → MLflow track/register/promote. Every
stage verifies existing checksums and reuses a completed result. It requires NASA access unless
`TURBINE_GUARD_CMAPSS_SOURCE_URL` points to a local `file://` archive. It is safe to rerun; force
rebuilds remain explicit in the underlying CLIs.

After the initial champion exists, normal use is:

```bash
make docker-up
docker compose ps
```

`docker compose up --build` is also valid on an already-bootstrapped volume. On a completely clean
volume it starts infrastructure and the API process, but readiness correctly stays unavailable
until the explicit bootstrap above is completed.

## Health and endpoints

PostgreSQL uses `pg_isready`; MLflow uses its `/health` endpoint. Compose marks the API healthy only
when `/health/ready` reports successful `database`, `model`, and `feature_contract` checks. Process
liveness remains independently available at `/health/live`.

```bash
curl http://127.0.0.1:8000/health/live
curl http://127.0.0.1:8000/health/ready
open http://127.0.0.1:8000/docs
open http://127.0.0.1:5000
curl http://127.0.0.1:8000/metrics
```

## Migrations, worker, and replay

Apply migrations only through the owner service:

```bash
make docker-migrate
docker compose run --rm migrate alembic current
```

The second command overrides the service command for inspection but still uses the migration
image/configuration. Do not add migration calls to API, worker, or replay startup.

Run a single safe monitoring window or inspect lifecycle state:

```bash
make docker-worker
docker compose --profile ops run --rm worker status
```

The worker is not a scheduler and does not loop. Its default `monitor` command can persist reports
and a trigger decision, but it never calls retraining or promotion implicitly.

Replay is manual. Its default command only reports state:

```bash
make docker-replay-status
docker compose --profile replay run --rm replay start --asset-id 9 --mode step
docker compose --profile replay run --rm replay status --all
docker compose --profile replay run --rm replay resume --run-id <UUID> --max-cycles 20
docker compose --profile replay run --rm replay start --asset-id 9 --mode accelerated
```

Only the final command streams the full held-out lifecycle; it is never a default startup action.

## Volumes, inspection, and reset

Named volumes are:

| Volume | Persists |
| --- | --- |
| `postgres_data` | TurbineGuard PostgreSQL schema and operational history |
| `mlflow_metadata` | MLflow experiment and registry metadata SQLite database |
| `mlflow_artifacts` | MLflow run/model artifacts served by the MLflow service |
| `turbine_guard_data` | Raw/processed/feature/model data and monitoring references |

Inspect state without deleting it:

```bash
docker volume ls --filter name=turbine-guard
docker volume inspect turbine-guard_postgres_data
docker volume inspect turbine-guard_mlflow_metadata
docker compose exec postgres psql -U turbine_guard -d turbine_guard -c '\dt'
docker compose exec mlflow ls -lh /var/lib/mlflow /mlartifacts
```

`make docker-down` stops containers and preserves volumes. This reset is destructive and removes
datasets, models, registry metadata, artifacts, and operational history:

```bash
docker compose --profile bootstrap --profile ops --profile replay down --volumes
```

Changing `POSTGRES_DB` after the PostgreSQL volume was initialized does not recreate the cluster;
reset the development volume or create the database deliberately. Changing the MLflow backend URI
requires a deliberate metadata migration or reset, not merely an environment edit.

## Logs, shutdown, and local CI equivalents

```bash
make docker-logs
make docker-down
make docker-test       # image user/import/CLI/settings/signal checks
make docker-smoke      # isolated fixture, migrations, readiness, request, restart, cleanup
```

The API receives a 35-second Compose grace period and Uvicorn receives 30 seconds for graceful
shutdown. `make docker-smoke` uses its own Compose project and always destroys only its ephemeral
smoke volumes.

GitHub Actions runs four jobs: Python quality/full tests; real-PostgreSQL migrations and guarded
integration tests; focused temporary local MLflow registry tests; and production-image plus
Compose smoke tests. The CI bootstrap fixture is generated deterministically, uses no NASA/network
data, and still exercises the real 552-feature, model-bundle, pyfunc, registry, alias, readiness,
and ingestion contracts.

## Troubleshooting and limitations

* `api` is `unhealthy`: inspect `docker compose logs api`; `/health/ready` identifies database,
  model, and feature-contract failures separately. Run explicit bootstrap if no champion exists.
* MLflow cannot start: inspect `docker compose logs mlflow`, confirm its metadata/artifact volumes
  are writable by UID 10001, and retain the four-slash absolute SQLite URI from `.env.example`.
* Model downloads fail: confirm the MLflow artifact volume is writable by UID 10001 and that the
  experiment uses `mlflow-artifacts:/...`, not an API-container-local file path.
* A port is occupied: change `POSTGRES_PORT`, `MLFLOW_PORT`, or `TURBINE_GUARD_API_PORT` in `.env`.
* Docker volume permission errors after manual modification: inspect ownership inside a one-shot
  application container; do not run bootstrap as root against the shared volume.
* This is a single-host development topology with example credentials, SQLite MLflow metadata,
  and filesystem artifacts.
  It provides no scheduler, distributed queue, TLS, authentication, secrets manager, backups,
  autoscaling, high availability, public deployment, or dashboard.
