#!/bin/sh
set -eu

root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$root"

export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-turbine-guard-smoke-$$}"
export TURBINE_GUARD_IMAGE="${TURBINE_GUARD_IMAGE:-turbine-guard:smoke}"
export TURBINE_GUARD_MLFLOW_EXPERIMENT_NAME="${TURBINE_GUARD_MLFLOW_EXPERIMENT_NAME:-TurbineGuard-CI-Smoke}"
export TURBINE_GUARD_MLFLOW_REGISTERED_MODEL_NAME="${TURBINE_GUARD_MLFLOW_REGISTERED_MODEL_NAME:-TurbineGuard-CI-RUL}"
export POSTGRES_PORT="${POSTGRES_PORT:-15432}"
export MLFLOW_PORT="${MLFLOW_PORT:-15000}"
export TURBINE_GUARD_API_PORT="${TURBINE_GUARD_API_PORT:-18000}"

cleanup() {
  docker compose --profile ci --profile ops --profile replay down --volumes --remove-orphans
}
trap cleanup EXIT INT TERM

if [ "${SKIP_BUILD:-0}" != "1" ]; then
  docker compose build
fi
docker compose up --detach --wait postgres mlflow
docker compose --profile ci run --rm bootstrap-ci
docker compose up --detach --wait api
python3 scripts/smoke_api.py --base-url "http://127.0.0.1:${TURBINE_GUARD_API_PORT:-8000}"
docker compose --profile ops run --rm worker status >/dev/null
docker compose --profile replay run --rm replay status --all >/dev/null

docker compose restart postgres
docker compose up --detach --wait postgres
docker compose restart mlflow
docker compose up --detach --wait mlflow
docker compose restart api
docker compose up --detach --wait api
python3 scripts/smoke_api.py --base-url "http://127.0.0.1:${TURBINE_GUARD_API_PORT:-8000}"

echo "Compose smoke and named-volume restart persistence passed."
