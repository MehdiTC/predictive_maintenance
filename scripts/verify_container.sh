#!/bin/sh
set -eu

image="${1:-turbine-guard:local}"
container_name="turbine-guard-contract-$$"

cleanup() {
  docker rm --force "$container_name" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

user="$(docker image inspect --format '{{.Config.User}}' "$image")"
case "$user" in
  ""|0|0:*|root|root:*)
    echo "container image must configure a non-root user; found '$user'" >&2
    exit 1
    ;;
esac

docker run --rm "$image" python -c \
  'import turbine_guard; from turbine_guard.api.app import create_app; assert callable(create_app)'
docker run --rm "$image" alembic --help >/dev/null
docker run --rm "$image" python scripts/replay_sensor_data.py --help >/dev/null
docker run --rm "$image" python scripts/model_lifecycle.py --help >/dev/null
docker run --rm "$image" python scripts/bootstrap.py --help >/dev/null
docker run --rm \
  -e TURBINE_GUARD_DATABASE_URL=postgresql+psycopg://user:pass@postgres:5432/turbine_guard \
  -e TURBINE_GUARD_MLFLOW_TRACKING_URI=http://mlflow:5000 \
  -e TURBINE_GUARD_REPLAY_API_BASE_URL=http://api:8000 \
  "$image" python -c \
  'from turbine_guard.config.settings import Settings; s=Settings(); assert s.database_url.split("@")[1].startswith("postgres:"); assert s.mlflow_tracking_uri == "http://mlflow:5000"; assert s.replay_api_base_url == "http://api:8000"'

docker run --detach --name "$container_name" \
  -e TURBINE_GUARD_ONLINE_INFERENCE_ENABLED=false \
  -e TURBINE_GUARD_MODEL_PRELOAD_ENABLED=false \
  "$image" >/dev/null

attempt=0
until docker exec "$container_name" python scripts/healthcheck.py live; do
  attempt=$((attempt + 1))
  if [ "$attempt" -ge 20 ]; then
    docker logs "$container_name" >&2
    exit 1
  fi
  sleep 1
done

docker stop --time 30 "$container_name" >/dev/null
exit_code="$(docker inspect --format '{{.State.ExitCode}}' "$container_name")"
if [ "$exit_code" -ne 0 ]; then
  docker logs "$container_name" >&2
  echo "API did not shut down cleanly (exit $exit_code)" >&2
  exit 1
fi

echo "container contract passed for $image (user $user)"
