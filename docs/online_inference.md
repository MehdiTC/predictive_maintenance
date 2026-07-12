# Online Inference and Asset Health API (Loop 7)

## Architecture and request lifecycle

```text
POST /v1/sensor-readings
  -> strict Pydantic validation + request ID
  -> resolve/create and lock generic asset
  -> idempotent PostgreSQL reading insert
  -> same-asset history through current cycle
  -> Loop 3 FeatureBuilder from verified manifest
  -> exact ordered 552-feature current row
  -> cached MLflow champion alias
  -> validate RUL / interval / risk output
  -> idempotent version-pinned prediction insert
  -> commit reading + prediction together
```

Routes do not call SQLAlchemy, pandas, or MLflow directly. `OnlineInferenceService` owns the use
case; focused Loop 6 repositories own persistence; `ChampionModelLoader` owns registry/cache
behavior; the original `FeatureBuilder` remains the only feature implementation.

## Asset and history policy

The first cycle auto-creates an active generic asset. Only the external ID is copied to the asset;
no engine type or C-MAPSS dataset is inferred. The reading records its supplied source.

New assets begin at cycle 1 and new cycles must be contiguous. A gap or new older cycle is rejected
with `409 history_conflict`. Exact retries remain valid after later cycles exist. Conflicting data at
an existing cycle is never overwritten and returns `409 sensor_reading_conflict`. This strict policy
means every stored online history is contiguous and no historical prediction becomes silently stale
after a backfill.

## Features and model loading

The service queries one asset ordered by cycle and bounds history at the submitted cycle. It loads
the exact Loop 3 feature configuration and ordered fields from
`data/features/cmapss/FD001/feature_manifest.json`, then calls `FeatureBuilder.transform_asset()`.
Future cycles and other assets cannot enter the frame. Early-cycle null features are retained and
handled by the saved training-only Ridge preprocessing.

The model URI is `models:/TurbineGuard-FD001-RUL@champion` by default. A thread-safe lazy cache
resolves the registry version/run, loads the pyfunc once, and verifies its signature and feature
version. Optional startup preload warms the cache. `refresh()` is explicit for a future controlled
promotion; there is no polling watcher.

## Transaction and idempotency policy

One transaction covers asset creation, reading, features, inference, and prediction. Any failure
rolls everything back. Repositories still never commit themselves.

Responses expose:

* `reading_idempotent`: the cycle already existed identically.
* `prediction_idempotent`: the current model version already predicted that reading.
* `idempotent`: both are true; no new operational row was created.

An exact full retry returns HTTP 200. A newly stored reading or a new champion-version prediction
returns HTTP 201. One reading may validly have multiple predictions when model versions differ.

## Endpoints

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/sensor-readings` | Atomically persist a cycle and champion prediction |
| `GET /v1/assets` | Bounded asset summaries, ordered by external ID |
| `GET /v1/assets/{id}` | Asset metadata, latest state, recent events |
| `GET /v1/assets/{id}/health` | RUL, interval, risk, staleness, prediction trend |
| `GET /v1/predictions/recent` | Recent-first predictions, optional asset filter |
| `GET /v1/models/current` | Cached registry identity and recorded evidence |
| `GET /v1/monitoring/summary` | Available service/database counts only |
| `GET /health/live` | Process liveness |
| `GET /health/ready` | Database, model, and feature-contract readiness |
| `GET /metrics` | Prometheus exposition |

`POST /v1/predictions` is deliberately deferred: with one champion it adds no distinct capability
over idempotent ingestion/re-scoring. Maintenance-event ingestion is Loop 8 delayed feedback and is
also absent.

## Request contract

The sensor body contains `external_asset_id`, positive `cycle`, optional timezone-aware
`observed_at`, `operating_setting_1..3`, anonymous `sensor_01..21`, `source`, optional
`ingestion_id`, and `schema_version`. Extra fields, naive timestamps, blanks, NaN, and infinities
are rejected. If time is omitted, receipt time is stored; an omitted-time retry reuses the original
stored observation time.

Risk is the model's constrained `healthy`, `warning`, or `critical` output. Failure-within-30/50
fields are boolean horizon indicators derived from point RUL, not probabilities.

## Errors and request correlation

Clients may send `X-Request-ID` using 1–128 safe characters; otherwise a UUID is generated. It is
returned as a header and inside structured errors. Logs include request ID, route, method, status,
latency, and safe model/error metadata, never full sensor payloads or database URLs.

| Status | Stable codes |
| --- | --- |
| 422 | `request_validation_failed` |
| 409 | `history_conflict`, `sensor_reading_conflict`, `prediction_conflict` |
| 404 | `asset_not_found` |
| 503 | `model_unavailable`, `feature_contract_unavailable`, `database_unavailable` |
| 500 | `internal_error` |

## Metrics and monitoring summary

Prometheus metrics cover HTTP request count/latency, accepted ingestion, prediction requests,
prediction/model/database/validation/conflict failures, model latency, current safe model identity,
and risk counts. Labels are bounded to route, method, status, and risk. Asset/request identifiers
and raw errors are prohibited.

The JSON monitoring summary combines process-local counters with persistent reading/prediction
counts, latest ingestion time, recent stored risk distribution, and current model version. It does
not calculate drift or delayed performance.

## Configuration and local setup

Use `.env.example` for all settings. Important values are the operational PostgreSQL URL, MLflow
tracking URI/model/alias, online/preload flags, staleness threshold, pagination/history/request
limits, restrictive CORS origins, trusted hosts, and docs exposure.

```bash
export TURBINE_GUARD_DATABASE_URL='postgresql+psycopg://localhost:5432/turbine_guard'
export TURBINE_GUARD_MLFLOW_TRACKING_URI='sqlite:///data/mlflow/mlflow.db'
uv run alembic upgrade head
uv run uvicorn --factory turbine_guard.api.app:create_app
```

Interactive contracts and examples are available at `/docs`. PostgreSQL integration tests require
`TURBINE_GUARD_DATABASE_TEST_URL` naming a database containing `test`.

Basic checks:

```bash
curl -s http://127.0.0.1:8000/health/live
curl -s http://127.0.0.1:8000/health/ready
curl -s http://127.0.0.1:8000/v1/models/current
curl -s http://127.0.0.1:8000/metrics
```

One complete cycle (channels remain anonymous):

```bash
curl -i -X POST http://127.0.0.1:8000/v1/sensor-readings \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: example-cycle-1' \
  -d '{
    "external_asset_id":"example-asset","cycle":1,
    "observed_at":"2026-07-12T12:00:00Z","source":"manual-example",
    "schema_version":"1",
    "operating_setting_1":0.0,"operating_setting_2":0.0,"operating_setting_3":100.0,
    "sensor_01":518.67,"sensor_02":641.82,"sensor_03":1589.7,
    "sensor_04":1400.6,"sensor_05":14.62,"sensor_06":21.61,"sensor_07":554.36,
    "sensor_08":2388.0,"sensor_09":9046.19,"sensor_10":1.3,"sensor_11":47.47,
    "sensor_12":521.66,"sensor_13":2388.02,"sensor_14":8138.62,"sensor_15":8.4195,
    "sensor_16":0.03,"sensor_17":392.0,"sensor_18":2388.0,"sensor_19":100.0,
    "sensor_20":39.06,"sensor_21":23.419
  }'
```

Repeat the identical body to receive HTTP 200 and all three idempotency flags true. Change one
sensor at cycle 1 to receive HTTP 409. Asset IDs returned by ingestion can be used with the detail
and health endpoints.

## Security and scope limitations

CORS is disabled by default, trusted hosts are explicit, request bodies are size-limited, docs can
be disabled, and production errors do not expose tracebacks. Authentication and authorization are
not part of the current project specification, but a real industrial deployment must add both,
plus TLS, rate limiting, asset enrollment, secret management, and network controls.

Loop 7 does not implement replay, maintenance/failure feedback, label backfill, drift, performance
monitoring, retraining, promotion, Prefect, Docker, deployment, or a dashboard.
