# Dashboard and zero-cost public deployment

## Public framing

TurbineGuard is an independently developed predictive-maintenance platform using public NASA
simulated turbofan degradation data. It contains no proprietary client data or implementation
details. C-MAPSS sensors remain anonymous (`sensor_01` through `sensor_21`); dashboard labels do not
assign temperature, vibration, or other invented physical meanings.

## Dashboard architecture and routes

FastAPI serves both the API and the dashboard. Jinja renders useful HTML first, local CSS provides
the responsive visual system, and small JavaScript modules use Plotly for chart enhancement. The
templates consume typed service projections; they do not query SQLAlchemy or reproduce inference,
replay, monitoring, or registry behavior.

| Page | Purpose |
| --- | --- |
| `/` | Redirect to the fleet dashboard |
| `/dashboard` | Fleet metrics/table, current alerts, and safe replay status/control |
| `/dashboard/assets/{asset_id}` | Asset state, RUL interval, risk/horizons, selected anonymous sensors, events, and history |
| `/dashboard/predictions` | Bounded versioned prediction history and filters |
| `/dashboard/models` | Champion evidence, aliases, lineage, and lifecycle activity |
| `/dashboard/monitoring` | Latest bounded drift detail and delayed online performance |

JSON data sources remain versioned under `/v1`: `/v1/fleet`,
`/v1/assets/{asset_id}/dashboard`, `/v1/predictions/history`, `/v1/alerts`,
`/v1/models/overview`, `/v1/monitoring/drift`, `/v1/monitoring/performance`, `/v1/replay`, and
`/v1/replay/actions`. Query limits never exceed the configured API maximum. Fleet latest-state
queries are set based rather than one query per asset. Only a configured subset of drift features
and sensor history is returned by default.

The browser receives no database URL, MLflow address, artifact path, application secret, or replay
token. Production responses add CSP, clickjacking, content-type, referrer, permissions, and HSTS
headers. CORS is empty by default and trusted hosts are explicit.

## Local use

After the Loop 10 stack is bootstrapped:

```bash
docker compose up --build
open http://localhost:8000/dashboard
```

Local Compose uses the dashboard defaults, so replay is read-only. To test the constrained public
policy in a gitignored `.env`, set:

```dotenv
TURBINE_GUARD_PUBLIC_DEMO_MODE=true
TURBINE_GUARD_REPLAY_CONTROLS_ENABLED=true
TURBINE_GUARD_REPLAY_DEMO_SOURCE_ASSET_ID=9
TURBINE_GUARD_APPLICATION_SECRET=replace-with-a-long-random-value
```

Never commit the value. Non-demo writable mode must also set
`TURBINE_GUARD_REPLAY_ADMIN_TOKEN` and requires operators to enter it for each form action or send
it as `X-Replay-Control-Token` to the JSON endpoint.

## Replay-control safety

The dashboard calls the existing Loop 8 orchestrator and therefore retains verified replay-split
selection, deterministic cycle payloads, exact retry reconciliation, durable leases, phase recovery,
and delayed-ground-truth isolation.

Public-demo restrictions are:

* only `TURBINE_GUARD_REPLAY_DEMO_SOURCE_ASSET_ID` is eligible;
* protected train, validation, calibration, and official-test assets cannot be selected;
* start does not advance implicitly;
* resume advances one cycle and accelerated mode advances at most 10 cycles per request by default;
* a per-client cooldown limits request bursts;
* an active lease rejects simultaneous advancement;
* reset needs explicit confirmation, creates append-only attempt history, and is capped at three
  public attempts by default;
* no shell command, source path, arbitrary API URL, or final source cycle is accepted or exposed;
* final-cycle ground truth appears only after the completed state.

Set `TURBINE_GUARD_REPLAY_CONTROLS_ENABLED=false` for a fully read-only public dashboard.

## Zero-cost deployment topology (ADR 0011)

The public demo costs $0/month: one free Render web service plus an external Neon free-tier
PostgreSQL database. There is no MLflow service, no persistent disk, and no Render database.

```text
Render Free Web Service (Docker, single instance, sleeps when idle)
        │
        ├── FastAPI dashboard and /v1 API
        ├── immutable deployment bundle (restored, checksum-verified at start)
        └── bounded public replay controls
        │
        ▼
Neon Free PostgreSQL (operational, replay, monitoring, lifecycle rows)
```

| Component | Provider/plan | Responsibility | Persistence |
| --- | --- | --- | --- |
| `turbine-guard-web` | Render free web service | FastAPI, dashboard, bundle serving, bounded replay | Ephemeral filesystem; bundle re-restored on every cold start |
| Neon database | Neon free plan | All PostgreSQL state | Durable (0.5 GB storage, monthly compute allowance) |
| Deployment bundle | Hugging Face dataset repo (or GitHub Release) | Immutable exported champion snapshot + serving inputs | Revision-pinned, checksum-pinned |

The full production-style topology — live MLflow registry, persistent volumes, worker and replay
profiles — remains the local Docker Compose stack; the public demo intentionally demonstrates the
serving, dashboard, replay, and persistence behavior, not registry mutation.

## The deployment bundle

Serving in the demo uses `TURBINE_GUARD_MODEL_SOURCE=deployment_bundle`: the champion is loaded
from an **immutable exported champion snapshot**, not a live registry. The bundle is a `tar.gz`
(about 1 MB) containing exactly what serving needs, each file checksummed in a typed
`deployment_manifest.json`:

* `models/cmapss/FD001/models/champion.joblib` — the checksummed Loop 4 `ModelBundle`;
* `features/cmapss/FD001/feature_manifest.json` and `split_manifest.json` — the shared serving
  feature contract and split provenance;
* `processed/cmapss/FD001/train_FD001.parquet` and `processing_report.json` — the verified replay
  trajectory source, so the Loop 8 integrity chain still verifies end to end.

The manifest also captures the registry identity at export time: registered model name, version,
champion alias and all alias assignments, source run ID, model family, target definition, RUL cap,
feature version/count, metrics (validation/replay/official RMSE, conformal target), git commit,
lineage ID, and the dataset/feature-manifest checksums. The dashboard models page labels this as an
exported snapshot; the public deployment cannot mutate any registry.

Export from a fully bootstrapped local stack (live MLflow reachable):

```bash
uv run python scripts/deployment_bundle.py export
```

Export loads the champion through the normal MLflow serving path first and refuses to package a
`champion.joblib` whose checksum differs from the registered version. The command logs the archive
path and its SHA-256 pin.

Publish the archive as a revision-pinned artifact. A Hugging Face dataset repository fits the
portfolio framing:

```bash
hf upload <user>/turbine-guard-demo-bundle \
  data/deployment/turbine-guard-deployment-bundle.tar.gz turbine-guard-deployment-bundle.tar.gz \
  --repo-type dataset
```

Then pin the *commit-revision* URL, never a moving branch:

```text
https://huggingface.co/datasets/<user>/turbine-guard-demo-bundle/resolve/<commit-sha>/turbine-guard-deployment-bundle.tar.gz
```

A versioned GitHub Release asset URL works identically. The application repository continues to
contain no generated data or model artifacts. Shipping a new champion to the demo is an explicit
release step: export, publish a new revision, update the pinned URL and SHA-256 in Render, redeploy.

## Cold start sequence

`render.yaml` starts the container with `python scripts/start_demo.py`, which:

1. downloads the pinned bundle and verifies the archive SHA-256 against
   `TURBINE_GUARD_DEPLOYMENT_BUNDLE_SHA256` (safe stdlib tar extraction; every file re-verified
   against the deployment manifest; completion marker written last, so an interrupted restore is
   retried, and a corrupted file is healed on the next start);
2. runs `alembic upgrade head` against Neon (idempotent; free plans have no pre-deploy phase, and
   the single-instance service cannot race its own migrations);
3. starts the normal API entry point.

Startup never downloads NASA data, never trains, and never registers models. No demo seeding is
required: dashboard pages present explicit empty states, the first visitor-driven replay creates
data through the bounded public controls, and everything written afterwards persists in Neon
across compute restarts and idle sleep.

## Persistence matrix

| Data | Location | Restart behavior |
| --- | --- | --- |
| Assets, readings, predictions, maintenance events | Neon PostgreSQL | Durable |
| Replay run/progress, outcomes, evaluations | Neon PostgreSQL | Durable |
| Quality, drift, online-performance reports and lifecycle audit | Neon PostgreSQL | Durable |
| Champion model, feature/split manifests, replay source | Deployment bundle | Immutable; re-restored and re-verified on every cold start |
| Registry identity, aliases, metrics, lineage | `deployment_manifest.json` in the bundle | Immutable snapshot from export time |
| Live MLflow experiments/registry | Local Docker Compose only | Not part of the public demo |
| Container filesystem, caches, temporary files | Ephemeral | Discarded safely |
| Application logs | Render logs | Subject to workspace retention, not application state |

Take periodic `pg_dump` exports of the Neon database if the accumulated demo history matters; the
bundle needs no backup because it is a published, pinned artifact.

## First deployment

1. Create a free Neon project and database (`turbine_guard`). Copy the standard `postgresql://`
   connection string; settings normalize the scheme to `postgresql+psycopg://`. Outbound port 5432
   from Render to Neon is unrestricted.
2. Export and publish the deployment bundle as above; note the pinned URL and SHA-256.
3. Create the Render Blueprint from `render.yaml` (one free web service; no payment approval is
   requested). Enter the three `sync: false` values in the Render dashboard:
   `TURBINE_GUARD_DATABASE_URL`, `TURBINE_GUARD_DEPLOYMENT_BUNDLE_URL`,
   `TURBINE_GUARD_DEPLOYMENT_BUNDLE_SHA256`. Render generates
   `TURBINE_GUARD_APPLICATION_SECRET` itself.
4. Deploy. Watch the start logs for `deployment_bundle_restored` and `demo_migrations_applied`.
5. Verify `https://<service>.onrender.com/health/live`, `/health/ready`, `/dashboard`, `/docs`,
   and `/v1/models/current`.
6. Start the predefined replay from the dashboard and advance a few accelerated batches so the
   fleet page has content for the next visitor.

Secrets live only in the Render and Neon dashboards. Rotate by updating the value there,
redeploying, verifying readiness, then revoking the old value. Never commit a connection string;
`render.yaml` deliberately contains no credential, bundle URL, or checksum.

## Health, logs, and restart behavior

* `/health/live` proves the process responds and is the Render routing check.
* `/health/ready` checks PostgreSQL, champion loading, and exact feature compatibility; a
  dependency failure returns 503.
* Dashboard routes show a bounded friendly degraded message and no traceback.
* Structured logs contain request IDs, routes, statuses, and bounded labels—not payloads or secrets.
* PostgreSQL replay leases expire and idempotent resume reconciles an interrupted cycle, so a
  mid-replay spin-down is safe.

## Free-tier tradeoffs (stated openly)

* The service sleeps after idle minutes; the first visit after sleep waits for a cold start
  (container boot, bundle restore, migration check — the bundle is ~1 MB, so this is seconds of
  restore on top of Render's own spin-up).
* Render free instances have 512 MB memory; bundle mode never imports MLflow, keeping the serving
  process within it.
* Monthly free allowances bound compute (Render free hours, Neon compute units). A dormant demo
  costs nothing; a heavily shared one can hit the caps.
* The public model registry is read-only by construction: no promotion, retraining, or alias
  mutation is possible on the demo. These remain demonstrable in the local Compose stack.
* Neon free-tier limits (0.5 GB, connection budget) are far above what the bounded demo writes.

## Troubleshooting

* **Liveness works, readiness fails:** inspect the readiness JSON. Check the start logs for a
  bundle restore or migration failure; a wrong SHA-256 pin or stale bundle URL fails loudly with
  `deployment_bundle_failed`.
* **`checksum mismatch` at start:** the pinned URL and SHA-256 disagree — usually a new bundle was
  published without updating both values. Update them as a pair.
* **Database URL rejected:** paste Neon's standard connection string into the Render environment;
  settings normalize the scheme. Do not put it in Git.
* **Cold start feels slow:** expected on the free plan after idle sleep; the recorded Loop 12 demo
  GIF/video is the zero-latency fallback for first impressions.
* **Replay rejects a source asset:** only the configured replay-split demo ID is public. This is a
  safety control, not a data-loading error.
* **Replay reports a conflict:** another request owns the short PostgreSQL lease. Wait for it to
  finish or expire and retry; do not delete the run.
* **Neon project suspended/idle:** Neon wakes automatically on the next connection; no action is
  needed.
