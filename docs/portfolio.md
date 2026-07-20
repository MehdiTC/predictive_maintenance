# Portfolio Summary

A recruiter-facing, one-page summary of TurbineGuard: the problem, the architecture, the model, the
outcome, how to reproduce it, and how to talk about it. It is an independently developed platform
inspired by industrial predictive-maintenance use cases, built on public NASA data with no
proprietary client data or implementation details.

## The one-minute version

**Problem.** Industrial turbines fail; reactive maintenance is expensive and unplanned. TurbineGuard
predicts each engine's Remaining Useful Life (RUL) from streaming sensor cycles and turns those
predictions into timely warning/critical maintenance alerts.

**System.** A full MLOps loop on public NASA C-MAPSS turbofan data: reproducible data acquisition →
leakage-safe feature engineering → multi-model training and evaluation → MLflow tracking/registry →
FastAPI online inference on PostgreSQL → held-out trajectory replay with delayed failure labels →
drift/performance monitoring → gated retraining and promotion → containerized local stack + CI →
a public dashboard and a zero-cost live demo.

**Model.** Four model families (constant baseline, Ridge, histogram gradient boosting, XGBoost) are
trained on identical engine-level splits and compared on explicit operational criteria. A Ridge
champion (RUL capped at 125) wins by a simplicity-tolerance rule — chosen by policy, not hard-coded.

**Outcome (all on public/simulated data).** Held-out replay RMSE ≈ 14 cycles with ~90% conformal
interval coverage; critical-alert recall ≈ 0.77 at under 1 false alarm per 1,000 cycles; and a
simulated predictive maintenance policy that is ~64% cheaper than reactive in normalized units with
zero missed failures. Live demo: <https://turbine-guard-web.onrender.com>.

## What it demonstrates (skills)

* **ML systems engineering**, not just modeling: data lineage, training-serving consistency,
  online/offline parity, delayed feedback, monitoring, and gated retraining.
* **Leakage discipline**: engine-level splits, held-out replay/calibration engines, features that
  depend only on past cycles, and explicit tests that future data cannot change past features.
* **Evaluation rigor**: MAE/RMSE plus NASA asymmetric score, failure-horizon precision/recall,
  false-alarm rate, alert lead time, lifecycle slices, calibrated intervals, and a maintenance-cost
  simulation — not a single aggregate metric.
* **Production surface**: versioned FastAPI with idempotent ingestion, readiness that proves
  dependencies, structured logging, Prometheus metrics, Alembic migrations, Docker Compose, and CI.
* **Architecture judgment**: Kafka/Kubernetes/Spark/feature-store/React were considered and
  deliberately excluded for a single-model system; the scale-up path is documented, not built.

## Resume bullets

Copy-ready, honest, and backed by reproducible artifacts in this repo:

* Built an end-to-end predictive-maintenance MLOps platform (Python, FastAPI, PostgreSQL, MLflow,
  Docker, GitHub Actions) that predicts turbine Remaining Useful Life from streaming sensor data and
  issues maintenance alerts, using public NASA C-MAPSS data.
* Engineered a leakage-safe time-series feature pipeline (552 trailing-window features) shared
  identically between offline training and online serving, with explicit tests proving future
  observations cannot alter past features.
* Trained and compared four model families on engine-level splits and selected a champion by
  explicit operational gates and a simplicity-tolerance rule; evaluated with NASA asymmetric score,
  failure-horizon precision/recall, false-alarm rate, lead time, and conformal prediction intervals
  (~90% empirical coverage).
* Implemented continuous trajectory replay with delayed failure labels, drift/performance
  monitoring against a training-only reference, and gated retraining/promotion with human approval,
  MLflow reload-equivalence checks, and rollback.
* Deployed a public, zero-cost live demo (Render + Neon + an immutable checksum-pinned model bundle)
  serving predictions identical to the tracked registry champion (max difference 0.0).

## Interview talking points (Q → A)

**Why does a linear model beat gradient-boosted trees here?** Selection uses a 2% RMSE tolerance
band and then picks the least-complex candidate; on FD001 Ridge lands inside that band, so the
policy prefers it. The point is the *policy*, not the winner — the champion is reproducibly chosen,
never hard-coded.

**How do you prevent data leakage?** Four structural defenses: split by engine (never by row),
hold replay and calibration engines out of training, compute features at cycle `t` from cycles `≤ t`
only, and hide the failure cycle from the inference path. Each is covered by a test, including
exhaustive per-cycle future-mutation checks.

**How do you guarantee training-serving consistency?** One `FeatureBuilder` implementation serves
both batch training and single-cycle online inference; offline and incremental outputs are proven
equal at every cycle. There is no separate notebook/production feature code.

**Why not Kafka/Kubernetes/Spark?** For one replay producer and one model at portfolio scale they
add configuration cost without ML value. Each has a documented trigger condition and a clean seam to
add it (see [scaling.md](scaling.md)) — the ingestion idempotency, stateless API, pure feature
function, and alias-based registry are already the primitives those tools would build on.

**What are the maintenance numbers, really?** Normalized, hypothetical units — not currency and not
claimed savings. The predictive policy is ~64% cheaper than reactive in the base scenario with zero
missed failures, but I report low/base/high failure-cost scenarios because the advantage is
assumption-sensitive.

**What's the honest online performance?** Against realized *uncapped* RUL the online metrics are
worse than capped replay by construction (early-life rows exceed the 125 cap). I report that view
explicitly rather than only the flattering capped-domain numbers.

## One-command local reproduction

Prerequisite: [uv](https://docs.astral.sh/uv/) (`brew install uv`). Then, from the repo root:

```bash
uv sync            # create the environment (Python 3.12 + locked deps)
make reproduce     # acquire -> process -> features -> train
```

`make reproduce` downloads the public NASA archive, validates and processes it, builds the
leakage-safe feature layer, and trains/evaluates every model family, regenerating all metrics under
`data/models/cmapss/FD001/reports/`. Every figure in the [model card](model_card.md) comes from
those files. Add MLflow tracking and verify registry equivalence with:

```bash
make train-tracked && make mlflow-verify   # registry champion == local bundle (diff 0.0)
```

The full multi-service stack (API, PostgreSQL, MLflow, replay, monitoring) runs with Docker Compose;
see [containers.md](containers.md). The live public demo needs nothing installed:
<https://turbine-guard-web.onrender.com>.

## A note on the demo recording

The live dashboard demonstrates continuous replay in the browser. A short screen-recorded GIF/video
of that flow is the one portfolio asset that must be captured by hand from the running demo.
Everything else on this page is reproducible from the repository.
