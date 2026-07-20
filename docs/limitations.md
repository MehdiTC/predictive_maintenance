# Limitations and Known Risks

TurbineGuard is a portfolio-grade demonstration of predictive-maintenance ML systems engineering. It
is deliberately honest about what it does and does not prove. This page consolidates the limitations
that are otherwise spread across the per-loop docs.

## Data and problem framing

* **Simulated data.** NASA C-MAPSS FD001 is *simulated* turbofan degradation data, not real
  power-plant or turbine sensor data. Nothing here demonstrates performance on physical assets.
* **Anonymous sensors.** The 21 sensor channels have no published physical meaning, so the project
  never assigns them one (no "vibration", "temperature", etc.). Feature and drift interpretation is
  by channel index only.
* **Single subset.** Only FD001 (single operating condition, single fault mode) is modeled. FD002–
  FD004 add operating regimes and fault modes and would need explicit handling; cross-subset
  comparison is backlog, not done.
* **One labeled final row per test engine.** The official NASA benchmark provides exactly one RUL
  label per test engine, so the official-test metrics are a small-sample, late-life view.

## Modeling and evaluation

* **Capped target.** The champion predicts RUL capped at 125 cycles and cannot recover true early-
  life RUL above 125. Capped and uncapped metrics are reported separately and are not comparable.
* **Conformal coverage is empirical.** Split-conformal intervals treat calibration rows as the
  calibration units, but rows within an engine are temporally dependent. Reported coverage (0.898
  on replay) is a useful empirical approximation, not a formal trajectory-level guarantee.
* **Feature interpretation is not causal.** Standardized Ridge coefficients and tree importances are
  affected by scaling, collinearity, and split mechanics. A concentrated top-feature share is noted
  as a future ablation hypothesis, not a causal claim, and the stable feature contract is not
  redesigned from importances alone.
* **Simulated costs.** All maintenance-policy costs are normalized, hypothetical units. They are not
  currency, not real industrial values, and not a claim of real savings. Results are sensitive to
  the assumed failure/repair/downtime costs, so multiple scenarios are always reported.

## Online, monitoring, and retraining

* **Online collection is simulated.** "Live" ingestion is historical trajectory replay through the
  real API, not a real sensor feed. Timestamps are deterministic simulations; no real-hour claim is
  implied.
* **Few labeled assets.** Only 10 held-out engines are available for delayed feedback. Retraining
  and promotion are deliberately blocked below configurable minimum-asset, minimum-row, and safe-
  holdout thresholds, so the live monitoring run correctly reaches a `blocked` decision rather than
  retraining on insufficient data.
* **Frozen conformal calibrator on retrain.** Retrained candidates reuse the champion's conformal
  calibrator because no new calibration role is available; poor promotion-holdout coverage blocks
  promotion instead.
* **Rejected inputs are not a durable stream.** The API stores accepted readings, so data-quality
  reporting can only count rejected inputs when its producer supplies them explicitly.

## Infrastructure and deployment

* **MLflow registry is single-host SQLite.** In both local host and Compose modes the MLflow
  metadata store is SQLite; it is persistent and service-owned under Compose but is not a
  high-availability registry. The serving-model cache refresh is per API process.
* **Public demo is intentionally reduced.** The zero-cost public deployment (ADR 0011) serves an
  immutable, checksum-pinned bundle instead of a live registry, so promotion and retraining are not
  available publicly — only in the local Compose stack, and the dashboard says so. Free-tier idle
  sleep/cold-start, an ephemeral compute filesystem (the bundle is re-restored on start), and
  bounded monthly allowances are accepted tradeoffs.
* **Image size.** The production image is ~696 MB because the locked scientific/ML runtime is
  included; shrinking it must preserve locked reproducibility and all model families.
* **Upstream deprecation warnings.** FastAPI's TestClient import, joblib's NumPy shape assignment on
  reload, and several MLflow warnings are known upstream noise; tests, model loading, prediction
  equality, and the UI all pass regardless.

## Source stability

NASA hosting for C-MAPSS has moved before. The source URL is configuration
(`TURBINE_GUARD_CMAPSS_SOURCE_URL`), and `file://` acquisition provides a manual fallback if the
mirror moves again.

## What would need domain validation before real use

Real deployment would require: real sensor data with documented physical channels, calibration
against real failure distributions, cost parameters agreed with maintenance stakeholders,
prospective (not replayed) online evaluation, formal exchangeability handling for prediction
intervals, and a governed, highly-available registry. None of that is claimed here.
