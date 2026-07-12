# ADR 0002 — Loop 3 labels, asset-level splits, and leakage-safe features

**Status:** Accepted (2026-07-12)

## Context

Loop 3 turns the validated FD001 Parquet trajectories (Loop 2) into reproducible, model-ready
feature tables: RUL labels, asset-level splits, and one shared feature-generation component. The
dominant risk is time-series leakage — using future observations to build a feature for an
earlier prediction — plus training-serving skew if offline and online feature logic diverge.
Several design choices needed fixing; this ADR records them.

No new dependencies were added: pandas, NumPy, PyArrow, Pydantic, and the standard library cover
the whole loop. Scikit-learn (for splitting), time-series feature libraries, and feature stores
were all considered and rejected as unnecessary and explicitly out of scope for the core project.

## Decisions

### 1. Split proportions and determinism

Training assets are partitioned **by `asset_id`, never by row**, into
**train 70 % / validation 15 % / calibration 5 % / replay 10 %**. For FD001's 100 training
engines this is exactly 70/15/5/10. Rationale:

* Row-level splitting leaks a trajectory across train and validation.
* Calibration is held separate now because Loop 4 uses split conformal prediction, which needs a
  calibration set disjoint from training and validation.
* Replay is held out entirely so Loop 8 can stream those engines as "live" data with delayed
  labels; they must never influence initial fitting, feature selection, calibration, or
  thresholds.

Determinism uses `numpy.random.default_rng(seed)` (a reproducible PCG64 stream) to permute the
sorted unique asset IDs, then largest-remainder rounding to convert fractions into integer counts
that always sum to the asset count. The seed is configurable; the strategy and asset lists are
recorded in the split manifest.

### 2. Missing-value policy: preserve rows, defer imputation to Loop 4

Rolling and lagged features are structurally undefined for early cycles (e.g. the first-cycle
delta, or rolling std with a single observation). The policy is:

* **Preserve every early-cycle row** — nothing is silently dropped.
* Leave structurally-unavailable values **null**.
* **Do not impute or scale in Loop 3.** Imputation and model-specific scaling are deferred to the
  Loop 4 model pipeline, where they can be fit on training assets only and applied identically to
  the other splits.

This keeps Loop 3 free of any fitted state, which in turn makes the fit-isolation and
training-serving-consistency guarantees structural rather than procedural. The manifest records
`imputation: null` to make the deferral explicit. `min_periods` defaults to 1 so rolling
mean/min/max are defined from the first cycle; delta, rolling std, and rolling slope are naturally
null or degenerate on the first cycle and documented as such.

### 3. Rolling-window and slope semantics

All windows are **trailing** (right-aligned, ending at the current cycle) and computed **grouped
by asset** so they never cross asset boundaries — no centered or forward-looking windows exist.
Rolling slope is the ordinary-least-squares slope of value against cycle over the trailing window,
computed vectorized from grouped rolling sums (no per-row Python loop). A single-observation or
otherwise degenerate window yields a deterministic slope of `0.0`; windows below `min_periods`
remain null. The only division (the slope denominator) is guarded, so no infinities are produced.

### 4. RUL capping is optional and off by default

The uncapped `rul = T_i − t` target is always produced. A capped `rul_capped = min(rul, cap)`
target is generated only when a cap is configured, and the uncapped column is always preserved.
The cap is a single configurable value, not hard-coded. Loop 4 will compare raw vs capped RUL as a
hyperparameter; Loop 3 only needs to be able to produce both.

### 5. Stateless FeatureBuilder with a history-replaying incremental path

`FeatureBuilder` holds **no fitted state**; a feature at cycle `t` depends only on that asset's
observations at cycles `≤ t`. The single-asset incremental interface (`IncrementalFeatureState`)
retains the asset's observation history and recomputes through the exact same batch code, so the
online current-cycle row equals the offline batch row **by construction** (verified by tests at
every cycle). Trajectories here are short (≤ 362 cycles), so full-history retention is cheap and
keeps the `adjust=True` EWM features exactly equal to the batch values. A production system with
very long histories could bound the buffer and keep a running EWM accumulator; that optimization
is deliberately out of scope.

### 6. Typed-dataclass configuration, not YAML

The feature/split/RUL configuration is expressed as frozen dataclasses (`FeatureConfig`,
`SplitConfig`, `RulConfig`, `BuildConfig`), matching the established `AcquisitionConfig` /
`ProcessingConfig` convention. The repository has no YAML configuration layer yet; introducing one
here would add a new pattern (and a parsing surface) for a handful of values that are clearer and
type-checked as code. The `PROJECT_SPEC.md` `configs/*.yaml` sketch remains a future option if a
loop genuinely needs externalized, non-code configuration.

## Consequences

* Leakage protections (future-row mutation, future-row append, cross-asset isolation,
  fit-isolation, replay exclusion) and offline-vs-incremental equality are covered by explicit
  tests and hold structurally.
* Feature and split manifests tie every output to the exact validated inputs (by checksum), the
  feature version, and the split, so a future training run can identify precisely what it used.
* Model-ready outputs live under `data/features/cmapss/FD001/` (gitignored), are checksummed, and
  regenerate idempotently with tamper detection and a `--force` rebuild — the same guarantees as
  the Loop 1/2 layers.
* Imputation and scaling remain a Loop 4 responsibility; downstream code must handle nulls in the
  early-cycle feature rows (or drop/​impute them inside the model pipeline).
