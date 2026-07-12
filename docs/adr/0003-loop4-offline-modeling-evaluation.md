# ADR 0003 — Loop 4 offline modeling, alerts, conformal calibration, and serialization

**Status:** Accepted (2026-07-12)

## Context

Loop 4 must train several RUL models against identical Loop 3 asset partitions, handle structural
early-cycle nulls without leakage, compare raw and capped targets, calibrate uncertainty, and
translate predictions into maintenance-oriented evidence. It must remain a local offline boundary:
MLflow, a registry, serving, replay infrastructure, and monitoring begin in later loops.

## Decisions

### Training-only preprocessing and candidate models

The manifest's ordered 552-feature list is the sole model matrix contract. Identifiers, metadata,
and targets are rejected if they enter that list, and exact schemas/checksums are verified before
fitting. Ridge uses median imputation with missing indicators and standard scaling, all fitted only
on training rows. Histogram gradient boosting and XGBoost use their native missing-value handling
without scaling. The constant model ignores features and learns only the training-target median.

Histogram gradient boosting is the primary sklearn tree baseline: it supports structural nulls,
has bounded CPU-friendly training, and avoids the memory cost of a large bagged forest over 552
features. XGBoost is a principal candidate, not a presumed winner. Each approach has only one or
two deliberate configurations; no search framework is introduced.

### Target comparison

Both uncapped RUL and `min(RUL, 125)` are evaluated. A capped prediction estimates this
piecewise target and is clipped to `[0, 125]`; it is never presented as reconstructed uncapped
early-life RUL. Ordinary metrics always use the matching target truth. Because full-range RMSEs
for capped and uncapped targets are not comparable, champion ranking uses the common validation
domain `uncapped RUL <= 125`, where both definitions have identical truth. Alert horizons 30 and
50 also lie inside this common domain.

### Alert events and policy intervention

A row alerts when predicted RUL is at or below its configured horizon. Consecutive alert rows for
one asset collapse into an episode; the first episode controls maintenance intervention. Its lead
time is true uncapped RUL at episode start. Lead above the horizon is early, lead below one cycle
is late, and no usable alert (including a first alert at failure) is a missed failure. Reports keep
row precision/recall/F1/PR-AUC separate from episode and asset metrics.

The policy simulator compares failure-only reactive maintenance with intervention on the first
critical alert. Costs are named normalized units, never currency: unplanned failure, inspection,
planned repair, failure downtime, early replacement per forfeited cycle, and missed failure. A
small explicit sensitivity set is reported.

### Conformal design

After validation-only selection, the frozen champion predicts every calibration row. The interval
radius is the finite-sample corrected order statistic of absolute residuals at 90% nominal
coverage. All calibration rows are used because FD001 has only five calibration assets; selecting
one row per asset would make a 90% quantile poorly resolved. This is a documented row-level
approximation: temporal dependence within trajectories violates strict exchangeability, so formal
coverage guarantees are not overclaimed. Empirical coverage is reported overall and by lifecycle
stage on held-out data.

### Champion selection

Selection uses validation data only. Candidates must meet minimum critical recall and maximum
false-alarm rate. The best common-domain RMSE defines a relative tolerance band; within that band,
lower declared complexity wins, followed by NASA score, MAE, latency, size, and ID as deterministic
tie-breakers. Calibration, replay, and official-test results are computed only after selection and
cannot alter it.

### Serialization and local artifacts

Every candidate pipeline and a champion bundle are serialized with joblib. The bundle includes
preprocessing, point model, ordered features, target definition, alert horizons, and conformal
state. JSON/CSV/Markdown reports and all model files are checksummed in a completion manifest.
Reruns verify inputs and outputs and return `already_trained`; `--force` is the intentional rebuild
path. Joblib uses pickle semantics and can execute code while loading, so only trusted,
checksum-verified files may be loaded.

## Consequences

* Validation determines the point model and fixed alert horizons; calibration only determines the
  interval radius; replay and official test remain final held-out evidence.
* The full feature set remains stable for Loop 3 compatibility. Coefficients and built-in XGBoost
  importance are diagnostic associations, not causal explanations or an automatic feature redesign.
* Runtime dependencies added are scikit-learn, XGBoost, and joblib. `scikit-learn-stubs` is
  development-only for the repository's strict mypy contract. Scikit-learn brings required SciPy
  numerical routines transitively.
* No Loop 5 tracking or registration functionality exists.
