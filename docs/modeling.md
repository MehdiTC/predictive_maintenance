# Offline RUL Modeling and Evaluation (Loop 4)

Loop 4 consumes the checksummed Loop 3 feature layer and produces reproducible local models and
reports. It implements no experiment tracker, registry, API inference, online replay, database,
monitoring, or deployment functionality.

## Dataset-role boundary

```text
Loop 3 model-ready features
        -> preprocessing fitted on training rows only
        -> bounded candidate training
        -> validation-only comparison and champion selection
        -> calibration-only conformal residual quantile
        -> frozen replay and official-final-row evaluation
        -> simulated maintenance-policy report
        -> checksummed local artifacts
```

| Role | Assets | Permitted use |
| --- | ---: | --- |
| Train | 70 | Fit preprocessing and point models |
| Validation | 15 | Compare configurations and select the champion/policy |
| Calibration | 5 | Fit only the conformal residual quantile |
| Replay | 10 | Final trajectory, alert, interval, and simulated-policy evaluation |
| Official NASA test | 100 | One final-row RUL benchmark prediction per asset |

The training command verifies feature/split manifests, every Loop 3 output checksum, exact column
order, split labels, row and asset counts, disjoint asset IDs, finite/non-infinite model inputs,
and the one-to-one official final-row label join. Features are never inferred by selecting numeric
columns: the manifest's ordered feature list is authoritative. The only accepted non-finite model
input is the documented structural `NaN` pattern.

## Missing values and models

Early-cycle rows remain present. Ridge uses a scikit-learn pipeline with training-fitted median
imputation, missing indicators, and standard scaling. Histogram gradient boosting and XGBoost use
native missing-value support. Nothing is fitted on validation, calibration, replay, or official
test features.

Candidates are:

* Constant training-median RUL baseline.
* Ridge at two regularization strengths.
* Histogram gradient boosting at two bounded tree/iteration sizes.
* XGBoost at two bounded depth/iteration sizes, CPU histogram mode, and one worker.

The manual grid is intentionally small and fully recorded in `candidate_comparison.json`. Fixed
seeds, disabled histogram-gradient early stopping, and fixed XGBoost CPU settings support
deterministic reruns on a normal laptop.

## Targets and champion comparison

Experiments use uncapped RUL and capped RUL `min(rul, 125)`. Capped predictions represent the
capped target and cannot recover early-life values above 125. Target-specific metrics always use
matching truth. Cross-target selection uses rows with uncapped RUL at or below 125 so its RMSE,
MAE, and NASA score are comparable.

Candidates first pass validation operational gates (critical recall and false alarms per 1,000
cycles). The lowest common-domain RMSE defines a 2% tolerance band. The least complex candidate in
that band wins; NASA score, MAE, latency, artifact size, and stable candidate ID break remaining
ties. No replay or official-test field is accepted by selection logic.

## Regression and NASA metrics

Reports contain MAE, RMSE, contextual R-squared, and the NASA asymmetric score overall,
asset-balanced, per asset, by early/middle/late lifecycle third, and by short/long trajectory.
Lifecycle and trajectory length use realized truth for evaluation only and are never model inputs.

For error `d = predicted RUL - true RUL`, the NASA score is:

```text
d < 0: exp(-d / 13) - 1
d >= 0: exp(d / 10) - 1
score = sum of row terms
```

Overprediction is penalized more steeply because it can delay maintenance.

## Alerts and lead time

Defaults are critical within 30 cycles and warning within 50 cycles. A row alerts when predicted
RUL is at or below its horizon. PR-AUC uses `-predicted_rul` as the continuous risk score.
Consecutive positive rows form one episode. Operational metrics use the first episode per asset:

* First-alert lead time is true uncapped RUL at episode start.
* Lead above the horizon is too early.
* Lead from 1 through the horizon is timely.
* Lead below one cycle is too late.
* No usable first alert is a missed failure.
* False-alarm episodes start before the true asset enters the horizon; their count per 1,000
  observed cycles is reported separately from row false positives.

No extra probability threshold is needed: configured RUL horizons are the fixed decision rule.

## Conformal intervals

For nominal coverage `c`, calibration absolute residuals are sorted and rank
`ceil((n + 1) * c)` is selected (bounded by `n`). The symmetric interval is prediction plus/minus
that residual, with the lower bound clipped at zero and the upper bound clipped to the selected
target cap when applicable. Reports show empirical coverage and average/median width overall and
by lifecycle stage.

The method treats calibration rows as the calibration units. Rows within an asset are temporally
dependent, so the strict exchangeability premise of classical split conformal prediction does not
hold. Coverage is therefore an empirical, useful approximation, not a formal trajectory-level
guarantee.

## Simulated maintenance policy

Reactive maintenance pays unplanned-failure and downtime-event costs at each asset failure.
Predictive maintenance intervenes on the first critical episode when it has at least one cycle of
lead; it pays inspection, planned repair, and early-replacement cost for forfeited remaining life.
No/late alerts pay unplanned failure, downtime, and missed-failure cost. Base, lower-failure-cost,
and higher-failure-cost scenarios are reported per asset and in aggregate.

All costs are normalized hypothetical units. They are not currency, Genelba values, or claims of
real savings.

## Artifacts and reproduction

```bash
uv run python scripts/train_models.py
# options: --data-dir DIR --output-dir DIR --seed N --rul-cap N
#          --critical-horizon N --warning-horizon N
#          --conformal-coverage FLOAT --force
```

Default outputs are under `data/models/cmapss/FD001/`:

```text
models/
  champion.joblib
  champion_metadata.json
  <target>--<candidate>.joblib
reports/
  candidate_comparison.{json,csv}
  validation_report.json
  champion_selection.json
  replay_evaluation.json
  official_test_benchmark.json
  conformal_metrics.json
  maintenance_simulation.json
  model_interpretation.json
  model_latency_size.json
  slice_metrics.csv
  feature_importance_coefficients.csv
  evaluation_summary.md
training_manifest.json
```

`training_manifest.json` records the Git SHA, evaluation/configuration versions, dataset/feature/
split checksums, every Loop 3 input checksum, champion ID, and checksums/sizes for all outputs.
An unchanged rerun verifies everything and returns `already_trained`; modified outputs fail rather
than silently repair. Use `--force` only for an intentional rebuild.

Joblib/pickle deserialization can execute arbitrary code. Load only trusted artifacts after
verifying their manifest checksum.

## Interpretation and limitations

Reports include standardized Ridge coefficients and built-in XGBoost importance, plus aggregation
by original anonymous source channel. Scaling, collinearity, and tree split mechanics affect these
values; they are not causal. A concentrated top-feature share is noted as a future ablation
hypothesis, but Loop 4 does not redesign the Loop 3 feature contract.

Other limitations: FD001 is simulated; official evaluation has only one labeled final row per
test asset; replay costs are assumption-sensitive; and no result here demonstrates online or
industrial performance.
