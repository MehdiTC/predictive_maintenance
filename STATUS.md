# STATUS.md

## Project Status

**Project:** TurbineGuard
**Current phase:** Loop 3 implemented and validated
**Active loop:** None — Loop 3 complete, awaiting review before Loop 4
**Overall status:** Foundation, reproducible acquisition, validated Parquet processing, EDA, RUL labels, deterministic asset-level splits, and a leakage-safe shared feature pipeline in place; no models yet
**Last updated:** 2026-07-12

---

## Project Objective

Build a public, production-style predictive-maintenance ML platform using NASA turbine degradation data.

The completed system will demonstrate:

* Reproducible data acquisition
* Data validation and processing
* Exploratory data analysis
* Leakage-safe time-series feature engineering
* Remaining Useful Life modeling
* Maintenance-oriented evaluation
* Experiment tracking
* Model versioning
* FastAPI deployment
* PostgreSQL persistence
* Continuous sensor replay
* Delayed outcome collection
* Drift and performance monitoring
* Conditional retraining
* Model promotion gates
* Dockerized local deployment
* CI/CD
* A public dashboard and demo

See `PROJECT_SPEC.md` for the complete design.

---

## Current Repository State

Loops 0–3 are implemented:

```text
├── src/turbine_guard/
│   ├── api/                    # create_app() factory, health routes/schemas
│   ├── config/settings.py      # typed BaseSettings (+ data_dir, cmapss_source_url)
│   ├── data/
│   │   ├── acquisition.py      # idempotent C-MAPSS FD001 acquisition (+ public verify_raw_layer)
│   │   ├── manifest.py         # provenance manifest models + persistence
│   │   ├── schema.py           # canonical column names/dtypes (SCHEMA_VERSION 1)
│   │   ├── parsing.py          # typed parser with line-numbered ParseError
│   │   ├── validation.py       # structured checks + stats; optional FD001 canonical profile
│   │   ├── processing.py       # verify → parse → validate → Parquet + JSON report
│   │   ├── cli.py              # acquisition CLI
│   │   └── process_cli.py      # processing CLI
│   ├── features/               # Loop 3: labels, splits, features (model-free)
│   │   ├── config.py           # frozen-dataclass FeatureConfig/SplitConfig/RulConfig/BuildConfig
│   │   ├── labels.py           # RUL = T_i − t, optional cap, validation, test benchmark
│   │   ├── splits.py           # deterministic asset-level 70/15/5/10 partitioning
│   │   ├── builder.py          # stateless FeatureBuilder + IncrementalFeatureState
│   │   ├── manifest.py         # SplitManifest + FeatureManifest models/persistence
│   │   ├── pipeline.py         # verify inputs → labels → split → features → outputs
│   │   └── build_cli.py        # feature-build CLI
│   ├── services/health.py
│   └── logging_config.py       # structured JSON logging
├── scripts/download_data.py     # thin wrappers over turbine_guard CLIs
├── scripts/process_data.py
├── scripts/build_features.py
├── notebooks/01_eda.ipynb       # the single primary EDA notebook, executed
├── docs/data_contract.md        # raw structure, canonical schema, validation rules, outputs
├── docs/features.md             # Loop 3 contract: labels, splits, features, manifests
├── docs/adr/0001-…, 0002-…      # Loop 2 + Loop 3 decision records
├── tests/                       # 183 tests: unit (offline) + local FD001 integration (auto-skip)
├── data/                        # gitignored: raw, manifests, processed, features
└── pyproject.toml / uv.lock / Makefile / .pre-commit-config.yaml / .env.example / README.md
```

Data layers on disk after `make acquire && make process && make features`:

```text
data/
├── raw/cmapss/FD001/            # immutable, read-only (0444), checksum-verified
├── manifests/cmapss_fd001.json
├── processed/cmapss/FD001/
│   ├── train_FD001.parquet      # 20,631 rows, 100 engines, canonical schema
│   ├── test_FD001.parquet       # 13,096 rows, 100 engines
│   ├── rul_FD001.parquet        # 100 official test RUL values
│   └── processing_report.json   # machine-readable checks/stats/checksums/provenance
└── features/cmapss/FD001/
    ├── train.parquet            # 14,407 rows, 70 engines: ids, split, rul, 552 features
    ├── validation.parquet       # 3,160 rows, 15 engines
    ├── calibration.parquet      # 909 rows, 5 engines
    ├── replay.parquet           # 2,155 rows, 10 engines (isolated for Loop 8)
    ├── test_features.parquet    # 13,096 rows, 100 engines, no targets
    ├── test_labels.parquet      # 100-row official RUL benchmark (evaluation only)
    ├── split_manifest.json      # seed, strategy, asset IDs/counts/rows per partition
    └── feature_manifest.json    # feature definition, columns, checksums, provenance
```

No models, database, MLflow, Prefect, replay service, monitoring, Docker, or CI functionality exists yet — deliberately.

---

## Current Loop

None active. Loop 3 is complete and awaiting review. The next loop (Loop 4 — Baseline Modeling and Evaluation) must not begin without explicit approval.

---

## Completed Work

* [x] Defined the overall project mission, dataset, target, and architecture (planning).
* [x] Implemented Loop 0 — repository foundation (2026-07-12).
* [x] Implemented Loop 1 — dataset acquisition and manifesting (2026-07-12).
* [x] Implemented Loop 2 — validation, processing, and EDA (2026-07-12).
* [x] Implemented Loop 3 — labels, asset-level splits, and leakage-safe features (2026-07-12).

---

## Loop 3 Implementation Notes

1. **RUL labels** (`features/labels.py`): uncapped `rul = T_i − t` (int64) always generated; optional `rul_capped = min(rul, cap)` only when a cap is configured (CLI `--rul-cap`), uncapped always preserved. Validation enforces non-negativity, RUL 0 at each final cycle, and exact −1 steps. The official test set gets **no fabricated per-row labels**: `test_labels.parquet` is a 100-row evaluation benchmark (`asset_id`, `final_cycle`, `rul`) encoding the positional row *i* ↔ asset *i+1* correspondence explicitly (asserted, documented as unverifiable from content).
2. **Splits** (`features/splits.py`): by `asset_id`, never by row. Default seed 42, `numpy.random.default_rng` permutation of sorted unique IDs, largest-remainder rounding. FD001: exactly **70 train / 15 validation / 5 calibration / 10 replay** engines (14,407 / 3,160 / 909 / 2,155 rows). `AssetSplit` construction proves disjointness; replay and calibration are isolated; the official test set is untouched.
3. **FeatureBuilder** (`features/builder.py`) is **stateless**: features at cycle `t` are a pure function of that asset's observations ≤ `t` (trailing windows, grouped per asset, sorted internally). This makes leakage-safety, cross-asset isolation, fit isolation, and training-serving consistency structural. `IncrementalFeatureState` feeds one cycle at a time through the *same* batch code; offline == incremental at every cycle (tested, incl. restart via `from_history`).
4. **Features:** 24 source columns × 23 = **552 features**: current, delta_1, rolling mean/std/min/max/range/slope over windows 5/10/20, EWM mean spans 5/10/20 (`adjust=True`). Families/windows/spans/min_periods configurable; deterministic names and stable order; `feature_columns()` is the authoritative contract. Rolling slope = OLS of value vs cycle over the trailing window, vectorized via grouped rolling sums; degenerate fits → 0.0; guarded division, no infinities.
5. **Early-cycle/missing-value policy** (ADR 0002): early rows preserved; structurally undefined values (first-cycle delta, single-observation std) left **null**; **no imputation or scaling in Loop 3** — deferred to the Loop 4 model pipeline (fit on train assets only). Manifest records `imputation: null` and per-file null counts.
6. **Outputs** (`features/pipeline.py`): six Parquet files + `split_manifest.json` + `feature_manifest.json` under `data/features/cmapss/FD001/`. Column order fixed: identifiers, `split`, targets (absent from test features), then features. Loop 2 inputs are checksum-verified before building; outputs are atomic, checksummed, idempotent (`already_built`), tamper-detected, `--force` rebuilds — mirroring Loops 1–2. Config changes (seed, cap, feature config) correctly trigger rebuilds.
7. **Manifests:** split manifest records seed/strategy/version, asset IDs+counts+row counts per partition, source-report checksum, git commit, timestamp. Feature manifest records feature/split/schema versions, full feature configuration, ordered feature columns, identifier/target/metadata column groups, RUL cap, imputation policy, inputs and outputs by SHA-256, row/asset counts by split, null counts, git commit, timestamp, seed — sufficient for a future training run to identify exactly what it used.
8. **Configuration** follows the established frozen-dataclass convention (`FeatureConfig`, `SplitConfig`, `RulConfig`, `BuildConfig`) — no YAML layer introduced (ADR 0002 §6); no scattered constants.
9. **Dependencies:** none added. pandas/NumPy/PyArrow/pydantic/stdlib only; scikit-learn, tsfresh-style libraries, feature stores all rejected as unnecessary.
10. **Tests:** 77 new (183 total). Labels, splits (determinism, coverage, isolation), feature families (each verified against manual/pandas reference), unsorted input, asset-boundary isolation, config effects, leakage (future-row mutation, future-row append incl. exhaustive per-cycle perturbation, cross-asset, fit isolation, replay exclusion), offline-vs-incremental equality at every cycle + restart reconstruction, persistence (round-trip, idempotency, tamper, force, input immutability), CLI (success, rerun, missing input, invalid config → exit 1). Three real-FD001 integration tests (auto-skip without local data).

---

## Loop 2 Implementation Notes

1. **Canonical schema** (`schema.py`, version 1): `asset_id`/`cycle` as int64; `operating_setting_1..3` and `sensor_01..21` as float64. Sensors stay anonymous — no physical interpretations assigned (dataset documentation provides none).
2. **Parser** (`parsing.py`): line-level whitespace splitting with exact field-count enforcement and line-numbered `ParseError`s; no silent row drops; trailing whitespace and blank lines handled; deterministic; preserves raw bytes. `NaN`/`inf` strings parse as numeric by design — rejecting non-finite values is the validation layer's job.
3. **Validation** (`validation.py`): structured pydantic `ValidationCheck`s and per-dataset stats instead of assertions. Required checks (block publication) are separated from warnings (constant/near-constant columns — reported, never deleted). General structural validation is separate from the optional FD001 canonical-count profile (`validate_canonical` flag; the CLI always applies it).
4. **Processing** (`processing.py`): raw layer re-verified against the manifest before parsing (refactored `verify_raw_layer` now public). Atomic writes (temp+rename), SHA-256 output checksums, idempotent re-runs (`already_processed`, nothing rewritten), tamper/missing-output detection with `--force` rebuild — mirroring the Loop 1 model. Trajectory outputs sorted by `(asset_id, cycle)`; source order never relied upon.
5. **Validated/processed layers collapsed into one step** for FD001: validation gates publication, so the Parquet under `data/processed/` *is* the validated output. Documented in `docs/data_contract.md`.
6. **Real FD001 results:** all 36 dataset checks + 1 cross-check pass. Train 20,631 rows / 100 engines (lives 128–362, median 199); test 13,096 rows / 100 engines (histories 31–303, median 133.5); RUL 100 values (7–145, median 86). Zero missing/duplicate/non-finite values. Constant: `operating_setting_3`, sensors 01/10/18/19; near-constant (relative std < 1e-4): sensors 05/06/08/13/16.
7. **Key EDA nuance:** sensors 08 and 13 are flagged near-constant only because their means are large — they show clear monotonic lifecycle trends (mean within-engine Spearman ≈ 0.65+). Loop 3 must select features by trend/information content, not raw variance. Strongest trends: sensors 11, 12, 04, 07; no-trend sensors match the constant/near-constant set.
8. **EDA notebook** (`notebooks/01_eda.ipynb`, committed executed): consumes only the Parquet outputs; covers counts, lengths, quality, constant columns, settings, lifecycle trends (Spearman-via-ranks, no SciPy), representative traces, distributions, correlation matrix, train-vs-test comparison (mechanical censoring, not drift), and a "Findings and implications for Loop 3" section. No RUL labels, RUL correlations, rolling features, splits, or models.
9. **Dependencies** (ADR 0001): runtime pandas + pyarrow; dev-only pandas-stubs, matplotlib, nbconvert, ipykernel. No validation framework (Great Expectations excluded by spec; typed Python checks instead). SciPy deliberately avoided. Note: uv resolved **pandas 3.0.3** — code targets the pandas 3 API.
10. **Tests:** 61 new (106 total). Parsing/validation/processing/CLI units are fully offline with 26-column fixtures; one integration test runs against the locally acquired FD001 data by copying it to a temp dir (repo `data/` never mutated) and auto-skips when the dataset is absent (`real_data` marker).

## Loop 1 Implementation Notes (retained)

1. **Source URL.** The historical `ti.arc.nasa.gov` download is gone and `data.nasa.gov` returns 404. The working source is NASA's S3 mirror `https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip` (12.4 MB), the configurable default (`TURBINE_GUARD_CMAPSS_SOURCE_URL` / `--url`). `file://` URLs are supported and used by tests.
2. **Nested archives.** Extraction searches nested zips (depth-limited); flat and nested layouts accepted.
3. **Immutability model.** Atomic writes, read-only (0444) raw files, checksum verification on re-run; mismatch → clear error, `--force` to replace.
4. **No pinned upstream checksums** — NASA publishes none; verify-what-you-acquired model with the manifest as source of truth.
5. **Manifest fields:** dataset/subset, source name+URL, UTC timestamp, versions, git commit, archive checksum, per-file SHA-256/size/record/asset counts.
6. **Acquired FD001 matches canonical characteristics** (verified again in Loop 2).
7. **Stdlib only in Loop 1**; `acquire()` is a plain callable (Prefect wrapper deferred).
8. **`data/` is gitignored** (root-anchored `/data/`).

## Loop 0 Implementation Notes (retained)

1. Repo initialized (`git init -b main`); Loop 0 committed as `d95b528`, Loop 1 as `94ae615`.
2. `uv` via Homebrew; Python 3.12.13 managed by uv.
3. Structured logging uses the standard library only (JSON formatter).
4. Pre-commit hooks are local `uv run` commands, versions synced with `uv.lock`.
5. `/health/ready` returns 200 with an empty check map; 503 path implemented and tested.
6. Package name `turbine-guard`; repo directory name differs (`predictive_maintenance`).

---

## Current Architecture Decisions

### Accepted

* Use NASA C-MAPSS FD001 for the first complete version.
* Split data by asset rather than individual rows.
* Keep replay assets out of initial model training.
* Use one shared feature pipeline for training and serving.
* Use tabular models before considering deep sequence models.
* Use MLflow for experiment tracking and model registration.
* Use Prefect for workflows.
* Use FastAPI for inference and operational endpoints.
* Use PostgreSQL for operational state.
* Use Docker Compose for local orchestration.
* Use a lightweight server-rendered dashboard.
* Use Render as the initial planned public deployment target.
* Avoid unnecessary distributed-system components.
* Loop 0: stdlib JSON logging; local `uv run` pre-commit hooks; application factory pattern.
* Loop 1: configurable source URL with `file://` support; verify-on-rerun checksum model; immutable read-only raw layer; manifests and datasets excluded from git.
* Loop 2 (ADR 0001): pandas + pyarrow runtime; typed hand-rolled validation (no Great Expectations/pandera); matplotlib/nbconvert/ipykernel dev-only; SciPy deferred; validated and processed layers collapsed into one gated step; required-vs-warning check separation; canonical FD001 profile separate from general validation.
* Loop 3 (ADR 0002): 70/15/5/10 asset-level split via seeded permutation + largest-remainder counts; preserve early-cycle rows with structural nulls, defer imputation/scaling to Loop 4; trailing asset-grouped windows with min_periods=1 and OLS rolling slope (degenerate → 0.0); optional off-by-default RUL cap alongside the always-present uncapped target; stateless FeatureBuilder with a history-replaying incremental state; frozen-dataclass configuration (no YAML layer).

### Implemented so far

Loops 0–3 only. All modeling/persistence/orchestration decisions remain design-only.

---

## Known Risks

1. NASA C-MAPSS is simulated turbofan data rather than actual power-plant sensor data.
2. Sensor columns are anonymous; no physical names are assigned.
3. Online collection will be simulated through historical trajectory replay.
4. Public free-tier deployment limitations may affect persistent MLflow artifacts or background workers.
5. Maintenance-cost results will be simulated and must not be presented as real industrial savings.
6. The system could become overengineered if future tools are introduced without a clear need.
7. Time-series leakage is a major modeling risk; Loop 3 added explicit structural protections and tests (future-row mutation/append, cross-asset isolation, fit isolation, replay exclusion), which future loops must keep passing.
8. Retraining on very few newly labeled replay assets may not provide meaningful improvements.
9. NASA hosting has moved before and may move again; the source URL is configuration, and `file://` acquisition provides a manual fallback.
10. Relative-variance heuristics mislead on FD001 (sensors 08/13). Loop 3 deliberately generates features for **all** columns (constant ones degrade to constants/zeros); trend/information-based feature *selection* is a Loop 4 modeling decision.
11. The shortest test history is 31 cycles; Loop 3 handles short-history warm-up via `min_periods=1` trailing windows and structural nulls, but Loop 4 models must handle those early-cycle nulls explicitly.
12. pandas 3.x is newer than most tutorials/snippets assume; future code must target the pandas 3 API.
13. Early-cycle feature rows contain structural nulls by design (no imputation in Loop 3); any Loop 4 imputation must be fit on training assets only and applied identically everywhere.

---

## Immediate Next Action

Review Loop 3. After explicit approval, begin **Loop 4 — Baseline Modeling and Evaluation**.

Loop 3 changes are left uncommitted pending review (same policy as Loops 0–2; Loop 2 was committed as `aa27dfc` after review).

---

## Loop 3 Exit Criteria — all satisfied

* [x] RUL labels correct: uncapped `rul = T_i − t`, final-cycle RUL 0, exact −1 steps, validated; optional capped target preserved alongside uncapped.
* [x] Official test set protected: no fabricated per-row labels; positional benchmark table documented as evaluation-only.
* [x] Splits deterministic (seed 42), asset-level, non-overlapping, complete coverage: 70/15/5/10 engines.
* [x] Replay and calibration assets isolated from all fitting-related partitions.
* [x] Feature generation uses only current and historical per-asset data (trailing windows, grouped by asset, sorted internally).
* [x] Offline and incremental feature outputs match at every cycle (tested on fixtures and a real engine; restart reconstruction covered).
* [x] Leakage tests pass: future-row mutation, future-row append, exhaustive per-cycle perturbation, cross-asset isolation, fit isolation, replay exclusion.
* [x] Model-ready Parquet outputs generated for train/validation/calibration/replay/test features/test labels.
* [x] Split and feature manifests generated with checksums, versions, configuration, and provenance.
* [x] Outputs reproducible and idempotent; tamper detection and `--force` rebuild work; raw and Loop 2 layers unchanged (checksum-verified).
* [x] Documentation matches implementation (`docs/features.md`, `docs/data_contract.md` boundary update, ADR 0002, README, Makefile).
* [x] No Loop 4 functionality implemented (no models, training, evaluation metrics, scaling, imputation, MLflow, conformal prediction).

---

## Loop 2 Exit Criteria — all satisfied

* [x] Data contract defined and documented (`docs/data_contract.md`, canonical schema v1).
* [x] Parser implemented in `src/`, reusable and independently tested (never notebook-only).
* [x] Structural + semantic validation with structured results; required failures block publication.
* [x] Invalid fixtures fail clearly (wrong column counts with line numbers, non-numeric values, duplicates, gaps, non-finite values, empty files, tampered outputs — all covered by tests).
* [x] Real FD001 data passes all checks; asset and cycle integrity verified (unique pairs, contiguous 1..n).
* [x] Validated Parquet outputs generated reproducibly (atomic, checksummed, idempotent, `--force` rebuild).
* [x] Machine-readable validation report generated (`processing_report.json`).
* [x] Single primary EDA notebook executes top to bottom on the processed Parquet; findings documented in the notebook and here.
* [x] Raw files unchanged (checksums verified before/after processing).
* [x] `STATUS.md`, `TASKS.md`, `README.md` updated; ADR 0001 added.
* [x] No Loop 3 functionality implemented (no labels, capped RUL, splits, rolling features, feature manifests, scaling, or models).

---

## Validation Status

All commands run on 2026-07-12 (macOS, Python 3.12.13 via uv), after Loop 3:

| Check                          | Status | Detail                                                        |
| ------------------------------ | ------ | ------------------------------------------------------------- |
| `uv sync`                      | Pass   | 106 packages resolved; no new dependencies in Loop 3           |
| Ruff format check              | Pass   | 54 files already formatted                                     |
| Ruff lint check                | Pass   | All checks passed                                              |
| Mypy (strict)                  | Pass   | No issues in 29 source files                                   |
| Pytest                         | Pass   | 183 passed (77 new; incl. 3 real-data feature integration tests) |
| Pre-commit (all files)         | Pass   | ruff format, ruff check, mypy                                  |
| Real FD001 feature build       | Pass   | `make features`: 6 Parquet outputs + 2 manifests, 552 features |
| Idempotent re-run demo         | Pass   | `already_built`; output mtimes unchanged; exit code 0          |
| Missing-input failure demo     | Pass   | Nonexistent data dir → exit code 1, no outputs                 |
| Parquet load verification      | Pass   | All 6 outputs load; shapes (14407/3160/909/2155, 556), (13096, 555), (100, 3) |
| Future-row leakage demo (real) | Pass   | Mutating sensor_04 after cycle 100 leaves features ≤ 100 identical |
| Offline vs incremental (real)  | Pass   | Engine 1, 192 cycles: equal at every cycle                     |
| Raw + processed integrity      | Pass   | All raw and Loop 2 Parquet checksums unchanged after the build |

Known warning (non-blocking, upstream, unchanged since Loop 0): importing `fastapi.testclient` with starlette 1.3.1 emits a `StarletteDeprecationWarning` recommending `httpx2`; originates in FastAPI's own compatibility import.

---

## Last Completed Loop

**Loop 3 — Labels, Asset-Level Splits, and Leakage-Safe Features** (2026-07-12).

---

## Next Planned Loop

After Loop 3 is reviewed and explicitly approved:

**Loop 4 — Baseline Modeling and Evaluation**

Do not begin Loop 4 automatically.
