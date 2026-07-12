# STATUS.md

## Project Status

**Project:** TurbineGuard
**Current phase:** Loop 2 implemented and validated
**Active loop:** None — Loop 2 complete, awaiting review before Loop 3
**Overall status:** Foundation, reproducible acquisition, validated Parquet processing, and EDA in place; no labels, features, or ML yet
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

Loops 0–2 are implemented:

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
│   ├── services/health.py
│   └── logging_config.py       # structured JSON logging
├── scripts/download_data.py     # thin wrappers over turbine_guard.data CLIs
├── scripts/process_data.py
├── notebooks/01_eda.ipynb       # the single primary EDA notebook, executed
├── docs/data_contract.md        # raw structure, canonical schema, validation rules, outputs
├── docs/adr/0001-…              # Loop 2 dependency + hand-rolled-validation decision
├── tests/                       # 106 tests: unit (offline) + local FD001 integration (auto-skip)
├── data/                        # gitignored: raw, manifests, processed
└── pyproject.toml / uv.lock / Makefile / .pre-commit-config.yaml / .env.example / README.md
```

Data layers on disk after `make acquire && make process`:

```text
data/
├── raw/cmapss/FD001/            # immutable, read-only (0444), checksum-verified
├── manifests/cmapss_fd001.json
└── processed/cmapss/FD001/
    ├── train_FD001.parquet      # 20,631 rows, 100 engines, canonical schema
    ├── test_FD001.parquet       # 13,096 rows, 100 engines
    ├── rul_FD001.parquet        # 100 official test RUL values
    └── processing_report.json   # machine-readable checks/stats/checksums/provenance
```

No labels, splits, features, models, database, MLflow, Prefect, replay, monitoring, Docker, or CI functionality exists yet — deliberately.

---

## Current Loop

None active. Loop 2 is complete and awaiting review. The next loop (Loop 3 — Labels, Splits, and Features) must not begin without explicit approval.

---

## Completed Work

* [x] Defined the overall project mission, dataset, target, and architecture (planning).
* [x] Implemented Loop 0 — repository foundation (2026-07-12).
* [x] Implemented Loop 1 — dataset acquisition and manifesting (2026-07-12).
* [x] Implemented Loop 2 — validation, processing, and EDA (2026-07-12).

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

### Implemented so far

Loops 0–2 only. All labeling/feature/modeling/persistence/orchestration decisions remain design-only.

---

## Known Risks

1. NASA C-MAPSS is simulated turbofan data rather than actual power-plant sensor data.
2. Sensor columns are anonymous; no physical names are assigned.
3. Online collection will be simulated through historical trajectory replay.
4. Public free-tier deployment limitations may affect persistent MLflow artifacts or background workers.
5. Maintenance-cost results will be simulated and must not be presented as real industrial savings.
6. The system could become overengineered if future tools are introduced without a clear need.
7. Time-series leakage is a major modeling risk and must be covered by explicit tests (Loop 3).
8. Retraining on very few newly labeled replay assets may not provide meaningful improvements.
9. NASA hosting has moved before and may move again; the source URL is configuration, and `file://` acquisition provides a manual fallback.
10. Relative-variance heuristics mislead on FD001 (sensors 08/13); Loop 3 feature selection must use trend/information content, not raw variance.
11. The shortest test history is 31 cycles; Loop 3 rolling windows must handle short-history warm-up or inference will fail on early-life engines.
12. pandas 3.x is newer than most tutorials/snippets assume; future code must target the pandas 3 API.

---

## Immediate Next Action

Review Loop 2. After explicit approval, begin **Loop 3 — Labels, Splits, and Features**.

Loop 2 changes are left uncommitted pending review (same policy as Loops 0–1).

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

All commands run on 2026-07-12 (macOS, Python 3.12.13 via uv):

| Check                        | Status | Detail                                                     |
| ---------------------------- | ------ | ---------------------------------------------------------- |
| `uv sync`                    | Pass   | 106 packages (pandas 3.0.3, pyarrow 25 + dev notebook deps) |
| Ruff format check            | Pass   | 36 files already formatted                                  |
| Ruff lint check              | Pass   | All checks passed                                           |
| Mypy (strict)                | Pass   | No issues in 21 source files                                |
| Pytest                       | Pass   | 106 passed (61 new; incl. real-data integration test)       |
| Pre-commit (all files)       | Pass   | ruff format, ruff check, mypy                               |
| Real FD001 processing        | Pass   | All 37 checks pass; 3 Parquet files + report written        |
| Idempotent re-run demo       | Pass   | `already_processed`; output mtimes unchanged                |
| Parquet load verification    | Pass   | (20631, 26) / (13096, 26) / (100, 1), canonical dtypes      |
| EDA notebook execution       | Pass   | `make eda`: all 12 code cells executed, no errors/stderr    |
| Raw-layer integrity          | Pass   | SHA-256 of all three raw files unchanged after processing   |

Known warning (non-blocking, upstream, unchanged since Loop 0): importing `fastapi.testclient` with starlette 1.3.1 emits a `StarletteDeprecationWarning` recommending `httpx2`; originates in FastAPI's own compatibility import.

---

## Last Completed Loop

**Loop 2 — Data Validation, Processing, and EDA** (2026-07-12).

---

## Next Planned Loop

After Loop 2 is reviewed and explicitly approved:

**Loop 3 — Labels, Splits, and Features**

Do not begin Loop 3 automatically.
