# STATUS.md

## Project Status

**Project:** TurbineGuard
**Current phase:** Loop 1 implemented and validated
**Active loop:** None — Loop 1 complete, awaiting review before Loop 2
**Overall status:** Foundation + reproducible dataset acquisition in place; no parsing or ML yet
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

Loops 0 and 1 are implemented:

```text
├── src/turbine_guard/
│   ├── api/                    # create_app() factory, health routes/schemas
│   ├── config/settings.py      # typed BaseSettings (+ data_dir, cmapss_source_url)
│   ├── data/
│   │   ├── acquisition.py      # idempotent C-MAPSS FD001 acquisition
│   │   ├── manifest.py         # provenance manifest models + persistence
│   │   └── cli.py              # argparse CLI used by scripts/download_data.py
│   ├── services/health.py
│   └── logging_config.py       # structured JSON logging
├── scripts/download_data.py    # thin wrapper over turbine_guard.data.cli
├── tests/                      # 45 tests, all offline (file:// fixtures)
├── data/                       # gitignored: raw layer, archive cache, manifests
├── pyproject.toml / uv.lock / Makefile / .pre-commit-config.yaml / .env.example
└── README.md                   # setup + dataset acquisition documentation
```

Raw data layout produced by `make acquire`:

```text
data/
├── raw/cmapss/
│   ├── 6._Turbofan_Engine_Degradation_Simulation_Data_Set.zip   # cached source archive
│   └── FD001/                                                    # immutable, read-only (0444)
│       ├── train_FD001.txt   # 20,631 records, 100 engine units
│       ├── test_FD001.txt    # 13,096 records, 100 engine units
│       └── RUL_FD001.txt     # 100 records
└── manifests/cmapss_fd001.json
```

No parsing/validation, EDA, labels, features, models, database, MLflow, Prefect, replay, monitoring, Docker, or CI functionality exists yet — deliberately.

---

## Current Loop

None active. Loop 1 is complete and awaiting review. The next loop (Loop 2 — Validation and EDA) must not begin without explicit approval.

---

## Completed Work

* [x] Defined the overall project mission, dataset, target, and architecture (planning).
* [x] Implemented Loop 0 — repository foundation (2026-07-12).
* [x] Implemented Loop 1 — dataset acquisition and manifesting (2026-07-12).

---

## Loop 1 Implementation Notes

1. **Source URL.** The historical `ti.arc.nasa.gov` download is gone (redirects away) and `data.nasa.gov` returns 404. The working source is NASA's S3 mirror `https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip` (12.4 MB), now the configurable default (`TURBINE_GUARD_CMAPSS_SOURCE_URL` / `--url`). `file://` URLs are supported for mirrors/offline use and are what the tests use, so unit tests never touch the network.
2. **Nested archives.** The S3 archive nests `CMAPSSData.zip` inside; extraction searches nested zips (depth-limited) and accepts both flat and nested layouts.
3. **Immutability model.** Raw files are written atomically (temp + rename) and marked read-only (0444). Re-runs verify SHA-256 checksums against the manifest: match → `already_acquired`, nothing downloaded or rewritten; mismatch or missing file → `AcquisitionError` with a clear message rather than silent repair; `--force` deliberately re-downloads and replaces the layer.
4. **No pinned upstream checksums.** NASA publishes none, and mirrors have re-zipped the archive over the years. Checksums are computed at acquisition time and enforced on every subsequent run (corruption/tamper detection). This is the verify-what-you-acquired model; the manifest is the source of truth.
5. **Manifest fields.** dataset name, subset, source name + URL, retrieval timestamp (UTC), acquisition version, acquiring tool version, git commit (when resolvable), archive checksum/size, and per file: SHA-256, size, record count, asset count. Record counts are content-neutral line counts; asset counts (distinct first-column unit IDs, per the dataset's own readme) are computed only for trajectory files and left `null` for `RUL_FD001.txt`. Full parsing with column semantics is Loop 2.
6. **Acquired FD001 matches canonical characteristics:** train 20,631 records / 100 units; test 13,096 records / 100 units; 100 RUL values.
7. **Provenance framing embedded in the manifest `notes`:** simulated turbofan degradation data; anonymous sensor channels; no physical interpretations assigned.
8. **Stdlib only** — `urllib`, `zipfile`, `hashlib`, `subprocess` (git SHA). Zero new dependencies; no ADR required.
9. **`acquire()` is a plain, directly-callable function.** The Prefect `acquire_dataset_flow` wrapper is deferred to the orchestration loop, per the loop plan.
10. **`data/` is gitignored (root-anchored `/data/`)** so datasets and manifests are never committed; the pattern deliberately does not catch the `src/turbine_guard/data/` package.

## Loop 0 Implementation Notes (retained)

1. Project directory was made its own git repository (`git init -b main`); Loop 0 was committed by the user as `d95b528`.
2. `uv` installed via Homebrew (0.11.24); Python 3.12.13 managed by uv.
3. Structured logging uses the standard library only (JSON formatter).
4. Pre-commit hooks are local `uv run` commands, keeping versions synced with `uv.lock`.
5. `/health/ready` returns 200 with an empty check map (no dependencies yet); 503 path implemented and tested.
6. Package name `turbine-guard` per spec; repo directory name differs (`predictive_maintenance`).

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
* Loop 1: configurable source URL with `file://` support; verify-on-rerun checksum model (no pinned upstream checksums); immutable read-only raw layer; nested-archive-tolerant extraction; manifests and datasets excluded from git.

### Implemented so far

Loops 0–1 only. All modeling/persistence/orchestration decisions remain design-only.

---

## Known Risks

1. NASA C-MAPSS is simulated turbofan data rather than actual power-plant sensor data.
2. Sensor columns may not have physical names such as vibration or temperature.
3. Online collection will be simulated through historical trajectory replay.
4. Public free-tier deployment limitations may affect persistent MLflow artifacts or background workers.
5. Maintenance-cost results will be simulated and must not be presented as real industrial savings.
6. The system could become overengineered if future tools are introduced without a clear need.
7. Time-series leakage is a major modeling risk and must be covered by explicit tests.
8. Retraining on very few newly labeled replay assets may not provide meaningful improvements.
9. NASA hosting has moved before and may move again; the source URL is configuration, and `file://` acquisition provides a manual fallback.

---

## Immediate Next Action

Review Loop 1. After explicit approval, begin **Loop 2 — Validation and EDA**.

Loop 1 changes are staged but not committed (same policy as Loop 0: commits are left for review).

---

## Loop 1 Exit Criteria — all satisfied

* [x] Dataset can be acquired from scratch with one command (`make acquire`; demonstrated against NASA S3, 12.4 MB archive).
* [x] Re-running does not corrupt or duplicate data (checksum verification → `already_acquired`; file mtimes and manifest untouched; covered by tests and demonstrated live).
* [x] Manifest contains required provenance fields (dataset name, subset, source name/URL, retrieval timestamp, filenames, SHA-256 checksums, file sizes, acquisition version, plus record/asset counts, git commit, tool version, archive record).
* [x] Tests use small fixtures rather than downloading the full dataset (in-memory fixture zips served via `file://`; zero network use in tests).
* [x] Clear failure handling (missing archive members, unreachable source, invalid zip, tampered/missing raw files — each raises `AcquisitionError` with an actionable message; CLI exits 1).
* [x] Documentation for acquisition added to `README.md`.
* [x] `STATUS.md` and `TASKS.md` updated.
* [x] No Loop 2+ functionality implemented (no parsing into tables, no EDA, no labels, no features, no models, no new services or dependencies).

---

## Validation Status

All commands run on 2026-07-12 (macOS, Python 3.12.13 via uv):

| Check                       | Status | Detail                                              |
| --------------------------- | ------ | ---------------------------------------------------- |
| `uv sync`                   | Pass   | 45 packages; no dependency changes in Loop 1          |
| Ruff format check           | Pass   | 25 files already formatted                            |
| Ruff lint check             | Pass   | All checks passed                                     |
| Mypy (strict)               | Pass   | No issues in 16 source files                          |
| Pytest                      | Pass   | 45 passed (21 new acquisition/manifest/CLI/settings)  |
| Pre-commit (all files)      | Pass   | ruff format, ruff check, mypy                         |
| Real acquisition demo       | Pass   | Downloaded from NASA S3; manifest written             |
| Idempotent re-run demo      | Pass   | `already_acquired`; nothing re-downloaded             |
| Raw-layer immutability      | Pass   | Files 0444; tamper/missing detection tested           |

Known warning (non-blocking, upstream, unchanged from Loop 0): importing `fastapi.testclient` with starlette 1.3.1 emits a `StarletteDeprecationWarning` recommending `httpx2`; originates in FastAPI's own compatibility import.

---

## Last Completed Loop

**Loop 1 — Dataset Acquisition and Manifesting** (2026-07-12).

---

## Next Planned Loop

After Loop 1 is reviewed and explicitly approved:

**Loop 2 — Validation and EDA**

Do not begin Loop 2 automatically.
