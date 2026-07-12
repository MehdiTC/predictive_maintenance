# Data Contract — NASA C-MAPSS FD001

This document is the contract between the raw data layer (Loop 1), the validated/processed
layer (Loop 2), and everything downstream. Code that produces or consumes FD001 data must
conform to it.

## Provenance

* **Dataset:** NASA C-MAPSS *Turbofan Engine Degradation Simulation*, subset **FD001**, from the
  NASA Prognostics Center of Excellence data repository.
* **Nature:** *simulated* turbofan run-to-failure data produced with the C-MAPSS simulator. It is
  not real power-plant data.
* **Anonymous sensors:** the dataset's documentation identifies each trajectory column only as
  "operational setting" or "sensor measurement". This project therefore does **not** assign
  physical meanings (vibration, temperature, pressure, …) to sensor channels. Canonical names are
  positional: `sensor_01` … `sensor_21`.
* **Acquisition** (Loop 1, `make acquire`) stores the files byte-for-byte in an immutable,
  read-only raw layer and records a provenance manifest (`data/manifests/cmapss_fd001.json`) with
  SHA-256 checksums, source URL, retrieval timestamp, and record/asset counts.

## Raw file structure

Whitespace-delimited text, no header, lines end with trailing spaces:

| File | Rows | Fields per row | Meaning |
| --- | --- | --- | --- |
| `train_FD001.txt` | 20,631 | 26 | run-to-failure trajectories, 100 engines |
| `test_FD001.txt` | 13,096 | 26 | truncated trajectories, 100 engines |
| `RUL_FD001.txt` | 100 | 1 | remaining cycles for each test engine at truncation |

Per the dataset readme, the 26 trajectory fields are: unit number, cycle number, three
operational settings, and twenty-one sensor measurements.

## Canonical schema

Defined in `src/turbine_guard/data/schema.py` (`SCHEMA_VERSION = "1"`).

### Trajectory rows (train and test)

| Column | Dtype | Constraints |
| --- | --- | --- |
| `asset_id` | `int64` | ≥ 1 |
| `cycle` | `int64` | ≥ 1; unique per asset; contiguous `1..n` within each asset |
| `operating_setting_1..3` | `float64` | finite, non-null |
| `sensor_01..21` | `float64` | finite, non-null |

Column order is fixed as listed. Processed trajectory rows are sorted by `(asset_id, cycle)`;
source row order is never relied on for correctness.

### Official test RUL file

| Column | Dtype | Constraints |
| --- | --- | --- |
| `rul` | `int64` | ≥ 0; row *i* corresponds to test `asset_id` *i + 1*; row count equals the test asset count |

The RUL file is parsed and validated **structurally** in Loop 2. Generating per-row RUL labels
from it (or from train failure cycles) is deliberately deferred to Loop 3 and must never happen
in the Loop 2 processing path.

## Validation rules

Implemented in `src/turbine_guard/data/validation.py` as structured, machine-readable checks.
A failed **required** check blocks publication: no Parquet output or report is written.

Required checks (per dataset):

* raw-layer integrity: manifest exists and every raw file matches its recorded SHA-256
  (verified via the acquisition layer before parsing);
* parseability: 26 (or 1) whitespace-delimited fields per line, numeric values, non-empty file
  (violations raise `ParseError` with line numbers — malformed rows are never silently dropped);
* schema: exact canonical columns, order, and dtypes; no unexpected columns; non-empty;
* asset/cycle integrity: positive `asset_id` and `cycle`, unique `(asset_id, cycle)` pairs,
  cycles contiguous from 1 within each asset (order-independent);
* numeric integrity: no missing and no non-finite values in settings/sensors;
* RUL: non-negative integers; cross-check: RUL row count equals test asset count.

Canonical-profile checks (separate from general validation, applied when the subset has a known
profile; FD001: train 20,631 rows / 100 assets, test 13,096 rows / 100 assets, 100 RUL values).
The library exposes `validate_canonical=False` for non-canonical inputs; the CLI always applies
the profile.

Warnings (recorded, never fatal, never auto-fixed):

* constant columns (zero std) and near-constant columns (relative std < 1e-4) — reported but
  **kept**; exclusion is a Loop 3 feature-selection decision;
* observed per-column min/max ranges are reported without enforcing invented physical bounds,
  and unusual values are never removed as "outliers".

## Generated outputs

`make process` (or `uv run python scripts/process_data.py`) writes to
`data/processed/cmapss/FD001/`:

| File | Content |
| --- | --- |
| `train_FD001.parquet` | validated train trajectories, canonical schema |
| `test_FD001.parquet` | validated test trajectories, canonical schema |
| `rul_FD001.parquet` | validated official RUL values |
| `processing_report.json` | machine-readable validation/processing report |

The report records: processing and schema versions, dataset/subset, UTC timestamp, tool version,
git commit, source-archive checksum, input and output filenames with SHA-256 checksums and
row/asset counts, per-dataset statistics (trajectory length min/median/max, duplicates, missing
and non-finite counts, constant/near-constant columns, column ranges), every check with its
pass/fail status, and warnings.

Example report shape (abridged):

```json
{
  "processing_version": "1",
  "schema_version": "1",
  "dataset_subset": "FD001",
  "processed_at": "2026-07-12T15:13:47Z",
  "git_commit": "…",
  "source_archive_sha256": "…",
  "inputs":  [{"filename": "train_FD001.txt", "sha256": "…", "record_count": 20631, "asset_count": 100}],
  "outputs": [{"filename": "train_FD001.parquet", "sha256": "…", "record_count": 20631, "asset_count": 100}],
  "datasets": [
    {
      "dataset": "train",
      "checks": [{"name": "unique_asset_cycle_pairs", "passed": true, "required": true, "message": "ok"}],
      "warnings": ["Constant columns (kept, not deleted): operating_setting_3, sensor_01, …"],
      "trajectory_stats": {"row_count": 20631, "asset_count": 100, "trajectory_length_min": 128, "…": "…"}
    }
  ],
  "cross_checks": [{"name": "rul_count_matches_test_assets", "passed": true, "required": true, "message": "ok"}],
  "passed": true
}
```

### Behavior guarantees

* **Raw immutability:** processing only reads the raw layer; raw files stay read-only and
  checksum-verified on every run.
* **Atomic writes:** outputs and the report are written to a temp file and renamed.
* **Idempotency:** re-running verifies the existing report against the current manifest and the
  outputs' checksums; when everything matches, nothing is rewritten (`already_processed`).
* **Tamper detection:** a modified or missing output fails with a clear error instead of silent
  repair; `--force` deliberately rebuilds the processed layer.
* Generated data is **never committed to git** (`/data/` is gitignored).

## How to run

```bash
make acquire   # Loop 1: download + immutable raw layer + manifest
make process   # Loop 2: parse, validate, write Parquet + report
make eda       # execute notebooks/01_eda.ipynb top to bottom
```

The EDA notebook consumes only the processed Parquet outputs, never the raw text files.

## Known limitations

* FD001 only; other C-MAPSS subsets (FD002–FD004) are out of scope for now. The parser and the
  general validation rules are subset-agnostic; only the canonical count profile is FD001-specific.
* The RUL ↔ asset correspondence (row *i* ↔ test unit *i + 1*) follows the dataset readme; it is
  positional and cannot be cross-validated from file content alone.
* Validated and processed layers are collapsed into one step for this dataset: validation gates
  publication, and the Parquet files under `data/processed/` are the validated output. A separate
  `data/validated/` stage would add a copy without adding a guarantee.

## Loop 2 / Loop 3 boundary

Loop 2 ends at *trustworthy, typed, validated Parquet plus an EDA notebook*. It deliberately does
**not** produce: per-row RUL labels, capped RUL, train/validation/calibration/replay splits,
rolling-window features, feature manifests, or model-oriented scaling. Those belong to Loop 3,
which consumes the Parquet outputs defined here.
