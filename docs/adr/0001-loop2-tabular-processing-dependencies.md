# ADR 0001 — Loop 2 tabular processing dependencies and hand-rolled validation

**Status:** Accepted (2026-07-12)

## Context

Loop 2 turns the immutable FD001 raw text files into typed, validated Parquet datasets plus an
executable EDA notebook. Loops 0–1 were deliberately stdlib-only; this is the first loop that
needs tabular processing, a Parquet engine, and plotting/notebook execution. A data-validation
framework could also be introduced here.

## Decision

Runtime dependencies (both in the approved core stack):

* **pandas** — tabular parsing, validation computations, and Parquet I/O. It is also the
  interchange format every later loop (features, models, monitoring) consumes.
* **pyarrow** — the Parquet engine behind `DataFrame.to_parquet` / `read_parquet`. Only pandas'
  Parquet interface is used; pyarrow is never imported directly.

Development-only dependencies (dev group; not installed in production images):

* **pandas-stubs** — the project runs mypy in strict mode; pandas needs type stubs to keep that.
* **matplotlib** — EDA plots. Chosen over Plotly for the notebook because executed notebooks
  render static images that display on GitHub without a JS bundle; Plotly remains the planned
  choice for the *dashboard* loop per the spec.
* **nbconvert + ipykernel** — headless top-to-bottom execution of `notebooks/01_eda.ipynb`
  (`make eda`), which keeps the notebook reproducible and CI-executable.

Validation is implemented as **typed Python check objects** (pydantic models in
`turbine_guard.data.validation`), not a validation framework:

* Great Expectations is explicitly excluded by the project spec.
* Alternatives (pandera, pydantic-on-rows) would add a dependency and a DSL to express ~15
  checks that are clear, testable, and machine-readable as plain code, and the structured
  report format is bespoke anyway (checksums, provenance, canonical profiles).

SciPy was deliberately **not** added: the one statistic that wanted it (Spearman correlation in
the EDA notebook) is computed as Pearson-on-ranks with pandas. SciPy joins when a loop needs it
substantively.

Notably, `uv` resolved **pandas 3.x** (not 2.x); the code is written against the pandas 3 API.

## Consequences

* Strict mypy still passes across `src/` with pandas in the type graph.
* Unit tests stay offline and fast; Parquet round-trips are covered by tests.
* Production runtime gains exactly two libraries; notebook/plotting tooling stays out of the
  runtime dependency set.
* If a future loop needs pyarrow directly (e.g., partitioned datasets), it is already pinned.
