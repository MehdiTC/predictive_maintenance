"""RUL labels, asset-level splits, and leakage-safe features.

This package turns the validated Parquet trajectories into reproducible,
model-ready feature tables. It is deliberately model-free: no training,
scaling, or feature selection happens here; those belong to model training.

Public surface:

* :mod:`turbine_guard.features.config` — typed feature/split configuration.
* :mod:`turbine_guard.features.labels` — Remaining Useful Life labels.
* :mod:`turbine_guard.features.splits` — deterministic asset-level splits.
* :mod:`turbine_guard.features.builder` — the shared :class:`FeatureBuilder`
  (offline batch and single-asset incremental generation).
* :mod:`turbine_guard.features.pipeline` — orchestrates labels + splits +
  features + manifests into idempotent Parquet outputs.
"""
