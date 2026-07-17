"""Checksum-pinned deployment bundle export, restore, and demo startup.

The free public demo deliberately has no live MLflow service and no
persistent compute disk. A deployment bundle is an immutable, exported
snapshot of the verified champion and the exact serving inputs it needs:
the champion ``ModelBundle``, the Loop 3 feature/split manifests, and the
Loop 2 processed replay source. The bundle's registry identity (model name,
version, aliases, source run, metrics, checksums) is captured from MLflow
at export time and served read-only.
"""
