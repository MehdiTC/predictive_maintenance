"""Tests for the explicit deterministic container bootstrap boundary."""

import zipfile
from pathlib import Path

from turbine_guard.bootstrap import _ci_fixture_bytes, _ensure_ci_fixture, _training_config
from turbine_guard.modeling.config import ModelKind


def test_ci_fixture_is_deterministic_schema_complete_and_idempotent(tmp_path: Path) -> None:
    first = _ci_fixture_bytes()
    second = _ci_fixture_bytes()
    assert first == second

    path = _ensure_ci_fixture(tmp_path)
    assert path.read_bytes() == first
    assert _ensure_ci_fixture(tmp_path) == path
    with zipfile.ZipFile(path) as archive:
        assert set(archive.namelist()) == {
            "train_FD001.txt",
            "test_FD001.txt",
            "RUL_FD001.txt",
        }
        first_row = archive.read("train_FD001.txt").decode().splitlines()[0].split()
        assert len(first_row) == 26
        assert len(archive.read("RUL_FD001.txt").decode().splitlines()) == 20


def test_ci_training_keeps_every_established_model_family(tmp_path: Path) -> None:
    config = _training_config(tmp_path, ci_fixture=True)
    assert {candidate.kind for candidate in config.candidates} == set(ModelKind)
    assert config.latency_repeats == 1
    assert config.selection.minimum_critical_recall == 0.0
