"""Deployment bundle export, pinned restore, and snapshot serving tests."""

import io
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from turbine_guard.config.settings import Settings
from turbine_guard.deployment.export import export_deployment_bundle
from turbine_guard.deployment.manifest import (
    CHAMPION_RELATIVE_PATH,
    FEATURE_MANIFEST_RELATIVE_PATH,
    MANIFEST_FILENAME,
    PROCESSING_REPORT_RELATIVE_PATH,
    REQUIRED_BUNDLE_FILES,
    SPLIT_MANIFEST_RELATIVE_PATH,
    STATE_FILENAME,
    TRAIN_PARQUET_RELATIVE_PATH,
    BundleError,
)
from turbine_guard.deployment.restore import RestoreStatus, restore_deployment_bundle
from turbine_guard.modeling.artifacts import serialize_joblib, sha256_path
from turbine_guard.modeling.conformal import SplitConformalCalibrator
from turbine_guard.modeling.estimators import MedianRulRegressor
from turbine_guard.modeling.pipeline import ModelBundle
from turbine_guard.serving.bundle_loader import DeploymentBundleLoader
from turbine_guard.serving.champion import (
    EXPORTED_SNAPSHOT_SOURCE,
    ModelMetadata,
    validate_prediction_output,
)

FEATURE_COLUMNS = ("sensor_01_current",)
ALIASES = {"champion": "7", "challenger": "7", "candidate": "7"}


def _tiny_bundle() -> ModelBundle:
    pipeline = Pipeline((("model", MedianRulRegressor()),))
    pipeline.fit(pd.DataFrame({FEATURE_COLUMNS[0]: [0.0, 1.0]}), [40.0, 40.0])
    conformal = SplitConformalCalibrator(0.9).fit([38.0, 41.0], [40.0, 40.0])
    return ModelBundle(
        pipeline,
        FEATURE_COLUMNS,
        "capped_125",
        125,
        30,
        50,
        conformal,
        {"model_kind": "constant_median", "model_configuration": {}},
    )


def _write_source_data_dir(data_dir: Path) -> None:
    contents = {
        CHAMPION_RELATIVE_PATH: serialize_joblib(_tiny_bundle()),
        FEATURE_MANIFEST_RELATIVE_PATH: b'{"feature_manifest": "fixture"}\n',
        SPLIT_MANIFEST_RELATIVE_PATH: b'{"split_manifest": "fixture"}\n',
        PROCESSING_REPORT_RELATIVE_PATH: b'{"processing_report": "fixture"}\n',
        TRAIN_PARQUET_RELATIVE_PATH: b"parquet-fixture-bytes",
    }
    for relative, payload in contents.items():
        path = data_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def _metadata(data_dir: Path) -> ModelMetadata:
    from datetime import UTC, datetime

    return ModelMetadata(
        model_name="TurbineGuard-FD001-RUL",
        version="7",
        alias="champion",
        source_run_id="run-7",
        target_definition="capped_125",
        rul_cap=125,
        feature_count=len(FEATURE_COLUMNS),
        feature_version="1",
        validation_rmse=1.0,
        replay_rmse=2.0,
        official_test_rmse=3.0,
        conformal_coverage_target=0.9,
        loaded_at=datetime.now(UTC),
        checksum=sha256_path(data_dir / CHAMPION_RELATIVE_PATH),
        lineage_id="lineage",
        model_family="constant_median",
        git_sha="abc1234",
        dataset_checksum="dataset-sha",
        feature_manifest_checksum=sha256_path(data_dir / FEATURE_MANIFEST_RELATIVE_PATH),
    )


def _install_fake_registry_loader(monkeypatch: pytest.MonkeyPatch, data_dir: Path) -> None:
    metadata = _metadata(data_dir)

    class FakeLoader:
        def __init__(self, settings: Settings) -> None:
            del settings

        def get(self) -> SimpleNamespace:
            return SimpleNamespace(metadata=metadata)

        def registry_aliases(self) -> dict[str, str]:
            return dict(ALIASES)

    monkeypatch.setattr("turbine_guard.serving.model_loader.ChampionModelLoader", FakeLoader)


def _export(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, str]:
    source_dir = tmp_path / "source-data"
    _write_source_data_dir(source_dir)
    _install_fake_registry_loader(monkeypatch, source_dir)
    settings = Settings(data_dir=source_dir, online_inference_enabled=False)
    result = export_deployment_bundle(settings, tmp_path / "bundle.tar.gz")
    return result.archive_path, result.archive_sha256


def _restore_settings(tmp_path: Path, archive: Path, sha256: str) -> Settings:
    return Settings(
        data_dir=tmp_path / "restored-data",
        online_inference_enabled=False,
        model_source="deployment_bundle",
        deployment_bundle_url=archive.as_uri(),
        deployment_bundle_sha256=sha256,
    )


def _fake_feature_manifest() -> SimpleNamespace:
    record = SimpleNamespace(
        feature_version="1",
        source_columns=("sensor_01",),
        families=("current",),
        windows=(5,),
        ewm_spans=(5,),
        min_periods=1,
    )
    return SimpleNamespace(feature_config=record, feature_columns=FEATURE_COLUMNS)


def test_export_records_and_checksums_every_required_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive, archive_sha256 = _export(monkeypatch, tmp_path)
    assert archive.is_file()
    assert len(archive_sha256) == 64
    with tarfile.open(archive, mode="r:gz") as tar:
        names = set(tar.getnames())
    assert MANIFEST_FILENAME in names
    assert {required.as_posix() for required in REQUIRED_BUNDLE_FILES} <= names


def test_export_refuses_champion_differing_from_registry_checksum(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_dir = tmp_path / "source-data"
    _write_source_data_dir(source_dir)
    _install_fake_registry_loader(monkeypatch, source_dir)
    (source_dir / CHAMPION_RELATIVE_PATH).write_bytes(b"tampered-after-registration")
    settings = Settings(data_dir=source_dir, online_inference_enabled=False)
    with pytest.raises(BundleError, match="registered champion checksum"):
        export_deployment_bundle(settings, tmp_path / "bundle.tar.gz")


def test_restore_round_trip_is_verified_and_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive, archive_sha256 = _export(monkeypatch, tmp_path)
    settings = _restore_settings(tmp_path, archive, archive_sha256)
    first = restore_deployment_bundle(settings)
    assert first.status is RestoreStatus.RESTORED
    assert first.manifest.registry_version == "7"
    for required in REQUIRED_BUNDLE_FILES:
        assert (settings.data_dir / required).is_file()
    assert (settings.data_dir / MANIFEST_FILENAME).is_file()
    assert (settings.data_dir / STATE_FILENAME).is_file()

    second = restore_deployment_bundle(settings)
    assert second.status is RestoreStatus.ALREADY_RESTORED

    (settings.data_dir / TRAIN_PARQUET_RELATIVE_PATH).write_bytes(b"corrupted")
    healed = restore_deployment_bundle(settings)
    assert healed.status is RestoreStatus.RESTORED
    assert (
        settings.data_dir / TRAIN_PARQUET_RELATIVE_PATH
    ).read_bytes() == b"parquet-fixture-bytes"


def test_restore_rejects_wrong_archive_pin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    archive, _ = _export(monkeypatch, tmp_path)
    settings = _restore_settings(tmp_path, archive, "0" * 64)
    with pytest.raises(BundleError, match="checksum mismatch"):
        restore_deployment_bundle(settings)
    assert not (settings.data_dir / STATE_FILENAME).exists()


def test_restore_rejects_member_tampered_after_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive, _ = _export(monkeypatch, tmp_path)
    repacked = tmp_path / "tampered.tar.gz"
    extracted = tmp_path / "tamper-work"
    with tarfile.open(archive, mode="r:gz") as tar:
        tar.extractall(extracted, filter="data")
    (extracted / TRAIN_PARQUET_RELATIVE_PATH).write_bytes(b"parquet-fixture-bytez")
    with tarfile.open(repacked, mode="w:gz") as tar:
        for path in sorted(extracted.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(extracted).as_posix())
    settings = _restore_settings(tmp_path, repacked, sha256_path(repacked))
    with pytest.raises(BundleError, match="checksum mismatch for"):
        restore_deployment_bundle(settings)


def test_restore_rejects_path_traversal_members(tmp_path: Path) -> None:
    malicious = tmp_path / "malicious.tar.gz"
    with tarfile.open(malicious, mode="w:gz") as tar:
        payload = b"outside"
        info = tarfile.TarInfo(name="../escape.txt")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    settings = _restore_settings(tmp_path, malicious, sha256_path(malicious))
    with pytest.raises(BundleError, match="extraction failed"):
        restore_deployment_bundle(settings)
    assert not (tmp_path / "escape.txt").exists()


def test_bundle_loader_serves_the_exported_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive, archive_sha256 = _export(monkeypatch, tmp_path)
    settings = _restore_settings(tmp_path, archive, archive_sha256)
    restore_deployment_bundle(settings)
    monkeypatch.setattr(
        "turbine_guard.serving.bundle_loader.load_feature_manifest",
        lambda _: _fake_feature_manifest(),
    )
    loader = DeploymentBundleLoader(settings)
    loaded = loader.get()
    assert loader.get() is loaded
    assert loaded.metadata.registry_source == EXPORTED_SNAPSHOT_SOURCE
    assert loaded.metadata.version == "7"
    assert loaded.metadata.rul_cap == 125
    assert loaded.feature_columns == FEATURE_COLUMNS
    assert loader.check_model() is True
    assert loader.check_feature_contract() is True
    assert loader.registry_aliases() == ALIASES

    frame = loaded.model.predict(pd.DataFrame({FEATURE_COLUMNS[0]: [0.5]}))
    point, lower, upper, risk = validate_prediction_output(frame)
    assert point == 40.0
    assert lower <= point <= upper
    assert risk == "warning"


def test_bundle_loader_rejects_tampered_champion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive, archive_sha256 = _export(monkeypatch, tmp_path)
    settings = _restore_settings(tmp_path, archive, archive_sha256)
    restore_deployment_bundle(settings)
    monkeypatch.setattr(
        "turbine_guard.serving.bundle_loader.load_feature_manifest",
        lambda _: _fake_feature_manifest(),
    )
    (settings.data_dir / CHAMPION_RELATIVE_PATH).write_bytes(b"tampered")
    loader = DeploymentBundleLoader(settings)
    with pytest.raises(RuntimeError, match="could not be loaded"):
        loader.get()
    assert loader.check_model() is False


def test_bundle_loader_failed_refresh_preserves_current_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive, archive_sha256 = _export(monkeypatch, tmp_path)
    settings = _restore_settings(tmp_path, archive, archive_sha256)
    restore_deployment_bundle(settings)
    monkeypatch.setattr(
        "turbine_guard.serving.bundle_loader.load_feature_manifest",
        lambda _: _fake_feature_manifest(),
    )
    loader = DeploymentBundleLoader(settings)
    current = loader.get()
    (settings.data_dir / CHAMPION_RELATIVE_PATH).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="could not be loaded"):
        loader.refresh()
    assert loader.get() is current
