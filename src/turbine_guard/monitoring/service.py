"""Durable Loop 9 monitoring, retraining, promotion, and rollback orchestration."""

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session, sessionmaker

from turbine_guard.config.settings import Settings
from turbine_guard.data.acquisition import current_git_commit
from turbine_guard.database.commands import (
    NewDataQualityReport,
    NewDriftReport,
    NewLifecycleAssetAssignment,
    NewLifecycleEvent,
    NewModelEvaluation,
    NewPipelineRun,
)
from turbine_guard.database.enums import (
    DataQualityStatus,
    EvaluationScope,
    LifecycleAssetRole,
    PipelineRunStatus,
    PipelineRunType,
)
from turbine_guard.database.models import PipelineRun
from turbine_guard.database.repositories import (
    DataQualityReportRepository,
    DriftReportRepository,
    LifecycleAssetAssignmentRepository,
    LifecycleEventRepository,
    ModelEvaluationRepository,
    PipelineRunRepository,
)
from turbine_guard.database.session import (
    DatabaseConfig,
    create_database_engine,
    create_session_factory,
    session_scope,
)
from turbine_guard.features.manifest import feature_config_from_manifest, load_feature_manifest
from turbine_guard.modeling.artifacts import load_joblib, sha256_bytes, sha256_path
from turbine_guard.modeling.config import AlertConfig, TrainingConfig
from turbine_guard.modeling.pipeline import ModelBundle
from turbine_guard.monitoring.candidate import (
    CandidateComparison,
    PromotionGateResult,
    TrainedCandidate,
    champion_candidate_config,
    compare_candidate,
    promotion_gates,
    train_candidate,
)
from turbine_guard.monitoring.config import LifecycleConfig
from turbine_guard.monitoring.data import (
    completed_labeled_assets,
    delayed_model_frame,
    feature_window,
    previously_trained_asset_ids,
    sensor_window,
)
from turbine_guard.monitoring.decisions import TriggerAction, TriggerDecision, decide_retraining
from turbine_guard.monitoring.drift import feature_drift_report
from turbine_guard.monitoring.performance import delayed_performance_report
from turbine_guard.monitoring.quality import data_quality_report
from turbine_guard.monitoring.reference import build_training_reference
from turbine_guard.monitoring.retraining import (
    RetrainingSplit,
    assemble_holdout_frame,
    assemble_training_frame,
    load_original_training_frame,
    split_labeled_assets,
)
from turbine_guard.serving.model_loader import ChampionModelLoader
from turbine_guard.tracking.config import MlflowConfig
from turbine_guard.tracking.lifecycle import (
    CandidateRegistration,
    aliases,
    attach_training_reference,
    champion_baseline_metrics,
    configured_mlflow,
    load_champion,
    promote_candidate,
    register_candidate,
    rollback_champion,
)


@dataclass(frozen=True)
class MonitoringRunResult:
    run_id: uuid.UUID
    status: str
    decision: TriggerDecision
    data_quality: dict[str, Any]
    drift: dict[str, Any]
    performance: dict[str, Any]


@dataclass(frozen=True)
class LifecycleRunResult:
    run_id: uuid.UUID
    phase: str
    status: str
    champion_version: str
    candidate_version: str | None
    gates: dict[str, Any] | None
    aliases: dict[str, str]


class LifecycleService:
    """Phase-checkpointed lifecycle service with PostgreSQL and MLflow idempotency."""

    def __init__(
        self,
        settings: Settings,
        *,
        sessions: sessionmaker[Session] | None = None,
        model_loader: ChampionModelLoader | None = None,
    ) -> None:
        self.settings = settings
        self.config = LifecycleConfig.from_settings(settings)
        self.mlflow = MlflowConfig.from_settings(settings)
        self._engine = None
        if sessions is None:
            self._engine = create_database_engine(DatabaseConfig.from_settings(settings))
            sessions = create_session_factory(self._engine)
        self.sessions = sessions
        self.model_loader = model_loader

    def close(self) -> None:
        if self._engine is not None:
            self._engine.dispose()

    def run_monitoring(
        self,
        *,
        window_start: datetime | None = None,
        window_end: datetime | None = None,
        manual_force: bool = False,
    ) -> MonitoringRunResult:
        """Persist one idempotent monitoring window and its structured trigger decision."""
        end = _aware(window_end or datetime.now(UTC))
        start = _aware(window_start or end - timedelta(days=self.config.window_days))
        if start >= end:
            raise ValueError("Monitoring window start must precede its end.")
        champion = load_champion(self.mlflow)
        key = _digest(
            "monitoring",
            champion.version,
            start.isoformat(),
            end.isoformat(),
            str(manual_force),
        )
        run_id = self._get_or_create_run(
            run_type=PipelineRunType.MONITORING,
            key=key,
            trigger="manual_force" if manual_force else "scheduled_or_manual",
            model_version=champion.version,
            metadata={
                "window_start": start.isoformat(),
                "window_end": end.isoformat(),
                "champion_version": champion.version,
                "manual_force": manual_force,
            },
        )
        with session_scope(self.sessions) as session:
            run = _required_run(PipelineRunRepository(session).get_for_update(run_id), run_id)
            if "trigger_decision" in run.run_metadata:
                return _monitoring_result(run)

        manifest = load_feature_manifest(
            self.settings.data_dir / "features" / "cmapss" / "FD001" / "feature_manifest.json"
        )
        feature_config = feature_config_from_manifest(manifest)
        reference = build_training_reference(
            data_dir=self.settings.data_dir,
            model_name=self.mlflow.registered_model_name,
            model_version=champion.version,
            expected_feature_version=feature_config.feature_version,
        )
        attach_training_reference(
            config=self.mlflow,
            champion=champion,
            reference_path=reference.path,
            reference_sha256=reference.sha256,
        )

        with session_scope(self.sessions) as session:
            run = _required_run(PipelineRunRepository(session).get_for_update(run_id), run_id)
            if "trigger_decision" in run.run_metadata:
                return _monitoring_result(run)
            raw = sensor_window(session, start, end)
            features = feature_window(session, start, end, feature_config)
            delayed = delayed_model_frame(
                session,
                start=start,
                end=end,
                model_name=self.mlflow.registered_model_name,
                model_version=champion.version,
            )
            quality = data_quality_report(
                raw,
                reference=reference.reference,
                minimum_rows=self.config.minimum_quality_rows,
                minimum_assets=self.config.minimum_quality_assets,
                sufficient_history_cycles=max(feature_config.windows, default=1),
                out_of_range_stddevs=self.config.out_of_range_stddevs,
            )
            drift = feature_drift_report(
                features,
                reference=reference.reference,
                thresholds=self.config.drift,
            )
            performance = delayed_performance_report(
                delayed,
                AlertConfig(
                    critical_horizon=champion.bundle.critical_horizon,
                    warning_horizon=champion.bundle.warning_horizon,
                ),
            )
            all_assets = completed_labeled_assets(session, feature_config)
            trained_ids = previously_trained_asset_ids(session)
            new_assets = [asset for asset in all_assets if asset.asset_id not in trained_ids]
            new_rows = sum(asset.row_count for asset in new_assets)
            safe_holdout = len(new_assets) > self.config.trigger.minimum_holdout_assets
            interval_elapsed = self._retraining_interval_elapsed(session, end)
            decision = decide_retraining(
                thresholds=self.config.trigger,
                data_quality_status=quality.status,
                drift_status=drift.status,
                drifted_feature_count=drift.drifted_feature_count,
                newly_labeled_assets=len(new_assets),
                newly_labeled_rows=new_rows,
                current_metrics=performance.metrics if performance.sample_count else None,
                baseline_metrics=champion_baseline_metrics(champion),
                interval_elapsed=interval_elapsed,
                safe_holdout_available=safe_holdout,
                manual_force=manual_force,
            )
            DataQualityReportRepository(session).create(
                NewDataQualityReport(
                    pipeline_run_id=run.id,
                    model_name=self.mlflow.registered_model_name,
                    model_version=champion.version,
                    feature_version=feature_config.feature_version,
                    window_start=start,
                    window_end=end,
                    status=quality.status,
                    record_count=quality.record_count,
                    asset_count=quality.asset_count,
                    failure_count=quality.failure_count,
                    details={**quality.details, "training_reference_sha256": reference.sha256},
                )
            )
            DriftReportRepository(session).create(
                NewDriftReport(
                    model_name=self.mlflow.registered_model_name,
                    model_version=champion.version,
                    feature_version=feature_config.feature_version,
                    window_start=start,
                    window_end=end,
                    status=drift.status,
                    drifted_feature_count=drift.drifted_feature_count,
                    max_psi=drift.max_psi,
                    max_wasserstein=drift.max_wasserstein,
                    details={
                        **drift.details,
                        "pipeline_run_id": str(run.id),
                        "training_reference_sha256": reference.sha256,
                    },
                )
            )
            if performance.sample_count:
                _persist_performance(
                    session,
                    run,
                    self.mlflow.registered_model_name,
                    champion.version,
                    start,
                    end,
                    performance.sample_count,
                    performance.metrics,
                )
            metadata = {
                **run.run_metadata,
                "training_reference_sha256": reference.sha256,
                "data_quality": {
                    "status": quality.status.value,
                    "record_count": quality.record_count,
                    "asset_count": quality.asset_count,
                    "failure_count": quality.failure_count,
                    "details": quality.details,
                },
                "drift": {
                    "status": drift.status.value,
                    "drifted_feature_count": drift.drifted_feature_count,
                    "max_psi": drift.max_psi,
                    "max_wasserstein": drift.max_wasserstein,
                },
                "performance": performance.metrics,
                "trigger_decision": decision.record(),
                "eligible_asset_ids": [str(asset.asset_id) for asset in new_assets],
            }
            repository = PipelineRunRepository(session)
            repository.checkpoint(run, phase="decision_recorded", metadata=metadata)
            repository.finish(
                run,
                status=PipelineRunStatus.SUCCEEDED,
                finished_at=datetime.now(UTC),
                output_manifest_checksum=_json_digest(metadata),
            )
            return _monitoring_result(run)

    def start_retraining(self, *, manual_force: bool = False) -> LifecycleRunResult:
        """Run or resume candidate fitting through gates, stopping for approval by default."""
        monitoring = self.run_monitoring(manual_force=manual_force)
        if monitoring.decision.action is not TriggerAction.RETRAIN:
            return LifecycleRunResult(
                monitoring.run_id,
                "trigger_decision",
                monitoring.decision.action.value,
                str(monitoring.data_quality.get("champion_version", "unknown")),
                None,
                None,
                self.registry_aliases(),
            )
        champion = load_champion(self.mlflow)
        manifest = load_feature_manifest(
            self.settings.data_dir / "features" / "cmapss" / "FD001" / "feature_manifest.json"
        )
        feature_config = feature_config_from_manifest(manifest)
        with session_scope(self.sessions) as session:
            all_assets = completed_labeled_assets(session, feature_config)
            trained_ids = previously_trained_asset_ids(session)
        by_id = {asset.asset_id: asset for asset in all_assets}
        new_assets = [asset for asset in all_assets if asset.asset_id not in trained_ids]
        split = split_labeled_assets(
            new_assets,
            holdout_fraction=self.config.holdout_fraction,
            minimum_holdout_assets=self.config.trigger.minimum_holdout_assets,
            seed=int(champion.run_params.get("random_seed", "42")),
        )
        prior_additions = tuple(by_id[asset_id] for asset_id in sorted(trained_ids, key=str))
        additions = (*prior_additions, *split.additions)
        key = _digest(
            "retraining",
            champion.version,
            *(str(asset.asset_id) for asset in split.additions),
            "holdout",
            *(str(asset.asset_id) for asset in split.holdout),
        )
        run_id = self._get_or_create_run(
            run_type=PipelineRunType.RETRAINING,
            key=key,
            trigger="manual_force" if manual_force else "monitoring_decision",
            model_version=champion.version,
            metadata={
                "monitoring_run_id": str(monitoring.run_id),
                "champion_version": champion.version,
                "data_split": RetrainingSplit(tuple(additions), split.holdout).record(),
                "new_asset_ids": [str(asset.asset_id) for asset in new_assets],
            },
        )
        existing_state = self.get_lifecycle(run_id)
        if (
            existing_state.status
            in {
                PipelineRunStatus.SUCCEEDED.value,
                PipelineRunStatus.CANCELLED.value,
                PipelineRunStatus.FAILED.value,
            }
            or existing_state.phase == "awaiting_approval"
        ):
            return existing_state
        self._persist_assignments(run_id, tuple(additions), split.holdout)
        paths = _lifecycle_paths(self.settings.data_dir, run_id)
        training_config = TrainingConfig(
            data_dir=self.settings.data_dir,
            random_seed=int(champion.run_params.get("random_seed", "42")),
            alerts=AlertConfig(
                critical_horizon=champion.bundle.critical_horizon,
                warning_horizon=champion.bundle.warning_horizon,
            ),
        )
        original = load_original_training_frame(
            self.settings.data_dir,
            expected_feature_columns=champion.bundle.feature_columns,
        )
        training_frame = assemble_training_frame(
            original,
            tuple(additions),
            feature_columns=champion.bundle.feature_columns,
        )
        holdout_frame = assemble_holdout_frame(
            split.holdout, feature_columns=champion.bundle.feature_columns
        )
        candidate_config = champion_candidate_config(
            model_family=champion.run_params["model_family"],
            parameters=champion.run_params,
            candidate_name=f"retrained_{champion.run_tags.get('candidate_id', 'champion')}",
        )
        trained = self._train_or_load_candidate(
            run_id,
            paths,
            training_frame,
            champion.bundle,
            candidate_config,
            training_config,
            additions,
        )
        comparison = self._compare_or_load(
            run_id,
            paths,
            trained,
            champion.bundle,
            training_frame,
            holdout_frame,
            training_config,
        )
        registration = register_candidate(
            config=self.mlflow,
            lifecycle_id=str(run_id),
            bundle_path=paths.candidate,
            bundle_sha256=sha256_path(paths.candidate),
            bundle=trained.bundle,
            input_example=holdout_frame.loc[:, list(champion.bundle.feature_columns)].head(2),
            comparison=comparison.record(),
            lineage={
                "feature_version": feature_config.feature_version,
                "feature_manifest_sha256": sha256_path(
                    self.settings.data_dir
                    / "features"
                    / "cmapss"
                    / "FD001"
                    / "feature_manifest.json"
                ),
                "target_configuration": json.dumps(
                    {"name": trained.target_config.name, "cap": trained.target_config.cap},
                    sort_keys=True,
                ),
                "git_sha": current_git_commit() or "unknown",
            },
        )
        self._checkpoint_registration(run_id, registration)
        gates = promotion_gates(
            comparison,
            thresholds=self.config.promotion,
            data_quality_passes=(monitoring.data_quality["status"] == DataQualityStatus.PASS.value),
            enough_labeled_data=(
                len(new_assets) >= self.config.trigger.minimum_assets
                and sum(asset.row_count for asset in new_assets) >= self.config.trigger.minimum_rows
                and len(split.holdout) >= self.config.trigger.minimum_holdout_assets
            ),
            artifact_valid=_valid_bundle(paths.candidate, champion.bundle.feature_columns),
            reload_equivalence_difference=registration.max_prediction_difference,
        )
        self._checkpoint_gates(run_id, gates, registration)
        if not gates.passed:
            return self.reject_candidate(
                run_id, reason="blocking gates: " + ", ".join(gates.blocking_failures)
            )
        if self.config.approval_required:
            with session_scope(self.sessions) as session:
                run = _required_run(PipelineRunRepository(session).get_for_update(run_id), run_id)
                PipelineRunRepository(session).checkpoint(
                    run,
                    phase="awaiting_approval",
                    status=PipelineRunStatus.PENDING,
                    metadata={**run.run_metadata, "approval_required": True},
                )
            return self.get_lifecycle(run_id)
        return self.approve_promotion(run_id, actor="automatic_policy")

    def approve_promotion(
        self, run_id: uuid.UUID, *, actor: str = "manual_cli"
    ) -> LifecycleRunResult:
        """Promote only a gate-passing candidate and append an approval audit event."""
        with session_scope(self.sessions) as session:
            run = _required_run(PipelineRunRepository(session).get(run_id), run_id)
            metadata = dict(run.run_metadata)
        gates = metadata.get("promotion_gates", {})
        if not bool(gates.get("passed")):
            raise ValueError("Candidate cannot be approved because blocking gates did not pass.")
        candidate_version = str(metadata["candidate_version"])
        champion_version = str(metadata["champion_version"])
        aliases_after = promote_candidate(
            config=self.mlflow,
            candidate_version=candidate_version,
            expected_champion=champion_version,
        )
        refresh: dict[str, Any] = {"attempted": self.model_loader is not None}
        if self.model_loader is not None:
            try:
                loaded = self.model_loader.refresh()
                refresh.update({"succeeded": True, "loaded_version": loaded.metadata.version})
            except Exception as exc:
                refresh.update({"succeeded": False, "error": str(exc)})
        with session_scope(self.sessions) as session:
            run = _required_run(PipelineRunRepository(session).get_for_update(run_id), run_id)
            LifecycleEventRepository(session).create(
                NewLifecycleEvent(
                    event_key=f"{run_id}:promotion-approved",
                    event_type="promotion_approved",
                    phase="promoted",
                    model_name=self.mlflow.registered_model_name,
                    actor=actor,
                    pipeline_run_id=run_id,
                    from_version=champion_version,
                    to_version=candidate_version,
                    details={"aliases": aliases_after, "serving_refresh": refresh},
                )
            )
            metadata = {
                **run.run_metadata,
                "approval": {"approved": True, "actor": actor},
                "aliases_after": aliases_after,
                "serving_refresh": refresh,
            }
            repository = PipelineRunRepository(session)
            repository.checkpoint(run, phase="promoted", metadata=metadata)
            repository.finish(
                run,
                status=PipelineRunStatus.SUCCEEDED,
                finished_at=datetime.now(UTC),
                output_manifest_checksum=_json_digest(metadata),
            )
        return self.get_lifecycle(run_id)

    def reject_candidate(
        self, run_id: uuid.UUID, *, reason: str, actor: str = "policy"
    ) -> LifecycleRunResult:
        """Record rejection without moving the champion alias."""
        if not reason.strip():
            raise ValueError("Candidate rejection requires a reason.")
        with session_scope(self.sessions) as session:
            run = _required_run(PipelineRunRepository(session).get_for_update(run_id), run_id)
            metadata = dict(run.run_metadata)
            expected = str(metadata["champion_version"])
            current = self.registry_aliases().get(self.mlflow.champion_alias)
            if current != expected:
                raise ValueError("Champion changed; rejection will not rewrite lifecycle history.")
            LifecycleEventRepository(session).create(
                NewLifecycleEvent(
                    event_key=f"{run_id}:candidate-rejected",
                    event_type="candidate_rejected",
                    phase="rejected",
                    model_name=self.mlflow.registered_model_name,
                    actor=actor,
                    pipeline_run_id=run_id,
                    from_version=expected,
                    to_version=metadata.get("candidate_version"),
                    details={"reason": reason},
                )
            )
            metadata.update({"rejection": {"reason": reason, "actor": actor}})
            repository = PipelineRunRepository(session)
            repository.checkpoint(run, phase="rejected", metadata=metadata)
            if run.status not in {
                PipelineRunStatus.CANCELLED,
                PipelineRunStatus.SUCCEEDED,
                PipelineRunStatus.FAILED,
            }:
                repository.finish(
                    run,
                    status=PipelineRunStatus.CANCELLED,
                    finished_at=datetime.now(UTC),
                    output_manifest_checksum=_json_digest(metadata),
                )
        return self.get_lifecycle(run_id)

    def rollback(self, target_version: str, *, actor: str = "manual_cli") -> LifecycleRunResult:
        """Restore a prior valid version and persist the alias transition."""
        current = self.registry_aliases().get(self.mlflow.champion_alias)
        if current is None:
            raise ValueError("There is no current champion to roll back.")
        existing = self._rollback_run_for_target(target_version)
        if current == target_version:
            if existing is None:
                raise ValueError(f"Model version {target_version} is already the champion.")
            if existing.status is PipelineRunStatus.SUCCEEDED:
                return self.get_lifecycle(existing.id)
            run_id = existing.id
            displaced = str(existing.run_metadata["champion_version"])
            aliases_after = self.registry_aliases()
        else:
            key = _digest("rollback", current, target_version)
            run_id = self._get_or_create_run(
                run_type=PipelineRunType.PROMOTION,
                key=key,
                trigger="manual_rollback",
                model_version=current,
                metadata={"champion_version": current, "rollback_target": target_version},
            )
            displaced, aliases_after = rollback_champion(
                config=self.mlflow, target_version=target_version
            )
        refresh: dict[str, Any] = {"attempted": self.model_loader is not None}
        if self.model_loader is not None:
            try:
                loaded = self.model_loader.refresh()
                refresh.update({"succeeded": True, "loaded_version": loaded.metadata.version})
            except Exception as exc:
                refresh.update({"succeeded": False, "error": str(exc)})
        with session_scope(self.sessions) as session:
            run = _required_run(PipelineRunRepository(session).get_for_update(run_id), run_id)
            LifecycleEventRepository(session).create(
                NewLifecycleEvent(
                    event_key=f"{run_id}:rollback",
                    event_type="champion_rollback",
                    phase="rolled_back",
                    model_name=self.mlflow.registered_model_name,
                    actor=actor,
                    pipeline_run_id=run_id,
                    from_version=current,
                    to_version=target_version,
                    details={
                        "displaced_version": displaced,
                        "aliases": aliases_after,
                        "serving_refresh": refresh,
                    },
                )
            )
            metadata = {
                **run.run_metadata,
                "aliases_after": aliases_after,
                "serving_refresh": refresh,
            }
            repository = PipelineRunRepository(session)
            repository.checkpoint(run, phase="rolled_back", metadata=metadata)
            if run.status not in {PipelineRunStatus.SUCCEEDED, PipelineRunStatus.CANCELLED}:
                repository.finish(
                    run,
                    status=PipelineRunStatus.SUCCEEDED,
                    finished_at=datetime.now(UTC),
                    output_manifest_checksum=_json_digest(metadata),
                )
        return self.get_lifecycle(run_id)

    def _rollback_run_for_target(self, target_version: str) -> PipelineRun | None:
        """Find the latest rollback checkpoint used to reconcile an already-moved alias."""
        with session_scope(self.sessions) as session:
            runs = session.scalars(
                select(PipelineRun)
                .where(
                    PipelineRun.run_type == PipelineRunType.PROMOTION,
                    PipelineRun.trigger == "manual_rollback",
                )
                .order_by(desc(PipelineRun.started_at))
            )
            return next(
                (
                    run
                    for run in runs
                    if str(run.run_metadata.get("rollback_target")) == target_version
                    and run.status is not PipelineRunStatus.CANCELLED
                    and run.status is not PipelineRunStatus.FAILED
                ),
                None,
            )

    def get_lifecycle(self, run_id: uuid.UUID) -> LifecycleRunResult:
        with session_scope(self.sessions) as session:
            run = _required_run(PipelineRunRepository(session).get(run_id), run_id)
            metadata = run.run_metadata
            return LifecycleRunResult(
                run_id=run.id,
                phase=run.phase or "created",
                status=run.status.value,
                champion_version=str(metadata.get("champion_version", run.model_version or "")),
                candidate_version=(
                    None
                    if metadata.get("candidate_version") is None
                    else str(metadata["candidate_version"])
                ),
                gates=metadata.get("promotion_gates"),
                aliases=self.registry_aliases(),
            )

    def recent_status(self, *, limit: int = 20) -> list[dict[str, Any]]:
        with session_scope(self.sessions) as session:
            return [
                {
                    "run_id": str(run.id),
                    "run_type": run.run_type.value,
                    "status": run.status.value,
                    "phase": run.phase,
                    "started_at": run.started_at.isoformat(),
                    "finished_at": None if run.finished_at is None else run.finished_at.isoformat(),
                    "model_version": run.model_version,
                    "metadata": run.run_metadata,
                }
                for run in PipelineRunRepository(session).recent(limit=limit)
            ]

    def registry_aliases(self) -> dict[str, str]:
        with configured_mlflow(self.mlflow) as client:
            return aliases(client, self.mlflow)

    def refresh_serving_model(self) -> str:
        loader = self.model_loader or ChampionModelLoader(self.settings)
        return loader.refresh().metadata.version

    def _get_or_create_run(
        self,
        *,
        run_type: PipelineRunType,
        key: str,
        trigger: str,
        model_version: str,
        metadata: dict[str, Any],
    ) -> uuid.UUID:
        with session_scope(self.sessions) as session:
            repository = PipelineRunRepository(session)
            existing = repository.get_by_idempotency_key(key)
            if existing is not None:
                return existing.id
            run = repository.create(
                NewPipelineRun(
                    run_type=run_type,
                    status=PipelineRunStatus.RUNNING,
                    started_at=datetime.now(UTC),
                    trigger=trigger,
                    idempotency_key=key,
                    phase="created",
                    git_sha=current_git_commit(),
                    model_version=model_version,
                    metadata=metadata,
                )
            )
            return run.id

    def _retraining_interval_elapsed(self, session: Session, at: datetime) -> bool:
        latest = session.scalar(
            select(PipelineRun)
            .where(
                PipelineRun.run_type == PipelineRunType.RETRAINING,
                PipelineRun.status == PipelineRunStatus.SUCCEEDED,
            )
            .order_by(desc(PipelineRun.finished_at))
            .limit(1)
        )
        return (
            latest is None
            or latest.finished_at is None
            or (at - latest.finished_at >= timedelta(days=self.config.trigger.interval_days))
        )

    def _persist_assignments(
        self,
        run_id: uuid.UUID,
        additions: tuple[Any, ...],
        holdout: tuple[Any, ...],
    ) -> None:
        with session_scope(self.sessions) as session:
            run = _required_run(PipelineRunRepository(session).get_for_update(run_id), run_id)
            repository = LifecycleAssetAssignmentRepository(session)
            if repository.for_run(run_id):
                return
            for role, assets in (
                (LifecycleAssetRole.RETRAINING_ADDITION, additions),
                (LifecycleAssetRole.PROMOTION_HOLDOUT, holdout),
            ):
                for asset in assets:
                    repository.create(
                        NewLifecycleAssetAssignment(
                            pipeline_run_id=run_id,
                            asset_id=asset.asset_id,
                            role=role,
                            row_count=asset.row_count,
                            source_asset_id=str(asset.source_asset_id),
                        )
                    )
            PipelineRunRepository(session).checkpoint(run, phase="data_assigned")

    def _train_or_load_candidate(
        self,
        run_id: uuid.UUID,
        paths: "LifecyclePaths",
        training_frame: Any,
        champion: ModelBundle,
        candidate_config: Any,
        training_config: TrainingConfig,
        additions: tuple[Any, ...],
    ) -> TrainedCandidate:
        if paths.candidate.exists():
            checksum = sha256_path(paths.candidate)
            value = load_joblib(paths.candidate)
            if not isinstance(value, ModelBundle):
                raise ValueError("Persisted lifecycle candidate is not a ModelBundle.")
            self._checkpoint_candidate_artifact(run_id, paths.candidate, checksum)
            return TrainedCandidate(
                value,
                candidate_config,
                champion_candidate_target(champion),
                paths.candidate.read_bytes(),
            )
        trained = train_candidate(
            training_frame=training_frame,
            feature_columns=champion.feature_columns,
            champion_bundle=champion,
            candidate_config=candidate_config,
            training_config=training_config,
            metadata={
                "candidate_id": candidate_config.name,
                "model_kind": candidate_config.kind.value,
                "model_configuration": candidate_config.params,
                "target_definition": {"name": champion.target_name, "cap": champion.target_cap},
                "feature_count": len(champion.feature_columns),
                "random_seed": training_config.random_seed,
                "lifecycle_run_id": str(run_id),
                "retraining_asset_ids": [str(asset.asset_id) for asset in additions],
                "conformal_policy": "frozen_champion_calibrator_no_protected_data_reread",
            },
        )
        _atomic_bytes(paths.candidate, trained.artifact_bytes)
        self._checkpoint_candidate_artifact(
            run_id, paths.candidate, sha256_bytes(trained.artifact_bytes)
        )
        return trained

    def _compare_or_load(
        self,
        run_id: uuid.UUID,
        paths: "LifecyclePaths",
        trained: TrainedCandidate,
        champion: ModelBundle,
        training_frame: Any,
        holdout_frame: Any,
        training_config: TrainingConfig,
    ) -> CandidateComparison:
        if paths.comparison.exists():
            record = json.loads(paths.comparison.read_text(encoding="utf-8"))
            comparison = CandidateComparison(
                holdout_sha256=record["holdout_sha256"],
                row_count=record["row_count"],
                asset_count=record["asset_count"],
                candidate=record["candidate"],
                champion=record["champion"],
                naive=record["naive"],
            )
            self._checkpoint_comparison(run_id, paths.comparison, comparison)
            return comparison
        comparison = compare_candidate(
            candidate=trained,
            champion=champion,
            training_frame=training_frame,
            holdout_frame=holdout_frame,
            training_config=training_config,
        )
        _atomic_json(paths.comparison, comparison.record())
        self._checkpoint_comparison(run_id, paths.comparison, comparison)
        return comparison

    def _checkpoint_candidate_artifact(self, run_id: uuid.UUID, path: Path, checksum: str) -> None:
        with session_scope(self.sessions) as session:
            run = _required_run(PipelineRunRepository(session).get_for_update(run_id), run_id)
            existing = run.run_metadata.get("candidate_artifact_sha256")
            if existing is not None:
                if existing != checksum:
                    raise ValueError("Persisted candidate checksum differs from lifecycle state.")
                return
            PipelineRunRepository(session).checkpoint(
                run,
                phase="candidate_trained",
                metadata={
                    **run.run_metadata,
                    "candidate_artifact": str(path),
                    "candidate_artifact_sha256": checksum,
                },
            )

    def _checkpoint_comparison(
        self, run_id: uuid.UUID, path: Path, comparison: CandidateComparison
    ) -> None:
        checksum = sha256_path(path)
        with session_scope(self.sessions) as session:
            run = _required_run(PipelineRunRepository(session).get_for_update(run_id), run_id)
            existing = run.run_metadata.get("comparison_sha256")
            if existing is not None:
                if existing != checksum:
                    raise ValueError("Persisted comparison checksum differs from lifecycle state.")
                return
            PipelineRunRepository(session).checkpoint(
                run,
                phase="candidate_evaluated",
                metadata={
                    **run.run_metadata,
                    "comparison_path": str(path),
                    "comparison_sha256": checksum,
                    "comparison": comparison.record(),
                },
            )

    def _checkpoint_registration(
        self, run_id: uuid.UUID, registration: CandidateRegistration
    ) -> None:
        with session_scope(self.sessions) as session:
            run = _required_run(PipelineRunRepository(session).get_for_update(run_id), run_id)
            if run.run_metadata.get("candidate_version") is not None:
                return
            PipelineRunRepository(session).checkpoint(
                run,
                phase="candidate_registered",
                metadata={
                    **run.run_metadata,
                    "candidate_version": registration.version,
                    "candidate_run_id": registration.run_id,
                    "reload_prediction_difference": registration.max_prediction_difference,
                    "aliases_after_registration": registration.aliases,
                },
            )

    def _checkpoint_gates(
        self,
        run_id: uuid.UUID,
        gates: PromotionGateResult,
        registration: CandidateRegistration,
    ) -> None:
        with session_scope(self.sessions) as session:
            run = _required_run(PipelineRunRepository(session).get_for_update(run_id), run_id)
            if run.run_metadata.get("promotion_gates") is not None:
                return
            LifecycleEventRepository(session).create(
                NewLifecycleEvent(
                    event_key=f"{run_id}:gates-evaluated",
                    event_type="promotion_gates_evaluated",
                    phase="gates_evaluated",
                    model_name=self.mlflow.registered_model_name,
                    actor="policy",
                    pipeline_run_id=run_id,
                    from_version=str(run.run_metadata["champion_version"]),
                    to_version=registration.version,
                    details=gates.record(),
                )
            )
            PipelineRunRepository(session).checkpoint(
                run,
                phase="gates_evaluated",
                metadata={**run.run_metadata, "promotion_gates": gates.record()},
            )


@dataclass(frozen=True)
class LifecyclePaths:
    root: Path
    candidate: Path
    comparison: Path


def _lifecycle_paths(data_dir: Path, run_id: uuid.UUID) -> LifecyclePaths:
    root = data_dir / "monitoring" / "lifecycle" / str(run_id)
    return LifecyclePaths(root, root / "candidate.joblib", root / "comparison.json")


def _persist_performance(
    session: Session,
    run: PipelineRun,
    model_name: str,
    model_version: str,
    start: datetime,
    end: datetime,
    sample_count: int,
    metrics: dict[str, Any],
) -> None:
    regression = metrics["regression"]
    critical = metrics["critical"]
    interval = metrics.get("interval") or {}
    ModelEvaluationRepository(session).create(
        NewModelEvaluation(
            model_name=model_name,
            model_version=model_version,
            evaluation_scope=EvaluationScope.ONLINE,
            dataset_subset="FD001",
            window_start=start,
            window_end=end,
            sample_count=sample_count,
            mae=regression["mae"],
            rmse=regression["rmse"],
            nasa_score=regression["nasa_score"],
            critical_precision=critical["precision"],
            critical_recall=critical["recall"],
            interval_coverage=interval.get("empirical_coverage"),
            metrics={**metrics, "pipeline_run_id": str(run.id), "aggregation": "online_window"},
        )
    )


def _monitoring_result(run: PipelineRun) -> MonitoringRunResult:
    metadata = run.run_metadata
    decision_record = metadata["trigger_decision"]
    decision = TriggerDecision(
        action=TriggerAction(decision_record["action"]),
        reasons=tuple(decision_record["reasons"]),
        checks={str(key): bool(value) for key, value in decision_record["checks"].items()},
        signals=dict(decision_record["signals"]),
    )
    quality = dict(metadata["data_quality"])
    quality["champion_version"] = metadata["champion_version"]
    return MonitoringRunResult(
        run.id,
        run.status.value,
        decision,
        quality,
        dict(metadata["drift"]),
        dict(metadata["performance"]),
    )


def _required_run(run: PipelineRun | None, run_id: uuid.UUID) -> PipelineRun:
    if run is None:
        raise ValueError(f"Pipeline run {run_id} does not exist.")
    return run


def _digest(*values: str) -> str:
    return hashlib.sha256("\x1f".join(values).encode()).hexdigest()


def _json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Lifecycle timestamps must be timezone-aware.")
    return value.astimezone(UTC)


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _valid_bundle(path: Path, feature_columns: tuple[str, ...]) -> bool:
    try:
        value = load_joblib(path)
    except Exception:
        return False
    return isinstance(value, ModelBundle) and value.feature_columns == feature_columns


def champion_candidate_target(champion: ModelBundle) -> Any:
    from turbine_guard.modeling.config import TargetConfig

    return TargetConfig(champion.target_name, champion.target_cap)
