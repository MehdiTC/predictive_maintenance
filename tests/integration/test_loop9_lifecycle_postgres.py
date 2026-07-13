"""Real PostgreSQL persistence and partial-phase recovery for Loop 9."""

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from turbine_guard.config.settings import Settings
from turbine_guard.database.commands import (
    NewAsset,
    NewDataQualityReport,
    NewLifecycleAssetAssignment,
    NewLifecycleEvent,
    NewPipelineRun,
)
from turbine_guard.database.enums import (
    DataQualityStatus,
    LifecycleAssetRole,
    PipelineRunStatus,
    PipelineRunType,
)
from turbine_guard.database.repositories import (
    AssetRepository,
    DataQualityReportRepository,
    LifecycleAssetAssignmentRepository,
    LifecycleEventRepository,
    PipelineRunRepository,
)
from turbine_guard.database.session import session_scope
from turbine_guard.monitoring.service import LifecycleService

pytestmark = pytest.mark.postgres
NOW = datetime(2026, 7, 13, 12, tzinfo=UTC)


def test_lifecycle_reports_assignments_events_recovery_and_idempotency(
    db_session: Session,
) -> None:
    key = "a" * 64
    runs = PipelineRunRepository(db_session)
    run = runs.create(
        NewPipelineRun(
            run_type=PipelineRunType.RETRAINING,
            status=PipelineRunStatus.RUNNING,
            started_at=NOW,
            trigger="performance_degradation",
            idempotency_key=key,
            phase="created",
            model_version="1",
            metadata={"champion_version": "1"},
        )
    )
    assert runs.get_by_idempotency_key(key) is run
    locked = runs.get_for_update(run.id)
    assert locked is run
    runs.checkpoint(
        run,
        phase="reports_persisted",
        metadata={**run.run_metadata, "reports_persisted": True},
    )

    report = DataQualityReportRepository(db_session).create(
        NewDataQualityReport(
            pipeline_run_id=run.id,
            model_name="model",
            model_version="1",
            feature_version="1",
            window_start=NOW,
            window_end=NOW,
            status=DataQualityStatus.PASS,
            record_count=100,
            asset_count=5,
            failure_count=0,
            details={"accepted": True},
        )
    )
    assert DataQualityReportRepository(db_session).for_model("model", "1") == [report]

    asset = AssetRepository(db_session).create(NewAsset(external_id=f"loop9-{uuid.uuid4()}"))
    assignments = LifecycleAssetAssignmentRepository(db_session)
    assignment = assignments.create(
        NewLifecycleAssetAssignment(
            pipeline_run_id=run.id,
            asset_id=asset.id,
            role=LifecycleAssetRole.PROMOTION_HOLDOUT,
            row_count=20,
            source_asset_id="9",
        )
    )
    assert assignments.for_run(run.id) == [assignment]

    events = LifecycleEventRepository(db_session)
    command = NewLifecycleEvent(
        event_key=f"{run.id}:gates",
        event_type="promotion_gates_evaluated",
        phase="gates_evaluated",
        model_name="model",
        actor="policy",
        pipeline_run_id=run.id,
        from_version="1",
        to_version="2",
        details={"passed": False},
    )
    first = events.create(command)
    assert events.create(command) is first
    assert events.for_run(run.id) == [first]

    # Simulated restart resumes the same row from its last committed phase.
    resumed = runs.get_by_idempotency_key(key)
    assert resumed is run
    assert resumed.phase == "reports_persisted"
    assert resumed.run_metadata["reports_persisted"] is True


def test_approved_promotion_and_rejection_are_audited_and_resumable(
    postgres_engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    connection = postgres_engine.connect()
    transaction = connection.begin()
    factory = sessionmaker(
        bind=connection,
        class_=Session,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )
    settings = Settings(online_inference_enabled=False)
    service = LifecycleService(settings, sessions=factory)
    monkeypatch.setattr(
        "turbine_guard.monitoring.service.promote_candidate",
        lambda **_: {"champion": "2", "archived": "1"},
    )
    monkeypatch.setattr(
        service,
        "registry_aliases",
        lambda: {"champion": "1", "candidate": "2", "challenger": "2"},
    )
    try:
        with session_scope(factory) as session:
            approved = PipelineRunRepository(session).create(
                NewPipelineRun(
                    run_type=PipelineRunType.RETRAINING,
                    status=PipelineRunStatus.PENDING,
                    started_at=datetime.now(UTC),
                    trigger="test",
                    idempotency_key="b" * 64,
                    phase="awaiting_approval",
                    model_version="1",
                    metadata={
                        "champion_version": "1",
                        "candidate_version": "2",
                        "promotion_gates": {"passed": True},
                    },
                )
            )
            approved_id = approved.id
        result = service.approve_promotion(approved_id, actor="integration-test")
        assert result.status == "succeeded"
        assert result.phase == "promoted"
        # Re-entering after the alias phase does not duplicate audit history.
        service.approve_promotion(approved_id, actor="integration-test")
        with session_scope(factory) as session:
            assert len(LifecycleEventRepository(session).for_run(approved_id)) == 1

        with session_scope(factory) as session:
            rejected = PipelineRunRepository(session).create(
                NewPipelineRun(
                    run_type=PipelineRunType.RETRAINING,
                    status=PipelineRunStatus.PENDING,
                    started_at=datetime.now(UTC),
                    trigger="test",
                    idempotency_key="c" * 64,
                    phase="awaiting_approval",
                    model_version="1",
                    metadata={
                        "champion_version": "1",
                        "candidate_version": "3",
                        "promotion_gates": {"passed": True},
                    },
                )
            )
            rejected_id = rejected.id
        rejected_result = service.reject_candidate(
            rejected_id, reason="operator rejected", actor="integration-test"
        )
        assert rejected_result.status == "cancelled"
        assert rejected_result.aliases["champion"] == "1"
        service.reject_candidate(rejected_id, reason="operator rejected", actor="integration-test")
        with session_scope(factory) as session:
            assert len(LifecycleEventRepository(session).for_run(rejected_id)) == 1

        # The MLflow alias moved before a crash, but the database phase did not. The
        # retry reconciles and finishes the original run instead of creating another.
        with session_scope(factory) as session:
            rollback = PipelineRunRepository(session).create(
                NewPipelineRun(
                    run_type=PipelineRunType.PROMOTION,
                    status=PipelineRunStatus.RUNNING,
                    started_at=datetime.now(UTC),
                    trigger="manual_rollback",
                    idempotency_key="d" * 64,
                    phase="created",
                    model_version="2",
                    metadata={"champion_version": "2", "rollback_target": "1"},
                )
            )
            rollback_id = rollback.id
        rolled_back = service.rollback("1", actor="integration-test")
        assert rolled_back.run_id == rollback_id
        assert rolled_back.status == "succeeded"
        assert rolled_back.phase == "rolled_back"
        assert service.rollback("1", actor="integration-test").run_id == rollback_id
        with session_scope(factory) as session:
            assert len(LifecycleEventRepository(session).for_run(rollback_id)) == 1
    finally:
        service.close()
        transaction.rollback()
        connection.close()
