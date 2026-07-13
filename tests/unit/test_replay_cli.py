"""Replay CLI: dispatch, output, JSON mode, and failure exit codes."""

import json
import uuid
from datetime import UTC, datetime
from typing import Any

import pytest

from turbine_guard.database.enums import ReplayMode, ReplayRunStatus
from turbine_guard.replay.cli import main
from turbine_guard.replay.engine import ReplayStatusReport
from turbine_guard.replay.errors import ReplayStateError
from turbine_guard.replay.state import ReplayRunState, StoredEvaluation

NOW = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)
RUN_ID = uuid.uuid4()


def _state(**overrides: Any) -> ReplayRunState:
    values: dict[str, Any] = {
        "run_id": RUN_ID,
        "dataset_name": "cmapss",
        "dataset_subset": "FD001",
        "source_asset_id": 9,
        "attempt": 1,
        "external_asset_id": "replay-FD001-009",
        "asset_id": uuid.uuid4(),
        "final_cycle": 6,
        "last_confirmed_cycle": 6,
        "status": ReplayRunStatus.COMPLETED,
        "mode": ReplayMode.ACCELERATED,
        "cycle_delay_seconds": 0.0,
        "simulated_cycle_duration_seconds": 1.0,
        "replay_started_at": NOW,
        "last_advanced_at": NOW,
        "completed_at": NOW,
        "failure_event_id": uuid.uuid4(),
        "labels_backfilled_at": NOW,
        "evaluation_completed_at": NOW,
        "error_message": None,
        "metadata": {},
    }
    values.update(overrides)
    return ReplayRunState(**values)


def _evaluation() -> StoredEvaluation:
    return StoredEvaluation(
        model_name="fake-rul",
        model_version="1",
        evaluation_scope="replay",
        sample_count=6,
        mae=0.4,
        rmse=0.63,
        nasa_score=0.18,
        critical_precision=1.0,
        critical_recall=1.0,
        interval_coverage=1.0,
        metrics={"aggregation": "replay_asset"},
        created_at=NOW,
    )


class StubOrchestrator:
    def __init__(self, state: ReplayRunState | None = None, fail: Exception | None = None):
        self.state = state or _state()
        self.fail = fail
        self.calls: list[tuple[str, Any]] = []

    def _maybe_fail(self) -> None:
        if self.fail is not None:
            raise self.fail

    def start(self, asset_id: int, **kwargs: Any) -> ReplayRunState:
        self._maybe_fail()
        self.calls.append(("start", (asset_id, kwargs)))
        return _state(status=ReplayRunStatus.CREATED, last_confirmed_cycle=0)

    def drive(self, run_id: uuid.UUID, **kwargs: Any) -> ReplayRunState:
        self.calls.append(("drive", (run_id, kwargs)))
        return self.state

    def step(self, run_id: uuid.UUID) -> ReplayRunState:
        self._maybe_fail()
        self.calls.append(("step", run_id))
        return self.state

    def resume(self, run_id: uuid.UUID, **kwargs: Any) -> ReplayRunState:
        self.calls.append(("resume", (run_id, kwargs)))
        return self.state

    def stop(self, run_id: uuid.UUID) -> ReplayRunState:
        self.calls.append(("stop", run_id))
        return self.state

    def status(self, run_id: uuid.UUID) -> ReplayStatusReport:
        self.calls.append(("status", run_id))
        return ReplayStatusReport(run=self.state, evaluations=(_evaluation(),))

    def list_status(self, **kwargs: Any) -> list[ReplayStatusReport]:
        self.calls.append(("list_status", kwargs))
        return []

    def replay_asset_ids(self) -> tuple[int, ...]:
        return (9, 13)

    def evaluate_aggregate(self) -> list[StoredEvaluation]:
        self._maybe_fail()
        self.calls.append(("evaluate_aggregate", None))
        return [_evaluation()]


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> StubOrchestrator:
    orchestrator = StubOrchestrator()
    monkeypatch.setattr(
        "turbine_guard.replay.cli._build_orchestrator",
        lambda *_args, **_kwargs: orchestrator,
    )
    return orchestrator


class TestStart:
    def test_start_single_asset_drives_non_step_modes(
        self, stub: StubOrchestrator, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["start", "--asset-id", "9"]) == 0
        names = [name for name, _ in stub.calls]
        assert names == ["start", "drive"]
        output = capsys.readouterr().out
        assert f"run {RUN_ID}" in output
        assert "status=completed" in output
        assert "cycle 6/6" in output

    def test_start_step_mode_does_not_drive(self, stub: StubOrchestrator) -> None:
        assert main(["start", "--asset-id", "9", "--mode", "step"]) == 0
        names = [name for name, _ in stub.calls]
        assert names == ["start"]

    def test_start_all_replays_every_replay_asset(self, stub: StubOrchestrator) -> None:
        assert main(["start", "--all"]) == 0
        started = [args[0] for name, args in stub.calls if name == "start"]
        assert started == [9, 13]

    def test_force_restart_flag_is_passed_through(self, stub: StubOrchestrator) -> None:
        assert main(["start", "--asset-id", "9", "--force-restart"]) == 0
        _, (_, kwargs) = stub.calls[0]
        assert kwargs["force_restart"] is True


class TestLifecycleCommands:
    def test_step_command(self, stub: StubOrchestrator) -> None:
        assert main(["step", "--run-id", str(RUN_ID)]) == 0
        assert stub.calls == [("step", RUN_ID)]

    @pytest.mark.usefixtures("stub")
    def test_resume_command_with_json_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["resume", "--run-id", str(RUN_ID), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["run_id"] == str(RUN_ID)
        assert payload["status"] == "completed"

    def test_stop_command(self, stub: StubOrchestrator) -> None:
        assert main(["stop", "--run-id", str(RUN_ID)]) == 0
        assert stub.calls == [("stop", RUN_ID)]

    @pytest.mark.usefixtures("stub")
    def test_status_includes_evaluations(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["status", "--run-id", str(RUN_ID)]) == 0
        output = capsys.readouterr().out
        assert "failure_event=yes labels=yes evaluation=yes" in output
        assert "evaluation fake-rul v1 [replay_asset]" in output

    @pytest.mark.usefixtures("stub")
    def test_status_all_with_no_runs(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["status", "--all"]) == 0
        assert "no replay runs exist" in capsys.readouterr().out

    @pytest.mark.usefixtures("stub")
    def test_evaluate_aggregate(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["evaluate-aggregate"]) == 0
        assert "aggregate fake-rul v1" in capsys.readouterr().out


class TestFailures:
    def test_replay_error_returns_nonzero_with_message(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        orchestrator = StubOrchestrator(fail=ReplayStateError("run is broken"))
        monkeypatch.setattr(
            "turbine_guard.replay.cli._build_orchestrator",
            lambda *_args, **_kwargs: orchestrator,
        )
        assert main(["step", "--run-id", str(RUN_ID)]) == 1
        assert "error: run is broken" in capsys.readouterr().out

    def test_missing_subcommand_is_a_usage_error(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main([])
        assert excinfo.value.code == 2

    def test_invalid_run_id_is_a_usage_error(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["step", "--run-id", "not-a-uuid"])
        assert excinfo.value.code == 2

    def test_start_requires_asset_selection(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["start"])
        assert excinfo.value.code == 2
