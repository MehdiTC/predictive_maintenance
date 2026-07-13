"""Command-line interface for held-out sensor replay and delayed feedback.

The thin ``scripts/replay_sensor_data.py`` wrapper calls :func:`main`; all
behavior lives here and in :mod:`turbine_guard.replay` so it stays testable.
Summaries go to stdout; structured diagnostics go to the JSON logger.
"""

import argparse
import json
import logging
import sys
import uuid
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

from turbine_guard.config.settings import Settings, get_settings
from turbine_guard.database.enums import ReplayMode
from turbine_guard.database.session import (
    DatabaseConfig,
    create_database_engine,
    create_session_factory,
)
from turbine_guard.logging_config import configure_logging
from turbine_guard.replay.client import ReplayClientConfig, ReplayIngestionClient
from turbine_guard.replay.engine import (
    ReplayEngineConfig,
    ReplayOrchestrator,
    ReplayStatusReport,
)
from turbine_guard.replay.errors import ReplayError
from turbine_guard.replay.source import ReplaySource, ReplaySourceConfig
from turbine_guard.replay.state import PostgresReplayStateStore, ReplayRunState

logger = logging.getLogger(__name__)


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    """Build the replay CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="replay_sensor_data",
        description=(
            "Replay held-out FD001 trajectories one cycle at a time through the "
            "running TurbineGuard API, then emit the delayed failure outcome, "
            "backfill realized RUL labels, and evaluate historical predictions."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=settings.data_dir,
        help="Base data directory holding the verified Loop 2/3 layers (default: %(default)s)",
    )
    parser.add_argument(
        "--api-base-url",
        default=settings.replay_api_base_url,
        help="Base URL of the running inference API (default: %(default)s)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    start = commands.add_parser("start", help="Start (or idempotently resume) a replay run.")
    selection = start.add_mutually_exclusive_group(required=True)
    selection.add_argument("--asset-id", type=int, help="Source asset ID from the replay split.")
    selection.add_argument(
        "--all", action="store_true", help="Replay every replay-split asset in order."
    )
    start.add_argument(
        "--mode",
        choices=[mode.value for mode in ReplayMode],
        default=ReplayMode.ACCELERATED.value,
        help="step advances only on demand; continuous waits --delay between "
        "cycles; accelerated streams without waiting (default: %(default)s)",
    )
    start.add_argument(
        "--delay",
        type=float,
        default=settings.replay_cycle_delay_seconds,
        help="Seconds between cycles in continuous mode (default: %(default)s)",
    )
    start.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Advance at most this many cycles before returning.",
    )
    start.add_argument(
        "--force-restart",
        action="store_true",
        help="Cancel an incomplete run (or supersede a completed one) and begin a "
        "new attempt with a fresh operational asset. Existing operational "
        "history is never deleted.",
    )
    start.add_argument("--json", action="store_true", help="Emit the run state as JSON.")

    step = commands.add_parser("step", help="Advance a run by exactly one cycle.")
    step.add_argument("--run-id", type=uuid.UUID, required=True)
    step.add_argument("--json", action="store_true", help="Emit the run state as JSON.")

    resume = commands.add_parser("resume", help="Resume a paused, failed, or interrupted run.")
    resume.add_argument("--run-id", type=uuid.UUID, required=True)
    resume.add_argument("--max-cycles", type=int, default=None)
    resume.add_argument("--json", action="store_true", help="Emit the run state as JSON.")

    status = commands.add_parser("status", help="Show durable run state and evaluations.")
    status_selection = status.add_mutually_exclusive_group(required=True)
    status_selection.add_argument("--run-id", type=uuid.UUID)
    status_selection.add_argument("--all", action="store_true")
    status.add_argument("--json", action="store_true", help="Emit the report as JSON.")

    stop = commands.add_parser("stop", help="Stop a run after its current cycle.")
    stop.add_argument("--run-id", type=uuid.UUID, required=True)
    stop.add_argument("--json", action="store_true", help="Emit the run state as JSON.")

    aggregate = commands.add_parser(
        "evaluate-aggregate",
        help="Persist aggregate delayed evaluation across all completed runs.",
    )
    aggregate.add_argument("--json", action="store_true", help="Emit results as JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one replay CLI command; return a process exit code."""
    settings = get_settings()
    configure_logging(settings.log_level)
    args = build_parser(settings).parse_args(argv)
    engine = create_database_engine(DatabaseConfig.from_settings(settings))
    http = httpx.Client(
        base_url=str(args.api_base_url),
        timeout=settings.replay_http_timeout_seconds,
    )
    try:
        orchestrator = _build_orchestrator(settings, args, engine, http)
        return _dispatch(orchestrator, args)
    except ReplayError as exc:
        logger.error("replay_command_failed", extra={"error": str(exc)})
        _write(f"error: {exc}\n")
        return 1
    finally:
        http.close()
        engine.dispose()


def _build_orchestrator(
    settings: Settings, args: argparse.Namespace, engine: Any, http: httpx.Client
) -> ReplayOrchestrator:
    store = PostgresReplayStateStore(create_session_factory(engine))
    source = ReplaySource(ReplaySourceConfig(data_dir=args.data_dir))
    client = ReplayIngestionClient(
        http,
        ReplayClientConfig(
            max_attempts=settings.replay_max_send_attempts,
            backoff_seconds=settings.replay_retry_backoff_seconds,
        ),
    )
    config = ReplayEngineConfig(
        lease_seconds=settings.replay_lease_seconds,
        default_cycle_delay_seconds=settings.replay_cycle_delay_seconds,
        simulated_cycle_duration_seconds=settings.replay_simulated_cycle_duration_seconds,
    )
    return ReplayOrchestrator(store, source, client, config)


def _dispatch(orchestrator: ReplayOrchestrator, args: argparse.Namespace) -> int:
    if args.command == "start":
        return _command_start(orchestrator, args)
    if args.command == "step":
        state = orchestrator.step(args.run_id)
        _emit_run(state, as_json=args.json)
        return 0
    if args.command == "resume":
        state = orchestrator.resume(args.run_id, max_cycles=args.max_cycles)
        _emit_run(state, as_json=args.json)
        return 0
    if args.command == "status":
        if args.run_id is not None:
            _emit_report(orchestrator.status(args.run_id), as_json=args.json)
        else:
            reports = orchestrator.list_status()
            if args.json:
                _write(json.dumps([_report_dict(report) for report in reports], indent=2) + "\n")
            else:
                for report in reports:
                    _emit_report(report, as_json=False)
                if not reports:
                    _write("no replay runs exist\n")
        return 0
    if args.command == "stop":
        state = orchestrator.stop(args.run_id)
        _emit_run(state, as_json=args.json)
        return 0
    if args.command == "evaluate-aggregate":
        evaluations = orchestrator.evaluate_aggregate()
        if args.json:
            _write(
                json.dumps([_evaluation_dict(evaluation) for evaluation in evaluations], indent=2)
                + "\n"
            )
        else:
            for evaluation in evaluations:
                _write(
                    f"aggregate {evaluation.model_name} v{evaluation.model_version}: "
                    f"samples={evaluation.sample_count} mae={_round(evaluation.mae)} "
                    f"rmse={_round(evaluation.rmse)} "
                    f"critical_recall={_round(evaluation.critical_recall)} "
                    f"coverage={_round(evaluation.interval_coverage)}\n"
                )
        return 0
    raise ReplayError(f"Unknown command {args.command!r}.")  # pragma: no cover


def _command_start(orchestrator: ReplayOrchestrator, args: argparse.Namespace) -> int:
    mode = ReplayMode(args.mode)
    asset_ids = orchestrator.replay_asset_ids() if args.all else (int(args.asset_id),)
    final_states: list[ReplayRunState] = []
    for asset_id in asset_ids:
        state = orchestrator.start(
            asset_id,
            mode=mode,
            cycle_delay_seconds=args.delay,
            force_restart=args.force_restart,
        )
        if mode is not ReplayMode.STEP and state.status.value not in ("completed",):
            state = orchestrator.drive(state.run_id, max_cycles=args.max_cycles)
        final_states.append(state)
    if args.json:
        _write(json.dumps([_run_dict(state) for state in final_states], indent=2) + "\n")
    else:
        for state in final_states:
            _emit_run(state, as_json=False)
    return 0


def _emit_run(state: ReplayRunState, *, as_json: bool) -> None:
    if as_json:
        _write(json.dumps(_run_dict(state), indent=2) + "\n")
    else:
        _write(_run_line(state))


def _emit_report(report: ReplayStatusReport, *, as_json: bool) -> None:
    if as_json:
        _write(json.dumps(_report_dict(report), indent=2) + "\n")
        return
    _write(_run_line(report.run))
    for evaluation in report.evaluations:
        _write(
            f"  evaluation {evaluation.model_name} v{evaluation.model_version} "
            f"[{evaluation.metrics.get('aggregation', 'unknown')}]: "
            f"samples={evaluation.sample_count} mae={_round(evaluation.mae)} "
            f"rmse={_round(evaluation.rmse)} nasa={_round(evaluation.nasa_score)} "
            f"critical_recall={_round(evaluation.critical_recall)} "
            f"coverage={_round(evaluation.interval_coverage)}\n"
        )


def _run_line(state: ReplayRunState) -> str:
    phases = (
        f"failure_event={'yes' if state.failure_event_id else 'no'} "
        f"labels={'yes' if state.labels_backfilled_at else 'no'} "
        f"evaluation={'yes' if state.evaluation_completed_at else 'no'}"
    )
    error = f" error={state.error_message!r}" if state.error_message else ""
    return (
        f"run {state.run_id} {state.dataset_subset} asset {state.source_asset_id} "
        f"attempt {state.attempt} [{state.mode.value}] status={state.status.value} "
        f"cycle {state.last_confirmed_cycle}/{state.final_cycle} {phases}{error}\n"
    )


def _run_dict(state: ReplayRunState) -> dict[str, Any]:
    payload = asdict(state)
    return {key: _jsonable(value) for key, value in payload.items()}


def _report_dict(report: ReplayStatusReport) -> dict[str, Any]:
    return {
        "run": _run_dict(report.run),
        "evaluations": [_evaluation_dict(evaluation) for evaluation in report.evaluations],
    }


def _evaluation_dict(evaluation: Any) -> dict[str, Any]:
    return {key: _jsonable(value) for key, value in asdict(evaluation).items()}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    if hasattr(value, "value") and not isinstance(value, (int, float, str, bool)):
        return value.value
    return value


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


def _write(text: str) -> None:
    sys.stdout.write(text)
