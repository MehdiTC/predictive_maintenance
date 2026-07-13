"""Concise management CLI for Loop 9 model-lifecycle operations."""

import argparse
import json
import sys
import uuid
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime
from typing import Any

from turbine_guard.config.settings import get_settings
from turbine_guard.logging_config import configure_logging
from turbine_guard.monitoring.service import LifecycleService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="model_lifecycle")
    commands = parser.add_subparsers(dest="command", required=True)
    monitor = commands.add_parser("monitor", help="Run and persist one monitoring window.")
    monitor.add_argument("--window-start", type=_timestamp)
    monitor.add_argument("--window-end", type=_timestamp)

    status = commands.add_parser("status", help="Inspect durable lifecycle status.")
    status.add_argument("--run-id", type=uuid.UUID)
    status.add_argument("--limit", type=int, default=20)

    commands.add_parser(
        "force-retraining", help="Force the trigger signal while retaining safety blocks."
    )
    commands.add_parser(
        "evaluate-candidate", help="Run or resume trigger-driven candidate evaluation."
    )

    dry_run = commands.add_parser("dry-run-promotion", help="Show persisted gates and aliases.")
    dry_run.add_argument("--run-id", type=uuid.UUID, required=True)

    approve = commands.add_parser("approve-promotion", help="Approve a gate-passing candidate.")
    approve.add_argument("--run-id", type=uuid.UUID, required=True)
    approve.add_argument("--actor", default="manual_cli")

    reject = commands.add_parser("reject-candidate", help="Reject without moving champion.")
    reject.add_argument("--run-id", type=uuid.UUID, required=True)
    reject.add_argument("--reason", required=True)
    reject.add_argument("--actor", default="manual_cli")

    rollback = commands.add_parser("rollback", help="Restore a prior valid numbered version.")
    rollback.add_argument("--version", required=True)
    rollback.add_argument("--actor", default="manual_cli")

    commands.add_parser("refresh-serving-model", help="Safely reload the configured champion.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    args = build_parser().parse_args(argv)
    service = LifecycleService(settings)
    try:
        payload = _dispatch(service, args)
    except Exception as exc:
        sys.stderr.write(json.dumps({"status": "error", "error": str(exc)}) + "\n")
        return 1
    finally:
        service.close()
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return 0


def _dispatch(service: LifecycleService, args: argparse.Namespace) -> Any:
    if args.command == "monitor":
        result = service.run_monitoring(
            window_start=args.window_start,
            window_end=args.window_end,
        )
        return {
            **asdict(result),
            "decision": result.decision.record(),
        }
    if args.command == "status":
        return (
            asdict(service.get_lifecycle(args.run_id))
            if args.run_id is not None
            else service.recent_status(limit=args.limit)
        )
    if args.command == "force-retraining":
        return asdict(service.start_retraining(manual_force=True))
    if args.command == "evaluate-candidate":
        return asdict(service.start_retraining())
    if args.command == "dry-run-promotion":
        lifecycle = service.get_lifecycle(args.run_id)
        return {
            "run_id": str(lifecycle.run_id),
            "phase": lifecycle.phase,
            "candidate_version": lifecycle.candidate_version,
            "gates": lifecycle.gates,
            "aliases_unchanged": lifecycle.aliases,
        }
    if args.command == "approve-promotion":
        return asdict(service.approve_promotion(args.run_id, actor=args.actor))
    if args.command == "reject-candidate":
        return asdict(service.reject_candidate(args.run_id, reason=args.reason, actor=args.actor))
    if args.command == "rollback":
        return asdict(service.rollback(args.version, actor=args.actor))
    if args.command == "refresh-serving-model":
        return {"status": "refreshed", "champion_version": service.refresh_serving_model()}
    raise ValueError(f"Unknown lifecycle command {args.command!r}.")


def _timestamp(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamps must include a UTC offset")
    return parsed
