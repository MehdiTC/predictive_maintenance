"""Machine-readable and concise human-readable Loop 4 reports."""

import csv
from pathlib import Path
from typing import Any

from turbine_guard.modeling.artifacts import write_json, write_text


def write_candidate_comparison(path: Path, candidates: list[dict[str, Any]]) -> None:
    """Write a flat candidate comparison CSV."""
    fields = [
        "candidate_id",
        "model_kind",
        "target_name",
        "target_cap",
        "validation_mae",
        "validation_rmse",
        "validation_nasa_score",
        "common_domain_mae",
        "common_domain_rmse",
        "common_domain_nasa_score",
        "critical_precision",
        "critical_recall",
        "critical_f1",
        "false_alarms_per_1000_cycles",
        "mean_alert_lead_time",
        "training_seconds",
        "prediction_latency_ms",
        "model_size_bytes",
        "complexity_rank",
        "preprocessing_policy",
    ]
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        regression = candidate["regression_metrics"]["row_weighted"]
        common = candidate["common_domain_metrics"]
        critical = candidate["critical_alert_metrics"]
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "model_kind": candidate["model_kind"],
                "target_name": candidate["target_definition"]["name"],
                "target_cap": candidate["target_definition"]["cap"],
                "validation_mae": regression["mae"],
                "validation_rmse": regression["rmse"],
                "validation_nasa_score": regression["nasa_score"],
                "common_domain_mae": common["mae"],
                "common_domain_rmse": common["rmse"],
                "common_domain_nasa_score": common["nasa_score"],
                "critical_precision": critical["precision"],
                "critical_recall": critical["recall"],
                "critical_f1": critical["f1"],
                "false_alarms_per_1000_cycles": critical["false_alarms_per_1000_cycles"],
                "mean_alert_lead_time": critical["mean_first_alert_lead_time"],
                "training_seconds": candidate["training_seconds"],
                "prediction_latency_ms": candidate["prediction_latency_ms"],
                "model_size_bytes": candidate["model_size_bytes"],
                "complexity_rank": candidate["complexity_rank"],
                "preprocessing_policy": candidate["preprocessing_policy"],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def write_human_report(
    path: Path,
    *,
    selection: dict[str, Any],
    replay: dict[str, Any],
    official: dict[str, Any],
    conformal: dict[str, Any],
    simulation: dict[str, Any],
) -> None:
    """Write the focused Markdown summary; detailed records remain in JSON/CSV."""
    selected = selection["selected_model"]
    replay_metrics = replay["regression_metrics"]["row_weighted"]
    official_metrics = official["regression_metrics"]["row_weighted"]
    base_sim = simulation["scenarios"][0]
    predictive = base_sim["predictive"]
    lines = [
        "# Loop 4 Model Evaluation",
        "",
        "All candidate selection used the validation role only. Calibration fitted only the "
        "conformal residual quantile; replay and official NASA test results did not affect "
        "selection.",
        "",
        "## Champion",
        "",
        f"- Selected candidate: `{selected}`",
        f"- Rationale: {selection['rationale']}",
        "",
        "## Held-out results",
        "",
        f"- Replay MAE / RMSE / NASA: {replay_metrics['mae']:.4f} / "
        f"{replay_metrics['rmse']:.4f} / {replay_metrics['nasa_score']:.4f}",
        f"- Official final-row MAE / RMSE / NASA: {official_metrics['mae']:.4f} / "
        f"{official_metrics['rmse']:.4f} / {official_metrics['nasa_score']:.4f}",
        f"- Replay conformal coverage / average width: "
        f"{conformal['replay']['empirical_coverage']:.3f} / "
        f"{conformal['replay']['average_width']:.3f}",
        "",
        "## Simulated maintenance policy",
        "",
        "These are normalized hypothetical cost units, not currency or claimed industrial savings.",
        "",
        f"- Base reactive total cost: {base_sim['reactive']['total_normalized_cost']:.3f}",
        f"- Base predictive total cost: {predictive['total_normalized_cost']:.3f}",
        f"- Relative change: {predictive['relative_cost_change_vs_reactive']:.2%}",
        f"- Planned interventions / unplanned failures / missed failures: "
        f"{predictive['planned_interventions']} / {predictive['unplanned_failures']} / "
        f"{predictive['missed_failures']}",
        "",
        "## Limitations",
        "",
        "- FD001 is simulated and has anonymous sensors.",
        "- Conformal calibration uses temporally dependent rows, so formal exchangeability "
        "guarantees are approximate.",
        "- Built-in coefficients/importances are associative diagnostics, not causal effects.",
        "- Policy costs are a sensitivity exercise using normalized assumptions.",
        "- Joblib artifacts must only be loaded from trusted, checksum-verified sources.",
        "",
    ]
    write_text(path, "\n".join(lines))


def write_report_set(root: Path, reports: dict[str, Any]) -> list[Path]:
    """Write the standard JSON report set and return all paths."""
    paths: list[Path] = []
    for name, payload in reports.items():
        path = root / f"{name}.json"
        write_json(path, payload)
        paths.append(path)
    return paths
