"""Reusable normalized-cost maintenance-policy simulation."""

from typing import Any

import numpy as np
import pandas as pd

from turbine_guard.data.schema import ASSET_ID_COLUMN, CYCLE_COLUMN
from turbine_guard.modeling.config import AlertConfig, SensitivityScenario


def simulate_maintenance_policies(
    frame: pd.DataFrame,
    alerts: AlertConfig,
    scenarios: tuple[SensitivityScenario, ...],
) -> dict[str, Any]:
    """Compare failure-only reactive and first-critical-alert predictive policies."""
    required = {ASSET_ID_COLUMN, CYCLE_COLUMN, "y_true_uncapped", "y_pred"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Simulation frame is missing columns: {sorted(missing)}.")
    if not scenarios:
        raise ValueError("At least one maintenance sensitivity scenario is required.")

    ordered = frame.sort_values([ASSET_ID_COLUMN, CYCLE_COLUMN], kind="stable")
    results = [_simulate_scenario(ordered, alerts, scenario) for scenario in scenarios]
    return {
        "label": "simulated_normalized_costs",
        "assumptions": {
            "reactive_policy": "maintenance occurs at failure",
            "predictive_policy": (
                "maintenance occurs at the first predicted-RUL critical alert; alerts with less "
                "than the minimum lead are treated as missed failures"
            ),
            "critical_horizon": alerts.critical_horizon,
            "minimum_lead_cycles": alerts.minimum_lead_cycles,
            "currency": None,
            "cost_units": "normalized, hypothetical, and not client-specific",
        },
        "scenarios": results,
    }


def _simulate_scenario(
    frame: pd.DataFrame, alerts: AlertConfig, scenario: SensitivityScenario
) -> dict[str, Any]:
    costs = scenario.costs
    reactive_assets: list[dict[str, Any]] = []
    predictive_assets: list[dict[str, Any]] = []
    for asset_id, group in frame.groupby(ASSET_ID_COLUMN, sort=True):
        reactive_cost = costs.unplanned_failure + costs.downtime_per_failure
        reactive_assets.append(
            {
                "asset_id": int(str(asset_id)),
                "normalized_cost": reactive_cost,
                "planned_intervention": False,
                "unplanned_failure": True,
                "missed_failure": False,
                "intervention_lead_time": None,
                "useful_life_forfeited": 0.0,
            }
        )

        alerted = group[group["y_pred"] <= alerts.critical_horizon]
        lead = None if alerted.empty else float(alerted.iloc[0]["y_true_uncapped"])
        usable = lead is not None and lead >= alerts.minimum_lead_cycles
        if usable:
            assert lead is not None
            predictive_cost = (
                costs.planned_inspection
                + costs.planned_repair
                + costs.early_replacement_per_cycle * lead
            )
            record: dict[str, Any] = {
                "asset_id": int(str(asset_id)),
                "normalized_cost": predictive_cost,
                "planned_intervention": True,
                "unplanned_failure": False,
                "missed_failure": False,
                "intervention_lead_time": lead,
                "useful_life_forfeited": lead,
            }
        else:
            predictive_cost = (
                costs.unplanned_failure + costs.downtime_per_failure + costs.missed_failure
            )
            record = {
                "asset_id": int(str(asset_id)),
                "normalized_cost": predictive_cost,
                "planned_intervention": False,
                "unplanned_failure": True,
                "missed_failure": True,
                "intervention_lead_time": lead,
                "useful_life_forfeited": 0.0,
            }
        predictive_assets.append(record)

    reactive = _aggregate_policy(reactive_assets)
    predictive = _aggregate_policy(predictive_assets)
    reactive_total = float(reactive["total_normalized_cost"])
    predictive["relative_cost_change_vs_reactive"] = (
        (float(predictive["total_normalized_cost"]) - reactive_total) / reactive_total
        if reactive_total
        else None
    )
    return {
        "name": scenario.name,
        "costs": {
            "unplanned_failure": costs.unplanned_failure,
            "planned_inspection": costs.planned_inspection,
            "planned_repair": costs.planned_repair,
            "downtime_per_failure": costs.downtime_per_failure,
            "early_replacement_per_cycle": costs.early_replacement_per_cycle,
            "missed_failure": costs.missed_failure,
        },
        "reactive": reactive,
        "predictive": predictive,
    }


def _aggregate_policy(records: list[dict[str, Any]]) -> dict[str, Any]:
    leads = [
        float(record["intervention_lead_time"])
        for record in records
        if record["intervention_lead_time"] is not None
    ]
    total = sum(float(record["normalized_cost"]) for record in records)
    return {
        "total_normalized_cost": total,
        "cost_per_asset": total / len(records) if records else 0.0,
        "planned_interventions": sum(bool(record["planned_intervention"]) for record in records),
        "unplanned_failures": sum(bool(record["unplanned_failure"]) for record in records),
        "missed_failures": sum(bool(record["missed_failure"]) for record in records),
        "mean_intervention_lead_time": float(np.mean(leads)) if leads else None,
        "useful_life_forfeited": sum(float(record["useful_life_forfeited"]) for record in records),
        "per_asset": records,
    }
