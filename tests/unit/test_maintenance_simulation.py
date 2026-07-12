"""Hand-calculated reactive/predictive normalized-cost simulation tests."""

import pandas as pd
import pytest

from turbine_guard.modeling.config import (
    AlertConfig,
    MaintenanceCosts,
    SensitivityScenario,
)
from turbine_guard.modeling.simulation import simulate_maintenance_policies


def simulation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asset_id": [1, 1, 1, 2, 2, 2],
            "cycle": [1, 2, 3, 1, 2, 3],
            "y_true_uncapped": [2, 1, 0, 2, 1, 0],
            # Asset 1 triggers at lead 2. Asset 2 never triggers.
            "y_pred": [1, 1, 0, 5, 4, 3],
        }
    )


def test_reactive_predictive_costs_and_first_alert_logic() -> None:
    scenario = SensitivityScenario("base", MaintenanceCosts())
    report = simulate_maintenance_policies(
        simulation_frame(),
        AlertConfig(critical_horizon=1, warning_horizon=2),
        (scenario,),
    )["scenarios"][0]

    assert report["reactive"]["total_normalized_cost"] == pytest.approx(24.0)
    # Asset 1: 0.5 + 3 + 0.03*2 = 3.56. Asset 2 missed: 10 + 2 + 4 = 16.
    assert report["predictive"]["total_normalized_cost"] == pytest.approx(19.56)
    assert report["predictive"]["planned_interventions"] == 1
    assert report["predictive"]["unplanned_failures"] == 1
    assert report["predictive"]["missed_failures"] == 1
    assert report["predictive"]["useful_life_forfeited"] == pytest.approx(2.0)
    assert report["predictive"]["mean_intervention_lead_time"] == pytest.approx(2.0)


def test_early_replacement_and_missed_penalties_are_configurable() -> None:
    low = SensitivityScenario(
        "low",
        MaintenanceCosts(early_replacement_per_cycle=0.0, missed_failure=0.0),
    )
    high = SensitivityScenario(
        "high",
        MaintenanceCosts(early_replacement_per_cycle=1.0, missed_failure=10.0),
    )
    report = simulate_maintenance_policies(
        simulation_frame(),
        AlertConfig(critical_horizon=1, warning_horizon=2),
        (low, high),
    )
    low_cost = report["scenarios"][0]["predictive"]["total_normalized_cost"]
    high_cost = report["scenarios"][1]["predictive"]["total_normalized_cost"]
    assert high_cost > low_cost
    assert report["label"] == "simulated_normalized_costs"
