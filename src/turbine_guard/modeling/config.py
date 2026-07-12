"""Typed configuration for Loop 4 offline model training and evaluation."""

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

EVALUATION_VERSION = "1"


class ModelKind(StrEnum):
    """Supported point-model approaches."""

    CONSTANT = "constant_median"
    RIDGE = "ridge"
    HIST_GRADIENT_BOOSTING = "hist_gradient_boosting"
    XGBOOST = "xgboost"


@dataclass(frozen=True)
class TargetConfig:
    """One explicit RUL target definition."""

    name: str
    cap: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Target name must not be empty.")
        if self.cap is not None and self.cap <= 0:
            raise ValueError("Target cap must be positive when configured.")

    def apply(self, values: Any) -> Any:
        """Apply this target definition to an array-like object."""
        return values if self.cap is None else values.clip(upper=self.cap)


@dataclass(frozen=True)
class CandidateConfig:
    """One deliberately bounded model configuration."""

    name: str
    kind: ModelKind
    parameters: tuple[tuple[str, int | float | str | bool], ...] = ()
    complexity_rank: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Candidate name must not be empty.")
        if self.complexity_rank < 0:
            raise ValueError("Candidate complexity rank must be non-negative.")

    @property
    def params(self) -> dict[str, int | float | str | bool]:
        """Parameters as a fresh dictionary suitable for an estimator."""
        return dict(self.parameters)


@dataclass(frozen=True)
class AlertConfig:
    """Operational alert semantics shared by evaluation and simulation."""

    critical_horizon: int = 30
    warning_horizon: int = 50
    minimum_lead_cycles: int = 1

    def __post_init__(self) -> None:
        if self.critical_horizon <= 0:
            raise ValueError("Critical horizon must be positive.")
        if self.warning_horizon <= self.critical_horizon:
            raise ValueError("Warning horizon must exceed critical horizon.")
        if self.minimum_lead_cycles < 0:
            raise ValueError("Minimum lead cycles must be non-negative.")


@dataclass(frozen=True)
class SelectionConfig:
    """Validation-only champion eligibility and tie-breaking rules."""

    minimum_critical_recall: float = 0.60
    maximum_false_alarms_per_1000_cycles: float = 250.0
    relative_rmse_tolerance: float = 0.02

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_critical_recall <= 1.0:
            raise ValueError("Minimum critical recall must be in [0, 1].")
        if self.maximum_false_alarms_per_1000_cycles < 0:
            raise ValueError("Maximum false-alarm rate must be non-negative.")
        if not 0.0 <= self.relative_rmse_tolerance <= 1.0:
            raise ValueError("RMSE tolerance must be in [0, 1].")


@dataclass(frozen=True)
class MaintenanceCosts:
    """Normalized simulation cost units; never interpreted as currency."""

    unplanned_failure: float = 10.0
    planned_inspection: float = 0.5
    planned_repair: float = 3.0
    downtime_per_failure: float = 2.0
    early_replacement_per_cycle: float = 0.03
    missed_failure: float = 4.0

    def __post_init__(self) -> None:
        if any(value < 0 for value in asdict(self).values()):
            raise ValueError("Maintenance cost components must be non-negative.")


@dataclass(frozen=True)
class SensitivityScenario:
    """A named set of normalized maintenance costs."""

    name: str
    costs: MaintenanceCosts


def default_candidates() -> tuple[CandidateConfig, ...]:
    """Small manual candidate grid used identically for both target definitions."""
    return (
        CandidateConfig("constant", ModelKind.CONSTANT, complexity_rank=0),
        CandidateConfig(
            "ridge_alpha_1",
            ModelKind.RIDGE,
            (("alpha", 1.0),),
            complexity_rank=1,
        ),
        CandidateConfig(
            "ridge_alpha_10",
            ModelKind.RIDGE,
            (("alpha", 10.0),),
            complexity_rank=1,
        ),
        CandidateConfig(
            "histgb_small",
            ModelKind.HIST_GRADIENT_BOOSTING,
            (("learning_rate", 0.08), ("max_iter", 120), ("max_leaf_nodes", 15)),
            complexity_rank=2,
        ),
        CandidateConfig(
            "histgb_medium",
            ModelKind.HIST_GRADIENT_BOOSTING,
            (("learning_rate", 0.05), ("max_iter", 180), ("max_leaf_nodes", 31)),
            complexity_rank=3,
        ),
        CandidateConfig(
            "xgboost_small",
            ModelKind.XGBOOST,
            (
                ("learning_rate", 0.06),
                ("max_depth", 3),
                ("n_estimators", 180),
                ("subsample", 0.9),
                ("colsample_bytree", 0.8),
            ),
            complexity_rank=3,
        ),
        CandidateConfig(
            "xgboost_medium",
            ModelKind.XGBOOST,
            (
                ("learning_rate", 0.04),
                ("max_depth", 5),
                ("n_estimators", 260),
                ("subsample", 0.9),
                ("colsample_bytree", 0.8),
            ),
            complexity_rank=4,
        ),
    )


def default_sensitivity_scenarios() -> tuple[SensitivityScenario, ...]:
    """Small, explicit cost sensitivity set around the normalized base case."""
    return (
        SensitivityScenario("base", MaintenanceCosts()),
        SensitivityScenario(
            "lower_failure_cost",
            MaintenanceCosts(unplanned_failure=6.0, missed_failure=2.0),
        ),
        SensitivityScenario(
            "higher_failure_cost",
            MaintenanceCosts(unplanned_failure=16.0, missed_failure=8.0),
        ),
    )


@dataclass(frozen=True)
class TrainingConfig:
    """Aggregate configuration for one deterministic Loop 4 training run."""

    data_dir: Path = Path("data")
    subset: str = "FD001"
    output_dir: Path | None = None
    random_seed: int = 42
    evaluation_version: str = EVALUATION_VERSION
    targets: tuple[TargetConfig, ...] = (
        TargetConfig("uncapped"),
        TargetConfig("capped_125", cap=125),
    )
    candidates: tuple[CandidateConfig, ...] = field(default_factory=default_candidates)
    imputation_strategy: str = "median_with_indicators"
    scale_ridge: bool = True
    alerts: AlertConfig = field(default_factory=AlertConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    conformal_coverage: float = 0.90
    maintenance_scenarios: tuple[SensitivityScenario, ...] = field(
        default_factory=default_sensitivity_scenarios
    )
    latency_repeats: int = 5
    force: bool = False

    def __post_init__(self) -> None:
        if not self.subset:
            raise ValueError("Dataset subset must not be empty.")
        if not self.targets:
            raise ValueError("At least one target definition is required.")
        if not any(target.cap is None for target in self.targets):
            raise ValueError("An uncapped target experiment is required.")
        if not any(target.cap is not None for target in self.targets):
            raise ValueError("At least one capped target experiment is required.")
        if not self.candidates:
            raise ValueError("At least one candidate configuration is required.")
        required = set(ModelKind)
        present = {candidate.kind for candidate in self.candidates}
        if not required <= present:
            missing = sorted(kind.value for kind in required - present)
            raise ValueError(f"Missing required model approaches: {missing}.")
        if not 0.0 < self.conformal_coverage < 1.0:
            raise ValueError("Conformal coverage must be strictly between 0 and 1.")
        if self.latency_repeats < 1:
            raise ValueError("Latency repeats must be positive.")

    @property
    def features_dir(self) -> Path:
        """Loop 3 input directory."""
        return self.data_dir / "features" / "cmapss" / self.subset

    @property
    def artifacts_dir(self) -> Path:
        """Loop 4 output directory."""
        return self.output_dir or self.data_dir / "models" / "cmapss" / self.subset


def config_record(config: TrainingConfig) -> dict[str, Any]:
    """JSON-safe configuration record with paths kept repository-relative as configured."""
    record = asdict(config)
    record["data_dir"] = str(config.data_dir)
    record["output_dir"] = str(config.output_dir) if config.output_dir is not None else None
    record["force"] = False  # execution control is not part of run identity
    return record
