"""Typed, configurable Loop 9 monitoring and lifecycle thresholds."""

from dataclasses import dataclass

from turbine_guard.config.settings import Settings


@dataclass(frozen=True)
class DriftThresholds:
    minimum_rows: int
    minimum_non_null: int
    psi_warning: float
    psi_detected: float
    normalized_wasserstein: float
    missingness_shift: float
    mean_shift: float
    std_shift: float


@dataclass(frozen=True)
class TriggerThresholds:
    minimum_assets: int
    minimum_rows: int
    minimum_holdout_assets: int
    drifted_feature_count: int
    performance_degradation: float
    minimum_critical_recall: float
    false_alarm_increase_tolerance: float
    minimum_coverage: float
    interval_days: int


@dataclass(frozen=True)
class PromotionThresholds:
    rmse_relative_tolerance: float
    nasa_relative_tolerance: float
    minimum_critical_recall: float
    maximum_false_alarms_per_1000: float
    minimum_coverage: float
    maximum_latency_ms: float
    maximum_artifact_size_bytes: int
    equivalence_tolerance: float


@dataclass(frozen=True)
class LifecycleConfig:
    window_days: int
    minimum_quality_rows: int
    minimum_quality_assets: int
    out_of_range_stddevs: float
    holdout_fraction: float
    approval_required: bool
    drift: DriftThresholds
    trigger: TriggerThresholds
    promotion: PromotionThresholds

    @classmethod
    def from_settings(cls, settings: Settings) -> "LifecycleConfig":
        """Build lifecycle configuration from the application's typed settings."""
        return cls(
            window_days=settings.monitoring_window_days,
            minimum_quality_rows=settings.monitoring_min_rows,
            minimum_quality_assets=settings.monitoring_min_assets,
            out_of_range_stddevs=settings.monitoring_out_of_range_stddevs,
            holdout_fraction=settings.retraining_holdout_fraction,
            approval_required=settings.promotion_approval_required,
            drift=DriftThresholds(
                minimum_rows=settings.monitoring_min_rows,
                minimum_non_null=settings.monitoring_min_feature_non_null,
                psi_warning=settings.monitoring_psi_warning,
                psi_detected=settings.monitoring_psi_detected,
                normalized_wasserstein=(settings.monitoring_wasserstein_normalized_threshold),
                missingness_shift=settings.monitoring_missingness_shift_threshold,
                mean_shift=settings.monitoring_mean_shift_threshold,
                std_shift=settings.monitoring_std_shift_threshold,
            ),
            trigger=TriggerThresholds(
                minimum_assets=settings.retraining_min_new_assets,
                minimum_rows=settings.retraining_min_new_rows,
                minimum_holdout_assets=settings.retraining_min_holdout_assets,
                drifted_feature_count=settings.monitoring_drifted_feature_trigger_count,
                performance_degradation=settings.retraining_performance_degradation,
                minimum_critical_recall=settings.retraining_minimum_critical_recall,
                false_alarm_increase_tolerance=(settings.retraining_false_alarm_increase_tolerance),
                minimum_coverage=settings.retraining_minimum_coverage,
                interval_days=settings.retraining_interval_days,
            ),
            promotion=PromotionThresholds(
                rmse_relative_tolerance=settings.promotion_rmse_relative_tolerance,
                nasa_relative_tolerance=settings.promotion_nasa_relative_tolerance,
                minimum_critical_recall=settings.promotion_minimum_critical_recall,
                maximum_false_alarms_per_1000=(settings.promotion_maximum_false_alarms_per_1000),
                minimum_coverage=settings.promotion_minimum_coverage,
                maximum_latency_ms=settings.promotion_maximum_latency_ms,
                maximum_artifact_size_bytes=(settings.promotion_maximum_artifact_size_bytes),
                equivalence_tolerance=settings.promotion_equivalence_tolerance,
            ),
        )
