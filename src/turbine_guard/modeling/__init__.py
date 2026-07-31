"""Offline model training, evaluation, and policy simulation.

This package consumes the checksummed feature layer and produces local, reproducible artifacts.
Optional MLflow tracking consumes its completed outputs; serving, replay infrastructure, and
monitoring consume them separately.
"""

from turbine_guard.modeling.config import TrainingConfig
from turbine_guard.modeling.pipeline import TrainingResult, train_models

__all__ = ["TrainingConfig", "TrainingResult", "train_models"]
