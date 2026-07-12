"""Loop 4 offline model training, evaluation, and policy simulation.

This package consumes the checksummed Loop 3 feature layer and deliberately
stops at local, reproducible artifacts. Experiment tracking, registration,
serving, replay infrastructure, and monitoring belong to later loops.
"""

from turbine_guard.modeling.config import TrainingConfig
from turbine_guard.modeling.pipeline import TrainingResult, train_models

__all__ = ["TrainingConfig", "TrainingResult", "train_models"]
