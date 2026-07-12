"""Loop 4 offline model training, evaluation, and policy simulation.

This package consumes the checksummed Loop 3 feature layer and deliberately
stops at local, reproducible artifacts. Optional Loop 5 tracking consumes its
completed outputs; serving, replay infrastructure, and monitoring remain later loops.
"""

from turbine_guard.modeling.config import TrainingConfig
from turbine_guard.modeling.pipeline import TrainingResult, train_models

__all__ = ["TrainingConfig", "TrainingResult", "train_models"]
