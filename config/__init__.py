"""
Configuration Management for SparseFusion

Centralized configuration for all hyperparameters and settings.

Quick start:
    from config.presets import load_preset
    config = load_preset("multi_task")
"""

from .model_config import ModelConfig
from .training_config import TrainingConfig
from .evaluation_config import EvaluationConfig
from .sparsity_config import SparsityConfig
from .task_config import TaskConfig
from .presets import load_preset, print_config, PRESETS

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "SparsityConfig",
    "TaskConfig",
    "load_preset",
    "print_config",
    "PRESETS",
]



Centralized configuration for all hyperparameters and settings.

Quick start:
    from config.presets import load_preset
    config = load_preset("multi_task")
"""

from .model_config import ModelConfig
from .training_config import TrainingConfig
from .evaluation_config import EvaluationConfig
from .sparsity_config import SparsityConfig
from .task_config import TaskConfig
from .presets import load_preset, print_config, PRESETS

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "SparsityConfig",
    "TaskConfig",
    "load_preset",
    "print_config",
    "PRESETS",
]

