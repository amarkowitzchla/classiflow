"""
classiflow: Production-grade ML toolkit for molecular subtype classification.

This package provides:
- Nested cross-validation training for binary and multiclass tasks
- Adaptive SMOTE handling for imbalanced datasets
- Meta-classifier training on binary task scores
- Comprehensive metrics and visualization
- CLI tools and Streamlit UI
"""

__version__ = "0.1.0"

from classiflow.training.binary import train_binary_task
from classiflow.training.meta import train_meta_classifier
from classiflow.tasks.builder import TaskBuilder
from classiflow.config import TrainConfig, MetaConfig

__all__ = [
    "__version__",
    "train_binary_task",
    "train_meta_classifier",
    "TaskBuilder",
    "TrainConfig",
    "MetaConfig",
]
