"""Project module for end-to-end clinical test workflows."""

from classiflow.projects.project_models import ProjectConfig, ThresholdsConfig
from classiflow.projects.project_fs import ProjectPaths
from classiflow.projects.dataset_registry import register_dataset
from classiflow.projects.orchestrator import (
    run_feasibility,
    run_technical_validation,
    build_final_model,
    run_independent_test,
)

__all__ = [
    "ProjectConfig",
    "ThresholdsConfig",
    "ProjectPaths",
    "register_dataset",
    "run_feasibility",
    "run_technical_validation",
    "build_final_model",
    "run_independent_test",
]
