"""Inference pipeline for applying trained models to new data."""

from classiflow.inference.api import run_inference
from classiflow.inference.config import InferenceConfig, RunManifest

__all__ = [
    "run_inference",
    "InferenceConfig",
    "RunManifest",
    "HierarchicalInference",
]


def __getattr__(name: str):
    if name == "HierarchicalInference":
        from classiflow.inference.hierarchical import HierarchicalInference

        return HierarchicalInference
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
