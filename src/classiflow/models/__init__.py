"""Model definitions and utilities.

This module intentionally avoids importing torch-backed components at import time.
Torch wrappers are exposed via lazy attribute access so that CPU-only workflows
and tooling (e.g., `classiflow --version`, linting) do not require a working
torch installation.
"""

from __future__ import annotations

from typing import Any

from classiflow.models.estimators import get_estimators, get_param_grids
from classiflow.models.smote import AdaptiveSMOTE, apply_smote

__all__ = [
    "get_estimators",
    "get_param_grids",
    "AdaptiveSMOTE",
    "apply_smote",
    "resolve_device",
    "TorchMLP",
    "TorchMLPWrapper",
    "TorchLinearClassifier",
    "TorchMLPClassifier",
]


def resolve_device(name: str) -> str:
    """
    Resolve device name to available hardware.

    Mirrors the semantics of `classiflow.models.torch_mlp.resolve_device` but
    imports torch lazily.
    """
    if name == "auto":
        try:
            import torch
        except Exception:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if name in {"cuda", "mps"}:
        try:
            import torch
        except Exception:
            return "cpu"
        if name == "mps" and not torch.backends.mps.is_available():
            return "cpu"
        if name == "cuda" and not torch.cuda.is_available():
            return "cpu"

    return name


def __getattr__(name: str) -> Any:
    if name in {"TorchMLP", "TorchMLPWrapper"}:
        from classiflow.models.torch_mlp import TorchMLP, TorchMLPWrapper

        return {"TorchMLP": TorchMLP, "TorchMLPWrapper": TorchMLPWrapper}[name]
    if name in {"TorchLinearClassifier", "TorchMLPClassifier"}:
        from classiflow.models.torch_multiclass import TorchLinearClassifier, TorchMLPClassifier

        return {"TorchLinearClassifier": TorchLinearClassifier, "TorchMLPClassifier": TorchMLPClassifier}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)

