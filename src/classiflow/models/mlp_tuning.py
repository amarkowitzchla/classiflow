"""Shared MLP tuning specifications."""

from __future__ import annotations

from typing import Dict, Iterable, Literal

MLPGridProfile = Literal["basic", "fast"]


def _unique_in_order(values: Iterable[object]) -> list[object]:
    seen: set[object] = set()
    ordered: list[object] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def build_torch_mlp_param_grid(
    profile: MLPGridProfile,
    *,
    expanded: bool = False,
) -> Dict[str, list]:
    """Return a shared torch MLP hyperparameter grid."""
    if profile not in {"basic", "fast"}:
        raise ValueError(f"Unsupported MLP grid profile: {profile}")

    if profile == "fast":
        epochs = [10] if not expanded else [25, 50]
        batch_sizes = [256] if not expanded else [128, 256]
        hidden_dims = [64] if not expanded else [256, 512]
        n_layers = [1] if not expanded else [2, 3]
        dropouts = [0.2] if not expanded else [0.1, 0.2, 0.3]
        learning_rates = [1e-3] if not expanded else [1e-4, 5e-4, 1e-3]
        weight_decays = [1e-4] if not expanded else [1e-6, 1e-5, 1e-4]
        activations = ["relu"] if not expanded else ["relu", "elu"]
        batchnorm = [False] if not expanded else [False, True]
    else:
        epochs = [100, 200] if not expanded else [50, 100, 200]
        batch_sizes = [256] if not expanded else [128, 256, 512]
        hidden_dims = [64, 128] if not expanded else [256, 512, 768]
        n_layers = [1, 2, 3] if not expanded else [2, 3, 4]
        dropouts = [0.2, 0.4] if not expanded else [0.1, 0.2, 0.3]
        learning_rates = [5e-4, 1e-3] if not expanded else [1e-4, 5e-4, 1e-3]
        weight_decays = [1e-4, 1e-3] if not expanded else [1e-6, 1e-5, 1e-4]
        activations = ["relu"] if not expanded else ["relu", "elu"]
        batchnorm = [False] if not expanded else [False, True]

    return {
        "lr": _unique_in_order(learning_rates),
        "weight_decay": _unique_in_order(weight_decays),
        "hidden_dim": _unique_in_order(hidden_dims),
        "n_layers": _unique_in_order(n_layers),
        "dropout": _unique_in_order(dropouts),
        "activation": _unique_in_order(activations),
        "use_batchnorm": _unique_in_order(batchnorm),
        "batch_size": _unique_in_order(batch_sizes),
        "epochs": _unique_in_order(epochs),
    }


def prefix_param_grid(
    grid: Dict[str, list],
    *,
    prefix: str,
) -> Dict[str, list]:
    """Prefix all grid keys with the given prefix."""
    return {f"{prefix}{key}": values for key, values in grid.items()}
