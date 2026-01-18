"""Unit tests for backend registry."""

from classiflow.backends.registry import get_model_set


def test_torch_registry_binary_fast() -> None:
    spec = get_model_set(
        command="train-binary",
        backend="torch",
        model_set="torch_fast",
        device="cpu",
    )
    assert "TorchLogisticRegression" in spec["estimators"]
    assert "TorchMLP" in spec["estimators"]
    assert "TorchLogisticRegression" in spec["param_grids"]


def test_sklearn_registry_meta_default() -> None:
    spec = get_model_set(
        command="train-meta",
        backend="sklearn",
        model_set="default",
        meta_C_grid=[0.1, 1.0],
    )
    assert "LogisticRegression" in spec["base_estimators"]
    assert "MultinomialLogReg" in spec["meta_estimators"]
    assert spec["meta_param_grids"]["MultinomialLogReg"]["C"] == [0.1, 1.0]
