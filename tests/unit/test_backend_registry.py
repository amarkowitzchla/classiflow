"""Unit tests for backend registry."""

from sklearn.ensemble import BaggingClassifier

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


def test_torch_registry_binary_bagged_wraps_estimators() -> None:
    spec = get_model_set(
        command="train-binary",
        backend="torch",
        model_set="torch_fast",
        device="cpu",
        final_estimator_strategy="bagged",
    )
    assert isinstance(spec["estimators"]["TorchMLP"], BaggingClassifier)
    assert "clf__estimator__hidden_dim" in spec["param_grids"]["TorchMLP"]


def test_torch_registry_meta_expanded_grid_includes_ccix_axes() -> None:
    spec = get_model_set(
        command="train-meta",
        backend="torch",
        model_set="torch_basic",
        device="cpu",
        expanded_mlp_tuning_grid=True,
    )
    grid = spec["meta_param_grids"]["TorchMLPMulticlass"]
    assert 512 in grid["hidden_dim"]
    assert 3 in grid["n_layers"]
    assert 0.2 in grid["dropout"]
    assert "elu" in grid["activation"]
    assert True in grid["use_batchnorm"]
    assert 1e-4 in grid["lr"]
    assert 1e-6 in grid["weight_decay"]
    assert 50 in grid["epochs"]
    assert 256 in grid["batch_size"]
