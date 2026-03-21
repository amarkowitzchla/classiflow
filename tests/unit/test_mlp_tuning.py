"""Unit tests for shared MLP tuning specs."""

from classiflow.models.mlp_tuning import build_torch_mlp_param_grid, prefix_param_grid


def test_basic_expanded_grid_includes_ccix_neighbors() -> None:
    grid = build_torch_mlp_param_grid("basic", expanded=True)

    assert 512 in grid["hidden_dim"]
    assert 3 in grid["n_layers"]
    assert 0.2 in grid["dropout"]
    assert "elu" in grid["activation"]
    assert True in grid["use_batchnorm"]
    assert 1e-4 in grid["lr"]
    assert 1e-6 in grid["weight_decay"]
    assert 50 in grid["epochs"]
    assert 256 in grid["batch_size"]


def test_prefix_param_grid_adds_expected_prefix() -> None:
    prefixed = prefix_param_grid({"hidden_dim": [128], "epochs": [50]}, prefix="clf__")
    assert prefixed == {"clf__hidden_dim": [128], "clf__epochs": [50]}
