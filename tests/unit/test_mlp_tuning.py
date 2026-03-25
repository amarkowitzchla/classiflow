"""Unit tests for shared MLP tuning specs."""

from classiflow.models.mlp_tuning import build_torch_mlp_param_grid, prefix_param_grid


def test_basic_expanded_grid_includes_ccix_neighbors() -> None:
    grid = build_torch_mlp_param_grid("basic", expanded=True)

    assert grid["lr"] == [1e-4, 1e-3]
    assert grid["weight_decay"] == [1e-5, 1e-4]
    assert grid["hidden_dim"] == [256, 512]
    assert grid["n_layers"] == [3, 4]
    assert grid["dropout"] == [0.1, 0.2]
    assert grid["activation"] == ["relu"]
    assert grid["use_batchnorm"] == [True]
    assert grid["batch_size"] == [256, 512]
    assert grid["epochs"] == [50, 100]


def test_prefix_param_grid_adds_expected_prefix() -> None:
    prefixed = prefix_param_grid({"hidden_dim": [128], "epochs": [50]}, prefix="clf__")
    assert prefixed == {"clf__hidden_dim": [128], "clf__epochs": [50]}
