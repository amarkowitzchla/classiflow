"""Integration test for torch backend in binary nested CV."""

from pathlib import Path

import numpy as np
import pandas as pd

from classiflow.config import TrainConfig
from classiflow.training.binary import train_binary_task


def _write_binary_manifest(path: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    rows = 30
    df = pd.DataFrame({
        "feat1": rng.normal(size=rows),
        "feat2": rng.normal(size=rows),
        "feat3": rng.normal(size=rows),
        "label": np.where(rng.uniform(size=rows) > 0.5, "pos", "neg"),
    })
    df.to_csv(path, index=False)


def test_torch_backend_binary_outputs_match_schema(tmp_path: Path) -> None:
    data_csv = tmp_path / "binary.csv"
    _write_binary_manifest(data_csv)

    sklearn_out = tmp_path / "sklearn_run"
    torch_out = tmp_path / "torch_run"

    sklearn_config = TrainConfig(
        data_csv=data_csv,
        label_col="label",
        outdir=sklearn_out,
        outer_folds=2,
        inner_splits=2,
        inner_repeats=1,
        random_state=7,
        smote_mode="off",
        max_iter=200,
        backend="sklearn",
        model_set="default",
    )
    train_binary_task(sklearn_config)

    torch_config = TrainConfig(
        data_csv=data_csv,
        label_col="label",
        outdir=torch_out,
        outer_folds=2,
        inner_splits=2,
        inner_repeats=1,
        random_state=7,
        smote_mode="off",
        max_iter=200,
        backend="torch",
        device="cpu",
        model_set="torch_fast",
        torch_num_workers=0,
        torch_dtype="float32",
    )
    train_binary_task(torch_config)

    expected_files = [
        "run.json",
        "metrics_inner_cv.csv",
        "metrics_inner_cv_splits.csv",
        "metrics_outer_binary_eval.csv",
    ]
    for name in expected_files:
        assert (torch_out / name).exists(), f"Missing torch output: {name}"

    sklearn_inner = pd.read_csv(sklearn_out / "metrics_inner_cv.csv")
    torch_inner = pd.read_csv(torch_out / "metrics_inner_cv.csv")
    # Both backends should have common required columns (hyperparameter columns differ by backend)
    required_inner_cols = ["fold", "sampler", "task", "model_name", "rank_test_f1", "mean_test_f1", "std_test_f1"]
    for col in required_inner_cols:
        assert col in sklearn_inner.columns, f"sklearn missing column: {col}"
        assert col in torch_inner.columns, f"torch missing column: {col}"

    sklearn_outer = pd.read_csv(sklearn_out / "metrics_outer_binary_eval.csv")
    torch_outer = pd.read_csv(torch_out / "metrics_outer_binary_eval.csv")
    assert list(torch_outer.columns) == list(sklearn_outer.columns)
