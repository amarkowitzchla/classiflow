"""Integration test for multiclass nested CV training."""

from pathlib import Path

import pandas as pd

from classiflow.config import MulticlassConfig
from classiflow.training.multiclass import train_multiclass_classifier


def test_train_multiclass_outputs(tmp_path):
    data_csv = Path(__file__).resolve().parents[2] / "data" / "iris_data.csv"
    outdir = tmp_path / "multiclass_run"

    config = MulticlassConfig(
        data_csv=data_csv,
        label_col="Species",
        outdir=outdir,
        outer_folds=2,
        inner_splits=2,
        inner_repeats=1,
        random_state=42,
        smote_mode="off",
        max_iter=500,
        device="cpu",
        estimator_mode="cpu_only",
    )

    train_multiclass_classifier(config)

    expected = [
        outdir / "run.json",
        outdir / "inner_cv_results.csv",
        outdir / "inner_cv_splits.csv",
        outdir / "outer_results.csv",
        outdir / "averaged_roc.png",
        outdir / "averaged_pr.png",
        outdir / "metrics_inner_cv.csv",
        outdir / "metrics_inner_cv_splits.csv",
        outdir / "metrics_outer_multiclass_eval.csv",
    ]
    for path in expected:
        assert path.exists(), f"Missing expected output: {path}"

    fold_dir = outdir / "fold1" / "multiclass_none"
    assert (fold_dir / "multiclass_model.joblib").exists()
    assert (fold_dir / "multiclass_model_name.txt").exists()
    assert (fold_dir / "classes.csv").exists()
    assert (fold_dir / "feature_list.csv").exists()
    assert (fold_dir / "confusion_matrix_fold1.png").exists()
    assert (fold_dir / "roc_multiclass_fold1.png").exists()
    assert (fold_dir / "pr_multiclass_fold1.png").exists()

    inner_df = pd.read_csv(outdir / "inner_cv_results.csv")
    for col in [
        "fold",
        "sampler",
        "task",
        "model_name",
        "rank_test_f1_macro",
        "mean_test_f1_macro",
        "std_test_f1_macro",
    ]:
        assert col in inner_df.columns

    split_df = pd.read_csv(outdir / "inner_cv_splits.csv")
    for col in ["task_model", "outer_fold", "inner_split", "Accuracy", "Balanced Accuracy", "F1 Macro", "F1 Weighted"]:
        assert col in split_df.columns

    outer_df = pd.read_csv(outdir / "outer_results.csv")
    for col in [
        "fold",
        "sampler",
        "phase",
        "task",
        "model_name",
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "f1_weighted",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "recall",
        "precision",
        "mcc",
        "roc_auc_ovr_macro",
    ]:
        assert col in outer_df.columns
