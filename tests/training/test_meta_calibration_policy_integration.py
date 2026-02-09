"""Integration-style test for meta calibration decision artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classiflow.config import MetaConfig
from classiflow.training.meta import _train_meta_model


def _ovr_tasks(classes: list[str]):
    tasks = {}
    for cls in classes:
        tasks[f"{cls}_vs_Rest"] = lambda y, pos=cls: (y == pos).astype(int)
    return tasks


def test_meta_calibration_auto_policy_writes_artifacts(tmp_path: Path):
    rng = np.random.default_rng(7)
    classes = ["A", "B", "C"]
    n_total = 90

    X = pd.DataFrame(
        rng.normal(size=(n_total, 6)),
        columns=[f"f{i}" for i in range(6)],
    )
    y = pd.Series(np.array(classes * (n_total // len(classes))), name="label")
    perm = rng.permutation(n_total)
    X = X.iloc[perm].reset_index(drop=True)
    y = y.iloc[perm].reset_index(drop=True)

    X_tr = X.iloc[:60]
    y_tr = y.iloc[:60]
    X_va = X.iloc[60:]
    y_va = y.iloc[60:]

    tasks = _ovr_tasks(classes)
    best_pipes = {}
    best_models = {}
    for task_name, labeler in tasks.items():
        y_bin = labeler(y_tr)
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=200, solver="liblinear", random_state=0)),
            ]
        )
        pipe.fit(X_tr, y_bin)
        key = f"{task_name}__lr"
        best_pipes[key] = pipe
        best_models[task_name] = "lr"

    config = MetaConfig(
        data_path=tmp_path / "dummy.csv",
        label_col="label",
        outdir=tmp_path,
        inner_splits=2,
        inner_repeats=1,
        random_state=11,
        smote_mode="off",
        calibration_enabled="auto",
        calibration_method="sigmoid",
        calibration_cv=2,
        calibration_bins=8,
        calibration_binning="quantile",
    )

    cv_inner = list(
        RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=11).split(X_tr, y_tr)
    )
    var_dir = tmp_path / "fold1" / "binary_none"
    var_dir.mkdir(parents=True, exist_ok=True)

    outer_meta_rows = []
    _, _, _, _, prob_quality = _train_meta_model(
        X_tr=X_tr,
        y_tr=y_tr,
        X_va=X_va,
        y_va=y_va,
        best_pipes=best_pipes,
        best_models=best_models,
        tasks=tasks,
        cv_inner=cv_inner,
        variant="none",
        fold=1,
        var_dir=var_dir,
        config=config,
        outer_meta_rows=outer_meta_rows,
        groups_tr=None,
        meta_estimators={
            "logreg": LogisticRegression(
                max_iter=300,
                class_weight="balanced",
                multi_class="auto",
                random_state=0,
            )
        },
        meta_param_grids={"logreg": {"C": [1.0]}},
    )

    summary = json.loads((var_dir / "calibration_summary.json").read_text(encoding="utf-8"))
    pq = summary["overall"]["probability_quality"]
    decision = pq["calibration_decision"]

    assert "uncalibrated" in pq
    assert "calibrated" in pq
    assert pq["final_variant"] == "uncalibrated"
    assert decision["enabled_final"] is False
    assert isinstance(decision["reasons"], list) and decision["reasons"]

    assert (var_dir / "calibration_curve.csv").exists()
    assert (var_dir / "calibration_curve_top1_uncalibrated.csv").exists()
    assert (var_dir / "calibration_curve_top1_calibrated.csv").exists()
    assert (var_dir / "base_ovr_proba_fold1.npz").exists()

    compat = pd.read_csv(var_dir / "calibration_curve.csv")
    expected = pd.read_csv(var_dir / "calibration_curve_top1_uncalibrated.csv")
    assert compat.equals(expected)
    assert prob_quality["final_variant"] == "uncalibrated"

    assert outer_meta_rows
    val_rows = [row for row in outer_meta_rows if row.get("phase") == "val"]
    assert val_rows
    assert "mcc" in val_rows[0]
    assert pd.notna(val_rows[0]["mcc"])
