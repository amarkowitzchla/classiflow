from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def test_run_inference_exports_bag_member_metrics(tmp_path):
    from classiflow.inference import InferenceConfig, run_inference

    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df["sample_id"] = [f"sample_{idx:03d}" for idx in range(len(df))]
    df["label"] = iris.target_names[df["target"]]
    df = df.drop(columns=["target"])

    feature_cols = iris.feature_names
    train_df, test_df = train_test_split(
        df,
        test_size=0.3,
        stratify=df["label"],
        random_state=42,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                BaggingClassifier(
                    estimator=LogisticRegression(max_iter=500),
                    n_estimators=3,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(train_df[feature_cols], train_df["label"])

    run_dir = tmp_path / "trained_run"
    model_dir = run_dir / "fold1" / "multiclass_none"
    model_dir.mkdir(parents=True)
    joblib.dump(model, model_dir / "multiclass_model.joblib")
    pd.Series(sorted(train_df["label"].unique())).to_csv(
        model_dir / "classes.csv", index=False, header=False
    )
    pd.Series(feature_cols).to_csv(
        model_dir / "feature_list.csv", index=False, header=False
    )
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "bagged-run",
                "timestamp": "2024-01-15T12:00:00",
                "training_data_hash": "abc123",
                "config": {"label_col": "label"},
                "task_type": "multiclass",
                "feature_list": feature_cols,
            }
        )
    )

    test_path = tmp_path / "iris_test.csv"
    test_df.to_csv(test_path, index=False)

    output_dir = tmp_path / "inference_output"
    results = run_inference(
        InferenceConfig(
            run_dir=run_dir,
            data_csv=test_path,
            output_dir=output_dir,
            id_col="sample_id",
            label_col="label",
            include_plots=False,
            include_excel=False,
            verbose=0,
        )
    )

    assert "bagging" in results
    assert results["bagging"]["member_count"] == 3
    assert len(results["bagging"]["members"]) == 3
    assert (output_dir / "bagging_summary.json").exists()
    assert (output_dir / "metrics" / "bag_member_metrics.csv").exists()

    summary = json.loads((output_dir / "bagging_summary.json").read_text())
    assert summary["member_count"] == 3
    assert summary["metrics_csv_path"] == "metrics/bag_member_metrics.csv"

    member_metrics = pd.read_csv(output_dir / "metrics" / "bag_member_metrics.csv")
    assert list(member_metrics["member_index"]) == [1, 2, 3]
    assert "accuracy" in member_metrics.columns
    assert "agreement_with_ensemble" in member_metrics.columns
