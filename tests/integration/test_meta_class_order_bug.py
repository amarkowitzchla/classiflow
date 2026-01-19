"""
Test to expose class ordering bugs in meta-classifier inference.

This test creates scenarios where class ordering might differ between
training and inference, which can cause silent prediction errors.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import pytest
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

RANDOM_STATE = 42


def create_iris_datasets(tmp_path: Path) -> Tuple[Path, Path]:
    """Create train and test datasets from Iris."""
    np.random.seed(RANDOM_STATE)

    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target_names[iris.target], name="label")

    X["sample_id"] = [f"sample_{i:03d}" for i in range(len(X))]
    X["label"] = y

    train_df, test_df = train_test_split(
        X, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_path = tmp_path / "iris_train.csv"
    test_path = tmp_path / "iris_test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_path, test_path


def test_class_order_in_saved_vs_model():
    """
    Test that meta_classes.csv matches model.classes_ exactly.

    The bug: When saving meta_classes.csv, if we use list(meta_model.classes_)
    but classes_ is a numpy array, the order should be preserved. However,
    if we sort the classes or derive them differently, there's a mismatch.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        train_path, test_path = create_iris_datasets(tmp_path)

        from classiflow.config import MetaConfig
        from classiflow.training.meta import train_meta_classifier

        config = MetaConfig(
            data_csv=train_path,
            label_col="label",
            outdir=tmp_path / "training",
            outer_folds=2,
            inner_splits=2,
            inner_repeats=1,
            random_state=RANDOM_STATE,
            smote_mode="off",
            backend="sklearn",
            calibrate_meta=True,
            calibration_method="sigmoid",
            calibration_cv=2,
        )

        train_meta_classifier(config)

        # Load saved artifacts
        fold_dir = tmp_path / "training" / "fold1" / "binary_none"

        # Load meta_classes.csv
        meta_classes = pd.read_csv(
            fold_dir / "meta_classes.csv", header=None
        ).iloc[:, 0].astype(str).tolist()

        # Load model and get classes_
        meta_model = joblib.load(fold_dir / "meta_model.joblib")
        model_classes = [str(c) for c in meta_model.classes_]

        print(f"meta_classes.csv: {meta_classes}")
        print(f"model.classes_:   {model_classes}")

        # Critical assertion
        assert meta_classes == model_classes, \
            f"Class order mismatch!\n  CSV: {meta_classes}\n  Model: {model_classes}"


def test_predict_proba_column_order_matches_classes():
    """
    Test that predict_proba columns are ordered according to model.classes_

    sklearn's predict_proba returns columns in the order of model.classes_.
    If we label them using a different class list, predictions are wrong.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        train_path, test_path = create_iris_datasets(tmp_path)

        from classiflow.config import MetaConfig
        from classiflow.training.meta import train_meta_classifier
        from classiflow.inference import InferenceConfig, run_inference

        config = MetaConfig(
            data_csv=train_path,
            label_col="label",
            outdir=tmp_path / "training",
            outer_folds=2,
            inner_splits=2,
            inner_repeats=1,
            random_state=RANDOM_STATE,
            smote_mode="off",
            backend="sklearn",
            calibrate_meta=True,
            calibration_method="sigmoid",
            calibration_cv=2,
        )

        train_meta_classifier(config)

        # Run inference
        fold_dir = tmp_path / "training" / "fold1"
        infer_config = InferenceConfig(
            run_dir=fold_dir,
            data_csv=test_path,
            output_dir=tmp_path / "inference",
            id_col="sample_id",
            label_col="label",
            include_excel=False,
            include_plots=False,
            verbose=0,
        )

        results = run_inference(infer_config)
        predictions = results["predictions"]

        # Get probability columns
        proba_cols = [c for c in predictions.columns
                      if c.startswith("predicted_proba_") and c != "predicted_proba"]
        proba_classes = [c.replace("predicted_proba_", "") for c in proba_cols]

        # Get saved class order
        meta_classes = pd.read_csv(
            fold_dir / "binary_none" / "meta_classes.csv", header=None
        ).iloc[:, 0].astype(str).tolist()

        print(f"Probability column classes: {proba_classes}")
        print(f"Saved meta_classes:         {meta_classes}")

        # Columns should match saved order
        assert proba_classes == meta_classes, \
            f"Probability columns don't match saved class order"


def test_argmax_prediction_uses_correct_class():
    """
    Test that argmax(predict_proba) maps to the correct class label.

    Critical bug scenario:
    - model.classes_ = ['setosa', 'versicolor', 'virginica']
    - predict_proba returns [0.1, 0.8, 0.1]
    - argmax = 1 -> should be 'versicolor'

    If class ordering is wrong, argmax could return the wrong class.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        train_path, test_path = create_iris_datasets(tmp_path)

        from classiflow.config import MetaConfig
        from classiflow.training.meta import train_meta_classifier
        from classiflow.inference import InferenceConfig, run_inference

        config = MetaConfig(
            data_csv=train_path,
            label_col="label",
            outdir=tmp_path / "training",
            outer_folds=2,
            inner_splits=2,
            inner_repeats=1,
            random_state=RANDOM_STATE,
            smote_mode="off",
            backend="sklearn",
            calibrate_meta=True,
            calibration_method="sigmoid",
            calibration_cv=2,
        )

        train_meta_classifier(config)

        # Run inference
        fold_dir = tmp_path / "training" / "fold1"
        infer_config = InferenceConfig(
            run_dir=fold_dir,
            data_csv=test_path,
            output_dir=tmp_path / "inference",
            id_col="sample_id",
            label_col="label",
            include_excel=False,
            include_plots=False,
            verbose=0,
        )

        results = run_inference(infer_config)
        predictions = results["predictions"]

        # For each sample, verify argmax(proba) = predicted_label
        proba_cols = [c for c in predictions.columns
                      if c.startswith("predicted_proba_") and c != "predicted_proba"]
        proba_classes = [c.replace("predicted_proba_", "") for c in proba_cols]

        mismatches = 0
        for idx, row in predictions.iterrows():
            probas = [row[c] for c in proba_cols]
            argmax_idx = np.argmax(probas)
            argmax_class = proba_classes[argmax_idx]
            predicted = row["predicted_label"]

            if argmax_class != predicted:
                mismatches += 1
                print(f"Sample {idx}: argmax({proba_classes})={argmax_class}, but predicted={predicted}")

        assert mismatches == 0, f"{mismatches} samples have argmax != predicted_label"


def test_calibrated_model_classes_order():
    """
    Test that CalibratedClassifierCV preserves class ordering.

    sklearn's CalibratedClassifierCV should preserve the classes_ attribute
    from the underlying estimator. Verify this is true.
    """
    # Create simple 3-class problem
    np.random.seed(RANDOM_STATE)
    X = np.random.randn(100, 4)
    # Use specific class order
    y = np.array(['c_first', 'b_second', 'a_third'] * 33 + ['c_first'])

    # Fit base model
    base_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    base_model.fit(X, y)
    base_classes = list(base_model.classes_)

    # Fit calibrated model
    calibrated = CalibratedClassifierCV(base_model, method='sigmoid', cv=2)
    calibrated.fit(X, y)
    calibrated_classes = list(calibrated.classes_)

    print(f"Base model classes:       {base_classes}")
    print(f"Calibrated model classes: {calibrated_classes}")

    assert base_classes == calibrated_classes, \
        f"Calibration changed class order! Base: {base_classes}, Calibrated: {calibrated_classes}"


def test_meta_features_order_preserved():
    """
    Test that meta-feature order is preserved through training and inference.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        train_path, test_path = create_iris_datasets(tmp_path)

        from classiflow.config import MetaConfig
        from classiflow.training.meta import train_meta_classifier

        config = MetaConfig(
            data_csv=train_path,
            label_col="label",
            outdir=tmp_path / "training",
            outer_folds=2,
            inner_splits=2,
            inner_repeats=1,
            random_state=RANDOM_STATE,
            smote_mode="off",
            backend="sklearn",
            calibrate_meta=True,
            calibration_method="sigmoid",
            calibration_cv=2,
        )

        train_meta_classifier(config)

        # Load saved meta-features order
        fold_dir = tmp_path / "training" / "fold1" / "binary_none"
        meta_features = pd.read_csv(
            fold_dir / "meta_features.csv", header=None
        ).iloc[:, 0].astype(str).tolist()

        # Load best_models to get task order
        bundle = joblib.load(fold_dir / "binary_pipes.joblib")
        best_models = bundle.get("best_models", {})
        task_order = list(best_models.keys())
        expected_features = [f"{task}_score" for task in task_order]

        print(f"Saved meta_features:   {meta_features}")
        print(f"Expected from tasks:   {expected_features}")

        # Features should be in the same order as tasks
        assert set(meta_features) == set(expected_features), \
            f"Meta-feature set mismatch"


def test_inference_reorders_features_correctly():
    """
    Test that inference correctly extracts features in saved order,
    even if binary predictions have different column order.
    """
    from classiflow.inference.predict import MetaPredictor

    # Create mock meta model
    np.random.seed(RANDOM_STATE)
    X = pd.DataFrame({
        'task_A_score': [0.1, 0.9, 0.5],
        'task_B_score': [0.9, 0.1, 0.5],
        'task_C_score': [0.5, 0.5, 0.9],
    })
    y = ['class_1', 'class_2', 'class_3']

    model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    model.fit(X, y)

    # Saved feature order
    saved_features = ['task_A_score', 'task_B_score', 'task_C_score']
    saved_classes = list(model.classes_)

    predictor = MetaPredictor(
        meta_model=model,
        meta_features=saved_features,
        meta_classes=saved_classes,
    )

    # Binary predictions with DIFFERENT column order
    shuffled_predictions = pd.DataFrame({
        'task_C_score': [0.5, 0.5, 0.9],  # C first
        'task_A_score': [0.1, 0.9, 0.5],  # A second
        'task_B_score': [0.9, 0.1, 0.5],  # B last
    })

    # Predict should work correctly despite column order difference
    result = predictor.predict(shuffled_predictions)

    # Compare with original order
    original_predictions = predictor.predict(X)

    # Results should be identical
    assert list(result['predicted_label']) == list(original_predictions['predicted_label']), \
        f"Column reordering caused different predictions!"


if __name__ == "__main__":
    print("Running class order tests...\n")

    print("=" * 60)
    print("Test 1: Class order in saved vs model")
    print("=" * 60)
    test_class_order_in_saved_vs_model()
    print("PASSED\n")

    print("=" * 60)
    print("Test 2: Predict proba column order matches classes")
    print("=" * 60)
    test_predict_proba_column_order_matches_classes()
    print("PASSED\n")

    print("=" * 60)
    print("Test 3: Argmax prediction uses correct class")
    print("=" * 60)
    test_argmax_prediction_uses_correct_class()
    print("PASSED\n")

    print("=" * 60)
    print("Test 4: Calibrated model classes order")
    print("=" * 60)
    test_calibrated_model_classes_order()
    print("PASSED\n")

    print("=" * 60)
    print("Test 5: Meta features order preserved")
    print("=" * 60)
    test_meta_features_order_preserved()
    print("PASSED\n")

    print("=" * 60)
    print("Test 6: Inference reorders features correctly")
    print("=" * 60)
    test_inference_reorders_features_correctly()
    print("PASSED\n")

    print("\nAll tests passed!")
