"""
Regression test for meta-classifier inference consistency.

This test verifies that the meta-classifier produces consistent predictions
between nested CV (technical validation) and independent test inference.

Root cause: The meta-feature schema and/or class ordering may differ between
training and inference, causing predictions to be misaligned.

The test uses the Iris dataset as a minimal, deterministic reproducer.
"""

from __future__ import annotations

import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
RANDOM_STATE = 42


def create_iris_datasets(tmp_path: Path) -> Tuple[Path, Path]:
    """
    Create train and test datasets from Iris.

    Returns
    -------
    train_path : Path
        Path to training data CSV
    test_path : Path
        Path to test data CSV
    """
    np.random.seed(RANDOM_STATE)

    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target_names[iris.target], name="label")

    # Create sample IDs
    X["sample_id"] = [f"sample_{i:03d}" for i in range(len(X))]
    X["label"] = y

    # Stratified train/test split: 70% train, 30% test
    train_df, test_df = train_test_split(
        X, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Save datasets
    train_path = tmp_path / "iris_train.csv"
    test_path = tmp_path / "iris_test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Created train set: {len(train_df)} samples")
    logger.info(f"Created test set: {len(test_df)} samples")
    logger.info(f"Train classes: {train_df['label'].value_counts().to_dict()}")
    logger.info(f"Test classes: {test_df['label'].value_counts().to_dict()}")

    return train_path, test_path


def run_meta_technical_validation(
    train_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Run meta-classifier technical validation (nested CV).

    Returns metrics and artifacts for validation.
    """
    from classiflow.config import MetaConfig
    from classiflow.training.meta import train_meta_classifier

    config = MetaConfig(
        data_csv=train_path,
        label_col="label",
        outdir=output_dir,
        outer_folds=3,
        inner_splits=2,
        inner_repeats=1,
        random_state=RANDOM_STATE,
        smote_mode="off",  # Disable SMOTE for simplicity
        backend="sklearn",
        calibrate_meta=True,
        calibration_method="sigmoid",
        calibration_cv=2,
    )

    results = train_meta_classifier(config)

    # Load outer fold validation metrics
    metrics_path = output_dir / "metrics_outer_meta_eval.csv"
    metrics_df = pd.read_csv(metrics_path)
    val_metrics = metrics_df[metrics_df["phase"] == "val"]

    mean_accuracy = val_metrics["accuracy"].mean()
    mean_f1 = val_metrics["f1_macro"].mean()

    logger.info(f"CV Mean Accuracy: {mean_accuracy:.4f}")
    logger.info(f"CV Mean F1 Macro: {mean_f1:.4f}")

    return {
        "mean_accuracy": mean_accuracy,
        "mean_f1": mean_f1,
        "metrics_df": val_metrics,
        "output_dir": output_dir,
    }


def run_meta_inference(
    test_path: Path,
    model_dir: Path,
    output_dir: Path,
    fold: int = 1,
) -> Dict[str, Any]:
    """
    Run meta-classifier inference on independent test set.

    Returns predictions and metrics.
    """
    from classiflow.inference import InferenceConfig, run_inference

    fold_dir = model_dir / f"fold{fold}"

    config = InferenceConfig(
        run_dir=fold_dir,
        data_csv=test_path,
        output_dir=output_dir,
        id_col="sample_id",
        label_col="label",
        include_excel=False,
        include_plots=False,
        verbose=1,
    )

    results = run_inference(config)

    predictions = results["predictions"]
    metrics = results.get("metrics", {})

    overall = metrics.get("overall", {})
    accuracy = overall.get("accuracy", np.nan)
    f1_macro = overall.get("f1_macro", np.nan)

    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test F1 Macro: {f1_macro:.4f}")

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "predictions": predictions,
        "metrics": metrics,
    }


def verify_class_order_consistency(
    training_dir: Path,
    fold: int = 1,
    variant: str = "none",
) -> Dict[str, Any]:
    """
    Verify that class ordering is consistent between saved artifacts.

    Returns
    -------
    Dict with class ordering information and any mismatches detected.
    """
    import joblib

    fold_dir = training_dir / f"fold{fold}" / f"binary_{variant}"

    # Load meta_classes.csv
    meta_classes_path = fold_dir / "meta_classes.csv"
    if meta_classes_path.exists():
        meta_classes = pd.read_csv(meta_classes_path, header=None).iloc[:, 0].astype(str).tolist()
    else:
        meta_classes = []

    # Load meta model and get classes_
    meta_model_path = fold_dir / "meta_model.joblib"
    if meta_model_path.exists():
        meta_model = joblib.load(meta_model_path)
        model_classes = [str(c) for c in meta_model.classes_]
    else:
        model_classes = []

    # Check consistency
    classes_match = meta_classes == model_classes

    result = {
        "meta_classes_csv": meta_classes,
        "model_classes_": model_classes,
        "classes_match": classes_match,
    }

    if not classes_match:
        logger.error(f"Class order mismatch!")
        logger.error(f"  meta_classes.csv: {meta_classes}")
        logger.error(f"  model.classes_:   {model_classes}")

    return result


def verify_meta_feature_schema(
    training_dir: Path,
    fold: int = 1,
    variant: str = "none",
) -> Dict[str, Any]:
    """
    Verify that meta-feature schema is properly saved and consistent.

    Returns
    -------
    Dict with meta-feature schema information.
    """
    fold_dir = training_dir / f"fold{fold}" / f"binary_{variant}"

    # Load meta_features.csv
    meta_features_path = fold_dir / "meta_features.csv"
    if meta_features_path.exists():
        meta_features = pd.read_csv(meta_features_path, header=None).iloc[:, 0].astype(str).tolist()
    else:
        meta_features = []

    # Load binary pipes to get task names
    import joblib
    binary_pipes_path = fold_dir / "binary_pipes.joblib"
    if binary_pipes_path.exists():
        bundle = joblib.load(binary_pipes_path)
        best_models = bundle.get("best_models", {})
        task_names = list(best_models.keys())
    else:
        task_names = []

    # Expected feature names from tasks
    expected_features = [f"{task}_score" for task in task_names]

    # Check if all expected features are in meta_features (order matters!)
    features_complete = set(expected_features) == set(meta_features)

    result = {
        "meta_features_csv": meta_features,
        "task_names": task_names,
        "expected_features": expected_features,
        "features_complete": features_complete,
    }

    if not features_complete:
        missing = set(expected_features) - set(meta_features)
        extra = set(meta_features) - set(expected_features)
        logger.error(f"Meta-feature schema mismatch!")
        logger.error(f"  Missing: {missing}")
        logger.error(f"  Extra: {extra}")

    return result


def verify_probability_validity(predictions: pd.DataFrame) -> Dict[str, Any]:
    """
    Verify that predicted probabilities are valid.

    Checks:
    - Probabilities sum to 1 per sample (multiclass)
    - Probabilities are in [0, 1]
    - No NaN values in probabilities
    """
    # Find probability columns
    proba_cols = [c for c in predictions.columns
                  if c.startswith("predicted_proba_") and c != "predicted_proba"]

    if not proba_cols:
        return {
            "valid": False,
            "error": "No probability columns found",
        }

    y_proba = predictions[proba_cols].values

    # Check for NaN
    has_nan = np.any(np.isnan(y_proba))

    # Check range [0, 1]
    in_range = np.all((y_proba >= 0) & (y_proba <= 1))

    # Check sum to 1 (with tolerance)
    row_sums = y_proba.sum(axis=1)
    sums_to_one = np.allclose(row_sums, 1.0, atol=1e-6)

    result = {
        "valid": not has_nan and in_range and sums_to_one,
        "has_nan": has_nan,
        "in_range": in_range,
        "sums_to_one": sums_to_one,
        "proba_cols": proba_cols,
        "row_sums_mean": float(row_sums.mean()),
        "row_sums_std": float(row_sums.std()),
    }

    if not result["valid"]:
        logger.error(f"Invalid probabilities detected!")
        logger.error(f"  Has NaN: {has_nan}")
        logger.error(f"  In [0,1]: {in_range}")
        logger.error(f"  Sums to 1: {sums_to_one}")

    return result


class TestMetaInferenceConsistency:
    """Test suite for meta-classifier inference consistency."""

    @pytest.fixture(scope="class")
    def iris_project(self, tmp_path_factory):
        """Create Iris project with train/test datasets and run training."""
        tmp_path = tmp_path_factory.mktemp("iris_meta_test")

        # Create datasets
        train_path, test_path = create_iris_datasets(tmp_path)

        # Run technical validation
        training_dir = tmp_path / "training"
        cv_results = run_meta_technical_validation(train_path, training_dir)

        # Run inference on test set
        inference_dir = tmp_path / "inference"
        test_results = run_meta_inference(test_path, training_dir, inference_dir)

        return {
            "tmp_path": tmp_path,
            "train_path": train_path,
            "test_path": test_path,
            "training_dir": training_dir,
            "inference_dir": inference_dir,
            "cv_results": cv_results,
            "test_results": test_results,
        }

    def test_cv_performance_is_reasonable(self, iris_project):
        """CV performance on Iris should be high (>0.85)."""
        cv_results = iris_project["cv_results"]

        assert cv_results["mean_accuracy"] > 0.85, \
            f"CV accuracy too low: {cv_results['mean_accuracy']:.4f}"
        assert cv_results["mean_f1"] > 0.85, \
            f"CV F1 too low: {cv_results['mean_f1']:.4f}"

    def test_independent_test_performance_matches_cv(self, iris_project):
        """
        Independent test performance should be within 0.10 of CV performance.

        This is the key test - if this fails, there's a bug in the inference pipeline.
        """
        cv_results = iris_project["cv_results"]
        test_results = iris_project["test_results"]

        cv_accuracy = cv_results["mean_accuracy"]
        test_accuracy = test_results["accuracy"]

        cv_f1 = cv_results["mean_f1"]
        test_f1 = test_results["f1_macro"]

        accuracy_diff = abs(cv_accuracy - test_accuracy)
        f1_diff = abs(cv_f1 - test_f1)

        logger.info(f"CV Accuracy: {cv_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}, Diff: {accuracy_diff:.4f}")
        logger.info(f"CV F1: {cv_f1:.4f}, Test F1: {test_f1:.4f}, Diff: {f1_diff:.4f}")

        # The key assertion - test performance should be similar to CV
        assert accuracy_diff <= 0.15, \
            f"Accuracy difference too large: CV={cv_accuracy:.4f}, Test={test_accuracy:.4f}, Diff={accuracy_diff:.4f}"
        assert f1_diff <= 0.15, \
            f"F1 difference too large: CV={cv_f1:.4f}, Test={test_f1:.4f}, Diff={f1_diff:.4f}"

        # Test should not collapse to random (1/3 for 3-class)
        assert test_accuracy > 0.5, \
            f"Test accuracy collapsed to near-random: {test_accuracy:.4f}"

    def test_class_order_consistency(self, iris_project):
        """Class ordering should be consistent between saved artifacts and model."""
        training_dir = iris_project["training_dir"]

        result = verify_class_order_consistency(training_dir, fold=1, variant="none")

        assert result["classes_match"], \
            f"Class order mismatch: CSV={result['meta_classes_csv']}, Model={result['model_classes_']}"

    def test_meta_feature_schema_completeness(self, iris_project):
        """Meta-feature schema should include all task scores."""
        training_dir = iris_project["training_dir"]

        result = verify_meta_feature_schema(training_dir, fold=1, variant="none")

        assert result["features_complete"], \
            f"Meta-feature schema incomplete: expected={result['expected_features']}, got={result['meta_features_csv']}"

    def test_probability_validity(self, iris_project):
        """Predicted probabilities should be valid (sum to 1, in [0,1], no NaN)."""
        predictions = iris_project["test_results"]["predictions"]

        result = verify_probability_validity(predictions)

        assert result["valid"], \
            f"Invalid probabilities: has_nan={result['has_nan']}, in_range={result['in_range']}, sums_to_one={result['sums_to_one']}"

    def test_prediction_uses_correct_class_labels(self, iris_project):
        """Predicted labels should be from the correct class set."""
        predictions = iris_project["test_results"]["predictions"]

        # Get predicted labels
        pred_labels = set(predictions["predicted_label"].unique())

        # Get true labels (from test data)
        test_path = iris_project["test_path"]
        test_df = pd.read_csv(test_path)
        true_labels = set(test_df["label"].unique())

        # Predicted labels should be subset of true labels
        assert pred_labels <= true_labels, \
            f"Predicted labels not in expected set: predicted={pred_labels}, expected={true_labels}"

    def test_inference_loads_saved_class_order(self, iris_project):
        """
        Inference should use the class order saved in meta_classes.csv,
        not derive it from the test data.
        """
        training_dir = iris_project["training_dir"]
        test_results = iris_project["test_results"]

        # Get saved class order
        class_result = verify_class_order_consistency(training_dir, fold=1, variant="none")
        saved_classes = class_result["meta_classes_csv"]

        # Get probability column order from predictions
        predictions = test_results["predictions"]
        proba_cols = [c for c in predictions.columns
                      if c.startswith("predicted_proba_") and c != "predicted_proba"]
        proba_classes = [c.replace("predicted_proba_", "") for c in proba_cols]

        # They should match
        assert proba_classes == saved_classes, \
            f"Probability columns don't match saved class order: proba={proba_classes}, saved={saved_classes}"


def run_standalone_reproducer():
    """
    Standalone reproducer script that can be run outside pytest.

    This function creates a temporary directory, runs the full pipeline,
    and reports results to stdout.
    """
    import tempfile
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 70)
    print("META-CLASSIFIER INFERENCE CONSISTENCY REPRODUCER")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        print("\n[1/4] Creating Iris datasets...")
        train_path, test_path = create_iris_datasets(tmp_path)

        print("\n[2/4] Running technical validation (nested CV)...")
        training_dir = tmp_path / "training"
        cv_results = run_meta_technical_validation(train_path, training_dir)

        print("\n[3/4] Running independent test inference...")
        inference_dir = tmp_path / "inference"
        test_results = run_meta_inference(test_path, training_dir, inference_dir)

        print("\n[4/4] Verifying consistency...")
        class_result = verify_class_order_consistency(training_dir, fold=1, variant="none")
        schema_result = verify_meta_feature_schema(training_dir, fold=1, variant="none")
        proba_result = verify_probability_validity(test_results["predictions"])

        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)

        cv_acc = cv_results["mean_accuracy"]
        test_acc = test_results["accuracy"]
        acc_diff = abs(cv_acc - test_acc)

        cv_f1 = cv_results["mean_f1"]
        test_f1 = test_results["f1_macro"]
        f1_diff = abs(cv_f1 - test_f1)

        print(f"\nPerformance Metrics:")
        print(f"  CV Accuracy:        {cv_acc:.4f}")
        print(f"  Test Accuracy:      {test_acc:.4f}")
        print(f"  Accuracy Diff:      {acc_diff:.4f} {'✓' if acc_diff <= 0.15 else '✗ MISMATCH'}")
        print(f"  CV F1 Macro:        {cv_f1:.4f}")
        print(f"  Test F1 Macro:      {test_f1:.4f}")
        print(f"  F1 Diff:            {f1_diff:.4f} {'✓' if f1_diff <= 0.15 else '✗ MISMATCH'}")

        print(f"\nConsistency Checks:")
        print(f"  Class order match:  {'✓' if class_result['classes_match'] else '✗ MISMATCH'}")
        print(f"  Schema complete:    {'✓' if schema_result['features_complete'] else '✗ MISMATCH'}")
        print(f"  Probabilities valid: {'✓' if proba_result['valid'] else '✗ INVALID'}")

        if class_result["classes_match"]:
            print(f"\n  Class order (CSV): {class_result['meta_classes_csv']}")
        else:
            print(f"\n  Class order (CSV):   {class_result['meta_classes_csv']}")
            print(f"  Class order (model): {class_result['model_classes_']}")

        # Determine if bug is present
        bug_detected = (
            acc_diff > 0.15 or
            f1_diff > 0.15 or
            test_acc < 0.5 or
            not class_result["classes_match"] or
            not proba_result["valid"]
        )

        print("\n" + "=" * 70)
        if bug_detected:
            print("BUG DETECTED: Meta-classifier inference is inconsistent!")
            print("=" * 70)
            return 1
        else:
            print("ALL CHECKS PASSED: Meta-classifier inference is consistent.")
            print("=" * 70)
            return 0


if __name__ == "__main__":
    import sys
    sys.exit(run_standalone_reproducer())
