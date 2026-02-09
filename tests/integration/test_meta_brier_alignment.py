"""
Test that Brier score and other probability-based metrics are computed
with proper alignment between y_proba columns and class labels.

The key issue: When label_binarize(y_true, classes=X) is called,
the resulting binary matrix has columns in the order of X.
If y_proba has a different column order, the Brier score will be wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize

from classiflow.metrics.calibration import compute_probability_quality


def test_brier_with_aligned_classes():
    """Test Brier score with properly aligned classes."""
    # Ground truth
    y_true = ["A", "B", "C", "A", "B", "C"]

    # Perfect predictions (each sample has 1.0 for correct class)
    # Column order: A, B, C
    y_proba = np.array(
        [
            [1.0, 0.0, 0.0],  # A
            [0.0, 1.0, 0.0],  # B
            [0.0, 0.0, 1.0],  # C
            [1.0, 0.0, 0.0],  # A
            [0.0, 1.0, 0.0],  # B
            [0.0, 0.0, 1.0],  # C
        ]
    )

    classes = ["A", "B", "C"]

    # Expected: Brier = 0 for perfect predictions
    y_bin = label_binarize(y_true, classes=classes)
    expected_brier = float(np.mean(np.sum((y_proba - y_bin) ** 2, axis=1)))

    print(f"y_bin:\n{y_bin}")
    print(f"y_proba:\n{y_proba}")
    print(f"Expected Brier: {expected_brier}")

    assert np.isclose(expected_brier, 0.0), f"Expected 0, got {expected_brier}"

    # Now compute using the function
    y_pred = [classes[np.argmax(p)] for p in y_proba]
    metrics, _ = compute_probability_quality(y_true, y_pred, y_proba, classes)

    print(f"Computed Brier: {metrics['brier']}")
    assert np.isclose(metrics["brier"], 0.0), f"Function returned {metrics['brier']}"


def test_brier_with_misaligned_classes():
    """
    Test that demonstrates the bug: when classes are misaligned,
    Brier score becomes incorrect.
    """
    # Ground truth
    y_true = ["A", "B", "C", "A", "B", "C"]

    # Perfect predictions with column order: A, B, C
    y_proba = np.array(
        [
            [1.0, 0.0, 0.0],  # Should be A
            [0.0, 1.0, 0.0],  # Should be B
            [0.0, 0.0, 1.0],  # Should be C
            [1.0, 0.0, 0.0],  # Should be A
            [0.0, 1.0, 0.0],  # Should be B
            [0.0, 0.0, 1.0],  # Should be C
        ]
    )

    # But what if we pass classes in WRONG order?
    # This is what happens if meta_classes.csv has different order than y_proba columns
    wrong_classes = ["C", "A", "B"]  # WRONG ORDER!

    y_bin = label_binarize(y_true, classes=wrong_classes)
    wrong_brier = float(np.mean(np.sum((y_proba - y_bin) ** 2, axis=1)))

    print(f"\nWith WRONG class order:")
    print(f"y_bin (wrong order):\n{y_bin}")
    print(f"y_proba:\n{y_proba}")
    print(f"Wrong Brier: {wrong_brier}")

    # The wrong Brier should NOT be 0 because columns are misaligned
    assert not np.isclose(
        wrong_brier, 0.0
    ), f"Misaligned Brier should not be 0, but got {wrong_brier}"

    print(f"\nThis demonstrates that class order matters for Brier score!")


def test_brier_compute_function_uses_passed_classes():
    """
    Verify that compute_probability_quality uses the exact class order passed.
    """
    y_true = ["A", "B", "C", "A", "B", "C"]

    # Probabilities where column order is: A, B, C
    # And predictions are correct
    y_proba = np.array(
        [
            [0.8, 0.1, 0.1],  # A (correct)
            [0.1, 0.8, 0.1],  # B (correct)
            [0.1, 0.1, 0.8],  # C (correct)
            [0.7, 0.2, 0.1],  # A (correct)
            [0.2, 0.7, 0.1],  # B (correct)
            [0.1, 0.2, 0.7],  # C (correct)
        ]
    )

    y_pred = ["A", "B", "C", "A", "B", "C"]  # All correct

    # With correct class order
    correct_classes = ["A", "B", "C"]
    metrics_correct, _ = compute_probability_quality(y_true, y_pred, y_proba, correct_classes)
    print(f"\nCorrect order - Brier: {metrics_correct['brier']:.4f}")

    # With wrong class order
    wrong_classes = ["C", "A", "B"]
    metrics_wrong, _ = compute_probability_quality(y_true, y_pred, y_proba, wrong_classes)
    print(f"Wrong order - Brier: {metrics_wrong['brier']:.4f}")

    # The metrics should differ
    assert not np.isclose(
        metrics_correct["brier"], metrics_wrong["brier"]
    ), "Brier scores should differ with different class orders"

    # Correct order should give lower (better) Brier
    assert (
        metrics_correct["brier"] < metrics_wrong["brier"]
    ), f"Correct order should give lower Brier: {metrics_correct['brier']} vs {metrics_wrong['brier']}"


def test_inference_class_order_extraction():
    """
    Test that simulates what happens during inference:
    1. Predictions DataFrame has columns like predicted_proba_A, predicted_proba_B, predicted_proba_C
    2. We extract class order from column names
    3. We compute metrics using that order

    The bug would occur if:
    - Column names have one order
    - But y_proba has a different order
    """
    # Simulate prediction DataFrame
    predictions = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3"],
            "true_label": ["A", "B", "C"],
            "predicted_label": ["A", "B", "C"],
            # These column names define the class order
            "predicted_proba_A": [0.8, 0.1, 0.1],
            "predicted_proba_B": [0.1, 0.8, 0.1],
            "predicted_proba_C": [0.1, 0.1, 0.8],
        }
    )

    # Extract class order from column names (this is what _compute_metrics does)
    proba_cols = [c for c in predictions.columns if c.startswith("predicted_proba_")]
    class_order = [c.replace("predicted_proba_", "") for c in proba_cols]
    y_proba = predictions[proba_cols].values

    print(f"\nColumn names: {proba_cols}")
    print(f"Extracted class order: {class_order}")
    print(f"y_proba:\n{y_proba}")

    y_true = predictions["true_label"].tolist()
    y_pred = predictions["predicted_label"].tolist()

    # Compute metrics
    metrics, _ = compute_probability_quality(y_true, y_pred, y_proba, class_order)

    print(f"Brier: {metrics['brier']:.4f}")

    # Verify alignment manually
    y_bin = label_binarize(y_true, classes=class_order)
    manual_brier_sum = float(np.mean(np.sum((y_proba - y_bin) ** 2, axis=1)))
    manual_brier_mean = manual_brier_sum / y_proba.shape[1]

    assert np.isclose(
        metrics["brier_multiclass_sum"], manual_brier_sum
    ), f"Function brier_multiclass_sum {metrics['brier_multiclass_sum']} != manual {manual_brier_sum}"
    assert np.isclose(
        metrics["brier"], manual_brier_mean
    ), f"Function brier(alias=brier_recommended) {metrics['brier']} != manual mean {manual_brier_mean}"


def test_meta_predictor_class_order():
    """
    Test that MetaPredictor creates probability columns in the correct order.
    """
    from classiflow.inference.predict import MetaPredictor
    from sklearn.linear_model import LogisticRegression

    # Create a simple model with known class order
    X_train = pd.DataFrame(
        {
            "feat1": [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
            "feat2": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        }
    )
    y_train = ["ClassZ", "ClassA", "ClassM", "ClassZ", "ClassA", "ClassM"]

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    # sklearn sorts classes
    print(f"\nModel classes_: {list(model.classes_)}")
    # Should be ['ClassA', 'ClassM', 'ClassZ'] (alphabetical)

    # Create MetaPredictor with explicit class order
    # This should match what's saved in meta_classes.csv
    saved_classes = list(model.classes_)

    predictor = MetaPredictor(
        meta_model=model,
        meta_features=["feat1", "feat2"],
        meta_classes=saved_classes,
    )

    # Predict
    X_test = pd.DataFrame(
        {
            "feat1": [0.2, 0.8],
            "feat2": [0.3, 0.7],
        }
    )

    predictions = predictor.predict(X_test)

    # Check that probability columns match saved class order
    proba_cols = [
        c
        for c in predictions.columns
        if c.startswith("predicted_proba_") and c != "predicted_proba"
    ]
    actual_classes = [c.replace("predicted_proba_", "") for c in proba_cols]

    print(f"Saved classes: {saved_classes}")
    print(f"Probability column classes: {actual_classes}")

    assert (
        actual_classes == saved_classes
    ), f"Column order {actual_classes} doesn't match saved order {saved_classes}"


if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: Brier with aligned classes")
    print("=" * 60)
    test_brier_with_aligned_classes()
    print("PASSED\n")

    print("=" * 60)
    print("Test 2: Brier with misaligned classes")
    print("=" * 60)
    test_brier_with_misaligned_classes()
    print("PASSED\n")

    print("=" * 60)
    print("Test 3: Brier compute function uses passed classes")
    print("=" * 60)
    test_brier_compute_function_uses_passed_classes()
    print("PASSED\n")

    print("=" * 60)
    print("Test 4: Inference class order extraction")
    print("=" * 60)
    test_inference_class_order_extraction()
    print("PASSED\n")

    print("=" * 60)
    print("Test 5: MetaPredictor class order")
    print("=" * 60)
    test_meta_predictor_class_order()
    print("PASSED\n")

    print("\nAll tests passed!")
