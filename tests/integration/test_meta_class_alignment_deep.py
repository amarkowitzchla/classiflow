"""
Deep test of class alignment through the meta-classifier pipeline.

This test manually traces through the entire pipeline to identify
any points where class ordering might become misaligned.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

RANDOM_STATE = 42


def create_controlled_dataset(tmp_path: Path) -> Tuple[Path, Path]:
    """
    Create dataset with known class order issues.

    The key is to have classes that would be differently sorted vs
    the order they appear in the data.
    """
    np.random.seed(RANDOM_STATE)

    # Create data with classes that DON'T sort to the same order
    # as they appear in the data
    n_per_class = 35
    X_data = []
    y_data = []

    # Create classes in a specific order that differs from sorted
    classes_in_data_order = ['Z_last', 'A_first', 'M_middle']

    for i, cls in enumerate(classes_in_data_order):
        # Create slightly separable features
        X_cls = np.random.randn(n_per_class, 4) + i * 2
        X_data.append(X_cls)
        y_data.extend([cls] * n_per_class)

    X = pd.DataFrame(
        np.vstack(X_data),
        columns=['feat_0', 'feat_1', 'feat_2', 'feat_3']
    )
    y = pd.Series(y_data, name='label')

    X['sample_id'] = [f'sample_{i:03d}' for i in range(len(X))]
    X['label'] = y

    # Shuffle to mix classes
    X = X.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # Stratified split
    train_df, test_df = train_test_split(
        X, test_size=0.30, stratify=X['label'], random_state=RANDOM_STATE
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_path = tmp_path / 'train.csv'
    test_path = tmp_path / 'test.csv'

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Created train set: {len(train_df)} samples")
    print(f"Created test set: {len(test_df)} samples")
    print(f"Classes in data: {list(X['label'].unique())}")
    print(f"Sorted classes: {sorted(X['label'].unique())}")

    return train_path, test_path


def trace_class_order_through_training(train_path: Path, training_dir: Path):
    """
    Trace exactly what class order is used at each stage of training.
    """
    from classiflow.config import MetaConfig
    from classiflow.training.meta import train_meta_classifier

    config = MetaConfig(
        data_csv=train_path,
        label_col='label',
        outdir=training_dir,
        outer_folds=2,
        inner_splits=2,
        inner_repeats=1,
        random_state=RANDOM_STATE,
        smote_mode='off',
        backend='sklearn',
        calibrate_meta=True,
        calibration_method='sigmoid',
        calibration_cv=2,
    )

    # Train
    train_meta_classifier(config)

    # Now trace through artifacts
    fold_dir = training_dir / 'fold1' / 'binary_none'

    print("\n=== CLASS ORDER TRACE ===")

    # 1. meta_classes.csv
    meta_classes = pd.read_csv(
        fold_dir / 'meta_classes.csv', header=None
    ).iloc[:, 0].astype(str).tolist()
    print(f"1. meta_classes.csv: {meta_classes}")

    # 2. model.classes_
    meta_model = joblib.load(fold_dir / 'meta_model.joblib')
    model_classes = [str(c) for c in meta_model.classes_]
    print(f"2. model.classes_: {model_classes}")

    # 3. What sorted(y_unique) would give
    train_df = pd.read_csv(train_path)
    sorted_classes = sorted(train_df['label'].unique())
    print(f"3. sorted(y_unique): {sorted_classes}")

    # 4. Check if there's a mismatch
    print(f"\n=== CONSISTENCY CHECK ===")
    print(f"meta_classes.csv == model.classes_: {meta_classes == model_classes}")
    print(f"meta_classes.csv == sorted: {meta_classes == sorted_classes}")

    # 5. Verify predict_proba column order (use meta-feature schema)
    meta_features_path = fold_dir / 'meta_features.csv'
    if meta_features_path.exists():
        meta_features = pd.read_csv(meta_features_path, header=None).iloc[:, 0].astype(str).tolist()
        X_test_meta = pd.DataFrame(0.0, index=train_df.index[:5], columns=meta_features)
    else:
        # Fallback: use raw features if meta schema is missing
        X_test_meta = train_df.drop(columns=['sample_id', 'label']).iloc[:5]
    y_proba = meta_model.predict_proba(X_test_meta)
    print(f"\n=== PREDICT_PROBA CHECK ===")
    print(f"Shape: {y_proba.shape}")
    print(f"Column order: {model_classes}")

    return {
        'meta_classes': meta_classes,
        'model_classes': model_classes,
        'sorted_classes': sorted_classes,
    }


def trace_class_order_through_inference(test_path: Path, training_dir: Path, inference_dir: Path):
    """
    Trace what class order is used during inference.
    """
    from classiflow.inference import InferenceConfig, run_inference
    from classiflow.inference.loader import ArtifactLoader

    fold_dir = training_dir / 'fold1'

    # 1. Load artifacts directly
    loader = ArtifactLoader(fold_dir)
    meta_model, meta_features, meta_classes, calibration_metadata = loader.load_meta_artifacts(variant='none')

    print("\n=== INFERENCE ARTIFACT LOAD ===")
    print(f"Loaded meta_classes: {meta_classes}")

    # 2. Run inference
    config = InferenceConfig(
        run_dir=fold_dir,
        data_csv=test_path,
        output_dir=inference_dir,
        id_col='sample_id',
        label_col='label',
        include_excel=False,
        include_plots=False,
        verbose=1,
    )

    results = run_inference(config)
    predictions = results['predictions']

    # 3. Check probability column order
    proba_cols = [c for c in predictions.columns
                  if c.startswith('predicted_proba_') and c != 'predicted_proba']
    proba_classes = [c.replace('predicted_proba_', '') for c in proba_cols]

    print(f"\n=== PREDICTION OUTPUT ===")
    print(f"Probability column classes: {proba_classes}")

    # 4. Check if predictions match ground truth
    test_df = pd.read_csv(test_path)
    y_true = test_df['label'].values
    y_pred = predictions['predicted_label'].values

    accuracy = (y_true == y_pred).mean()
    print(f"Accuracy: {accuracy:.4f}")

    # 5. Verify argmax matches predicted_label
    y_proba = predictions[proba_cols].values
    argmax_indices = np.argmax(y_proba, axis=1)
    argmax_classes = [proba_classes[i] for i in argmax_indices]

    mismatches = 0
    for i, (pred, argmax) in enumerate(zip(y_pred, argmax_classes)):
        if pred != argmax:
            mismatches += 1
            print(f"  MISMATCH at {i}: predicted_label={pred}, argmax={argmax}")

    print(f"\nArgmax vs predicted_label mismatches: {mismatches}")

    return {
        'loaded_meta_classes': meta_classes,
        'proba_classes': proba_classes,
        'accuracy': accuracy,
        'argmax_mismatches': mismatches,
    }


def verify_brier_score_correctness(test_path: Path, training_dir: Path, inference_dir: Path):
    """
    Manually verify Brier score calculation with explicit class alignment.
    """
    from classiflow.inference import InferenceConfig, run_inference
    from classiflow.metrics.calibration import compute_probability_quality

    fold_dir = training_dir / 'fold1'

    config = InferenceConfig(
        run_dir=fold_dir,
        data_csv=test_path,
        output_dir=inference_dir,
        id_col='sample_id',
        label_col='label',
        include_excel=False,
        include_plots=False,
        verbose=0,
    )

    results = run_inference(config)
    predictions = results['predictions']

    # Get classes and probabilities
    proba_cols = [c for c in predictions.columns
                  if c.startswith('predicted_proba_') and c != 'predicted_proba']
    class_order = [c.replace('predicted_proba_', '') for c in proba_cols]

    y_true = predictions['label'].astype(str).values
    y_pred = predictions['predicted_label'].astype(str).values
    y_proba = predictions[proba_cols].values

    print("\n=== BRIER SCORE VERIFICATION ===")
    print(f"Class order: {class_order}")
    print(f"y_proba shape: {y_proba.shape}")

    # Compute Brier score using the function
    metrics, _ = compute_probability_quality(
        y_true=y_true.tolist(),
        y_pred=y_pred.tolist(),
        y_proba=y_proba,
        classes=class_order,
        bins=10,
    )
    computed_brier = metrics['brier']

    # Manual Brier score computation
    y_bin = label_binarize(y_true, classes=class_order)
    manual_brier = float(np.mean(np.sum((y_proba - y_bin) ** 2, axis=1)))

    print(f"Computed Brier: {computed_brier:.6f}")
    print(f"Manual Brier:   {manual_brier:.6f}")

    # Check if y_bin is aligned with y_proba
    print("\n=== ALIGNMENT CHECK ===")
    for i in range(min(5, len(y_true))):
        print(f"  Sample {i}: true={y_true[i]}")
        print(f"    y_bin:   {y_bin[i]}")
        print(f"    y_proba: {[f'{p:.3f}' for p in y_proba[i]]}")

    return {
        'computed_brier': computed_brier,
        'manual_brier': manual_brier,
        'classes': class_order,
    }


def test_full_pipeline_class_alignment():
    """
    Full test tracing class order through entire pipeline.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        print("=" * 70)
        print("CREATING CONTROLLED DATASET")
        print("=" * 70)
        train_path, test_path = create_controlled_dataset(tmp_path)

        print("\n" + "=" * 70)
        print("TRACING TRAINING CLASS ORDER")
        print("=" * 70)
        training_dir = tmp_path / 'training'
        training_info = trace_class_order_through_training(train_path, training_dir)

        print("\n" + "=" * 70)
        print("TRACING INFERENCE CLASS ORDER")
        print("=" * 70)
        inference_dir = tmp_path / 'inference'
        inference_info = trace_class_order_through_inference(test_path, training_dir, inference_dir)

        print("\n" + "=" * 70)
        print("VERIFYING BRIER SCORE")
        print("=" * 70)
        brier_info = verify_brier_score_correctness(test_path, training_dir, inference_dir)

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        # Critical assertions
        assert training_info['meta_classes'] == training_info['model_classes'], \
            "meta_classes.csv != model.classes_"

        assert inference_info['loaded_meta_classes'] == inference_info['proba_classes'], \
            "Loaded meta_classes != probability column classes"

        assert inference_info['argmax_mismatches'] == 0, \
            f"argmax mismatches: {inference_info['argmax_mismatches']}"

        assert abs(brier_info['computed_brier'] - brier_info['manual_brier']) < 1e-6, \
            "Brier score mismatch"

        assert inference_info['accuracy'] > 0.5, \
            f"Accuracy collapsed: {inference_info['accuracy']}"

        print("\nAll checks passed!")


if __name__ == "__main__":
    test_full_pipeline_class_alignment()
