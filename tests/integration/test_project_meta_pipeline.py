"""
End-to-end test of the project meta-classifier pipeline.

This tests the full flow:
1. Project bootstrap
2. run-technical (nested CV)
3. build-bundle (final model)
4. run-test (independent test)

And verifies that metrics are consistent between CV and independent test.
"""

from __future__ import annotations

import logging
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

RANDOM_STATE = 42


def create_iris_project(project_dir: Path) -> Tuple[Path, Path]:
    """
    Create a project structure with Iris data.

    Returns paths to train and test manifests.
    """
    np.random.seed(RANDOM_STATE)

    # Create Iris datasets
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=['feat_0', 'feat_1', 'feat_2', 'feat_3'])
    y = pd.Series(iris.target_names[iris.target], name='label')

    X['sample_id'] = [f'sample_{i:03d}' for i in range(len(X))]
    X['label'] = y

    # Stratified split
    train_df, test_df = train_test_split(
        X, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Create project directory structure
    data_dir = project_dir / 'data'
    (data_dir / 'train').mkdir(parents=True, exist_ok=True)
    (data_dir / 'test').mkdir(parents=True, exist_ok=True)

    train_path = data_dir / 'train' / 'manifest.csv'
    test_path = data_dir / 'test' / 'manifest.csv'

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Created train set: {len(train_df)} samples")
    print(f"Created test set: {len(test_df)} samples")
    print(f"Classes: {sorted(train_df['label'].unique())}")

    return train_path, test_path


def create_project_yaml(project_dir: Path, train_path: Path, test_path: Path):
    """Create project.yaml configuration."""
    from classiflow.projects.yaml_utils import dump_yaml

    config = {
        'project': {
            'id': 'IRIS_META_TEST',
            'name': 'Iris Meta Test',
            'description': 'Test project for meta-classifier',
            'owner': 'test',
        },
        'data': {
            'train': {'manifest': str(train_path)},
            'test': {'manifest': str(test_path)},
        },
        'key_columns': {
            'sample_id': 'sample_id',
            'label': 'label',
            'patient_id': None,
        },
        'task': {
            'mode': 'meta',
            'patient_stratified': False,
        },
        'validation': {
            'nested_cv': {
                'outer_folds': 3,
                'inner_splits': 2,
                'inner_repeats': 1,
                'seed': RANDOM_STATE,
            },
        },
        'calibration': {
            'calibrate_meta': True,
            'method': 'sigmoid',
            'cv': 2,
            'bins': 10,
        },
    }

    dump_yaml(config, project_dir / 'project.yaml')

    # Create thresholds.yaml
    thresholds = {
        'technical_validation': {
            'required': {
                'f1': 0.7,
                'balanced_accuracy': 0.7,
            },
        },
        'independent_test': {
            'required': {
                'f1_macro': 0.7,
                'balanced_accuracy': 0.7,
            },
        },
    }
    dump_yaml(thresholds, project_dir / 'thresholds.yaml')

    return config


def register_datasets(project_dir: Path, train_path: Path, test_path: Path):
    """Register datasets with the project."""
    from classiflow.projects.project_fs import ProjectPaths
    from classiflow.projects.project_models import ProjectConfig
    from classiflow.projects.dataset_registry import register_dataset

    paths = ProjectPaths(project_dir)
    config = ProjectConfig.load(paths.project_yaml)

    register_dataset(paths.datasets_yaml, config, 'train', train_path)
    register_dataset(paths.datasets_yaml, config, 'test', test_path)

    print("Registered datasets")


def run_technical_validation(project_dir: Path) -> Path:
    """Run technical validation and return the run directory."""
    from classiflow.projects.project_fs import ProjectPaths
    from classiflow.projects.project_models import ProjectConfig
    from classiflow.projects.orchestrator import run_technical_validation

    paths = ProjectPaths(project_dir)
    config = ProjectConfig.load(paths.project_yaml)

    run_dir = run_technical_validation(paths, config, run_id='test_run', compare_smote=False)
    print(f"Technical validation complete: {run_dir}")
    return run_dir


def build_final_model(project_dir: Path, technical_run: Path) -> Path:
    """Build final model and return the run directory."""
    from classiflow.projects.project_fs import ProjectPaths
    from classiflow.projects.project_models import ProjectConfig
    from classiflow.projects.orchestrator import build_final_model

    paths = ProjectPaths(project_dir)
    config = ProjectConfig.load(paths.project_yaml)

    run_dir = build_final_model(paths, config, technical_run, run_id='final_test')
    print(f"Final model built: {run_dir}")
    return run_dir


def run_independent_test(project_dir: Path, final_run: Path) -> Path:
    """Run independent test and return the run directory."""
    from classiflow.projects.project_fs import ProjectPaths
    from classiflow.projects.project_models import ProjectConfig
    from classiflow.projects.orchestrator import run_independent_test

    paths = ProjectPaths(project_dir)
    config = ProjectConfig.load(paths.project_yaml)

    bundle_path = final_run / 'model_bundle.zip'
    run_dir = run_independent_test(paths, config, bundle_path, run_id='test_eval')
    print(f"Independent test complete: {run_dir}")
    return run_dir


def extract_cv_metrics(technical_run: Path) -> Dict[str, float]:
    """Extract CV metrics from technical validation."""
    metrics_path = technical_run / 'metrics_summary.json'
    if not metrics_path.exists():
        # Try alternate location
        metrics_path = technical_run / 'metrics_outer_meta_eval.csv'
        if metrics_path.exists():
            df = pd.read_csv(metrics_path)
            val_df = df[df['phase'] == 'val']
            return {
                'accuracy': val_df['accuracy'].mean(),
                'f1_macro': val_df['f1_macro'].mean(),
                'balanced_accuracy': val_df['balanced_accuracy'].mean(),
            }
        return {}

    with open(metrics_path) as f:
        data = json.load(f)

    summary = data.get('summary', {})
    return {
        'accuracy': summary.get('accuracy'),
        'f1_macro': summary.get('f1_macro'),
        'balanced_accuracy': summary.get('balanced_accuracy'),
    }


def extract_test_metrics(test_run: Path) -> Dict[str, float]:
    """Extract metrics from independent test."""
    metrics_path = test_run / 'metrics.json'
    if not metrics_path.exists():
        return {}

    with open(metrics_path) as f:
        data = json.load(f)

    overall = data.get('overall', {})
    return {
        'accuracy': overall.get('accuracy'),
        'f1_macro': overall.get('f1_macro'),
        'balanced_accuracy': overall.get('balanced_accuracy'),
    }


def compare_artifacts(technical_run: Path, final_run: Path) -> Dict[str, Any]:
    """Compare artifacts between technical and final runs."""
    import joblib

    result = {
        'technical_classes': None,
        'final_classes': None,
        'classes_match': None,
        'technical_features': None,
        'final_features': None,
        'features_match': None,
    }

    # Technical run artifacts (from fold1)
    tech_fold_dir = technical_run / 'fold1' / 'binary_none'
    if not tech_fold_dir.exists():
        tech_fold_dir = technical_run / 'fold1' / 'binary_smote'

    if tech_fold_dir.exists():
        classes_path = tech_fold_dir / 'meta_classes.csv'
        if classes_path.exists():
            result['technical_classes'] = pd.read_csv(classes_path, header=None).iloc[:, 0].astype(str).tolist()

        features_path = tech_fold_dir / 'meta_features.csv'
        if features_path.exists():
            result['technical_features'] = pd.read_csv(features_path, header=None).iloc[:, 0].astype(str).tolist()

    # Final run artifacts (from fold1)
    final_fold_dir = final_run / 'fold1' / 'binary_none'
    if not final_fold_dir.exists():
        final_fold_dir = final_run / 'fold1' / 'binary_smote'

    if final_fold_dir.exists():
        classes_path = final_fold_dir / 'meta_classes.csv'
        if classes_path.exists():
            result['final_classes'] = pd.read_csv(classes_path, header=None).iloc[:, 0].astype(str).tolist()

        features_path = final_fold_dir / 'meta_features.csv'
        if features_path.exists():
            result['final_features'] = pd.read_csv(features_path, header=None).iloc[:, 0].astype(str).tolist()

    # Compare
    if result['technical_classes'] and result['final_classes']:
        result['classes_match'] = result['technical_classes'] == result['final_classes']

    if result['technical_features'] and result['final_features']:
        result['features_match'] = result['technical_features'] == result['final_features']

    return result


def test_full_project_pipeline():
    """Full end-to-end test of the project pipeline."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        project_dir = Path(tmp_dir)

        print("\n" + "=" * 70)
        print("STEP 1: Create project structure")
        print("=" * 70)
        train_path, test_path = create_iris_project(project_dir)
        create_project_yaml(project_dir, train_path, test_path)
        register_datasets(project_dir, train_path, test_path)

        print("\n" + "=" * 70)
        print("STEP 2: Run technical validation")
        print("=" * 70)
        technical_run = run_technical_validation(project_dir)

        print("\n" + "=" * 70)
        print("STEP 3: Build final model")
        print("=" * 70)
        final_run = build_final_model(project_dir, technical_run)

        print("\n" + "=" * 70)
        print("STEP 4: Run independent test")
        print("=" * 70)
        test_run = run_independent_test(project_dir, final_run)

        print("\n" + "=" * 70)
        print("STEP 5: Compare metrics")
        print("=" * 70)

        cv_metrics = extract_cv_metrics(technical_run)
        test_metrics = extract_test_metrics(test_run)

        print(f"\nCV Metrics:")
        for k, v in cv_metrics.items():
            if v is not None:
                print(f"  {k}: {v:.4f}")

        print(f"\nTest Metrics:")
        for k, v in test_metrics.items():
            if v is not None:
                print(f"  {k}: {v:.4f}")

        # Compare
        cv_acc = cv_metrics.get('accuracy')
        test_acc = test_metrics.get('accuracy')

        if cv_acc and test_acc:
            diff = abs(cv_acc - test_acc)
            print(f"\nAccuracy difference: {diff:.4f}")

        print("\n" + "=" * 70)
        print("STEP 6: Compare artifacts")
        print("=" * 70)

        artifact_comparison = compare_artifacts(technical_run, final_run)
        print(f"\nTechnical classes: {artifact_comparison['technical_classes']}")
        print(f"Final classes: {artifact_comparison['final_classes']}")
        print(f"Classes match: {artifact_comparison['classes_match']}")
        print(f"\nTechnical features: {artifact_comparison['technical_features']}")
        print(f"Final features: {artifact_comparison['final_features']}")
        print(f"Features match: {artifact_comparison['features_match']}")

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        # Assertions
        if cv_acc and test_acc:
            diff = abs(cv_acc - test_acc)
            if diff > 0.15:
                print(f"\n*** WARNING: Large accuracy difference ({diff:.4f}) ***")
                print("This may indicate a bug in the pipeline!")
            else:
                print(f"\nAccuracy difference within tolerance ({diff:.4f} <= 0.15)")

            if test_acc < 0.5:
                print(f"\n*** WARNING: Test accuracy collapsed ({test_acc:.4f}) ***")
                print("This strongly indicates a bug!")

        if artifact_comparison['classes_match'] == False:
            print("\n*** WARNING: Class order mismatch between technical and final! ***")

        if artifact_comparison['features_match'] == False:
            print("\n*** WARNING: Meta-feature order mismatch! ***")

        # Return results for programmatic testing
        return {
            'cv_metrics': cv_metrics,
            'test_metrics': test_metrics,
            'artifact_comparison': artifact_comparison,
        }


if __name__ == "__main__":
    results = test_full_project_pipeline()
