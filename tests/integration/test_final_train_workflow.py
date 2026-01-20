"""
Integration tests for the new final_train workflow.

These tests verify:
1. Final models are trained from scratch (not reusing fold pipelines)
2. Per-task configs are correctly extracted and used
3. Sanity checks detect degenerate models
4. The complete pipeline: run-technical -> build-bundle -> run-test
5. SMOTE option works correctly
6. Bundle contents are complete for inference
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def iris_data():
    """Create Iris-based train/test datasets."""
    np.random.seed(RANDOM_STATE)

    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=['feat_0', 'feat_1', 'feat_2', 'feat_3'])
    y = pd.Series(iris.target_names[iris.target], name='label')

    X['sample_id'] = [f'sample_{i:03d}' for i in range(len(X))]
    X['label'] = y

    train_df, test_df = train_test_split(
        X, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df


@pytest.fixture
def iris_project(iris_data, tmp_path):
    """Create a complete Iris project structure."""
    train_df, test_df = iris_data

    # Create directories
    data_dir = tmp_path / 'data'
    (data_dir / 'train').mkdir(parents=True)
    (data_dir / 'test').mkdir(parents=True)

    train_path = data_dir / 'train' / 'manifest.csv'
    test_path = data_dir / 'test' / 'manifest.csv'

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Create project.yaml
    from classiflow.projects.yaml_utils import dump_yaml

    config = {
        'project': {
            'id': 'IRIS_FINAL_TRAIN_TEST',
            'name': 'Iris Final Train Test',
            'description': 'Test project for final_train workflow',
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
                'outer_folds': 2,  # Reduced for faster tests
                'inner_folds': 2,
                'repeats': 1,
                'seed': RANDOM_STATE,
            },
        },
        'calibration': {
            'calibrate_meta': True,
            'method': 'sigmoid',
            'cv': 2,
            'bins': 10,
        },
        'final_model': {
            'sanity_min_std': 0.02,
            'sanity_max_mean_deviation': 0.15,
        },
    }

    dump_yaml(config, tmp_path / 'project.yaml')

    # Create thresholds.yaml
    thresholds = {
        'technical_validation': {
            'required': {
                'f1': 0.5,
                'balanced_accuracy': 0.5,
            },
        },
        'independent_test': {
            'required': {
                'f1_macro': 0.5,
                'balanced_accuracy': 0.5,
            },
        },
    }
    dump_yaml(thresholds, tmp_path / 'thresholds.yaml')

    return {
        'project_dir': tmp_path,
        'train_path': train_path,
        'test_path': test_path,
        'train_df': train_df,
        'test_df': test_df,
    }


# =============================================================================
# Unit Tests for final_train Module
# =============================================================================


class TestSelectedConfigExtraction:
    """Tests for extracting per-task configs from technical validation."""

    def test_extract_per_task_configs(self, tmp_path):
        """Test that per-task configurations are correctly extracted."""
        from classiflow.projects.final_train import (
            extract_selected_configs_from_technical_run,
        )

        # Create mock inner CV metrics with different best configs per task
        metrics_df = pd.DataFrame([
            {'fold': 1, 'sampler': 'none', 'task': 'setosa_vs_Rest', 'model_name': 'lr', 'mean_test_score': 0.95, 'C': 1.0},
            {'fold': 1, 'sampler': 'none', 'task': 'setosa_vs_Rest', 'model_name': 'svm', 'mean_test_score': 0.90, 'C': 0.1},
            {'fold': 1, 'sampler': 'none', 'task': 'versicolor_vs_Rest', 'model_name': 'lr', 'mean_test_score': 0.70, 'C': 1.0},
            {'fold': 1, 'sampler': 'none', 'task': 'versicolor_vs_Rest', 'model_name': 'svm', 'mean_test_score': 0.85, 'C': 0.1},
            {'fold': 1, 'sampler': 'none', 'task': 'virginica_vs_Rest', 'model_name': 'lr', 'mean_test_score': 0.80, 'C': 1.0},
            {'fold': 1, 'sampler': 'none', 'task': 'virginica_vs_Rest', 'model_name': 'svm', 'mean_test_score': 0.75, 'C': 0.1},
        ])

        tech_run = tmp_path / 'technical_run'
        tech_run.mkdir()
        metrics_df.to_csv(tech_run / 'metrics_inner_cv.csv', index=False)

        binary_configs, meta_config = extract_selected_configs_from_technical_run(
            tech_run, variant='none'
        )

        # Verify per-task selection
        assert 'setosa_vs_Rest' in binary_configs
        assert binary_configs['setosa_vs_Rest'].model_name == 'lr'  # 0.95 > 0.90

        assert 'versicolor_vs_Rest' in binary_configs
        assert binary_configs['versicolor_vs_Rest'].model_name == 'svm'  # 0.85 > 0.70

        assert 'virginica_vs_Rest' in binary_configs
        assert binary_configs['virginica_vs_Rest'].model_name == 'lr'  # 0.80 > 0.75

    def test_save_and_load_selected_configs(self, tmp_path):
        """Test round-trip save/load of selected configs."""
        from classiflow.projects.final_train import (
            SelectedBinaryConfig,
            SelectedMetaConfig,
            save_selected_configs,
            load_selected_configs,
        )

        binary_configs = {
            'TaskA_vs_Rest': SelectedBinaryConfig(
                task_name='TaskA_vs_Rest',
                model_name='lr',
                params={'C': 1.0},
                mean_score=0.9,
                sampler='none',
            ),
            'TaskB_vs_Rest': SelectedBinaryConfig(
                task_name='TaskB_vs_Rest',
                model_name='svm',
                params={'C': 0.1},
                mean_score=0.85,
                sampler='none',
            ),
        }

        meta_config = SelectedMetaConfig(
            model_name='LogisticRegression',
            params={'class_weight': 'balanced'},
            calibration_method='sigmoid',
            calibration_cv=3,
            calibration_bins=10,
        )

        # Save
        save_selected_configs(tmp_path, binary_configs, meta_config)

        # Verify files exist
        assert (tmp_path / 'registry' / 'selected_binary_configs.json').exists()
        assert (tmp_path / 'registry' / 'selected_meta_config.json').exists()

        # Load
        loaded_binary, loaded_meta = load_selected_configs(tmp_path / 'registry')

        # Verify
        assert len(loaded_binary) == 2
        assert loaded_binary['TaskA_vs_Rest'].model_name == 'lr'
        assert loaded_binary['TaskB_vs_Rest'].model_name == 'svm'
        assert loaded_meta.calibration_method == 'sigmoid'


class TestSanityChecks:
    """Tests for sanity check functionality."""

    def test_sanity_checks_detect_random_predictions(self):
        """Test that sanity checks catch near-random predictions."""
        from classiflow.projects.final_train import run_sanity_checks, validate_sanity_checks

        class RandomClassifier:
            def predict_proba(self, X):
                n = len(X)
                proba = 0.5 + np.random.randn(n) * 0.01  # Low variance
                return np.column_stack([1 - proba, proba])

        X = pd.DataFrame(np.random.randn(100, 10), columns=[f'f{i}' for i in range(10)])
        y = pd.Series(['A'] * 50 + ['B'] * 50)

        tasks = {'Test_vs_Rest': lambda y: (y == 'A').astype(int)}
        best_pipes = {'Test_vs_Rest__RandomModel': RandomClassifier()}
        best_models = {'Test_vs_Rest': 'RandomModel'}

        results = run_sanity_checks(
            best_pipes, best_models, X, y, tasks,
            min_std=0.02, max_mean_deviation=0.15
        )

        passed, failures = validate_sanity_checks(results)
        assert not passed, "Sanity checks should fail for random predictions"
        assert len(failures) > 0

    def test_sanity_checks_pass_for_good_predictions(self):
        """Test that sanity checks pass for discriminative predictions."""
        from classiflow.projects.final_train import run_sanity_checks, validate_sanity_checks

        class GoodClassifier:
            def predict_proba(self, X):
                n = len(X)
                proba = np.random.rand(n)  # High variance
                return np.column_stack([1 - proba, proba])

        X = pd.DataFrame(np.random.randn(100, 10), columns=[f'f{i}' for i in range(10)])
        y = pd.Series(['A'] * 50 + ['B'] * 50)

        tasks = {'Test_vs_Rest': lambda y: (y == 'A').astype(int)}
        best_pipes = {'Test_vs_Rest__GoodModel': GoodClassifier()}
        best_models = {'Test_vs_Rest': 'GoodModel'}

        results = run_sanity_checks(
            best_pipes, best_models, X, y, tasks,
            min_std=0.02, max_mean_deviation=0.15
        )

        passed, failures = validate_sanity_checks(results)
        assert passed, "Sanity checks should pass for good predictions"


# =============================================================================
# Integration Tests
# =============================================================================


class TestEndToEndWorkflow:
    """End-to-end integration tests for the new workflow."""

    def test_full_pipeline_meta_mode(self, iris_project):
        """Test complete pipeline: run-technical -> build-bundle -> run-test."""
        from classiflow.projects.project_fs import ProjectPaths
        from classiflow.projects.project_models import ProjectConfig
        from classiflow.projects.dataset_registry import register_dataset
        from classiflow.projects.orchestrator import (
            run_technical_validation,
            build_final_model,
            run_independent_test,
        )

        project_dir = iris_project['project_dir']
        paths = ProjectPaths(project_dir)
        config = ProjectConfig.load(paths.project_yaml)

        # Register datasets
        register_dataset(paths.datasets_yaml, config, 'train', iris_project['train_path'])
        register_dataset(paths.datasets_yaml, config, 'test', iris_project['test_path'])

        # 1. Run technical validation
        technical_run = run_technical_validation(
            paths, config, run_id='tech_test', compare_smote=False
        )

        # Verify technical validation artifacts
        assert (technical_run / 'run.json').exists()
        assert (technical_run / 'metrics_inner_cv.csv').exists()

        # 2. Build final model
        final_run = build_final_model(
            paths, config, technical_run,
            run_id='final_test', sampler='none'
        )

        # Verify final model artifacts
        assert (final_run / 'run.json').exists()
        assert (final_run / 'model_bundle.zip').exists()
        assert (final_run / 'sanity_checks.json').exists()
        assert (final_run / 'registry' / 'selected_binary_configs.json').exists()

        # Verify sanity checks passed
        sanity_data = json.loads((final_run / 'sanity_checks.json').read_text())
        failed_checks = [c for c in sanity_data if not c['passed'] and c['check_type'] != 'data']
        assert len(failed_checks) == 0, f"Sanity checks failed: {failed_checks}"

        # 3. Run independent test
        bundle_path = final_run / 'model_bundle.zip'
        test_run = run_independent_test(
            paths, config, bundle_path, run_id='test_eval'
        )

        # Verify test results
        assert (test_run / 'metrics.json').exists()

        metrics = json.loads((test_run / 'metrics.json').read_text())
        overall = metrics.get('overall', {})

        # Assert reasonable performance (Iris is easy)
        assert overall.get('accuracy', 0) > 0.7, "Accuracy should be > 0.7"
        assert overall.get('f1_macro', 0) > 0.6, "F1 macro should be > 0.6"

    def test_smote_option(self, iris_project):
        """Test that SMOTE option works correctly."""
        from classiflow.projects.project_fs import ProjectPaths
        from classiflow.projects.project_models import ProjectConfig
        from classiflow.projects.dataset_registry import register_dataset
        from classiflow.projects.orchestrator import (
            run_technical_validation,
            build_final_model,
        )

        project_dir = iris_project['project_dir']
        paths = ProjectPaths(project_dir)
        config = ProjectConfig.load(paths.project_yaml)

        # Register datasets
        register_dataset(paths.datasets_yaml, config, 'train', iris_project['train_path'])
        register_dataset(paths.datasets_yaml, config, 'test', iris_project['test_path'])

        # Run technical validation without SMOTE comparison (just validate SMOTE flag works)
        technical_run = run_technical_validation(
            paths, config, run_id='tech_smote', compare_smote=False
        )

        # Build final model with SMOTE flag
        final_run_smote = build_final_model(
            paths, config, technical_run,
            run_id='final_smote', sampler='smote'
        )

        # Verify SMOTE was used in final model config
        run_json = json.loads((final_run_smote / 'run.json').read_text())
        final_model_config = run_json.get('config', {}).get('final_model', {})
        assert final_model_config.get('sampler') == 'smote'

        # Verify bundle exists and is valid
        assert (final_run_smote / 'model_bundle.zip').exists()

    def test_bundle_contents_complete(self, iris_project):
        """Test that bundle contains all required artifacts."""
        import zipfile
        from classiflow.projects.project_fs import ProjectPaths
        from classiflow.projects.project_models import ProjectConfig
        from classiflow.projects.dataset_registry import register_dataset
        from classiflow.projects.orchestrator import (
            run_technical_validation,
            build_final_model,
        )

        project_dir = iris_project['project_dir']
        paths = ProjectPaths(project_dir)
        config = ProjectConfig.load(paths.project_yaml)

        # Register datasets
        register_dataset(paths.datasets_yaml, config, 'train', iris_project['train_path'])
        register_dataset(paths.datasets_yaml, config, 'test', iris_project['test_path'])

        # Run technical validation
        technical_run = run_technical_validation(
            paths, config, run_id='tech_bundle', compare_smote=False
        )

        # Build final model
        final_run = build_final_model(
            paths, config, technical_run,
            run_id='final_bundle', sampler='none'
        )

        bundle_path = final_run / 'model_bundle.zip'

        # Check bundle contents
        with zipfile.ZipFile(bundle_path, 'r') as zf:
            names = zf.namelist()

            # Required files
            assert 'run.json' in names
            assert 'version.txt' in names
            assert 'artifacts.json' in names

            # Schema and sanity files
            assert 'class_order.json' in names
            assert 'feature_schema.json' in names
            assert 'sanity_checks.json' in names

            # Registry files
            assert any('registry/selected_binary_configs.json' in n for n in names)

            # Fold artifacts
            fold_artifacts = [n for n in names if n.startswith('fold1/binary_')]
            assert len(fold_artifacts) > 0
            assert any('binary_pipes.joblib' in n for n in fold_artifacts)
            assert any('meta_model.joblib' in n for n in fold_artifacts)
            assert any('meta_features.csv' in n for n in fold_artifacts)
            assert any('meta_classes.csv' in n for n in fold_artifacts)


class TestConfigSourceOfTruth:
    """Tests for verifying config source of truth is correctly used."""

    def test_per_task_configs_used_not_global(self, iris_project):
        """Verify that per-task configs are used, not a global best config."""
        from classiflow.projects.project_fs import ProjectPaths
        from classiflow.projects.project_models import ProjectConfig
        from classiflow.projects.dataset_registry import register_dataset
        from classiflow.projects.orchestrator import run_technical_validation
        from classiflow.projects.final_train import (
            extract_selected_configs_from_technical_run,
        )

        project_dir = iris_project['project_dir']
        paths = ProjectPaths(project_dir)
        config = ProjectConfig.load(paths.project_yaml)

        # Register datasets
        register_dataset(paths.datasets_yaml, config, 'train', iris_project['train_path'])

        # Run technical validation
        technical_run = run_technical_validation(
            paths, config, run_id='tech_configs', compare_smote=False
        )

        # Extract configs
        binary_configs, _ = extract_selected_configs_from_technical_run(
            technical_run, variant='none'
        )

        # Verify we have per-task configs
        assert len(binary_configs) > 0

        # Check that tasks potentially have different configs
        # (may or may not differ depending on data, but structure should support it)
        for task_name, cfg in binary_configs.items():
            assert cfg.task_name == task_name
            assert cfg.model_name is not None
            assert cfg.mean_score is not None


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility."""

    def test_legacy_config_loading(self, tmp_path):
        """Test that projects without final_model config still work."""
        from classiflow.projects.project_models import ProjectConfig
        from classiflow.projects.yaml_utils import dump_yaml

        # Create legacy config without final_model section
        legacy_config = {
            'project': {
                'id': 'LEGACY_TEST',
                'name': 'Legacy Test',
            },
            'data': {
                'train': {'manifest': 'data/train.csv'},
            },
            'key_columns': {
                'label': 'label',
            },
            'task': {
                'mode': 'meta',
            },
        }

        config_path = tmp_path / 'project.yaml'
        dump_yaml(legacy_config, config_path)

        # Should load without error
        config = ProjectConfig.load(config_path)

        # final_model should have defaults
        assert config.final_model.sanity_min_std == 0.02
        assert config.final_model.train_from_scratch == True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
