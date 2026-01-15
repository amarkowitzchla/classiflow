"""Unit tests for data compatibility assessment."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from classiflow.config import MetaConfig, HierarchicalConfig
from classiflow.io.compatibility import (
    assess_data_compatibility,
    print_compatibility_report,
    CompatibilityResult,
)


@pytest.fixture
def temp_csv():
    """Fixture to create temporary CSV files."""
    temp_files = []

    def _create_csv(data: pd.DataFrame) -> Path:
        """Create a temporary CSV file and return its path."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_path = Path(temp_file.name)
        data.to_csv(temp_path, index=False)
        temp_files.append(temp_path)
        return temp_path

    yield _create_csv

    # Cleanup
    for temp_file in temp_files:
        if temp_file.exists():
            temp_file.unlink()


class TestMetaCompatibility:
    """Tests for meta-classifier data compatibility."""

    def test_valid_meta_data(self, temp_csv):
        """Test that valid data passes compatibility check."""
        # Create valid data with 3 classes, 30 samples
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.randn(30),
            'feature2': np.random.randn(30),
            'feature3': np.random.randn(30),
            'diagnosis': ['ClassA'] * 10 + ['ClassB'] * 10 + ['ClassC'] * 10
        })

        csv_path = temp_csv(data)
        config = MetaConfig(data_csv=csv_path, label_col='diagnosis')

        result = assess_data_compatibility(config)

        assert result.is_compatible
        assert result.mode == 'meta'
        assert len(result.errors) == 0
        assert result.data_summary['n_samples'] == 30
        assert result.data_summary['n_features'] == 3
        assert result.data_summary['n_classes'] == 3

    def test_too_few_classes(self, temp_csv):
        """Test that binary data (2 classes) is rejected for meta mode."""
        data = pd.DataFrame({
            'feature1': np.random.randn(20),
            'feature2': np.random.randn(20),
            'diagnosis': ['ClassA'] * 10 + ['ClassB'] * 10
        })

        csv_path = temp_csv(data)
        config = MetaConfig(data_csv=csv_path, label_col='diagnosis')

        result = assess_data_compatibility(config)

        assert not result.is_compatible
        assert any('Too few classes' in err for err in result.errors)
        assert any('meta-classifier requires at least 3' in err for err in result.errors)

    def test_too_few_samples(self, temp_csv):
        """Test that datasets with < 10 samples are rejected."""
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'feature2': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'diagnosis': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
        })

        csv_path = temp_csv(data)
        config = MetaConfig(data_csv=csv_path, label_col='diagnosis')

        result = assess_data_compatibility(config)

        assert not result.is_compatible
        assert any('Too few samples' in err for err in result.errors)

    def test_single_sample_class(self, temp_csv):
        """Test that classes with only 1 sample are rejected."""
        data = pd.DataFrame({
            'feature1': np.random.randn(20),
            'feature2': np.random.randn(20),
            'diagnosis': ['ClassA'] * 10 + ['ClassB'] * 9 + ['ClassC']  # ClassC has only 1 sample
        })

        csv_path = temp_csv(data)
        config = MetaConfig(data_csv=csv_path, label_col='diagnosis')

        result = assess_data_compatibility(config)

        assert not result.is_compatible
        assert any('< 2 samples' in err for err in result.errors)

    def test_no_numeric_features(self, temp_csv):
        """Test that data with no numeric features is rejected."""
        data = pd.DataFrame({
            'text_col': ['a', 'b', 'c'] * 10,
            'category_col': ['x', 'y', 'z'] * 10,
            'diagnosis': ['ClassA'] * 10 + ['ClassB'] * 10 + ['ClassC'] * 10
        })

        csv_path = temp_csv(data)
        config = MetaConfig(data_csv=csv_path, label_col='diagnosis')

        result = assess_data_compatibility(config)

        assert not result.is_compatible
        assert any('No numeric feature columns found' in err for err in result.errors)

    def test_missing_label_column(self, temp_csv):
        """Test that missing label column is detected."""
        data = pd.DataFrame({
            'feature1': np.random.randn(30),
            'feature2': np.random.randn(30),
        })

        csv_path = temp_csv(data)
        config = MetaConfig(data_csv=csv_path, label_col='diagnosis')

        result = assess_data_compatibility(config)

        assert not result.is_compatible
        assert any('not found in CSV' in err for err in result.errors)

    def test_infinite_values(self, temp_csv):
        """Test that infinite values are detected."""
        data = pd.DataFrame({
            'feature1': [1, 2, np.inf, 4, 5] * 6,
            'feature2': np.random.randn(30),
            'diagnosis': ['ClassA'] * 10 + ['ClassB'] * 10 + ['ClassC'] * 10
        })

        csv_path = temp_csv(data)
        config = MetaConfig(data_csv=csv_path, label_col='diagnosis')

        result = assess_data_compatibility(config)

        assert not result.is_compatible
        assert any('infinite values' in err for err in result.errors)

    def test_class_imbalance_warning(self, temp_csv):
        """Test that severe class imbalance triggers warning."""
        data = pd.DataFrame({
            'feature1': np.random.randn(112),
            'feature2': np.random.randn(112),
            'diagnosis': ['ClassA'] * 100 + ['ClassB'] * 10 + ['ClassC'] * 2  # 50:5:1 ratio
        })

        csv_path = temp_csv(data)
        config = MetaConfig(data_csv=csv_path, label_col='diagnosis')

        result = assess_data_compatibility(config)

        assert result.is_compatible
        assert any('class imbalance' in warn.lower() for warn in result.warnings)
        assert any('SMOTE' in sug for sug in result.suggestions)

    def test_constant_features_warning(self, temp_csv):
        """Test that constant features trigger warning."""
        data = pd.DataFrame({
            'feature1': np.random.randn(30),
            'constant_feature': [5.0] * 30,  # Constant
            'diagnosis': ['ClassA'] * 10 + ['ClassB'] * 10 + ['ClassC'] * 10
        })

        csv_path = temp_csv(data)
        config = MetaConfig(data_csv=csv_path, label_col='diagnosis')

        result = assess_data_compatibility(config)

        assert result.is_compatible
        assert any('constant' in warn.lower() for warn in result.warnings)

    def test_missing_values_warning(self, temp_csv):
        """Test that missing values trigger warning."""
        data = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5] * 6,
            'feature2': np.random.randn(30),
            'diagnosis': ['ClassA'] * 10 + ['ClassB'] * 10 + ['ClassC'] * 10
        })

        csv_path = temp_csv(data)
        config = MetaConfig(data_csv=csv_path, label_col='diagnosis')

        result = assess_data_compatibility(config)

        assert result.is_compatible
        assert any('missing values' in warn.lower() for warn in result.warnings)

    def test_class_filtering(self, temp_csv):
        """Test that class filtering works correctly."""
        data = pd.DataFrame({
            'feature1': np.random.randn(40),
            'feature2': np.random.randn(40),
            'diagnosis': ['ClassA'] * 10 + ['ClassB'] * 10 + ['ClassC'] * 10 + ['ClassD'] * 10
        })

        csv_path = temp_csv(data)
        config = MetaConfig(
            data_csv=csv_path,
            label_col='diagnosis',
            classes=['ClassA', 'ClassB', 'ClassC']
        )

        result = assess_data_compatibility(config)

        assert result.is_compatible
        assert result.data_summary['n_samples'] == 30  # Only 3 classes kept
        assert result.data_summary['n_classes'] == 3


class TestHierarchicalCompatibility:
    """Tests for hierarchical classifier data compatibility."""

    def test_valid_hierarchical_data(self, temp_csv):
        """Test that valid hierarchical data passes compatibility check."""
        np.random.seed(42)
        data = pd.DataFrame({
            'svs_id': [f'patient_{i//3}' for i in range(30)],  # 10 patients, 3 samples each
            'feature1': np.random.randn(30),
            'feature2': np.random.randn(30),
            'tumor_type': ['TypeA'] * 15 + ['TypeB'] * 15,
            'subtype': ['SubA1'] * 8 + ['SubA2'] * 7 + ['SubB1'] * 8 + ['SubB2'] * 7,
        })

        csv_path = temp_csv(data)
        config = HierarchicalConfig(
            data_csv=csv_path,
            patient_col='svs_id',
            label_l1='tumor_type',
            label_l2='subtype'
        )

        result = assess_data_compatibility(config)

        assert result.is_compatible
        assert result.mode == 'hierarchical'
        assert len(result.errors) == 0
        assert result.data_summary['n_patients'] == 10
        assert result.data_summary['hierarchical']

    def test_single_level_hierarchical(self, temp_csv):
        """Test single-level (non-hierarchical) mode."""
        data = pd.DataFrame({
            'svs_id': [f'patient_{i//3}' for i in range(30)],
            'feature1': np.random.randn(30),
            'feature2': np.random.randn(30),
            'diagnosis': ['ClassA'] * 10 + ['ClassB'] * 10 + ['ClassC'] * 10,
        })

        csv_path = temp_csv(data)
        config = HierarchicalConfig(
            data_csv=csv_path,
            patient_col='svs_id',
            label_l1='diagnosis',
            label_l2=None  # Single-level
        )

        result = assess_data_compatibility(config)

        assert result.is_compatible
        assert not result.data_summary['hierarchical']

    def test_missing_patient_column(self, temp_csv):
        """Test that missing patient column is detected."""
        data = pd.DataFrame({
            'feature1': np.random.randn(30),
            'diagnosis': ['ClassA'] * 10 + ['ClassB'] * 10 + ['ClassC'] * 10,
        })

        csv_path = temp_csv(data)
        config = HierarchicalConfig(
            data_csv=csv_path,
            patient_col='svs_id',
            label_l1='diagnosis'
        )

        result = assess_data_compatibility(config)

        assert not result.is_compatible
        assert any('Patient column' in err and 'not found' in err for err in result.errors)

    def test_missing_l1_label(self, temp_csv):
        """Test that missing L1 label column is detected."""
        data = pd.DataFrame({
            'svs_id': [f'patient_{i}' for i in range(30)],
            'feature1': np.random.randn(30),
        })

        csv_path = temp_csv(data)
        config = HierarchicalConfig(
            data_csv=csv_path,
            patient_col='svs_id',
            label_l1='diagnosis'
        )

        result = assess_data_compatibility(config)

        assert not result.is_compatible
        assert any('L1 label column' in err and 'not found' in err for err in result.errors)

    def test_too_few_patients_for_cv(self, temp_csv):
        """Test that insufficient patients for CV is detected."""
        # Only 2 patients, but outer_folds=3
        data = pd.DataFrame({
            'svs_id': ['patient_1'] * 5 + ['patient_2'] * 5,
            'feature1': np.random.randn(10),
            'feature2': np.random.randn(10),
            'diagnosis': ['ClassA'] * 5 + ['ClassB'] * 5,
        })

        csv_path = temp_csv(data)
        config = HierarchicalConfig(
            data_csv=csv_path,
            patient_col='svs_id',
            label_l1='diagnosis',
            outer_folds=3
        )

        result = assess_data_compatibility(config)

        assert not result.is_compatible
        assert any('Too few patients' in err for err in result.errors)
        assert any('Reduce outer_folds' in sug for sug in result.suggestions)

    def test_insufficient_l2_per_branch_warning(self, temp_csv):
        """Test warning when L2 branches have insufficient classes."""
        # TypeA has only 1 L2 class (< min_l2_classes_per_branch=2)
        data = pd.DataFrame({
            'svs_id': [f'patient_{i//3}' for i in range(30)],
            'feature1': np.random.randn(30),
            'tumor_type': ['TypeA'] * 15 + ['TypeB'] * 15,
            'subtype': ['SubA1'] * 15 + ['SubB1'] * 8 + ['SubB2'] * 7,  # TypeA has 1 L2, TypeB has 2
        })

        csv_path = temp_csv(data)
        config = HierarchicalConfig(
            data_csv=csv_path,
            patient_col='svs_id',
            label_l1='tumor_type',
            label_l2='subtype',
            min_l2_classes_per_branch=2
        )

        result = assess_data_compatibility(config)

        assert result.is_compatible  # Still compatible, but with warnings
        assert any('L1 branches' in warn and 'L2 classes' in warn for warn in result.warnings)

    def test_file_not_found(self):
        """Test that missing file is detected."""
        config = MetaConfig(
            data_csv=Path('/nonexistent/file.csv'),
            label_col='diagnosis'
        )

        result = assess_data_compatibility(config)

        assert not result.is_compatible
        assert any('not found' in err for err in result.errors)


class TestCompatibilityResult:
    """Tests for CompatibilityResult dataclass."""

    def test_result_string_representation(self):
        """Test that result can be converted to string."""
        result = CompatibilityResult(
            is_compatible=True,
            mode='meta',
            warnings=['Warning 1'],
            errors=[],
            suggestions=['Suggestion 1'],
            data_summary={'n_samples': 30, 'n_features': 5, 'n_classes': 3}
        )

        result_str = str(result)

        assert 'COMPATIBLE' in result_str
        assert 'META MODE' in result_str
        assert 'Warning 1' in result_str
        assert 'Suggestion 1' in result_str

    def test_result_to_dict(self):
        """Test that result can be converted to dict."""
        result = CompatibilityResult(
            is_compatible=False,
            mode='hierarchical',
            errors=['Error 1', 'Error 2'],
            suggestions=['Fix 1']
        )

        result_dict = result.to_dict()

        assert result_dict['is_compatible'] is False
        assert result_dict['mode'] == 'hierarchical'
        assert len(result_dict['errors']) == 2

    def test_incompatible_result_format(self):
        """Test formatting of incompatible result."""
        result = CompatibilityResult(
            is_compatible=False,
            mode='meta',
            errors=['Critical error'],
            suggestions=['How to fix']
        )

        result_str = str(result)

        assert 'INCOMPATIBLE' in result_str
        assert 'Critical error' in result_str
        assert 'How to fix' in result_str


class TestPrintCompatibilityReport:
    """Tests for print_compatibility_report convenience function."""

    def test_print_compatibility_report(self, temp_csv, capsys):
        """Test that print function works and outputs results."""
        data = pd.DataFrame({
            'feature1': np.random.randn(30),
            'feature2': np.random.randn(30),
            'diagnosis': ['ClassA'] * 10 + ['ClassB'] * 10 + ['ClassC'] * 10
        })

        csv_path = temp_csv(data)
        config = MetaConfig(data_csv=csv_path, label_col='diagnosis')

        result = print_compatibility_report(config)

        assert result.is_compatible
        captured = capsys.readouterr()
        assert 'COMPATIBLE' in captured.out
        assert 'META MODE' in captured.out
