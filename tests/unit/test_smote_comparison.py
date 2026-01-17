"""
Unit tests for SMOTE comparison module.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from classiflow.evaluation.smote_comparison import (
    SMOTEComparison,
    SMOTEComparisonResult,
)


@pytest.fixture
def sample_smote_data():
    """Create sample SMOTE training results."""
    np.random.seed(42)
    n_folds = 3
    n_tasks = 5

    rows = []
    for fold in range(1, n_folds + 1):
        for task in [f"task_{i}" for i in range(n_tasks)]:
            rows.append({
                "fold": fold,
                "task": task,
                "sampler": "smote",
                "accuracy": np.random.uniform(0.7, 0.9),
                "f1": np.random.uniform(0.65, 0.85),
                "roc_auc": np.random.uniform(0.75, 0.95),
                "precision": np.random.uniform(0.6, 0.9),
                "recall": np.random.uniform(0.6, 0.9),
            })

    return pd.DataFrame(rows)


@pytest.fixture
def sample_no_smote_data():
    """Create sample no-SMOTE training results (slightly lower performance)."""
    np.random.seed(43)
    n_folds = 3
    n_tasks = 5

    rows = []
    for fold in range(1, n_folds + 1):
        for task in [f"task_{i}" for i in range(n_tasks)]:
            rows.append({
                "fold": fold,
                "task": task,
                "sampler": "none",
                "accuracy": np.random.uniform(0.65, 0.85),
                "f1": np.random.uniform(0.6, 0.8),
                "roc_auc": np.random.uniform(0.7, 0.9),
                "precision": np.random.uniform(0.55, 0.85),
                "recall": np.random.uniform(0.55, 0.85),
            })

    return pd.DataFrame(rows)


@pytest.fixture
def smote_comparison(sample_smote_data, sample_no_smote_data):
    """Create SMOTEComparison instance."""
    return SMOTEComparison(
        smote_data=sample_smote_data,
        no_smote_data=sample_no_smote_data,
        model_type="meta",
    )


class TestSMOTEComparison:
    """Test SMOTEComparison class."""

    def test_initialization(self, sample_smote_data, sample_no_smote_data):
        """Test initialization of SMOTEComparison."""
        comparison = SMOTEComparison(
            smote_data=sample_smote_data,
            no_smote_data=sample_no_smote_data,
            model_type="binary",
        )

        assert comparison.model_type == "binary"
        assert comparison.n_folds == 3
        assert len(comparison.metric_columns) == 5
        assert "accuracy" in comparison.metric_columns
        assert "f1" in comparison.metric_columns

    def test_missing_fold_column(self):
        """Test error when fold column is missing."""
        smote_data = pd.DataFrame({"accuracy": [0.8, 0.9]})
        no_smote_data = pd.DataFrame({"accuracy": [0.7, 0.85]})

        with pytest.raises(ValueError, match="must have 'fold' column"):
            SMOTEComparison(smote_data, no_smote_data, "meta")

    def test_compute_statistics(self, smote_comparison):
        """Test statistical computation."""
        stats = smote_comparison.compute_statistics()

        assert "accuracy" in stats
        assert "f1" in stats
        assert "roc_auc" in stats

        # Check structure
        for metric, values in stats.items():
            assert "smote_mean" in values
            assert "no_smote_mean" in values
            assert "delta" in values
            assert "paired_t_pval" in values
            assert "wilcoxon_pval" in values
            assert "cohens_d" in values

            # Check reasonable values
            assert 0 <= values["smote_mean"] <= 1
            assert 0 <= values["no_smote_mean"] <= 1
            assert 0 <= values["paired_t_pval"] <= 1

    def test_compute_statistics_specific_metrics(self, smote_comparison):
        """Test computing statistics for specific metrics only."""
        stats = smote_comparison.compute_statistics(metrics=["f1", "accuracy"])

        assert len(stats) == 2
        assert "f1" in stats
        assert "accuracy" in stats
        assert "roc_auc" not in stats

    def test_detect_overfitting_none(self, smote_comparison):
        """Test overfitting detection when no overfitting present."""
        overfitting, metrics, reason = smote_comparison.detect_overfitting(
            primary_metric="f1",
            secondary_metric="roc_auc",
            delta_threshold=0.5,  # High threshold - unlikely to trigger
        )

        assert overfitting is False
        assert len(metrics) == 0
        assert reason is None

    def test_detect_overfitting_concurrent_drops(self):
        """Test overfitting detection with concurrent performance drops."""
        # Create data where SMOTE performs worse
        np.random.seed(42)
        smote_data = pd.DataFrame({
            "fold": [1, 2, 3],
            "f1": [0.6, 0.62, 0.61],  # Lower than no-SMOTE
            "roc_auc": [0.65, 0.67, 0.66],
        })
        no_smote_data = pd.DataFrame({
            "fold": [1, 2, 3],
            "f1": [0.7, 0.72, 0.71],  # Higher
            "roc_auc": [0.75, 0.77, 0.76],
        })

        comparison = SMOTEComparison(smote_data, no_smote_data, "meta")

        overfitting, metrics, reason = comparison.detect_overfitting(
            primary_metric="f1",
            secondary_metric="roc_auc",
            delta_threshold=0.03,
        )

        assert overfitting is True
        assert "f1" in metrics
        assert "roc_auc" in metrics
        assert reason is not None
        assert "concurrent drops" in reason.lower()

    def test_generate_recommendation_use_smote(self):
        """Test recommendation generation when SMOTE is better."""
        # Create data where SMOTE clearly better
        smote_data = pd.DataFrame({
            "fold": [1, 2, 3],
            "f1": [0.85, 0.87, 0.86],
        })
        no_smote_data = pd.DataFrame({
            "fold": [1, 2, 3],
            "f1": [0.70, 0.72, 0.71],
        })

        comparison = SMOTEComparison(smote_data, no_smote_data, "meta")
        stats = comparison.compute_statistics()

        recommendation, confidence, reasoning = comparison.generate_recommendation(
            stats, primary_metric="f1"
        )

        assert recommendation == "use_smote"
        assert confidence in ["high", "medium", "low"]
        assert len(reasoning) > 0

    def test_generate_recommendation_no_smote(self):
        """Test recommendation when no-SMOTE is better."""
        # Create data where no-SMOTE is better
        smote_data = pd.DataFrame({
            "fold": [1, 2, 3],
            "f1": [0.70, 0.72, 0.71],
        })
        no_smote_data = pd.DataFrame({
            "fold": [1, 2, 3],
            "f1": [0.85, 0.87, 0.86],
        })

        comparison = SMOTEComparison(smote_data, no_smote_data, "meta")
        stats = comparison.compute_statistics()

        recommendation, confidence, reasoning = comparison.generate_recommendation(
            stats, primary_metric="f1"
        )

        assert recommendation == "no_smote"
        assert len(reasoning) > 0

    def test_generate_recommendation_equivalent(self):
        """Test recommendation when performance is equivalent."""
        # Create data with identical performance
        smote_data = pd.DataFrame({
            "fold": [1, 2, 3, 4, 5],
            "f1": [0.80, 0.81, 0.80, 0.81, 0.80],
        })
        no_smote_data = pd.DataFrame({
            "fold": [1, 2, 3, 4, 5],
            "f1": [0.80, 0.81, 0.80, 0.81, 0.80],
        })

        comparison = SMOTEComparison(smote_data, no_smote_data, "meta")
        stats = comparison.compute_statistics()

        recommendation, confidence, reasoning = comparison.generate_recommendation(
            stats, primary_metric="f1", significance_level=0.05, min_effect_size=0.2
        )

        # With identical values, should be equivalent
        assert recommendation == "equivalent"
        assert confidence == "high"

    def test_generate_report(self, smote_comparison):
        """Test full report generation."""
        result = smote_comparison.generate_report(
            primary_metric="f1",
            secondary_metric="roc_auc",
            overfitting_threshold=0.03,
        )

        assert isinstance(result, SMOTEComparisonResult)
        assert result.model_type == "meta"
        assert result.n_folds == 3
        assert len(result.metrics) > 0
        assert result.recommendation in ["use_smote", "no_smote", "equivalent", "insufficient_data"]
        assert result.confidence in ["high", "medium", "low"]
        assert len(result.reasoning) > 0

    def test_save_report(self, smote_comparison, tmp_path):
        """Test saving reports to files."""
        result = smote_comparison.generate_report()

        files = smote_comparison.save_report(result, tmp_path)

        assert "txt" in files
        assert "json" in files
        assert "csv" in files

        # Check files exist
        assert files["txt"].exists()
        assert files["json"].exists()
        assert files["csv"].exists()

        # Check JSON is valid
        with open(files["json"]) as f:
            data = json.load(f)
            assert "model_type" in data
            assert "recommendation" in data

        # Check CSV has expected structure
        df = pd.read_csv(files["csv"])
        assert "metric" in df.columns
        assert "smote_mean" in df.columns
        assert "no_smote_mean" in df.columns
        assert "delta" in df.columns

    def test_from_directory_missing_files(self, tmp_path):
        """Test from_directory with missing files."""
        with pytest.raises(FileNotFoundError, match="No metrics files found"):
            SMOTEComparison.from_directory(tmp_path)

    def test_from_directory_success(self, tmp_path, sample_smote_data, sample_no_smote_data):
        """Test loading from directory structure."""
        # Create fold directories
        for fold in [1, 2, 3]:
            fold_dir = tmp_path / f"fold{fold}"
            fold_dir.mkdir()

            # Combine SMOTE and no-SMOTE for this fold
            fold_data = pd.concat([
                sample_smote_data[sample_smote_data["fold"] == fold],
                sample_no_smote_data[sample_no_smote_data["fold"] == fold],
            ], ignore_index=True)

            fold_data.to_csv(fold_dir / "metrics_outer_meta_eval.csv", index=False)

        # Load
        comparison = SMOTEComparison.from_directory(tmp_path, model_type="meta")

        assert comparison.n_folds == 3
        assert comparison.model_type == "meta"
        assert len(comparison.smote_data) > 0
        assert len(comparison.no_smote_data) > 0

    def test_from_directory_combined_multiclass(self, tmp_path, sample_smote_data, sample_no_smote_data):
        """Test loading combined multiclass metrics file."""
        combined = pd.concat(
            [
                sample_smote_data,
                sample_no_smote_data,
            ],
            ignore_index=True,
        )
        combined.to_csv(tmp_path / "metrics_outer_multiclass_eval.csv", index=False)

        comparison = SMOTEComparison.from_directory(tmp_path)

        assert comparison.n_folds == 3
        assert "accuracy" in comparison.metric_columns
        assert comparison.smote_data["sampler"].nunique() == 1
        assert comparison.no_smote_data["sampler"].nunique() == 1


class TestSMOTEComparisonResult:
    """Test SMOTEComparisonResult dataclass."""

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = SMOTEComparisonResult(
            model_type="meta",
            n_folds=3,
            metrics=["f1", "accuracy"],
            smote_means={"f1": 0.85, "accuracy": 0.80},
            no_smote_means={"f1": 0.80, "accuracy": 0.75},
            deltas={"f1": 0.05, "accuracy": 0.05},
            paired_t_pvalues={"f1": 0.03, "accuracy": 0.04},
            wilcoxon_pvalues={"f1": 0.04, "accuracy": 0.05},
            effect_sizes={"f1": 0.5, "accuracy": 0.4},
            recommendation="use_smote",
            confidence="high",
            reasoning=["SMOTE improves F1 significantly"],
        )

        result_dict = result.to_dict()

        assert result_dict["model_type"] == "meta"
        assert result_dict["n_folds"] == 3
        assert result_dict["recommendation"] == "use_smote"
        assert "f1" in result_dict["metrics"]

    def test_summary_text(self):
        """Test generating summary text."""
        result = SMOTEComparisonResult(
            model_type="binary",
            n_folds=5,
            metrics=["f1"],
            smote_means={"f1": 0.85},
            no_smote_means={"f1": 0.80},
            deltas={"f1": 0.05},
            paired_t_pvalues={"f1": 0.01},
            wilcoxon_pvalues={"f1": 0.02},
            effect_sizes={"f1": 0.6},
            recommendation="use_smote",
            confidence="high",
            reasoning=["Significant improvement"],
        )

        text = result.summary_text()

        assert "SMOTE VS NO-SMOTE COMPARISON SUMMARY" in text
        assert "Model Type: BINARY" in text
        assert "Folds: 5" in text
        assert "f1:" in text
        assert "USE SMOTE" in text
        assert "Confidence: HIGH" in text

    def test_summary_text_with_overfitting(self):
        """Test summary text includes overfitting information."""
        result = SMOTEComparisonResult(
            model_type="meta",
            n_folds=3,
            metrics=["f1", "roc_auc"],
            smote_means={"f1": 0.75, "roc_auc": 0.80},
            no_smote_means={"f1": 0.80, "roc_auc": 0.85},
            deltas={"f1": -0.05, "roc_auc": -0.05},
            paired_t_pvalues={"f1": 0.01, "roc_auc": 0.02},
            wilcoxon_pvalues={"f1": 0.02, "roc_auc": 0.03},
            effect_sizes={"f1": -0.5, "roc_auc": -0.4},
            overfitting_detected=True,
            overfitting_metrics=["f1", "roc_auc"],
            overfitting_reason="Concurrent drops detected",
            recommendation="no_smote",
            confidence="high",
            reasoning=["Overfitting detected"],
        )

        text = result.summary_text()

        assert "OVERFITTING DETECTED" in text
        assert "Concurrent drops detected" in text


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_fold(self):
        """Test with single fold (statistical tests may be limited)."""
        smote_data = pd.DataFrame({
            "fold": [1],
            "f1": [0.85],
        })
        no_smote_data = pd.DataFrame({
            "fold": [1],
            "f1": [0.80],
        })

        comparison = SMOTEComparison(smote_data, no_smote_data, "meta")
        stats = comparison.compute_statistics()

        # Should still compute means and deltas
        assert "f1" in stats
        assert not np.isnan(stats["f1"]["smote_mean"])
        assert not np.isnan(stats["f1"]["no_smote_mean"])

        # But p-values may be NaN with only 1 sample
        assert np.isnan(stats["f1"]["paired_t_pval"])

    def test_missing_values(self):
        """Test handling of missing values in metrics."""
        smote_data = pd.DataFrame({
            "fold": [1, 2, 3],
            "f1": [0.85, np.nan, 0.87],
        })
        no_smote_data = pd.DataFrame({
            "fold": [1, 2, 3],
            "f1": [0.80, 0.82, np.nan],
        })

        comparison = SMOTEComparison(smote_data, no_smote_data, "meta")
        stats = comparison.compute_statistics()

        # Should handle NaN gracefully - after removing NaN pairs, only fold1 remains
        assert "f1" in stats
        # With only 1 valid pair, stats should have means but NaN for tests
        assert not np.isnan(stats["f1"]["smote_mean"])
        assert not np.isnan(stats["f1"]["no_smote_mean"])
        assert np.isnan(stats["f1"]["paired_t_pval"])  # Need ≥2 samples for t-test

    def test_empty_metric_columns(self):
        """Test with no numeric columns."""
        smote_data = pd.DataFrame({
            "fold": [1, 2, 3],
            "task": ["A", "B", "C"],
        })
        no_smote_data = pd.DataFrame({
            "fold": [1, 2, 3],
            "task": ["A", "B", "C"],
        })

        comparison = SMOTEComparison(smote_data, no_smote_data, "meta")

        # Should have empty metric columns
        assert len(comparison.metric_columns) == 0

        # Statistics should be empty
        stats = comparison.compute_statistics()
        assert len(stats) == 0
