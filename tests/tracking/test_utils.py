"""Tests for tracking utility functions."""

import pytest
from dataclasses import dataclass
from pathlib import Path

from classiflow.tracking.utils import (
    flatten_dict,
    sanitize_metric_name,
    extract_loggable_params,
    summarize_metrics,
)


class TestFlattenDict:
    """Tests for flatten_dict function."""

    def test_flat_dict_unchanged(self):
        """Flat dict should be returned as-is."""
        d = {"a": 1, "b": 2}
        result = flatten_dict(d)
        assert result == {"a": 1, "b": 2}

    def test_nested_dict_flattened(self):
        """Nested dict should be flattened with / separator."""
        d = {"a": {"b": 1, "c": 2}, "d": 3}
        result = flatten_dict(d)
        assert result == {"a/b": 1, "a/c": 2, "d": 3}

    def test_deeply_nested_dict(self):
        """Deeply nested dict should be fully flattened."""
        d = {"a": {"b": {"c": 1}}}
        result = flatten_dict(d)
        assert result == {"a/b/c": 1}

    def test_custom_separator(self):
        """Custom separator should be used."""
        d = {"a": {"b": 1}}
        result = flatten_dict(d, sep=".")
        assert result == {"a.b": 1}

    def test_parent_key_prefix(self):
        """parent_key should prefix all keys."""
        d = {"a": 1}
        result = flatten_dict(d, parent_key="prefix")
        assert result == {"prefix/a": 1}

    def test_empty_dict(self):
        """Empty dict should return empty dict."""
        assert flatten_dict({}) == {}

    def test_mixed_values(self):
        """Mixed value types should be preserved."""
        d = {"a": 1, "b": "string", "c": True, "d": 1.5}
        result = flatten_dict(d)
        assert result == {"a": 1, "b": "string", "c": True, "d": 1.5}


class TestSanitizeMetricName:
    """Tests for sanitize_metric_name function."""

    def test_valid_name_unchanged(self):
        """Valid names should be unchanged."""
        assert sanitize_metric_name("accuracy") == "accuracy"
        assert sanitize_metric_name("f1_score") == "f1_score"
        assert sanitize_metric_name("roc_auc") == "roc_auc"

    def test_brackets_replaced(self):
        """Brackets should be replaced with underscores."""
        assert sanitize_metric_name("fold[1]") == "fold_1_"
        assert sanitize_metric_name("fold[0]/accuracy") == "fold_0_/accuracy"

    def test_spaces_replaced(self):
        """Spaces should be replaced with underscores."""
        assert sanitize_metric_name("f1 score") == "f1_score"

    def test_slashes_preserved(self):
        """Slashes should be preserved (valid in hierarchical metrics)."""
        assert sanitize_metric_name("fold1/accuracy") == "fold1/accuracy"

    def test_special_chars_removed(self):
        """Invalid special characters should be removed."""
        assert sanitize_metric_name("metric@name") == "metricname"
        assert sanitize_metric_name("metric#1") == "metric1"


class TestExtractLoggableParams:
    """Tests for extract_loggable_params function."""

    def test_dict_input(self):
        """Dict input should be converted."""
        d = {"a": 1, "b": "string"}
        result = extract_loggable_params(d)
        assert result == {"a": 1, "b": "string"}

    def test_path_converted_to_string(self):
        """Path objects should be converted to strings."""
        d = {"path": Path("/some/path")}
        result = extract_loggable_params(d)
        assert result["path"] == "/some/path"

    def test_none_converted_to_string(self):
        """None should be converted to 'None' string."""
        d = {"value": None}
        result = extract_loggable_params(d)
        assert result["value"] == "None"

    def test_list_converted_to_string(self):
        """Lists should be converted to comma-separated strings."""
        d = {"items": [1, 2, 3]}
        result = extract_loggable_params(d)
        assert result["items"] == "1,2,3"

    def test_empty_list_converted(self):
        """Empty list should be converted to '[]'."""
        d = {"items": []}
        result = extract_loggable_params(d)
        assert result["items"] == "[]"

    def test_nested_dict_flattened(self):
        """Nested dict should be flattened."""
        d = {"outer": {"inner": 1}}
        result = extract_loggable_params(d)
        assert result["outer/inner"] == 1

    def test_dataclass_input(self):
        """Dataclass input should be converted."""
        @dataclass
        class Config:
            lr: float = 0.01
            epochs: int = 100

        config = Config()
        result = extract_loggable_params(config)
        assert result["lr"] == 0.01
        assert result["epochs"] == 100

    def test_object_with_to_dict(self):
        """Object with to_dict method should be supported."""
        class ConfigLike:
            def to_dict(self):
                return {"a": 1, "b": 2}

        result = extract_loggable_params(ConfigLike())
        assert result == {"a": 1, "b": 2}

    def test_invalid_type_raises_error(self):
        """Invalid input type should raise TypeError."""
        with pytest.raises(TypeError):
            extract_loggable_params("not a dict")


class TestSummarizeMetrics:
    """Tests for summarize_metrics function."""

    def test_numeric_values_extracted(self):
        """Numeric values should be extracted."""
        d = {"accuracy": 0.95, "loss": 0.05}
        result = summarize_metrics(d)
        assert result == {"accuracy": 0.95, "loss": 0.05}

    def test_non_numeric_values_filtered(self):
        """Non-numeric values should be filtered out."""
        d = {"accuracy": 0.95, "model_name": "logistic_regression"}
        result = summarize_metrics(d)
        assert result == {"accuracy": 0.95}

    def test_booleans_filtered(self):
        """Boolean values should be filtered out."""
        d = {"accuracy": 0.95, "converged": True}
        result = summarize_metrics(d)
        assert result == {"accuracy": 0.95}

    def test_nested_metrics_flattened(self):
        """Nested metrics should be flattened."""
        d = {"fold1": {"accuracy": 0.95}, "fold2": {"accuracy": 0.96}}
        result = summarize_metrics(d)
        assert result == {"fold1/accuracy": 0.95, "fold2/accuracy": 0.96}

    def test_prefix_added(self):
        """Prefix should be added to metric names."""
        d = {"accuracy": 0.95}
        result = summarize_metrics(d, prefix="val_")
        assert result == {"val_accuracy": 0.95}

    def test_metric_names_sanitized(self):
        """Metric names should be sanitized."""
        d = {"fold[1] accuracy": 0.95}
        result = summarize_metrics(d)
        # Key should be sanitized
        assert "fold_1__accuracy" in result

    def test_integers_converted_to_float(self):
        """Integer values should be converted to float."""
        d = {"epochs": 100}
        result = summarize_metrics(d)
        assert result["epochs"] == 100.0
        assert isinstance(result["epochs"], float)
