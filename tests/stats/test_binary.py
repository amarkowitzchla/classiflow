"""Tests for binary stats pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from classiflow.stats.binary import binary_feature_tests
from classiflow.stats.normality import check_normality_all_features
from classiflow.stats.config import StatsConfig
from classiflow.stats import api as stats_api


def _make_binary_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    a = rng.normal(loc=0.0, scale=1.0, size=30)
    b = rng.normal(loc=1.2, scale=1.0, size=30)
    return pd.DataFrame(
        {
            "label": ["A"] * len(a) + ["B"] * len(b),
            "feat1": np.concatenate([a, b]),
        }
    )


def test_binary_ttest_path():
    """Binary mode selects Welch t-test when both classes pass normality."""
    df = _make_binary_df()
    features = ["feat1"]
    classes = ["A", "B"]

    _, normality_by_class = check_normality_all_features(
        df, features, "label", classes, alpha=0.001, min_n=3
    )
    results = binary_feature_tests(
        df,
        features,
        "label",
        classes,
        normality_by_class,
        alpha=0.001,
        min_n=3,
        p_adjust="holm",
    )

    assert results.loc[0, "test_type"] == "ttest_welch"
    assert results.loc[0, "p_value"] < 0.05
    assert np.isfinite(results.loc[0, "hedges_g"])
    assert np.isfinite(results.loc[0, "p_adj"])


def test_binary_mannwhitney_path():
    """Binary mode selects Mann–Whitney U for non-normal data."""
    rng = np.random.default_rng(1)
    a = rng.exponential(scale=1.0, size=25)
    b = rng.exponential(scale=2.0, size=25)
    df = pd.DataFrame(
        {
            "label": ["A"] * len(a) + ["B"] * len(b),
            "feat1": np.concatenate([a, b]),
        }
    )
    features = ["feat1"]
    classes = ["A", "B"]

    _, normality_by_class = check_normality_all_features(
        df, features, "label", classes, alpha=0.05, min_n=3
    )
    results = binary_feature_tests(
        df,
        features,
        "label",
        classes,
        normality_by_class,
        alpha=0.05,
        min_n=3,
        p_adjust="holm",
    )

    assert results.loc[0, "test_type"] == "mannwhitney"
    assert np.isfinite(results.loc[0, "rank_biserial"])
    assert np.isfinite(results.loc[0, "p_adj"])


def test_binary_stats_workbook_smoke(tmp_path):
    """Binary stats run writes expected workbook outputs."""
    df = _make_binary_df()
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    config = StatsConfig(
        data_csv=data_path,
        label_col="label",
        outdir=tmp_path,
        classes=["A", "B"],
        alpha=0.05,
        min_n=3,
        dunn_adjust="holm",
        feature_whitelist=None,
        feature_blacklist=None,
        top_n_features=5,
        write_legacy_csv=False,
        write_legacy_xlsx=False,
    )

    results = stats_api.run_stats_from_config(config)
    assert results["publication_xlsx"].exists()

    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.load_workbook(results["publication_xlsx"])
    try:
        assert "Run_Manifest" in wb.sheetnames
        assert "Pairwise_Summary" in wb.sheetnames
    finally:
        wb.close()


def test_dispatch_binary_pipeline(monkeypatch, tmp_path):
    """Binary class count routes to binary pipeline."""
    df = _make_binary_df()
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    called = {"binary": False}

    def fake_binary(*args, **kwargs):
        called["binary"] = True
        return pd.DataFrame(
            [
                {
                    "feature": "feat1",
                    "group1": "A",
                    "group2": "B",
                    "n1": 10,
                    "n2": 10,
                    "mean1": 0.0,
                    "mean2": 1.0,
                    "sd1": 1.0,
                    "sd2": 1.0,
                    "median1": 0.0,
                    "median2": 1.0,
                    "normality_p1": 0.5,
                    "normality_p2": 0.5,
                    "normality": "Normal",
                    "test_type": "ttest_welch",
                    "statistic": 1.0,
                    "p_value": 0.01,
                    "log2fc": 0.0,
                    "fc_center1": 0.0,
                    "fc_center2": 0.0,
                    "hedges_g": 0.5,
                    "rank_biserial": np.nan,
                    "delta_mean": -1.0,
                    "delta_median": -1.0,
                    "p_adj": 0.02,
                    "reject": True,
                }
            ]
        )

    def fake_write(outdir, *args, **kwargs):
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / "publication_stats.xlsx"
        path.write_text("stub")
        return path

    monkeypatch.setattr(stats_api, "binary_feature_tests", fake_binary)
    monkeypatch.setattr(stats_api, "write_publication_workbook", fake_write)
    monkeypatch.setattr(stats_api, "write_legacy_workbook", fake_write)

    config = StatsConfig(
        data_csv=data_path,
        label_col="label",
        outdir=tmp_path,
        classes=["A", "B"],
        alpha=0.05,
        min_n=3,
        dunn_adjust="holm",
        top_n_features=5,
        write_legacy_csv=False,
        write_legacy_xlsx=False,
    )

    results = stats_api.run_stats_from_config(config)
    assert called["binary"] is True
    assert "test_type" in results["pairwise_summary"].columns


def test_dispatch_multiclass_pipeline(monkeypatch, tmp_path):
    """Multiclass counts route to existing pipeline."""
    df = pd.DataFrame(
        {
            "label": ["A", "B", "C"] * 5,
            "feat1": np.arange(15, dtype=float),
        }
    )
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    def fake_binary(*args, **kwargs):
        raise AssertionError("binary pipeline should not be called")

    def fake_nonparam(*args, **kwargs):
        return [], []

    def fake_write(outdir, *args, **kwargs):
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / "publication_stats.xlsx"
        path.write_text("stub")
        return path

    monkeypatch.setattr(stats_api, "binary_feature_tests", fake_binary)
    monkeypatch.setattr(stats_api, "run_nonparametric_tests", fake_nonparam)
    monkeypatch.setattr(stats_api, "write_publication_workbook", fake_write)
    monkeypatch.setattr(stats_api, "write_legacy_workbook", fake_write)

    config = StatsConfig(
        data_csv=data_path,
        label_col="label",
        outdir=tmp_path,
        classes=["A", "B", "C"],
        alpha=0.05,
        min_n=3,
        dunn_adjust="holm",
        top_n_features=5,
        write_legacy_csv=False,
        write_legacy_xlsx=False,
    )

    results = stats_api.run_stats_from_config(config)
    assert results["n_classes"] == 3
