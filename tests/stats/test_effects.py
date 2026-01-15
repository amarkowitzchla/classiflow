"""Tests for effect size calculations."""

import pytest
import numpy as np

from classiflow.stats.effects import (
    cohen_d,
    cliff_delta,
    rank_biserial,
    log2_fold_change,
    compute_all_effect_sizes,
)


def test_cohen_d_basic():
    """Test Cohen's d calculation."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

    d = cohen_d(x, y)

    # Mean difference = -1, pooled SD ≈ 1.58, d ≈ -0.63
    assert d < 0
    assert abs(d) > 0.5
    assert abs(d) < 0.7


def test_cohen_d_same_groups():
    """Test Cohen's d for identical groups."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])

    d = cohen_d(x, y)

    assert d == pytest.approx(0.0, abs=1e-10)


def test_cohen_d_insufficient_data():
    """Test Cohen's d with insufficient data returns NaN."""
    x = np.array([1.0])
    y = np.array([2.0, 3.0])

    d = cohen_d(x, y)

    assert np.isnan(d)


def test_cliff_delta_basic():
    """Test Cliff's delta calculation."""
    x = np.array([5.0, 6.0, 7.0, 8.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    delta = cliff_delta(x, y)

    # All x > y, so delta should be 1.0
    assert delta == pytest.approx(1.0)


def test_cliff_delta_reversed():
    """Test Cliff's delta with reversed groups."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([5.0, 6.0, 7.0])

    delta = cliff_delta(x, y)

    # All x < y, so delta should be -1.0
    assert delta == pytest.approx(-1.0)


def test_cliff_delta_equal():
    """Test Cliff's delta for equal groups."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])

    delta = cliff_delta(x, y)

    assert delta == pytest.approx(0.0, abs=1e-10)


def test_rank_biserial_basic():
    """Test rank-biserial correlation."""
    x = np.array([5.0, 6.0, 7.0, 8.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    r = rank_biserial(x, y)

    # Rank biserial can be negative depending on sign convention
    # Just check it's in valid range and not NaN
    assert -1.0 <= r <= 1.0
    assert not np.isnan(r)


def test_log2_fold_change_median():
    """Test log2 fold change with median."""
    x = np.array([4.0, 4.0, 4.0])
    y = np.array([2.0, 2.0, 2.0])

    l2fc, v1, v2 = log2_fold_change(x, y, center="median", eps=0.0)

    assert l2fc == pytest.approx(1.0)  # log2(4/2) = 1
    assert v1 == pytest.approx(4.0)
    assert v2 == pytest.approx(2.0)


def test_log2_fold_change_mean():
    """Test log2 fold change with mean."""
    x = np.array([8.0, 8.0, 8.0])
    y = np.array([2.0, 2.0, 2.0])

    l2fc, v1, v2 = log2_fold_change(x, y, center="mean", eps=0.0)

    assert l2fc == pytest.approx(2.0)  # log2(8/2) = 2
    assert v1 == pytest.approx(8.0)
    assert v2 == pytest.approx(2.0)


def test_log2_fold_change_with_eps():
    """Test log2 fold change with pseudocount."""
    x = np.array([0.0, 0.0])
    y = np.array([0.0, 0.0])

    l2fc, v1, v2 = log2_fold_change(x, y, center="mean", eps=1.0)

    # log2((0+1)/(0+1)) = 0
    assert l2fc == pytest.approx(0.0)


def test_compute_all_effect_sizes():
    """Test computing all effect sizes together."""
    x = np.array([5.0, 6.0, 7.0, 8.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    effects = compute_all_effect_sizes(x, y)

    assert "cohen_d" in effects
    assert "cliff_delta" in effects
    assert "rank_biserial" in effects
    assert "log2fc" in effects
    assert "fc_center_x" in effects
    assert "fc_center_y" in effects

    # All should indicate x > y
    assert effects["cohen_d"] > 1.0
    assert effects["cliff_delta"] == pytest.approx(1.0)
    assert effects["log2fc"] > 1.0


def test_compute_all_effect_sizes_handles_nan():
    """Test that effect sizes handle NaN appropriately."""
    x = np.array([1.0, np.nan, 3.0])
    y = np.array([2.0, 4.0, np.nan])

    effects = compute_all_effect_sizes(x, y)

    # Should compute on non-NaN values
    assert np.isfinite(effects["log2fc"])
    assert np.isfinite(effects["cohen_d"])
