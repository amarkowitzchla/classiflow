"""Pytest configuration and fixtures."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


@pytest.fixture
def sample_binary_data():
    """Generate sample binary classification data."""
    np.random.seed(42)
    n_samples = 100
    n_features = 20

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    y = pd.Series(np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]))

    return X, y


@pytest.fixture
def sample_multiclass_data():
    """Generate sample multiclass data."""
    np.random.seed(42)
    n_samples = 150
    n_features = 20

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    y = pd.Series(np.random.choice(["A", "B", "C"], size=n_samples))

    return X, y


@pytest.fixture
def temp_outdir(tmp_path):
    """Provide temporary output directory."""
    outdir = tmp_path / "derived"
    outdir.mkdir()
    return outdir
