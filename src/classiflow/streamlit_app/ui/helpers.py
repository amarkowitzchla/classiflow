"""UI helper functions for Streamlit app."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import streamlit as st

RELEVANT_DIRS = [
    "best_task_sheets",
    "binary_rocs",
    "cv_summary",
    "feature_importance",
    "fold1",
    "fold2",
    "fold3",
    "stats_results",
    "umap",
    "viz",
]


@st.cache_data(show_spinner=False)
def list_outputs(derived: Path) -> Dict[str, List[str]]:
    """
    List output files organized by directory.

    Parameters
    ----------
    derived : Path
        Derived outputs directory

    Returns
    -------
    outputs : Dict[str, List[str]]
        Directory name -> list of relative file paths
    """
    out = {}
    if not derived.exists():
        return out

    for sub in derived.iterdir():
        if sub.is_dir() and sub.name in RELEVANT_DIRS:
            files = [str(p.relative_to(derived)) for p in sorted(sub.rglob("*")) if p.is_file()]
            if files:
                out[sub.name] = files
        elif sub.is_file():
            # Top-level files like metrics_*.csv
            out.setdefault("root", []).append(str(sub.name))

    return out
