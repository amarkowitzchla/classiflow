"""Main Streamlit application entry point."""

from pathlib import Path

import streamlit as st

from classiflow.streamlit_app.ui.helpers import list_outputs
from classiflow.streamlit_app.ui.style import use_theme

st.set_page_config(page_title="SubtypeML", page_icon="🔬", layout="wide")
use_theme()

st.title("SubtypeML")

# Project root = current working directory
root = Path.cwd()

# Ensure data/ and derived/ exist
DATA = root / "data"
DERIVED = root / "derived"

DATA.mkdir(parents=True, exist_ok=True)
DERIVED.mkdir(parents=True, exist_ok=True)

# Status
cols = st.columns(3)
with cols[0]:
    st.metric("data/", "✅ created" if DATA.exists() else "missing")
with cols[1]:
    st.metric("derived/", "✅ created" if DERIVED.exists() else "missing")
with cols[2]:
    st.metric("classiflow", "✅ package mode")

st.markdown("---")

st.subheader("Current outputs (auto-detected)")
items = list_outputs(DERIVED)
if not items:
    st.info("No outputs yet. Go to **Train Models** to get started.")
else:
    for group, files in items.items():
        with st.expander(group, expanded=False):
            for f in files:
                st.write(f)

st.markdown("---")
st.markdown(
    """
**Workflow overview**

1. **Train Models** → fit binary (OvR & pairwise) + meta-classifier, export metrics
2. **Statistics** → normality tests, parametric/nonparametric comparisons, post-hoc
3. **Visualizations** → UMAP, ROC/Confusion, volcano/box/heatmap, fold-changes
4. **Publication Exports** → build a crisp ZIP of spreadsheets and figures

**Running the app:**

```bash
# After installation with [app] extras:
streamlit run -m classiflow.streamlit_app.app

# Or from the installed package directory:
cd $(python -c "import classiflow.streamlit_app; print(classiflow.streamlit_app.__path__[0])")
streamlit run app.py
```
"""
)
